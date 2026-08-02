#!/usr/bin/env python3
"""
ProcTHOR dual-agent main entry.

Design goals:
- Reuse SpatialWorld single-agent provider / parser / environment stack.
- Support one shared model or two per-agent models via --agent1 / --agent2.
- Keep output artifacts compatible with benchmark usage (log.json + dual_episode_*.json).
- Ported from ai2thor/main.py: history-feedback, llm-history-feedback,
  partner-view and image-scaling helpers.
"""


"""
python -m mllm_base_agent.dual_agent.procthor.run_csv_benchmark \
  --csv "./experiments/csv/procthor/dual/Spatial-Annotation-procthor.csv" \
  --config "experiments/configs/procthor/dual/config_close_gpt-5.yaml"
"""

import argparse
import base64
import json
import os
import re
import sys
import time
from copy import deepcopy
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from actions.max_steps import derive_dual_golden_steps
from configs.procthor.load_config import load_config
from actions.parser import parse_action_string
from mllm_base_agent.llm.provider import get_vlm
from mllm_base_agent.environments.procthor.wrapper import ProcTHOREnvWrapper
from mllm_base_agent.tools.memory import MemoryLibrary
from evaluation.procthor.base import create_evaluator_from_config
from scripts.evaluate_actions_procthor import load_init_actions_for_task

from .prompts import get_dual_procthor_prompt


LOCAL_RETRY_CONFIG = {
    "max_retries": 3,
    "api_max_retries": 5,
    "retry_delay": 2,
    "api_retry_delay": 5,
}
MODEL_HISTORY_TURNS = 29


# Map logical agent id ("agent_1" / "agent_2") to AI2-THOR embodied agentId (0 / 1).
AGENT_TO_THOR_ID = {"agent_1": 0, "agent_2": 1}


# ---------------------------------------------------------------------------
# LLM History Analyzer Agent
# ---------------------------------------------------------------------------
#
# A second LLM agent (built from the SAME VLM as the actors) that *manages and
# analyzes* the acting agent's recent action-outcome history.  When enabled via
# ``--llm-history-feedback`` / ``dual_agent.llm_history_feedback``, the
# analyzer's per-step annotation REPLACES the raw
# "Your previous action: X / Result: FAILED - Y" text that ``history_feedback``
# injects into each history entry — turning low-level action/error strings into
# high-information, actionable guidance for the acting agent.
#
# IMPORTANT: this is a pure superset.  When the flag is OFF the existing
# ``history_feedback`` behavior is unchanged.

HISTORY_ANALYZER_SYSTEM_PROMPT = """You are the History Analysis Agent for an embodied ProcTHOR dual-agent task.
You receive the task instruction and the acting agent's recent step sequence. Each step has the action it attempted and the environment result (SUCCESS or an error string).

Your job: produce a concise, actionable per-step analysis that will REPLACE the raw "action + error" text fed back to the acting agent. For EACH step write ONE short sentence (<=30 words) that captures (1) what the step tried / achieved (or why it failed) and (2) the concrete takeaway for the next move.

Grounding rules (follow strictly):
- Distance + "not in view": if distance < 1.0m the object is CLOSE but off-screen -> takeaway is to ROTATE (RotateLeft/RotateRight) or LookUp/LookDown, NOT to move closer. If distance >= 1.0m, move closer (MoveAhead(Small/Medium/Large)) while keeping it in view.
- "No valid positions to place object": the agent is too close / standing on the target surface -> step back (MoveBack / MoveBack(Large)) or use a different receptacle.
- "already" clean/off/on/sliced/open/closed: that subgoal is ALREADY satisfied -> move on, do not repeat.
- agent/object "is blocking": go around (MoveLeft / MoveRight) or step back; do not repeat the blocked move.
- DONE rejected: success conditions NOT met -> the agent must personally perform a successful state-changing interaction; do not trust partner claims.
- Success: note which objective was met so the agent does not redo it.

Output format: STRICT JSON only — a JSON array of strings, one per input step, in the same order. No markdown, no commentary.

Example:
Input:
1. PickupObject(Egg) | ERROR: Egg not in view, distance: 0.8m
2. RotateRight | SUCCESS
3. PickupObject(Egg) | SUCCESS
Output:
["Tried PickupObject(Egg) but Egg within 1.0m yet not in view -> rotate to frame it, do NOT move closer.", "Rotated view -> success; re-locate the Egg before interacting.", "Picked up Egg -> success; Egg in hand, advance to next objective."]
"""


class HistoryAnalyzerAgent:
    """LLM-based history manager that turns raw action/error strings into
    concise, actionable per-step annotations.

    The analyzer is text-only (no images) so the extra API call per step stays
    cheap.  ``analyze`` returns ``(annotations, response)`` where
    ``annotations`` is a list aligned with the input history entries (entries
    may be ``None`` when unavailable) or ``None`` on any failure so the caller
    can fall back to the raw ``history_feedback`` text.  ``response`` is the raw
    VLM response (for token accounting) or ``None``.
    """

    def __init__(self, vlm: Any):
        self._vlm = vlm
        # Bounded cache: signature -> annotations.  An episode produces at most
        # ~2 * per_agent_max_steps distinct windows, so this stays small.
        self._cache: Dict[Tuple, List[Optional[str]]] = {}

    @staticmethod
    def _signature(entries: List[dict]) -> Tuple:
        return tuple(
            (
                int(e.get("step", 0) or 0),
                str(e.get("action_string", "") or ""),
                str(e.get("error_message", "") or ""),
            )
            for e in entries
        )

    @staticmethod
    def _build_user_prompt(entries: List[dict], task_prompt: str) -> str:
        lines = [f"Task: {task_prompt}", "", "Recent steps (oldest -> newest):"]
        for i, e in enumerate(entries, 1):
            action = str(e.get("action_string", "") or "(no action)")
            err = e.get("error_message")
            result = "SUCCESS" if not err else f"ERROR: {err}"
            lines.append(f"{i}. {action} | {result}")
        lines.append("")
        lines.append(
            "Return a JSON array of strings, one per step above, each <=30 words, "
            "in the same order."
        )
        return "\n".join(lines)

    @staticmethod
    def _parse_annotations(text: str, expected_len: int) -> Optional[List[Optional[str]]]:
        if not text:
            return None
        stripped = text.strip()
        # Strip markdown code fences if present.
        if stripped.startswith("```"):
            first_newline = stripped.find("\n")
            if first_newline != -1:
                stripped = stripped[first_newline + 1 :]
            if stripped.rstrip().endswith("```"):
                stripped = stripped.rstrip()[:-3]
        # Isolate the first JSON array in the response.
        start = stripped.find("[")
        end = stripped.rfind("]")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            data = json.loads(stripped[start : end + 1])
        except Exception:
            return None
        if not isinstance(data, list):
            return None
        annotations: List[Optional[str]] = []
        for item in data:
            if isinstance(item, str):
                annotations.append(item.strip() or None)
            else:
                annotations.append(str(item).strip() or None)
        # Align with expected length: pad / truncate defensively.
        if len(annotations) < expected_len:
            annotations.extend([None] * (expected_len - len(annotations)))
        elif len(annotations) > expected_len:
            annotations = annotations[:expected_len]
        return annotations

    def analyze(
        self, history_entries: List[dict], task_prompt: str
    ) -> Tuple[Optional[List[Optional[str]]], Any]:
        """Return ``(annotations, raw_response)`` aligned with ``history_entries``.

        ``annotations`` is ``None`` when the analyzer could not produce a usable
        result, so the caller can fall back to the raw history_feedback text.
        """
        if not history_entries:
            return None, None

        sig = self._signature(history_entries)
        if sig in self._cache:
            return self._cache[sig], None

        from mllm_base_agent.llm.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=HISTORY_ANALYZER_SYSTEM_PROMPT),
            HumanMessage(content=self._build_user_prompt(history_entries, task_prompt)),
        ]

        try:
            response = self._vlm.invoke(messages)
        except Exception as exc:
            print(f"⚠️  History analyzer VLM call failed: {exc}")
            return None, None

        text = response.content if hasattr(response, "content") else str(response)
        annotations = self._parse_annotations(text, len(history_entries))
        if annotations is None:
            print(
                f"⚠️  History analyzer produced unparseable output: {(text or '')[:200]}"
            )
            return None, response

        self._cache[sig] = annotations
        return annotations, response


# ---------------------------------------------------------------------------
# Token-usage accounting (kept compatible with benchmark log parsing)
# ---------------------------------------------------------------------------

def _normalize_token_usage(raw_usage: Optional[dict]) -> Dict[str, int]:
    usage = raw_usage or {}

    def to_int(value: Any) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    prompt_tokens = to_int(usage.get("prompt_tokens"))
    completion_tokens = to_int(usage.get("completion_tokens"))
    total_tokens = to_int(usage.get("total_tokens")) or prompt_tokens + completion_tokens
    api_calls = to_int(usage.get("api_calls")) or (1 if total_tokens else 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "api_calls": api_calls,
    }


def _extract_token_usage_from_response(response: Any) -> Dict[str, int]:
    if response is None:
        return _normalize_token_usage({})
    metadata = getattr(response, "response_metadata", None) or {}
    if isinstance(metadata, dict) and metadata.get("token_usage"):
        return _normalize_token_usage(metadata["token_usage"])
    usage_metadata = getattr(response, "usage_metadata", None)
    if isinstance(usage_metadata, dict):
        return _normalize_token_usage(usage_metadata)
    additional_kwargs = getattr(response, "additional_kwargs", None) or {}
    if isinstance(additional_kwargs, dict):
        return _normalize_token_usage(
            additional_kwargs.get("token_usage") or additional_kwargs.get("usage")
        )
    return _normalize_token_usage({})


def _accumulate_token_usage(state: dict, token_usage: Dict[str, int]) -> None:
    if "token_usage" not in state or not isinstance(state.get("token_usage"), dict):
        state["token_usage"] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "api_calls": 0,
        }
    usage = state["token_usage"]
    normalized = _normalize_token_usage(token_usage)
    for key in ("prompt_tokens", "completion_tokens", "total_tokens", "api_calls"):
        usage[key] = int(usage.get(key, 0) or 0) + normalized[key]


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _resolve_image_scale(config: dict) -> float:
    """Return the image downscale factor from config (``image.scale``).

    Defaults to ``1.0`` (no scaling). Values are clamped to ``(0, 1.0]``.
    """
    try:
        scale = float(config.get("image", {}).get("scale", 1.0))
    except (TypeError, ValueError):
        return 1.0
    if scale <= 0.0 or scale > 1.0:
        return 1.0
    return scale


def _resolve_image_recent_steps(config: dict) -> int:
    """Return the number of recent history steps kept at original resolution."""
    try:
        recent_steps = int(config.get("image", {}).get("recent_steps", 0))
    except (TypeError, ValueError):
        return 0
    if recent_steps < 0:
        return 0
    return recent_steps


def _read_image_as_data_url(image_path: str, scale: float = 1.0) -> str:
    """Read an image file and return a ``data:image/...;base64,...`` URL.

    When ``scale`` is a value in ``(0, 1)`` the image is downscaled by that
    factor with PIL (LANCZOS) before base64-encoding. When ``scale`` is ``None``
    or ``>= 1.0`` the original raw bytes are returned unchanged.
    """
    # Fast path: no scaling -> original raw bytes, zero behavior change.
    if not scale or scale >= 1.0:
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/png;base64,{image_data}"

    # Downscale path: PIL resize -> PNG re-encode. Falls back to raw bytes on
    # any failure (missing PIL, corrupt image, ...) so the episode never breaks.
    try:
        from PIL import Image
    except ImportError:
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/png;base64,{image_data}"

    try:
        resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.LANCZOS)
        with Image.open(image_path) as img:
            w, h = img.size
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            img = img.convert("RGB")
            if (new_w, new_h) != (w, h):
                img = img.resize((new_w, new_h), resample)
            buf = BytesIO()
            img.save(buf, format="PNG", optimize=True)
            image_data = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{image_data}"
    except Exception:
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/png;base64,{image_data}"


class APIRetryError(Exception):
    """API retries exhausted."""


def load_config_dict(config_path: str) -> dict:
    """Load config file into a plain dict."""
    config = load_config(config_path)
    return config.get_all() if hasattr(config, "get_all") else config.config


def extract_model_config_from_file(config_path: str) -> dict:
    """Extract model.vlm from a single-agent config file."""
    config_dict = load_config_dict(config_path)
    model_config = config_dict.get("model", {}).get("vlm", {})
    if not model_config:
        raise ValueError(f"        model.vlm: {config_path}")
    return dict(model_config)


def apply_agent_model_overrides(
    config_dict: dict,
    agent1_config_path: Optional[str] = None,
    agent2_config_path: Optional[str] = None,
) -> dict:
    """Apply agent-specific model config overrides from single-agent config files."""
    if not agent1_config_path and not agent2_config_path:
        config_dict.setdefault("dual_agent", {})
        config_dict["dual_agent"].setdefault("use_separate_models", False)
        return config_dict

    if not agent1_config_path and agent2_config_path:
        agent1_config_path = agent2_config_path
    if not agent2_config_path and agent1_config_path:
        agent2_config_path = agent1_config_path

    agent1_path = str(Path(agent1_config_path).resolve())
    agent2_path = str(Path(agent2_config_path).resolve())

    for path in [agent1_path, agent2_path]:
        if not Path(path).exists():
            raise FileNotFoundError(f"         : {path}")

    agent1_model_config = extract_model_config_from_file(agent1_path)
    agent2_model_config = extract_model_config_from_file(agent2_path)

    config_dict.setdefault("model", {})
    config_dict.setdefault("dual_agent", {})

    if agent1_path == agent2_path:
        config_dict["model"]["vlm"] = agent1_model_config
        config_dict["dual_agent"]["use_separate_models"] = False
        print("✓         ")
        print(f"  Shared model config: {agent1_path}")
        print(f"     : {agent1_model_config.get('model_name')}")
    else:
        config_dict["model"]["vlm"] = agent1_model_config
        config_dict["dual_agent"]["use_separate_models"] = True
        config_dict["dual_agent"]["agent_1"] = agent1_model_config
        config_dict["dual_agent"]["agent_2"] = agent2_model_config
        print("✓        ")
        print(f"  Agent 1 config: {agent1_path}")
        print(f"  Agent 1    : {agent1_model_config.get('model_name')}")
        print(f"  Agent 2 config: {agent2_path}")
        print(f"  Agent 2    : {agent2_model_config.get('model_name')}")

    return config_dict


def create_vlm_from_config(vlm_config: dict):
    """Create a VLM instance from a config dict."""
    return get_vlm(
        provider=vlm_config.get("provider", "openai"),
        model_name=vlm_config.get("model_name", "gpt-4o"),
        temperature=vlm_config.get("temperature", 0.2),
        top_p=vlm_config.get("top_p"),
        max_tokens=vlm_config.get("max_tokens", 2000),
        base_url=vlm_config.get("base_url"),
        api_key=vlm_config.get("api_key"),
        proxy_url=vlm_config.get("proxy_url"),
    )


def get_agent_vlms(config_dict: dict) -> Tuple[Dict[str, Any], bool]:
    """Create either one shared model or two separate models."""
    model_config = dict(config_dict.get("model", {}).get("vlm", {}))
    dual_config = config_dict.get("dual_agent", {})
    use_separate_models = bool(dual_config.get("use_separate_models", False))

    def _vlm_display_name(vlm, cfg: dict) -> str:
        for attr in ("model_name", "_model", "model"):
            val = getattr(vlm, attr, None)
            if val:
                return str(val)
        return str(cfg.get("model_name") or cfg.get("model") or type(vlm).__name__)

    if not use_separate_models:
        shared = create_vlm_from_config(model_config)
        print(f"✓     : {_vlm_display_name(shared, model_config)}")
        return {"agent_1": shared, "agent_2": shared}, False

    agent_1_config = {**model_config, **(dual_config.get("agent_1", {}) or {})}
    agent_2_config = {**model_config, **(dual_config.get("agent_2", {}) or {})}
    agent_1_vlm = create_vlm_from_config(agent_1_config)
    agent_2_vlm = create_vlm_from_config(agent_2_config)
    print("✓         ")
    print(f"  Agent 1   : {_vlm_display_name(agent_1_vlm, agent_1_config)}")
    print(f"  Agent 2   : {_vlm_display_name(agent_2_vlm, agent_2_config)}")
    return {"agent_1": agent_1_vlm, "agent_2": agent_2_vlm}, True


def load_init_actions_from_task_folder(task_folder_path: str):
    """Load init actions from tasks/<task_id>/task.json."""
    task_file = os.path.join(task_folder_path, "task.json")
    if not os.path.isfile(task_file):
        return None
    return load_init_actions_for_task(task_file)


def perform_final_evaluation(env, task_config: dict, observation=None) -> tuple:
    """Run evaluator on the final environment state."""
    if not task_config:
        return False, 0.0

    if (not observation or not getattr(observation, "metadata", None)) and hasattr(env, "_get_current_observation"):
        try:
            observation = env._get_current_observation()
        except Exception:
            observation = observation

    if not observation or not getattr(observation, "metadata", None):
        return False, 0.0

    try:
        evaluator = create_evaluator_from_config(task_config)
        score = evaluator.evaluate(env, observation.metadata)
        return score >= 1.0, score
    except Exception as e:
        print(f"❌ Evaluation error: {e}")
        return False, 0.0


def _extract_tag_block(text: str, tag: str) -> Optional[str]:
    start_token = f"<{tag}>"
    end_token = f"</{tag}>"
    start = text.find(start_token)
    end = text.find(end_token)
    if start == -1 or end == -1 or end <= start:
        return None
    start += len(start_token)
    return text[start:end]


_READ_MEMORY_PATTERN = re.compile(r"^read\s*memory\s*\(\s*(.*?)\s*\)$", re.IGNORECASE)


def _parse_read_memory_action(action_string: str) -> Optional[str]:
    """Return the requested memory file name if ``action_string`` is a
    ``ReadMemory(<file_name>)`` call, else ``None``.

    Mirrors ``mllm_base_agent/dual_agent/ai2thor/main.py``: this is a
    pseudo-action that never reaches :func:`actions.parser.parse_action_string`
    or the ProcTHOR environment. It lets the agent consult the on-disk memory
    library (see :mod:`mllm_base_agent.tools.memory`) through the same
    lightweight text-action grammar it already uses for everything else.
    """
    match = _READ_MEMORY_PATTERN.match(action_string.strip())
    if not match:
        return None
    file_name = match.group(1).strip().strip("'\"")
    return file_name or "MEMORY.md"


def _build_memory_nudge_text(current_agent: dict) -> Optional[str]:
    """Return an escalating, situation-specific nudge towards ``ReadMemory``.

    Mirrors ``mllm_base_agent/dual_agent/ai2thor/main.py::_build_memory_nudge_text``.
    A generic one-line mention of the memory library in the system prompt is
    easy for the model to skim past, especially once it is buried under a
    long system prompt + image history. This produces a short, blunt
    reminder appended to the *current* turn (i.e. right next to where the
    model is about to decide its next action) whose urgency scales with
    ``consecutive_failures`` and whether this agent has ever opened the
    library at all this episode:

    - 0 consecutive failures: no nudge (avoid nagging on every single step).
    - 1 failure: a soft suggestion.
    - 2 failures: a strong, explicit instruction to read memory before
      retrying, naming the free-lookup guarantee to remove any "it costs a
      step" hesitation.
    - >=3 failures AND the agent has *never* used ReadMemory this episode:
      the most insistent framing, calling out the zero-usage fact directly
      -- this is the single strongest predictor that the agent is stuck in a
      loop it does not know how to break out of on its own.

    Returns ``None`` when no nudge should be shown this turn.
    """
    consecutive_failures = int(current_agent.get("consecutive_failures", 0) or 0)
    memory_reads_used = int(current_agent.get("memory_reads_used", 0) or 0)

    if consecutive_failures <= 0:
        return None
    if consecutive_failures == 1:
        return (
            "Tip: if you're not sure why that failed, `ReadMemory(<file_name>)` "
            "is a free lookup (no step cost) into the lessons-learned library "
            "indexed above."
        )
    if consecutive_failures >= 3 and memory_reads_used == 0:
        return (
            f"**STOP AND READ MEMORY.** You have failed {consecutive_failures} actions "
            "in a row this episode and have not opened the memory library even "
            "once. Repeating the same kind of action is very unlikely to start "
            "working on its own. Before your next action, output "
            "`ReadMemory(<file_name>)` for the entry in the index above that "
            "matches this error (it costs zero step budget) -- do not attempt "
            "another world-changing action until you have."
        )
    return (
        f"You have failed {consecutive_failures} actions in a row. Before "
        "retrying, output `ReadMemory(<file_name>)` for the matching entry in "
        "the memory index above -- it is a free lookup and repeated failures "
        "are exactly the situation it exists for."
    )


def _build_done_checklist_nudge_text(current_agent: dict) -> Optional[str]:
    """Return a one-time reminder to re-read the DONE-verification memory entry.

    Mirrors ``mllm_base_agent/dual_agent/ai2thor/main.py::_build_done_checklist_nudge_text``.
    Fires exactly once per agent, the first time its remaining step budget
    drops to (or below) 20% of ``max_steps``. This is the point in an
    episode where a model is most tempted to claim ``DONE`` prematurely to
    avoid running out of budget, and it is exactly the situation
    ``feedback_done_verification.md`` (see the memory library) targets. A
    single well-timed nudge here is cheap and does not spam every remaining
    step (tracked via ``current_agent["done_checklist_nudged"]``).
    """
    if current_agent.get("done_checklist_nudged"):
        return None
    max_steps = int(current_agent.get("max_steps", 0) or 0)
    step_count = int(current_agent.get("step_count", 0) or 0)
    if max_steps <= 0:
        return None
    remaining_fraction = (max_steps - step_count) / max_steps
    if remaining_fraction > 0.2:
        return None
    current_agent["done_checklist_nudged"] = True
    return (
        "**Budget check:** you are in the final ~20% of your step budget. "
        "Before outputting DONE, run `ReadMemory(feedback_done_verification.md)` "
        "(free lookup) and re-verify every subgoal yourself -- do not claim "
        "DONE based on assumption or your partner's report alone."
    )


# Hard-gate threshold: mirrors
# ``mllm_base_agent/dual_agent/ai2thor/main.py::FORCE_READ_MEMORY_FAILURE_THRESHOLD``.
# Once an agent has this many consecutive action failures, its next
# submitted action is REQUIRED to be ReadMemory(...) -- anything else is
# rejected (re-prompted, not executed) up to
# ``FORCE_READ_MEMORY_MAX_REJECTIONS`` times. Exists because softer text
# nudges alone let models paraphrase the memory index's one-line summaries
# in <THINK> without ever emitting a real ReadMemory(<file_name>) action.
FORCE_READ_MEMORY_FAILURE_THRESHOLD = 2
# Safety valve: if the model still refuses to call ReadMemory after this many
# rejected attempts, stop forcing the issue and let normal failure-handling
# run, rather than burning the whole iteration_cap on one stuck turn.
FORCE_READ_MEMORY_MAX_REJECTIONS = 3


def _should_force_read_memory(current_agent: dict) -> bool:
    """Whether the current agent's NEXT action is required to be ``ReadMemory(...)``.

    Mirrors ``mllm_base_agent/dual_agent/ai2thor/main.py::_should_force_read_memory``.
    True once ``consecutive_failures`` has reached
    ``FORCE_READ_MEMORY_FAILURE_THRESHOLD`` for the CURRENT failure streak AND
    the agent has not yet consulted memory during this same streak (tracked
    via ``memory_consulted_for_streak``). Forces AT LEAST one real lookup per
    failure streak, then gets out of the way instead of re-triggering every
    turn while the streak count stays flat.
    """
    rejections = int(current_agent.get("forced_memory_rejections", 0) or 0)
    if rejections >= FORCE_READ_MEMORY_MAX_REJECTIONS:
        return False
    consecutive_failures = int(current_agent.get("consecutive_failures", 0) or 0)
    if consecutive_failures < FORCE_READ_MEMORY_FAILURE_THRESHOLD:
        return False
    return not current_agent.get("memory_consulted_for_streak", False)


def _force_read_memory_rejection_text(current_agent: dict) -> str:
    """Feedback shown when a non-ReadMemory action is rejected by the hard gate."""
    consecutive_failures = int(current_agent.get("consecutive_failures", 0) or 0)
    return (
        f"**Action rejected -- not executed.** You have {consecutive_failures} consecutive "
        "action failures, which requires consulting the memory library before any other "
        "action is accepted. Your submitted action was discarded (no step budget was "
        "spent). Your ONLY valid <ACTION> this turn is `ReadMemory(<file_name>)` -- pick "
        "the entry from the index above that matches your current error. Any other "
        "action will continue to be rejected."
    )


def parse_dual_agent_response(response_text: str, enable_summary: bool = False) -> Dict[str, Any]:
    """Parse THINK/ACTION/(COMMUNICATE)/(SUMMARY) from a model response."""
    think_block = _extract_tag_block(response_text, "THINK")
    action_block = _extract_tag_block(response_text, "ACTION")
    communicate_block = _extract_tag_block(response_text, "COMMUNICATE")
    summary_block = _extract_tag_block(response_text, "SUMMARY") if enable_summary else None

    if not action_block:
        raise ValueError("Missing <ACTION> tag")
    action_string = action_block.strip()
    if not action_string:
        raise ValueError("ACTION tag content is empty")

    thinking_text = (
        think_block.strip()
        if think_block and think_block.strip()
        else "(No <THINK> block; action will still be executed)"
    )
    communication_text = communicate_block.strip() if communicate_block and communicate_block.strip() else ""
    updated_summary = summary_block.strip() if summary_block and summary_block.strip() else ""

    memory_file_name = _parse_read_memory_action(action_string)
    if memory_file_name is not None:
        parsed_action = {
            "action_type": "memory_lookup",
            "action_name": "ReadMemory",
            "file_name": memory_file_name,
        }
    else:
        parsed_action = parse_action_string(action_string)
        if parsed_action.get("action_type") == "communication":
            communication_text = parsed_action.get("message", communication_text)
    return {
        "thinking_text": thinking_text,
        "action_string": action_string,
        "parsed_action": parsed_action,
        "communication_text": communication_text,
        "updated_summary": updated_summary,
    }


def initialize_agent_state(
    agent_id: str,
    vlm: Any,
    observation: Any,
    max_steps: int,
) -> dict:
    return {
        "agent_id": agent_id,
        "vlm": vlm,
        "observation": observation,
        "step_count": 0,
        "max_steps": max_steps,
        "short_term_history": [],
        "long_term_summary": "",
        "structured_trajectory": [],
        "should_continue": True,
        "failure_type": None,
        "fail_reason": None,
        "consecutive_failures": 0,
        "last_error_message": None,
        # Counts real ReadMemory(<file_name>) lookups this agent has made so
        # far (see the "memory_lookup" branch below). Used purely to drive
        # the strength of the in-prompt nudge towards consulting the memory
        # library -- it is NOT a hard requirement/gate, just a way to make
        # the reminder more specific/insistent the longer an agent goes
        # without ever opening it (models tend to ignore generic, one-off
        # "you may use X" suggestions but respond better to "you have never
        # done X and are now failing" framing).
        "memory_reads_used": 0,
        # Set once _build_done_checklist_nudge_text has fired for this agent
        # (near step-budget exhaustion), so the reminder is shown only once
        # rather than on every remaining step.
        "done_checklist_nudged": False,
        # Count of non-ReadMemory actions rejected by the hard gate in
        # _should_force_read_memory (see FORCE_READ_MEMORY_MAX_REJECTIONS).
        "forced_memory_rejections": 0,
        # Whether a real ReadMemory lookup has happened during the CURRENT
        # consecutive-failure streak. Gates _should_force_read_memory so the
        # hard gate forces exactly one real lookup per streak.
        "memory_consulted_for_streak": False,
    }


def build_shared_context(state: dict, current_agent_id: str) -> str:
    """Build shared context for the current agent from recent communications."""
    communications = state.get("communication_history", [])
    relevant = []
    for msg in communications[-8:]:
        sender = msg.get("sender", "unknown")
        receiver = msg.get("receiver", "unknown")
        message = msg.get("message", "")
        step = msg.get("global_step", "?")
        if receiver == current_agent_id or sender != current_agent_id:
            relevant.append(f"[Step {step}] {sender} -> {receiver}: {message}")
    return "\n".join(relevant) if relevant else "No messages from partner yet."


def consume_pending_messages(state: dict, current_agent_id: str) -> List[dict]:
    """Pop pending messages for the current agent."""
    pending = []
    remaining = []
    for msg in state.get("message_queue", []):
        if msg.get("receiver") == current_agent_id:
            pending.append(msg)
        else:
            remaining.append(msg)
    state["message_queue"] = remaining
    return pending


def agent_can_continue(state: dict, agent_id: str) -> bool:
    agent_state = state[agent_id]
    return agent_state.get("should_continue", True) and agent_state.get("step_count", 0) < agent_state.get("max_steps", 0)


def ensure_system_step_started(state: dict, current_agent_id: str) -> None:
    """Initialize expected agents for the current system step."""
    if state.get("system_step_expected_agents"):
        return

    other_agent_id = "agent_2" if current_agent_id == "agent_1" else "agent_1"
    expected = [current_agent_id]
    if agent_can_continue(state, other_agent_id):
        expected.append(other_agent_id)

    state["system_step_expected_agents"] = expected
    state["system_step_completed_agents"] = []


def finalize_system_step_if_complete(state: dict) -> None:
    """Increase system step count when all expected active agents have acted."""
    expected = state.get("system_step_expected_agents", [])
    completed = state.get("system_step_completed_agents", [])

    if expected and all(agent_id in completed for agent_id in expected):
        state["global_step_count"] = state.get("global_step_count", 0) + 1
        state["system_step_expected_agents"] = []
        state["system_step_completed_agents"] = []


def refresh_system_step_expected_agents(state: dict) -> None:
    """Drop agents that died mid-step and close the step if all remaining agents have acted."""
    expected = state.get("system_step_expected_agents", [])
    if not expected:
        return

    completed = state.get("system_step_completed_agents", [])
    refreshed = []
    for agent_id in expected:
        if agent_id in completed or agent_can_continue(state, agent_id):
            refreshed.append(agent_id)

    state["system_step_expected_agents"] = refreshed
    finalize_system_step_if_complete(state)


def record_system_step_progress(state: dict, current_agent_id: str) -> None:
    """Record one agent action toward the current system step."""
    ensure_system_step_started(state, current_agent_id)

    completed = state.get("system_step_completed_agents", [])
    if current_agent_id not in completed:
        completed.append(current_agent_id)
    state["system_step_completed_agents"] = completed

    finalize_system_step_if_complete(state)


def finalize_partial_system_step(state: dict) -> None:
    """Count the last partial system step when execution terminates mid-round."""
    expected = state.get("system_step_expected_agents", [])
    completed = state.get("system_step_completed_agents", [])

    if expected and completed:
        state["global_step_count"] = state.get("global_step_count", 0) + 1

    state["system_step_expected_agents"] = []
    state["system_step_completed_agents"] = []


def maybe_switch_agent(state: dict, switch_interval: int) -> Optional[str]:
    """Choose current agent according to collaboration mode and availability."""
    current_agent = state["current_agent"]
    other_agent = "agent_2" if current_agent == "agent_1" else "agent_1"
    mode = state.get("collaboration_mode", "alternating")

    if mode == "alternating" and state.get("current_turn_steps", 0) >= switch_interval and agent_can_continue(state, other_agent):
        state["current_agent"] = other_agent
        state["current_turn_steps"] = 0
        state["turn_count"] += 1
        current_agent = other_agent

    if not agent_can_continue(state, current_agent):
        if agent_can_continue(state, other_agent):
            state["current_agent"] = other_agent
            state["current_turn_steps"] = 0
            state["turn_count"] += 1
            return other_agent
        return None

    if mode == "sequential" and current_agent == "agent_1" and not agent_can_continue(state, "agent_1"):
        if agent_can_continue(state, "agent_2"):
            state["current_agent"] = "agent_2"
            state["current_turn_steps"] = 0
            state["turn_count"] += 1
            return "agent_2"
        return None

    return current_agent


def handoff_agent_or_finish(
    state: dict,
    current_agent_id: str,
    reason: str,
    failure_type: Optional[str] = None,
) -> bool:
    """Deactivate current agent and hand off to the partner if possible."""
    other_agent_id = "agent_2" if current_agent_id == "agent_1" else "agent_1"
    current_agent = state[current_agent_id]

    current_agent["should_continue"] = False
    current_agent["fail_reason"] = reason
    current_agent["last_error_message"] = reason
    if failure_type is not None:
        current_agent["failure_type"] = failure_type

    if agent_can_continue(state, other_agent_id):
        state["current_agent"] = other_agent_id
        state["current_turn_steps"] = 0
        state["turn_count"] += 1
        print(f"🔄     {other_agent_id}: {reason}")
        return True

    if not state.get("fail_reason"):
        state["fail_reason"] = reason
    if failure_type is not None and not state.get("failure_type"):
        state["failure_type"] = failure_type
    print(f"⚠️      ：{other_agent_id}         : {reason}")
    return False


def mark_agent_failure(state: dict, agent_id: str, failure_type: str, fail_reason: str):
    """Mark current agent as failed and maybe update global failure hints."""
    agent_state = state[agent_id]
    agent_state["should_continue"] = False
    agent_state["failure_type"] = failure_type
    agent_state["fail_reason"] = fail_reason
    if not state.get("failure_type"):
        state["failure_type"] = failure_type
    if not state.get("fail_reason"):
        state["fail_reason"] = fail_reason


def save_dual_conversation_log(state: dict, output_dir: str):
    """Save log.json for dual-agent runs."""
    log_file = os.path.join(output_dir, "log.json")
    conversation_json = {
        "metadata": {
            "task_description": state.get("task_prompt", ""),
            "task_result": "success" if state.get("success") else "failure",
            "fail_reason": state.get("fail_reason"),
            "failure_type": state.get("failure_type"),
            "total_steps": state.get("global_step_count", 0),
            "max_steps": state.get("max_global_steps", 0),
            "agent_1_steps": state.get("agent_1", {}).get("step_count", 0),
            "agent_2_steps": state.get("agent_2", {}).get("step_count", 0),
            "communication_events": len(state.get("communication_history", [])),
            "mode": "dual_agent",
            "token_usage": state.get("token_usage", {}),
        },
        "messages": [],
        "images": [],
    }

    trajectory = []
    for agent_id in ["agent_1", "agent_2"]:
        for entry in state.get(agent_id, {}).get("structured_trajectory", []):
            copied = dict(entry)
            copied["agent_id"] = agent_id
            trajectory.append(copied)
    trajectory.sort(key=lambda x: (x.get("global_step", 0), x.get("agent_id", "")))

    for entry in trajectory:
        step_id = entry.get("global_step", entry.get("step", 0))
        image_path = entry.get("image_path", "")
        raw_response = entry.get("raw_response", "")
        action_string = entry.get("action_string", "")
        reward = entry.get("reward", 0)
        error_message = entry.get("error_message")
        agent_id = entry.get("agent_id", "agent")
        conversation_json["messages"].append(
            {
                "role": "user",
                "content": f"[{agent_id}] Step {step_id}" + ("\n<image>" if image_path else ""),
                "step": step_id,
                "image_path": image_path,
            }
        )
        conversation_json["messages"].append(
            {
                "role": "assistant",
                "content": raw_response,
                "step": step_id,
                "agent_id": agent_id,
                "action_executed": action_string,
                "reward": reward,
                "error_message": error_message,
                "communication": entry.get("communication", ""),
            }
        )
        if image_path:
            conversation_json["images"].append(image_path)

    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(conversation_json, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Conversation log saved: {log_file}")


def save_dual_episode_log(state: dict, output_dir: str, env) -> None:
    """Save a detailed dual-agent episode json."""
    scene_name = "ProcTHOR"
    if hasattr(env, "scene") and env.scene is not None:
        scene_name = str(env.scene) if not isinstance(env.scene, dict) else env.scene.get("sceneName", "ProcTHOR")
    scene_short = scene_name.replace(" ", "_").replace("/", "_")[:50]
    task_name = (state.get("config") or {}).get("task", {}).get("name", "task") or "task"
    task_short = task_name.replace(" ", "_").replace("/", "_")
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"dual_episode_{scene_short}_{task_short}_{timestamp_str}.json"
    filepath = os.path.join(output_dir, filename)

    trajectory = []
    for agent_id in ["agent_1", "agent_2"]:
        for entry in state.get(agent_id, {}).get("structured_trajectory", []):
            copied = dict(entry)
            copied["agent_id"] = agent_id
            trajectory.append(copied)
    trajectory.sort(key=lambda x: (x.get("global_step", 0), x.get("agent_id", "")))

    episode_log = {
        "task": state.get("task_prompt", ""),
        "scene": scene_name,
        "success": state.get("success", False),
        "fail_reason": state.get("fail_reason"),
        "failure_type": state.get("failure_type"),
        "global_step_count": state.get("global_step_count", 0),
        "max_global_steps": state.get("max_global_steps", 0),
        "agent_1_steps": state.get("agent_1", {}).get("step_count", 0),
        "agent_2_steps": state.get("agent_2", {}).get("step_count", 0),
        "turn_count": state.get("turn_count", 0),
        "communication_history": state.get("communication_history", []),
        "action_sequence": env.get_action_sequence() if hasattr(env, "get_action_sequence") else "(no action records)",
        "trajectory": trajectory,
        "token_usage": state.get("token_usage", {}),
        "timestamp": datetime.now().isoformat(),
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(episode_log, f, ensure_ascii=False, indent=2)
    print(f"✓ Dual episode log saved: {filepath}")


def run_dual_agent_loop(
    env,
    agent_vlms: Dict[str, Any],
    task_config: dict,
    task_output_dir: str,
    config: dict,
    collaboration_mode: str = "alternating",
    switch_interval: int = 1,
):
    """Run a lightweight dual-agent collaboration loop on one shared ProcTHOR env.

    Feature flags (read from ``config["dual_agent"]`` / ``config["image"]``):
      * ``history_feedback``    — inject each step's action + result into history.
      * ``llm_history_feedback``— a second LLM analyzes recent history and emits
        concise per-step annotations that replace the raw action+error text.
      * ``partner_view``        — inject a fresh first-person image rendered from
        the partner body's camera at each decision step.
      * ``image.scale`` / ``image.recent_steps`` / ``image.partner_view_scale`` —
        control image resolution to bound request-body size on long episodes.
    """
    task_prompt = task_config.get("instruction") or task_config.get("description") or "Complete the task."
    per_agent_steps = int(task_config.get("max_steps", config.get("max_steps", 30)))
    max_global_steps = 2 * per_agent_steps
    golden_steps = derive_dual_golden_steps(task_config)
    if golden_steps is not None:
        print(
            f"✓ Per-agent max steps (10+n, n=golden_actions.steps={int(golden_steps)}): "
            f"{per_agent_steps} | Global cap: {max_global_steps}"
        )
    else:
        print(f"✓ Per-agent max steps: {per_agent_steps} | Global cap: {max_global_steps}")
    enable_summary = config.get("context_management", {}).get("enable_long_term_summary", False)
    configured_history = int(config.get("context_management", {}).get("short_term_history_window_size", MODEL_HISTORY_TURNS))
    max_history = min(MODEL_HISTORY_TURNS, max(0, configured_history))

    initial_observation = env.reset(task_prompt)

    #   ：  agent2     agent1   ；                （    agent1/   agent2/）
    agent_1_observation = initial_observation
    agent_2_observation = initial_observation
    if getattr(env, "agent_count", 1) > 1:
        if hasattr(env, "relocate_second_agent_near_agent1"):
            env.relocate_second_agent_near_agent1()
        if hasattr(env, "get_observation_for_agent"):
            try:
                agent_1_observation = env.get_observation_for_agent(0)
                agent_2_observation = env.get_observation_for_agent(1)
                print(
                    f"✓     : agent1={agent_1_observation.image_path} | "
                    f"agent2={agent_2_observation.image_path}"
                )
            except Exception as e:
                print(f"⚠️            ，       : {e}")

    state = {
        "task_prompt": task_prompt,
        "config": config,
        "collaboration_mode": collaboration_mode,
        "switch_interval": switch_interval,
        "global_step_count": 0,
        "max_global_steps": max_global_steps,
        "system_step_expected_agents": [],
        "system_step_completed_agents": [],
        "current_agent": "agent_1",
        "current_turn_steps": 0,
        "turn_count": 0,
        "communication_history": [],
        "message_queue": [],
        "success": False,
        "fail_reason": None,
        "failure_type": None,
        "last_acting_agent": None,
        "token_usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "api_calls": 0,
        },
        # Skill-memory library (mirrors ai2thor/main.py + RPent's
        # resources/<env>/memory/): a reviewed MEMORY.md index plus
        # feedback_*.md leaf notes. The index is embedded into the system
        # prompt every turn; leaf notes are fetched on demand via the
        # ReadMemory(<file_name>) pseudo-action intercepted below.
        "memory_library": MemoryLibrary.for_env("procthor", agent_mode="dual") if bool((config.get("memory") or {}).get("enabled", True)) else None,
        "history_feedback": bool(config.get("dual_agent", {}).get("history_feedback", False)),
        "llm_history_feedback": bool(config.get("dual_agent", {}).get("llm_history_feedback", False)),
        # Image downscale factor for VLM inputs (``image.scale``). 1.0 = no scaling.
        "image_scale": _resolve_image_scale(config),
        # Number of recent history steps kept at original resolution; older
        # history images are downscaled to ``image_scale``. Current obs is
        # always full-resolution.
        "image_recent_steps": _resolve_image_recent_steps(config),
        # Partner-view injection: when enabled, each decision step also feeds
        # the model a fresh first-person image from the partner body's camera.
        "partner_view": bool(config.get("dual_agent", {}).get("partner_view", False)),
        "partner_view_scale": float(
            (config.get("image", {}) or {}).get("partner_view_scale")
            or _resolve_image_scale(config)
        ),
        "agent_1": initialize_agent_state("agent_1", agent_vlms["agent_1"], agent_1_observation, per_agent_steps),
        "agent_2": initialize_agent_state("agent_2", agent_vlms["agent_2"], agent_2_observation, per_agent_steps),
    }

    from mllm_base_agent.llm.messages import AIMessage, HumanMessage, SystemMessage

    # LLM History Analyzer: one analyzer per agent, built from that agent's own
    # VLM ("use the current LLM").  Only instantiated when the feature is on.
    if state.get("llm_history_feedback"):
        state["history_analyzers"] = {
            aid: HistoryAnalyzerAgent(vlm) for aid, vlm in agent_vlms.items() if vlm is not None
        }
    else:
        state["history_analyzers"] = {}

    while state["global_step_count"] < state["max_global_steps"]:
        refresh_system_step_expected_agents(state)
        if state["global_step_count"] >= state["max_global_steps"]:
            break

        current_agent_id = maybe_switch_agent(state, switch_interval)
        if current_agent_id is None:
            if not state.get("fail_reason"):
                state["fail_reason"] = "Both agents stopped before success"
            break

        current_agent = state[current_agent_id]
        other_agent_id = "agent_2" if current_agent_id == "agent_1" else "agent_1"

        #   ：    "   agent"           （  agent      ，
        #     step_with_action_dict            frame，      Pass）
        if getattr(env, "agent_count", 1) > 1 and hasattr(env, "get_observation_for_agent"):
            last_acting = state.get("last_acting_agent")
            needs_refresh = (
                current_agent.get("observation") is None
                or (last_acting is not None and last_acting != current_agent_id)
            )
            if needs_refresh:
                try:
                    thor_aid = AGENT_TO_THOR_ID.get(current_agent_id, 0)
                    current_agent["observation"] = env.get_observation_for_agent(thor_aid)
                except Exception as e:
                    print(f"⚠️  Refresh obs for {current_agent_id} failed: {e}")

        observation = current_agent.get("observation") or state["agent_1"].get("observation")

        print(f"\n{'=' * 60}\n🧠 {current_agent_id.upper()} Step {current_agent['step_count'] + 1}\n{'=' * 60}")

        memory_library: Optional[MemoryLibrary] = state.get("memory_library")
        memory_index_block = memory_library.index_prompt_block() if memory_library else ""
        system_prompt = get_dual_procthor_prompt(enable_summary=enable_summary).format(
            task_prompt=task_prompt,
            shared_context=build_shared_context(state, current_agent_id),
            memory_index_block=memory_index_block,
        )
        messages = [SystemMessage(content=system_prompt)]
        if memory_library is None:
            system_prompt += "\n\nThis run has no skill-memory library. ReadMemory is unavailable; never output it."

        history_feedback = state.get("history_feedback", False)
        llm_history_feedback = state.get("llm_history_feedback", False)
        image_scale = state.get("image_scale", 1.0)
        image_recent_steps = state.get("image_recent_steps", 0)

        recent_history = current_agent.get("short_term_history", [])[-max_history:]
        n_history = len(recent_history)

        # LLM History Analyzer: produce per-step annotations that REPLACE the
        # raw "Your previous action: X / Result: FAILED - Y" text.  Only runs
        # when --llm-history-feedback is enabled; otherwise the original
        # history_feedback injection below is used unchanged.  On any failure
        # ``analyzer_annotations`` stays None and we gracefully fall back to
        # the raw text.
        analyzer_annotations = None
        if llm_history_feedback and recent_history:
            analyzer = state.get("history_analyzers", {}).get(current_agent_id)
            if analyzer is not None:
                try:
                    analyzer_annotations, analyzer_response = analyzer.analyze(recent_history, task_prompt)
                    if analyzer_response is not None:
                        _accumulate_token_usage(
                            state, _extract_token_usage_from_response(analyzer_response)
                        )
                except Exception as exc:
                    print(f"⚠️  History analyzer failed: {exc}")
                    analyzer_annotations = None

        for idx, entry in enumerate(recent_history):
            content = []
            img_path = entry.get("image_path")
            if img_path and os.path.exists(img_path):
                # Distance-based scaling: entries within ``image_recent_steps``
                # of the current step keep full resolution; older entries are
                # downscaled to ``image_scale`` to control request-body size.
                distance = n_history - idx
                entry_scale = 1.0 if distance <= image_recent_steps else image_scale
                try:
                    content.append(
                        {"type": "image_url", "image_url": {"url": _read_image_as_data_url(img_path, scale=entry_scale)}}
                    )
                except Exception:
                    content.append({"type": "text", "text": "[Image unavailable]"})
            step_text = f"Step {entry.get('step', 0)}"

            memory_lookup_result = entry.get("memory_lookup_result")
            if memory_lookup_result:
                # ReadMemory(...) result: shown in full regardless of
                # history_feedback/analyzer settings, since it is the direct
                # answer to a lookup the agent itself just requested (not an
                # action outcome to summarize).
                step_text += f"\n{memory_lookup_result}"
                content.append({"type": "text", "text": step_text})
                messages.append(HumanMessage(content=content))
                messages.append(AIMessage(content=entry.get("raw_response", "")))
                continue

            annotation = (
                analyzer_annotations[idx]
                if (analyzer_annotations is not None and idx < len(analyzer_annotations))
                else None
            )
            if annotation:
                # LLM history analyzer mode: replace the raw action + error
                # string with the analyzer's concise, actionable per-step
                # insight.
                step_text += f"\nHistory analysis: {annotation}"
            elif history_feedback:
                prev_action = entry.get("action_string", "")
                prev_error = entry.get("error_message")
                if prev_action:
                    step_text += f"\nYour previous action: {prev_action}"
                    if prev_error:
                        step_text += f"\nResult: FAILED - {prev_error}"
                    else:
                        step_text += "\nResult: SUCCESS"
            content.append({"type": "text", "text": step_text})
            messages.append(HumanMessage(content=content))
            messages.append(AIMessage(content=entry.get("raw_response", "")))

        pending_messages = consume_pending_messages(state, current_agent_id)
        current_content = []
        if pending_messages:
            current_content.append(
                {
                    "type": "text",
                    "text": "**Messages from Partner:**\n" + "\n".join(f"- {msg.get('message', '')}" for msg in pending_messages),
                }
            )

        last_error = current_agent.get("last_error_message")
        if last_error:
            # Shown on every step (independent of history_feedback), so it
            # works even when --history-feedback is off.
            error_banner = f"**Last action error:** {last_error}\nAdjust your plan before repeating the same action."
            memory_nudge = _build_memory_nudge_text(current_agent)
            if memory_nudge:
                error_banner += f"\n{memory_nudge}"
            current_content.append(
                {
                    "type": "text",
                    "text": error_banner,
                }
            )

        done_checklist_nudge = _build_done_checklist_nudge_text(current_agent)
        if done_checklist_nudge:
            current_content.append(
                {
                    "type": "text",
                    "text": done_checklist_nudge,
                }
            )

        # Partner-view injection: render a fresh first-person image from the
        # partner body's camera so the acting agent can observe the shared
        # scene from a different angle.  Only when --partner-view is enabled
        # and the env actually has 2 embodied agents.
        partner_view = state.get("partner_view", False)
        partner_view_scale = state.get("partner_view_scale", image_scale)
        if partner_view and getattr(env, "agent_count", 1) > 1:
            partner_thor_id = AGENT_TO_THOR_ID.get(other_agent_id, 1)
            try:
                partner_obs = env.get_observation_for_agent(partner_thor_id)
                partner_image_path = getattr(partner_obs, "image_path", None)
                if partner_image_path and os.path.exists(partner_image_path):
                    current_content.append(
                        {
                            "type": "text",
                            "text": (
                                "**Partner's current view** (rendered fresh from your "
                                "partner body's camera — use it to observe the shared "
                                "scene from a different angle and cross-check your "
                                "partner's reported findings):"
                            ),
                        }
                    )
                    current_content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": _read_image_as_data_url(
                                    partner_image_path, scale=partner_view_scale
                                )
                            },
                        }
                    )
            except Exception as exc:
                print(f"⚠️  Partner-view render failed: {exc}")

        image_path = observation.image_path if observation else None
        if not image_path or not os.path.exists(image_path):
            mark_agent_failure(state, current_agent_id, "env_error", "Missing observation image")
            continue

        # Current observation: always full resolution — it is the most critical
        # visual input for the agent's immediate decision.  When image_scale <
        # 1.0 only *historical* images are downscaled, never the current frame.
        current_content.append(
            {"type": "image_url", "image_url": {"url": _read_image_as_data_url(image_path, scale=1.0)}}
        )
        current_content.append(
            {
                "type": "text",
                "text": f"Current step is Step {current_agent['step_count'] + 1}. "
                "Please output <THINK> and <ACTION> using the unified action space.",
            }
        )
        messages.append(HumanMessage(content=current_content))

        # --- Call VLM with API retries + token accounting -----------------
        response_text = None
        last_api_error = None
        step_token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "api_calls": 0,
        }
        for api_attempt in range(LOCAL_RETRY_CONFIG["api_max_retries"]):
            try:
                print(f"📡 Calling VLM... (API attempt {api_attempt + 1}/{LOCAL_RETRY_CONFIG['api_max_retries']})")
                response = current_agent["vlm"].invoke(messages)
                response_text = response.content if hasattr(response, "content") else str(response)
                usage = _extract_token_usage_from_response(response)
                _accumulate_token_usage(state, usage)
                for key in step_token_usage:
                    step_token_usage[key] += usage.get(key, 0)
                break
            except Exception as api_error:
                err_str = str(api_error)
                is_api = any(k in err_str.lower() for k in ["api", "request failed", "connection", "timeout", "timed out", "http", "429", "500", "400"])
                if is_api and api_attempt < LOCAL_RETRY_CONFIG["api_max_retries"] - 1:
                    delay = LOCAL_RETRY_CONFIG["api_retry_delay"] if "400" in err_str else LOCAL_RETRY_CONFIG["retry_delay"]
                    print(f"⚠️  API Error (attempt {api_attempt + 1}/{LOCAL_RETRY_CONFIG['api_max_retries']}): {err_str[:200]}")
                    print(f"   Waiting {delay}s before retry...")
                    time.sleep(delay)
                    continue
                last_api_error = api_error
                break

        if response_text is None:
            reason = f"API error after retries: {last_api_error}"
            current_agent["structured_trajectory"].append(
                {
                    "step": current_agent["step_count"] + 1,
                    "global_step": state["global_step_count"] + 1,
                    "thinking": "",
                    "action_string": "",
                    "raw_response": "",
                    "llm_token_usage": dict(step_token_usage),
                    "parse_error": str(last_api_error),
                    "failure_type": "api_error",
                    "image_path": image_path,
                    "communication": "",
                }
            )
            mark_agent_failure(state, current_agent_id, "api_error", reason)
            continue

        # --- Parse response with parse retries ----------------------------
        parsed = None
        parse_error = None
        for parse_attempt in range(LOCAL_RETRY_CONFIG["max_retries"]):
            try:
                parsed = parse_dual_agent_response(response_text, enable_summary=enable_summary)
                if parse_attempt > 0:
                    print(f"✓ Success after {parse_attempt + 1} parse attempts")
                break
            except ValueError as e:
                parse_error = e
                print(f"⚠️  Parse Error (parse attempt {parse_attempt + 1}/{LOCAL_RETRY_CONFIG['max_retries']}): {e}")
                if parse_attempt < LOCAL_RETRY_CONFIG["max_retries"] - 1:
                    print(f"   Waiting {LOCAL_RETRY_CONFIG['retry_delay']}s before re-calling VLM...")
                    time.sleep(LOCAL_RETRY_CONFIG["retry_delay"])
                    try:
                        response = current_agent["vlm"].invoke(messages)
                        response_text = response.content if hasattr(response, "content") else str(response)
                        usage = _extract_token_usage_from_response(response)
                        _accumulate_token_usage(state, usage)
                        for key in step_token_usage:
                            step_token_usage[key] += usage.get(key, 0)
                    except Exception as recall_error:
                        parse_error = recall_error
                    continue
                break

        if parsed is None:
            parse_error_text = str(parse_error)
            reason = f"Parse error: {parse_error}"
            current_agent["structured_trajectory"].append(
                {
                    "step": current_agent["step_count"] + 1,
                    "global_step": state["global_step_count"] + 1,
                    "thinking": "",
                    "action_string": "",
                    "raw_response": (response_text or "")[:2000],
                    "llm_token_usage": dict(step_token_usage),
                    "parse_error": str(parse_error),
                    "failure_type": "parse_error",
                    "image_path": image_path,
                    "communication": "",
                }
            )

            missing_action = (
                "Missing <ACTION> tag" in parse_error_text
                or "ACTION tag content is empty" in parse_error_text
            )
            if missing_action:
                handoff_agent_or_finish(
                    state,
                    current_agent_id,
                    "Missing ACTION after 3 retries; hand off to partner",
                    failure_type="model_error",
                )
            else:
                mark_agent_failure(state, current_agent_id, "parse_error", reason)
            continue

        communication_text = parsed["communication_text"]
        if communication_text:
            comm = {
                "sender": current_agent_id,
                "receiver": other_agent_id,
                "message": communication_text,
                "global_step": state["global_step_count"] + 1,
            }
            state["communication_history"].append(comm)
            state["message_queue"].append(comm)
            print(f"📨 Communication sent to {other_agent_id}: {communication_text[:120]}...")

        action_dict = parsed["parsed_action"]
        action_string = parsed["action_string"]
        thinking_text = parsed["thinking_text"]

        print(f"✓ Thinking: {thinking_text[:200]}{'...' if len(thinking_text) > 200 else ''}")
        print(f"✓ Action String: {action_string}")
        print(f"✓ Parsed Action: {action_dict}")

        # --- Hard gate: force a real ReadMemory(...) after repeated failures --
        # Mirrors mllm_base_agent/dual_agent/ai2thor/main.py. Once
        # consecutive_failures reaches FORCE_READ_MEMORY_FAILURE_THRESHOLD,
        # any action OTHER than a real ReadMemory(...) is discarded here --
        # not executed, no step consumed, no handoff -- and the agent is
        # re-prompted with an explicit rejection message on its immediate
        # next turn. Bounded by FORCE_READ_MEMORY_MAX_REJECTIONS.
        if state.get("memory_library") is not None and action_dict.get("action_type") != "memory_lookup" and _should_force_read_memory(current_agent):
            current_agent["forced_memory_rejections"] = current_agent.get("forced_memory_rejections", 0) + 1
            rejection_text = _force_read_memory_rejection_text(current_agent)
            print(f"🚫 {rejection_text}")
            current_agent["short_term_history"].append(
                {
                    "step": current_agent["step_count"],
                    "image_path": None,
                    "raw_response": response_text,
                    "action_string": action_string,
                    "error_message": None,
                    "memory_lookup_result": f"[Memory Gate] {rejection_text}",
                }
            )
            current_agent["short_term_history"] = current_agent["short_term_history"][-max_history:]
            # Same agent retries immediately (no handoff, no step consumed),
            # exactly like a real ReadMemory lookup would.
            continue

        # --- Memory lookup: free action, no env step, no step-budget cost --
        # ReadMemory(<file_name>) never reaches the ProcTHOR environment and
        # does NOT increment current_agent["step_count"] / global_step_count
        # (mirrors ai2thor/main.py + RPent's "reading memory is free"
        # design -- see mllm_base_agent/tools/memory.py). The looked-up
        # content is folded into short_term_history as a synthetic "tool
        # result" turn so the SAME agent sees it on its immediate next call,
        # then keeps its turn (does not hand off to the partner) since no
        # world-state action was taken yet.
        if action_dict.get("action_type") == "memory_lookup":
            file_name = action_dict.get("file_name", "MEMORY.md")
            memory_library = state.get("memory_library")
            if memory_library is not None:
                lookup_result = memory_library.read_entry(file_name)
            else:
                lookup_result = {"error": "memory library not configured"}

            if "error" in lookup_result:
                memory_text = f"[Memory] ReadMemory({file_name}) failed: {lookup_result['error']}"
                print(f"⚠️  {memory_text}")
            else:
                memory_text = (
                    f"[Memory] Contents of {file_name}:\n{lookup_result.get('content', '')}"
                )
                print(f"📖 Memory lookup: {file_name} ({lookup_result.get('size', 0)} chars)")

            current_agent["short_term_history"].append(
                {
                    "step": current_agent["step_count"],
                    "image_path": None,
                    "raw_response": response_text,
                    "action_string": action_string,
                    "error_message": None,
                    "memory_lookup_result": memory_text,
                }
            )
            current_agent["short_term_history"] = current_agent["short_term_history"][-max_history:]
            current_agent["memory_reads_used"] = current_agent.get("memory_reads_used", 0) + 1
            # A real ReadMemory lookup satisfies the hard gate for THIS failure
            # streak: mark it consulted and reset the rejection counter so a
            # LATER, separate failure streak gets the full budget again.
            current_agent["memory_consulted_for_streak"] = True
            current_agent["forced_memory_rejections"] = 0
            # Same agent keeps acting next iteration (no handoff, no step
            # consumed): just loop back to the top of the while-loop.
            continue

        trajectory_entry = {
            "step": current_agent["step_count"] + 1,
            "global_step": state["global_step_count"] + 1,
            "thinking": thinking_text,
            "action_string": action_string,
            "action": action_dict,
            "raw_response": response_text,
            "llm_token_usage": dict(step_token_usage),
            "image_path": image_path,
            "communication": communication_text,
            "updated_summary": parsed.get("updated_summary", ""),
        }

        current_agent["step_count"] += 1
        record_system_step_progress(state, current_agent_id)
        state["current_turn_steps"] += 1

        if enable_summary and parsed.get("updated_summary"):
            current_agent["long_term_summary"] = parsed["updated_summary"]

        # --- Communication / Pass: no env step, hand off ------------------
        if action_dict.get("action_type") == "communication":
            trajectory_entry["reward"] = 0.0
            trajectory_entry["error_message"] = None
            current_agent["structured_trajectory"].append(trajectory_entry)
            current_agent["short_term_history"].append(
                {
                    "step": current_agent["step_count"],
                    "image_path": image_path,
                    "raw_response": response_text,
                    "action_string": trajectory_entry.get("action_string", ""),
                    "error_message": trajectory_entry.get("error_message"),
                }
            )
            current_agent["short_term_history"] = current_agent["short_term_history"][-max_history:]
            handoff_agent_or_finish(state, current_agent_id, "communication action")
            continue

        # --- Task completion (DONE / FAIL) --------------------------------
        if action_dict.get("action_type") == "task_completion":
            trajectory_entry["reward"] = 0.0
            trajectory_entry["error_message"] = None
            current_agent["structured_trajectory"].append(trajectory_entry)
            current_agent["short_term_history"].append(
                {
                    "step": current_agent["step_count"],
                    "image_path": image_path,
                    "raw_response": response_text,
                    "action_string": trajectory_entry.get("action_string", ""),
                    "error_message": trajectory_entry.get("error_message"),
                }
            )

            if action_dict.get("action_name") == "DONE":
                success, score = perform_final_evaluation(env, task_config, observation)
                if success:
                    state["success"] = True
                    state["fail_reason"] = None
                    state["failure_type"] = None
                    print(f"✅ DONE verified (score={score:.2f})")
                    break
                done_reject_msg = "DONE was rejected by evaluator"
                current_agent["last_error_message"] = done_reject_msg
                print(f"❌ DONE rejected by evaluator (score={score:.2f})")
                handoff_agent_or_finish(
                    state,
                    current_agent_id,
                    "Model claimed DONE but success conditions were not met",
                    failure_type="model_error",
                )
                continue

            handoff_agent_or_finish(
                state,
                current_agent_id,
                "Model indicated FAIL before task completion",
                failure_type="model_error",
            )
            continue

        # --- Execute the action on the current agent's body ---------------
        thor_agent_id = AGENT_TO_THOR_ID.get(current_agent_id, 0)
        try:
            #       thor_agent_id   ；            
            if getattr(env, "agent_count", 1) > 1:
                observation, error_message = env.step_with_action_dict(
                    action_dict, thor_agent_id=thor_agent_id
                )
            else:
                observation, error_message = env.step_with_action_dict(action_dict)
        except Exception as e:
            observation = None
            error_message = str(e)

        reward = 0.0 if error_message else 0.1
        trajectory_entry["reward"] = reward
        trajectory_entry["error_message"] = error_message
        trajectory_entry["thor_agent_id"] = thor_agent_id
        current_agent["structured_trajectory"].append(trajectory_entry)
        current_agent["short_term_history"].append(
            {
                "step": current_agent["step_count"],
                "image_path": image_path,
                "raw_response": response_text,
                "action_string": trajectory_entry.get("action_string", ""),
                "error_message": error_message,
            }
        )
        current_agent["short_term_history"] = current_agent["short_term_history"][-max_history:]

        #   ：         agent      ；         
        if observation is not None:
            if getattr(env, "agent_count", 1) > 1:
                current_agent["observation"] = observation
            else:
                state["agent_1"]["observation"] = observation
                state["agent_2"]["observation"] = observation

        #           agent，               
        state["last_acting_agent"] = current_agent_id

        if error_message:
            current_agent["consecutive_failures"] += 1
            current_agent["last_error_message"] = error_message
            # A new failure within the streak means whatever memory entry was
            # read before (if any) did not resolve it -- re-arm the hard gate
            # so it can force a (possibly different) lookup again.
            current_agent["memory_consulted_for_streak"] = False
            print(f"  ⚠️  Action failed: {error_message}")
        else:
            current_agent["consecutive_failures"] = 0
            current_agent["last_error_message"] = None
            current_agent["memory_consulted_for_streak"] = False
            print("✓ Action executed successfully")

        if current_agent["consecutive_failures"] >= 4:
            mark_agent_failure(
                state,
                current_agent_id,
                "action_error",
                f"Consecutive {current_agent['consecutive_failures']} action failures (early stop)",
            )
            print(f"🛑 Early stop for {current_agent_id}: consecutive action failures")
            continue

        if observation is None:
            mark_agent_failure(state, current_agent_id, "env_error", "Environment step returned None")
            continue

        if current_agent["step_count"] >= current_agent["max_steps"] and not state["success"]:
            handoff_agent_or_finish(
                state,
                current_agent_id,
                "Reached max steps before task completion",
            )

    finalize_partial_system_step(state)

    final_observation = (
        state.get(state.get("current_agent", "agent_1"), {}).get("observation")
        or state.get("agent_1", {}).get("observation")
        or state.get("agent_2", {}).get("observation")
    )
    final_success, final_score = perform_final_evaluation(env, task_config, final_observation)
    if final_success:
        state["success"] = True
        state["fail_reason"] = None
        state["failure_type"] = None
        print(f"✅ Final terminal-state evaluation succeeded (score={final_score:.2f})")
    else:
        state["success"] = False
        if not state.get("fail_reason"):
            state["fail_reason"] = f"Final evaluation failed (score={final_score:.2f})"
        print(f"❌ Final terminal-state evaluation failed (score={final_score:.2f})")

    return state


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="ProcTHOR dual-agent main entry")
    parser.add_argument("--config", type=str, default="experiments/configs/procthor/dual/config_close_gpt-5.yaml", help="Base config file path")
    parser.add_argument("--tasks", type=str, nargs="+", default=None, help="Task ID(s), e.g. procthor00001")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--headless", action="store_true", help="Headless mode")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max steps")
    parser.add_argument("--switch-interval", type=int, default=1, help="Agent switch interval")
    parser.add_argument("--collaboration-mode", type=str, default="alternating", choices=["alternating", "sequential"], help="Dual-agent collaboration mode")
    parser.add_argument("--agent1", type=str, default=None, help="Agent 1 single-agent config path")
    parser.add_argument("--agent2", type=str, default=None, help="Agent 2 single-agent config path")
    parser.add_argument("--print-config", action="store_true", help="Print config and exit")
    parser.add_argument(
        "--history-feedback",
        action="store_true",
        help="Inject each step's action + result (incl. distance error) into history so the agent can reason over the full action-outcome sequence",
    )
    parser.add_argument(
        "--llm-history-feedback",
        action="store_true",
        help=(
            "Enable the LLM History Analyzer Agent: spawn a second LLM (the SAME "
            "model as the actors) that manages and analyzes each agent's recent "
            "action-outcome history and emits concise, actionable per-step "
            "annotations. These annotations REPLACE the raw 'Your previous "
            "action: X / Result: FAILED - Y' text that --history-feedback "
            "injects, giving the acting agent high-information guidance instead "
            "of low-level action/error strings. Does NOT alter --history-feedback "
            "when disabled; works with or without it."
        ),
    )
    parser.add_argument(
        "--image-scale",
        type=float,
        default=1.0,
        help=(
            "Downscale factor for images sent to the VLM (0 < scale <= 1.0). "
            "e.g. --image-scale 0.5 resizes 800x600 -> 400x300 before base64 "
            "encoding, shrinking the request body and avoiding HTTP 413 on long "
            "multi-image episodes. Default 1.0 = no scaling (original behavior)."
        ),
    )
    parser.add_argument(
        "--image-recent-steps",
        type=int,
        default=0,
        help=(
            "Number of recent history steps whose images are kept at original "
            "resolution (full 800x600). Older history images are downscaled to "
            "--image-scale. The current observation is ALWAYS full-resolution "
            "regardless of this setting. Default 0 = only the current observation "
            "is full-resolution; all history images are downscaled (prior behavior)."
        ),
    )
    parser.add_argument(
        "--partner-view",
        action="store_true",
        help=(
            "Enable partner-view injection: at each decision step, also feed the "
            "model a fresh first-person image rendered from the partner body's "
            "current camera (via env.get_observation_for_agent). This lets the "
            "acting agent observe the shared scene from a different angle and "
            "reason about what its partner can see. Only effective when the env "
            "has 2 embodied agents. Default off (original behavior: each agent "
            "only sees its own view)."
        ),
    )
    parser.add_argument(
        "--partner-view-scale",
        type=float,
        default=None,
        help=(
            "Downscale factor for the partner-view image (0 < scale <= 1.0). "
            "When --partner-view is enabled, the partner's current observation "
            "image is resized by this factor before base64 encoding. Defaults to "
            "the value of --image-scale (so partner images are downscaled the "
            "same as historical images). Set to 1.0 to keep partner images at "
            "full 800x600 resolution."
        ),
    )
    args = parser.parse_args()

    print(f"\n{'=' * 60}\n🔧         \n{'=' * 60}\n    : {args.config}")
    config = load_config(args.config)
    config_dict = config.get_all() if hasattr(config, "get_all") else config.config
    config_dict = apply_agent_model_overrides(config_dict, args.agent1, args.agent2)

    if args.print_config:
        print(json.dumps(config_dict, ensure_ascii=False, indent=2))
        return

    task_names = args.tasks or config.get_all_task_names()
    if not task_names:
        print("❌ No tasks specified and no task names from config")
        return

    dual_config = config_dict.setdefault("dual_agent", {})
    dual_config.setdefault("equal_collaboration", True)
    dual_config["collaboration_mode"] = args.collaboration_mode

    # --- Feature flags (ported from ai2thor/main.py) ---------------------
    dual_config["history_feedback"] = bool(args.history_feedback)
    if args.history_feedback:
        print("✓ History feedback: enabled (action + result injected into per-step history)")

    dual_config["llm_history_feedback"] = bool(args.llm_history_feedback)
    if args.llm_history_feedback:
        print(
            "✓ LLM history feedback: enabled (a second LLM analyzes the recent "
            "action/outcome history and replaces the raw action+error text with "
            "actionable per-step annotations)"
        )

    # Image downscale: clamp to (0, 1.0]; values outside this range are treated
    # as 1.0 (no scaling) so existing behavior is preserved by default.
    _image_scale = float(args.image_scale or 1.0)
    if _image_scale <= 0.0 or _image_scale > 1.0:
        print(f"⚠️  --image-scale {_image_scale} out of range (0, 1.0]; ignoring (no scaling)")
        _image_scale = 1.0
    config_dict.setdefault("image", {})["scale"] = _image_scale
    if _image_scale < 1.0:
        print(
            f"✓ Image downscale: enabled (scale={_image_scale}, 800x600 -> "
            f"{int(round(800 * _image_scale))}x{int(round(600 * _image_scale))})"
        )

    # Image recent-steps: K most-recent history images stay at original
    # resolution; older ones are downscaled to --image-scale.
    _image_recent_steps = int(args.image_recent_steps or 0)
    if _image_recent_steps < 0:
        print(f"⚠️  --image-recent-steps {_image_recent_steps} < 0; treating as 0")
        _image_recent_steps = 0
    config_dict.setdefault("image", {})["recent_steps"] = _image_recent_steps
    if _image_recent_steps > 0 and _image_scale < 1.0:
        print(
            f"✓ Image recent-steps: keeping the {_image_recent_steps} most-recent "
            f"history images at original 800x600, older ones at scale={_image_scale}. "
            f"Current observation is always full-resolution."
        )
    elif _image_recent_steps > 0 and _image_scale >= 1.0:
        print(
            f"ℹ️  --image-recent-steps {_image_recent_steps} has no effect when "
            f"--image-scale is 1.0 (no downscaling configured)"
        )

    # Partner-view injection: feed the acting agent a fresh image rendered from
    # the partner body's camera.  ``--partner-view-scale`` defaults to
    # ``--image-scale``.
    dual_config["partner_view"] = bool(args.partner_view)
    _partner_view_scale = args.partner_view_scale
    if _partner_view_scale is None:
        _partner_view_scale = _image_scale
    _partner_view_scale = float(_partner_view_scale or 1.0)
    if _partner_view_scale <= 0.0 or _partner_view_scale > 1.0:
        print(
            f"⚠️  --partner-view-scale {_partner_view_scale} out of range (0, 1.0]; "
            f"ignoring (using --image-scale={_image_scale})"
        )
        _partner_view_scale = _image_scale
    config_dict.setdefault("image", {})["partner_view_scale"] = _partner_view_scale
    if args.partner_view:
        print(
            f"✓ Partner view: enabled (injects the partner body's current "
            f"first-person image at scale={_partner_view_scale} into each step)"
        )

    output_dir = args.output_dir or config.get("experiment.output_dir", "outputs")
    if args.output_dir and len(task_names) == 1:
        run_output_dir = args.output_dir
    else:
        run_output_dir = os.path.join(output_dir, f"dual_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_output_dir, exist_ok=True)

    agent_vlms, separate_models = get_agent_vlms(config_dict)

    all_results = []
    for task_idx, task_name in enumerate(task_names, 1):
        print(f"\n{'=' * 60}\n📋 Task {task_idx}/{len(task_names)}: {task_name}\n{'=' * 60}")
        task_config = config.apply_task_by_name(task_name, dual_agent=True)
        if args.max_steps is not None:
            task_config["max_steps"] = args.max_steps
            print(f"✓ max_steps overridden: {args.max_steps}")

        task_output_dir = os.path.join(run_output_dir, task_name) if len(task_names) > 1 else run_output_dir
        os.makedirs(task_output_dir, exist_ok=True)

        task_folder_path = task_config.get("task_folder_path") or os.path.join("tasks", task_name)
        init_actions = load_init_actions_from_task_folder(task_folder_path)

        full_config = deepcopy(config_dict)
        full_config["task"] = task_config
        full_config["init_actions"] = init_actions or []
        #      two-bodies：   agentCount=2；relocate agent2 near agent1
        full_config.setdefault("env", {})
        full_config["env"].setdefault("agent_count", 2)
        full_config.setdefault("dual_agent", {})
        full_config["dual_agent"].setdefault("relocate_agent2_near_agent1", True)
        full_config["dual_agent"].setdefault("second_agent_spawn_offset_m", 0.75)

        #    agent1/   agent2/      （AI2-THOR   ）
        os.makedirs(os.path.join(task_output_dir, "agent1"), exist_ok=True)
        os.makedirs(os.path.join(task_output_dir, "agent2"), exist_ok=True)

        try:
            env = ProcTHOREnvWrapper(
                scene_index=task_config.get("scene_index", 0),
                output_dir=task_output_dir,
                config=full_config,
                headless=args.headless,
            )
        except Exception as e:
            print(f"❌ Failed to create environment: {e}")
            import traceback

            traceback.print_exc()
            all_results.append({"task_name": task_name, "success": False, "step_count": 0, "fail_reason": str(e)})
            continue

        try:
            state = run_dual_agent_loop(
                env=env,
                agent_vlms=agent_vlms,
                task_config=task_config,
                task_output_dir=task_output_dir,
                config=full_config,
                collaboration_mode=args.collaboration_mode,
                switch_interval=args.switch_interval,
            )
            save_dual_conversation_log(state, task_output_dir)
            save_dual_episode_log(state, task_output_dir, env)
            all_results.append(
                {
                    "task_name": task_name,
                    "success": state.get("success", False),
                    "step_count": state.get("global_step_count", 0),
                    "fail_reason": state.get("fail_reason"),
                }
            )
            print(
                f"\n📊 Result: {'✅ Success' if state['success'] else '❌ Failure'}"
                f" | Steps: {state['global_step_count']}/{state['max_global_steps']}"
                f" | Agent1: {state['agent_1']['step_count']}"
                f" | Agent2: {state['agent_2']['step_count']}"
                f" | Separate models: {separate_models}"
            )
        except Exception as e:
            print(f"❌ Task error: {e}")
            import traceback

            traceback.print_exc()
            all_results.append({"task_name": task_name, "success": False, "step_count": 0, "fail_reason": str(e)})
        finally:
            env.close()

    success_count = sum(1 for r in all_results if r["success"])
    print(f"\n{'=' * 80}\n🎉 All Dual-Agent Tasks Completed\n{'=' * 80}")
    print(f"Total: {len(all_results)} | Success: {success_count} | Failure: {len(all_results) - success_count}")
    if all_results:
        print(f"Success Rate: {success_count / len(all_results) * 100:.1f}%")
    print(f"Output: {run_output_dir}\n{'=' * 80}\n")


if __name__ == "__main__":
    main()
