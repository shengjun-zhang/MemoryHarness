#!/usr/bin/env python3
"""AI2-THOR dual-agent main entry.

This module runs a true two-body collaboration loop (alternating or sequential
handoff) on a shared AI2-THOR scene. Unlike the legacy path that delegated to
the single-agent ``AgentRunner`` (which silently ignored agent 2), the loop
here explicitly:

* tracks per-agent state (observation, step count, short-term history, ...),
* switches the active agent via :func:`maybe_switch_agent` /
  :func:`handoff_agent_or_finish` according to ``switch_interval`` /
  ``collaboration_mode``,
* routes every action through ``env.step_with_action_dict(..., thor_agent_id=)``
  so each body acts on its own camera, and refreshes a body's observation with
  ``env.get_observation_for_agent()`` when control is handed over,
* verifies task completion with the terminal-state evaluator before trusting a
  model-emitted ``DONE``.

The public ``main()`` CLI is preserved so ``run_benchmark.py`` / subprocess
invocations keep working unchanged.
"""

import argparse
import base64
import json
import os
import re
import sys
import time
from copy import deepcopy
from io import BytesIO
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from dotenv import load_dotenv

_AI2THOR_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[3]
for _path in (str(_AI2THOR_ROOT), str(_REPO_ROOT)):
    while _path in sys.path:
        sys.path.remove(_path)
sys.path.insert(0, str(_REPO_ROOT))
sys.path.append(str(_AI2THOR_ROOT))

from actions.max_steps import (
    compute_dual_agent_max_steps_from_steps,
    derive_dual_golden_steps,
)
from actions.parser import parse_action_string
from mllm_base_agent.tools.memory import MemoryLibrary

from .config import load_config
from .core.prompts.dual_agent import COLLABORATIVE_AGENT_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Stall watchdog: periodically dumps every thread's live Python call stack to
# *stdout* (not a side file) so it flows through the existing
# subprocess_streaming.py capture straight into the cluster job log and
# live.log on the shared filesystem -- no need to separately fetch a file from
# whichever worker container happened to run the task. This turns a cluster
# hang (e.g. stuck right after "AI2ThorEnvWrapper.__init__ starting" with no
# further log lines) into a self-diagnosing one: the exact C/Python frame
# everything is blocked in shows up in live.log on its own, no guessing from
# source review or live process attach required. Cheap (a background thread +
# faulthandler, both stdlib) and always-on.
# ---------------------------------------------------------------------------
def _start_stall_watchdog(interval_seconds: float = 120.0) -> None:
    import faulthandler
    import threading

    def _dump_forever():
        while True:
            time.sleep(interval_seconds)
            try:
                print(
                    f"\n🩺 [WATCHDOG] all-thread stack dump @ "
                    f"{datetime.now().isoformat(timespec='seconds')} pid={os.getpid()} "
                    "(printed periodically so a hang shows exactly which line it's "
                    "stuck in; harmless/ignorable if the process is progressing "
                    "normally) =====",
                    flush=True,
                )
                faulthandler.dump_traceback(file=sys.stdout, all_threads=True)
                sys.stdout.flush()
            except Exception:
                pass

    t = threading.Thread(target=_dump_forever, name="stall-watchdog", daemon=True)
    t.start()
    print(
        f"  🩺 Stall watchdog started: will print all-thread stack dumps to stdout "
        f"every {interval_seconds:.0f}s for the lifetime of this process",
        flush=True,
    )


# Map logical agent id ("agent_1" / "agent_2") to AI2-THOR embodied agentId (0 / 1).
AGENT_TO_THOR_ID = {"agent_1": 0, "agent_2": 1}

LOCAL_RETRY_CONFIG = {
    "max_retries": 3,
    "api_max_retries": 5,
    "retry_delay": 2,
    "api_retry_delay": 5,
}
MODEL_HISTORY_TURNS = 29


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

HISTORY_ANALYZER_SYSTEM_PROMPT = """You are the History Analysis Agent for an embodied AI2-THOR dual-agent task.
You receive the task instruction and the acting agent's recent step sequence. Each step has the action it attempted and the environment result (SUCCESS or an error string).

Your job: produce a concise, actionable per-step analysis that will REPLACE the raw "action + error" text fed back to the acting agent. For EACH step write ONE short sentence (<=30 words) that captures (1) what the step tried / achieved (or why it failed) and (2) the concrete takeaway for the next move.

Grounding rules (follow strictly):
- Distance + "not in view": if distance < 1.0m the object is CLOSE but off-screen -> takeaway is to ROTATE (RotateLeft/RotateRight) or LookUp/LookDown, NOT to move closer. If distance >= 1.0m, move closer while keeping it in view.
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
# Config / model helpers
# ---------------------------------------------------------------------------

def get_vlm_display_name(vlm) -> str:
    """Get a human-readable model name for different chat model implementations."""
    return (
        getattr(vlm, "model_name", None)
        or getattr(vlm, "_model", None)
        or getattr(vlm, "model", None)
        or type(vlm).__name__
    )


def resolve_task_dir(task_id: str) -> Path:
    """Resolve a dual-agent task id or explicit task directory."""
    raw_path = Path(task_id).expanduser()
    candidates = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.extend([Path.cwd() / raw_path, _REPO_ROOT / raw_path])

    task_names = [raw_path.name]
    if "_" in raw_path.name:
        task_names.append(raw_path.name.replace("_", ""))
    elif raw_path.name.lower().startswith("ai2thor"):
        task_names.append(raw_path.name.replace("ai2thor", "ai2thor_", 1))

    for root in (
        _REPO_ROOT / "data" / "ai2thor" / "dual" / "tasks",
        _AI2THOR_ROOT / "tasks",
        _REPO_ROOT / "data" / "ai2thor" / "tasks",
    ):
        for name in task_names:
            candidates.append(root / name)

    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_dir() and (candidate / "task.json").exists():
            return candidate

    return _AI2THOR_ROOT / "tasks" / raw_path.name


def load_task_info(task_id: str) -> dict:
    """Load task.json + init.json for a dual-agent task id."""
    task_dir = resolve_task_dir(task_id)
    task_file = task_dir / "task.json"
    init_file = task_dir / "init.json"

    if not task_file.exists():
        raise FileNotFoundError(f"Task file not found: {task_file}")

    with open(task_file, "r", encoding="utf-8") as f:
        task_info = json.load(f)

    init_actions = []
    scene_name = None
    if init_file.exists():
        with open(init_file, "r", encoding="utf-8") as f:
            init_data = json.load(f)
            if isinstance(init_data, list):
                init_actions = init_data
            elif isinstance(init_data, dict):
                init_actions = init_data.get("actions", [])
                scene_name = init_data.get("scene")

    if init_actions and init_actions[-1].strip().upper() == "DONE":
        init_actions = init_actions[:-1]

    golden_action_steps = derive_dual_golden_steps(task_info)
    recommended_max_steps = (
        compute_dual_agent_max_steps_from_steps(golden_action_steps)
        if golden_action_steps is not None
        else None
    )

    return {
        "task_id": task_id,
        "instruction": task_info.get("instruction", ""),
        "scene": scene_name or task_info.get("scene", "FloorPlan1"),
        "init_actions": init_actions,
        "task_info": task_info,
        "golden_action_steps": golden_action_steps,
        "recommended_max_steps": recommended_max_steps,
    }


def execute_init_actions(env, init_actions: list):
    """Run init actions on agent 1 (agentId=0) to set up the scene."""
    if not init_actions:
        return 0, None

    print(f"\n{'=' * 60}")
    print(f"📁 Init actions ({len(init_actions)} steps)")
    print(f"{'=' * 60}")

    init_count = 0
    last_observation = None
    for i, action_str in enumerate(init_actions, 1):
        action_str = action_str.strip()
        if not action_str or action_str.upper() == "DONE":
            break

        print(f"  {i}. {action_str}")
        try:
            action_dict = parse_action_string(action_str)
            # Init actions always run on agent 1 (agentId=0).
            observation, error = env.step_with_action_dict(
                action_dict, thor_agent_id=0
            )
            last_observation = observation
            init_count += 1

            if error:
                print(f"     ⚠️  {error}")
        except Exception as e:
            print(f"     ❌ parse/step error: {e}")

    print(f"✓ Executed {init_count} init actions\n")
    return init_count, last_observation


def load_config_dict(config_path: str) -> dict:
    """Load config file into a plain dict."""
    config = load_config(config_path)
    return config.get_all() if hasattr(config, "get_all") else config.config


def extract_model_config_from_file(config_path: str) -> dict:
    """Extract model.vlm from a single-agent config file."""
    config_dict = load_config_dict(config_path)
    model_config = config_dict.get("model", {}).get("vlm", {})
    if not model_config:
        raise ValueError(f"Config missing model.vlm: {config_path}")
    return dict(model_config)


def apply_agent_model_overrides(
    config_dict: dict,
    agent1_config_path: str = None,
    agent2_config_path: str = None,
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
            raise FileNotFoundError(f"Agent config not found: {path}")

    agent1_model_config = extract_model_config_from_file(agent1_path)
    agent2_model_config = extract_model_config_from_file(agent2_path)

    config_dict.setdefault("model", {})
    config_dict.setdefault("dual_agent", {})

    if agent1_path == agent2_path:
        config_dict["model"]["vlm"] = agent1_model_config
        config_dict["dual_agent"]["use_separate_models"] = False
        print("✓ Shared model config")
        print(f"  Shared config: {agent1_path}")
        print(f"  Model: {agent1_model_config.get('model_name')}")
    else:
        config_dict["model"]["vlm"] = agent1_model_config
        config_dict["dual_agent"]["use_separate_models"] = True
        config_dict["dual_agent"]["agent_1"] = agent1_model_config
        config_dict["dual_agent"]["agent_2"] = agent2_model_config
        print("✓ Separate per-agent models")
        print(f"  Agent 1 config: {agent1_path}")
        print(f"  Agent 1 model: {agent1_model_config.get('model_name')}")
        print(f"  Agent 2 config: {agent2_path}")
        print(f"  Agent 2 model: {agent2_model_config.get('model_name')}")

    return config_dict


def create_vlm_from_config(vlm_config: Mapping[str, Any]):
    """Build a VLM, forwarding model_kwargs (e.g. doubao thinking=disabled)."""
    from mllm_base_agent.llm.provider import get_vlm

    kwargs = {
        "provider": vlm_config.get("provider", "openai"),
        "model_name": vlm_config.get("model_name") or vlm_config.get("model"),
        "temperature": vlm_config.get("temperature", 0.7),
        "max_tokens": vlm_config.get("max_tokens", 2000),
        "top_p": vlm_config.get("top_p"),
        "base_url": vlm_config.get("base_url") or vlm_config.get("api_base"),
        "api_key": vlm_config.get("api_key"),
    }
    # Critical: forward model_kwargs so provider-specific flags (e.g. doubao
    # `thinking: disabled`) are actually applied. The legacy wrapper dropped
    # this field, which caused long thinking traces and missing <ACTION> tags.
    model_kwargs = vlm_config.get("model_kwargs")
    if model_kwargs:
        kwargs["model_kwargs"] = dict(model_kwargs)
    return get_vlm(**kwargs)


def get_agent_vlms(config_dict: dict) -> Tuple[Dict[str, Any], bool]:
    """Create either one shared model or two separate per-agent models."""
    model_config = dict(config_dict.get("model", {}).get("vlm", {}) or {})
    dual_config = config_dict.get("dual_agent", {}) or {}
    use_separate_models = bool(dual_config.get("use_separate_models", False))

    def _vlm_display_name(vlm, cfg: dict) -> str:
        for attr in ("model_name", "_model", "model"):
            val = getattr(vlm, attr, None)
            if val:
                return str(val)
        return str(cfg.get("model_name") or cfg.get("model") or type(vlm).__name__)

    if not use_separate_models:
        shared = create_vlm_from_config(model_config)
        print(f"✓ Shared VLM: {_vlm_display_name(shared, model_config)}")
        return {"agent_1": shared, "agent_2": shared}, False

    agent_1_config = {**model_config, **(dual_config.get("agent_1", {}) or {})}
    agent_2_config = {**model_config, **(dual_config.get("agent_2", {}) or {})}
    agent_1_vlm = create_vlm_from_config(agent_1_config)
    agent_2_vlm = create_vlm_from_config(agent_2_config)
    print("✓ Separate per-agent VLMs")
    print(f"  Agent 1 model: {_vlm_display_name(agent_1_vlm, agent_1_config)}")
    print(f"  Agent 2 model: {_vlm_display_name(agent_2_vlm, agent_2_config)}")
    return {"agent_1": agent_1_vlm, "agent_2": agent_2_vlm}, True


def compute_recursion_limit(per_agent_max_steps: int) -> int:
    """Safety iteration cap for the dual-agent loop (no longer a graph limit)."""
    total_action_cap = max(1, 2 * int(per_agent_max_steps))
    return max(500, 15 * total_action_cap)


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
# Response parsing (THINK / ACTION / COMMUNICATE / SUMMARY)
# ---------------------------------------------------------------------------

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

    This is a pseudo-action (mirrors the ``Pass()`` turn-skip handling right
    below): it never reaches :func:`actions.parser.parse_action_string` or the
    AI2-THOR environment. It lets the agent consult the on-disk memory library
    (see :mod:`mllm_base_agent.tools.memory`) through the same lightweight
    text-action grammar it already uses for everything else, without needing a
    full OpenAI-style tool-calling loop.
    """
    match = _READ_MEMORY_PATTERN.match(action_string.strip())
    if not match:
        return None
    file_name = match.group(1).strip().strip("'\"")
    return file_name or "MEMORY.md"


def _build_memory_nudge_text(current_agent: dict) -> Optional[str]:
    """Return an escalating, situation-specific nudge towards ``ReadMemory``.

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


# Hard-gate threshold: once an agent has this many consecutive action
# failures, its next submitted action is REQUIRED to be ReadMemory(...) --
# anything else is rejected (re-prompted, not executed) up to
# ``FORCE_READ_MEMORY_MAX_REJECTIONS`` times. This exists because the softer
# text nudges (``_build_memory_nudge_text``) turned out to be necessary but
# not sufficient in practice: benchmark analysis showed models will happily
# *reference* the memory index's one-line summaries in their <THINK> text
# ("According to the memory entry, I should...") without ever actually
# emitting a ReadMemory(<file_name>) action to read the full entry, because
# the index summary alone often "feels" like enough context. Gating on
# action_type here forces the model to consult the real (more detailed,
# situation-specific) leaf note instead of free-associating from the index.
FORCE_READ_MEMORY_FAILURE_THRESHOLD = 2
# Safety valve: if the model still refuses to call ReadMemory after this many
# rejected attempts (e.g. it is stuck emitting malformed actions), stop
# forcing the issue and let the normal failure-handling path run, rather than
# burning the whole iteration_cap on a single stuck turn.
FORCE_READ_MEMORY_MAX_REJECTIONS = 3


def _should_force_read_memory(current_agent: dict) -> bool:
    """Whether the current agent's NEXT action is required to be ``ReadMemory(...)``.

    True once ``consecutive_failures`` has reached
    ``FORCE_READ_MEMORY_FAILURE_THRESHOLD`` for the CURRENT failure streak AND
    the agent has not yet consulted memory during this same streak (tracked
    via ``memory_consulted_for_streak``, set by a real ReadMemory lookup and
    reset to False whenever ``consecutive_failures`` increments -- see the
    action-execution error-handling block below). This "one free pass per
    streak" design is deliberate: the gate should force AT LEAST one real
    lookup per failure streak, then get out of the way and let the agent act
    on what it just read, rather than re-triggering on every single turn
    while the streak count stays flat (which would otherwise force it to
    ReadMemory again immediately after already doing so).
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
    """Parse THINK / ACTION / (COMMUNICATE) / (SUMMARY) blocks from a model response."""
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
    communication_text = (
        communicate_block.strip() if communicate_block and communicate_block.strip() else ""
    )
    updated_summary = summary_block.strip() if summary_block and summary_block.strip() else ""

    # Pass() is a turn-skip in the dual prompt; map it to a no-op so the env
    # wrapper does not receive an unknown action name.
    normalized = action_string
    memory_file_name = _parse_read_memory_action(normalized)
    if memory_file_name is not None:
        parsed_action = {
            "action_type": "memory_lookup",
            "action_name": "ReadMemory",
            "file_name": memory_file_name,
        }
    elif normalized.lower().rstrip("()") == "pass":
        parsed_action = {"action_type": "communication", "action_name": "Pass", "message": communication_text or "skip turn"}
    else:
        parsed_action = parse_action_string(normalized)
        if parsed_action.get("action_type") == "communication":
            communication_text = parsed_action.get("message", communication_text)

    return {
        "thinking_text": thinking_text,
        "action_string": action_string,
        "parsed_action": parsed_action,
        "communication_text": communication_text,
        "updated_summary": updated_summary,
    }


# ---------------------------------------------------------------------------
# Per-agent state + coordination primitives (ported from procthor dual loop)
# ---------------------------------------------------------------------------

def initialize_agent_state(agent_id: str, vlm: Any, observation: Any, max_steps: int) -> dict:
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
        # consecutive-failure streak (reset to False on every failure/success
        # transition; set True by a real ReadMemory lookup). Gates
        # _should_force_read_memory so the hard gate forces exactly one real
        # lookup per streak rather than re-triggering every turn.
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


def build_partner_trajectory(state: dict, current_agent_id: str) -> str:
    """Summarize the partner's recent actions for the current agent's prompt."""
    partner_id = "agent_2" if current_agent_id == "agent_1" else "agent_1"
    entries = state.get(partner_id, {}).get("structured_trajectory", []) or []
    entries = entries[-5:]
    if not entries:
        return "No partner actions yet."
    lines = []
    for e in entries:
        step = e.get("global_step", e.get("step", "?"))
        act = e.get("action_string", "")
        comm = e.get("communication", "")
        line = f"[{partner_id}] Step {step}: {act}"
        if comm:
            line += f" | said: {comm[:160]}"
        lines.append(line)
    return "\n".join(lines)


def consume_pending_messages(state: dict, current_agent_id: str) -> List[dict]:
    """Pop pending messages addressed to the current agent."""
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
    return agent_state.get("should_continue", True) and agent_state.get(
        "step_count", 0
    ) < agent_state.get("max_steps", 0)


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
    """Drop agents that died mid-step and close the step if all remaining acted."""
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
    """Choose the current agent according to collaboration mode and availability."""
    current_agent = state["current_agent"]
    other_agent = "agent_2" if current_agent == "agent_1" else "agent_1"
    mode = state.get("collaboration_mode", "alternating")

    if (
        mode == "alternating"
        and state.get("current_turn_steps", 0) >= switch_interval
        and agent_can_continue(state, other_agent)
    ):
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
        print(f"🔄 Handoff to {other_agent_id}: {reason}")
        return True

    if not state.get("fail_reason"):
        state["fail_reason"] = reason
    if failure_type is not None and not state.get("failure_type"):
        state["failure_type"] = failure_type
    print(f"⚠️  Partner {other_agent_id} cannot continue either: {reason}")
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


# ---------------------------------------------------------------------------
# Terminal-state evaluation
# ---------------------------------------------------------------------------

def perform_final_evaluation(env, task_config: dict, observation=None) -> Tuple[bool, float]:
    """Run the evaluator on the final environment state."""
    if not task_config:
        return False, 0.0

    if (not observation or not getattr(observation, "metadata", None)) and hasattr(
        env, "get_observation_for_agent"
    ):
        try:
            observation = env.get_observation_for_agent(0)
        except Exception:
            pass

    if not observation or not getattr(observation, "metadata", None):
        return False, 0.0

    try:
        from evaluation import create_evaluator_from_config

        evaluator = create_evaluator_from_config(task_config)
        score = evaluator.evaluate(env, observation.metadata)
        return score >= 1.0, score
    except Exception as e:
        print(f"❌ Evaluation error: {e}")
        return False, 0.0


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _resolve_image_scale(config: dict) -> float:
    """Return the image downscale factor from config (``image.scale``).

    Defaults to ``1.0`` (no scaling -> original raw bytes, zero behavior
    change). Values are clamped to ``(0, 1.0]`` so an accidental 0 / negative /
    >1 value disables scaling rather than crashing PIL or enlarging images.
    """
    try:
        scale = float(config.get("image", {}).get("scale", 1.0))
    except (TypeError, ValueError):
        return 1.0
    if scale <= 0.0 or scale > 1.0:
        return 1.0
    return scale


def _resolve_image_recent_steps(config: dict) -> int:
    """Return the number of recent history steps kept at original resolution.

    Reads ``image.recent_steps`` (default ``0``). When ``image.scale`` < 1.0,
    the ``K`` most-recent *history* images are sent at full resolution while
    older history images are downscaled to ``image.scale``. The current
    observation is **always** full-resolution regardless of this setting.

    ``K=0`` (default): only the current observation is full-resolution; every
    history image is downscaled (preserves the prior behavior for history).
    ``K=3``: the 3 most-recent history images + current observation are
    full-resolution; older history images are downscaled.
    """
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
    factor with PIL (LANCZOS) before base64-encoding. This substantially
    reduces the request body size on long multi-image episodes and helps avoid
    HTTP 413 (Request Entity Too Large) errors from the API gateway.

    When ``scale`` is ``None`` or ``>= 1.0`` the original raw bytes are
    returned unchanged, preserving the prior behavior exactly (no PIL
    re-encode, no quality loss, no extra dependency on the hot path).
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
        # Pillow >= 9.1 prefers Image.Resampling.LANCZOS; older versions expose
        # Image.LANCZOS directly. Resolve the best available constant.
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


# ---------------------------------------------------------------------------
# The dual-agent collaboration loop
# ---------------------------------------------------------------------------

def run_dual_agent_loop(
    env,
    agent_vlms: Dict[str, Any],
    task_config: dict,
    task_output_dir: str,
    config: dict,
    collaboration_mode: str = "alternating",
    switch_interval: int = 1,
    iteration_cap: int = 1000,
) -> dict:
    """Run a lightweight dual-agent collaboration loop on one shared AI2-THOR env.

    Both agents are equal peers (two bodies in the same scene). Control alternates
    every ``switch_interval`` actions; each body acts through its own ``agentId``
    and sees only its own first-person view, learning about the partner via
    ``<COMMUNICATE>`` messages.
    """
    task_prompt = (
        task_config.get("instruction")
        or task_config.get("description")
        or "Complete the task."
    )
    per_agent_steps = int(task_config.get("max_steps", config.get("max_steps", 30)))
    max_global_steps = 2 * per_agent_steps

    enable_summary = config.get("context_management", {}).get("enable_long_term_summary", False)
    configured_history = int(
        config.get("context_management", {}).get("short_term_history_window_size", MODEL_HISTORY_TURNS)
    )
    max_history = min(MODEL_HISTORY_TURNS, max(0, configured_history))

    initial_observation = env.reset(task_prompt)

    # Agent 2 is placed near agent 1; each body gets its own first-person view.
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
                    f"✓ Per-agent obs: agent1={agent_1_observation.image_path} | "
                    f"agent2={agent_2_observation.image_path}"
                )
            except Exception as e:
                print(f"⚠️  Per-agent observation refresh failed, sharing initial obs: {e}")

    state = {
        "task_prompt": task_prompt,
        "config": config,
        "task": task_config,
        "collaboration_mode": collaboration_mode,
        "switch_interval": switch_interval,
        "global_step_count": 0,
        "max_global_steps": max_global_steps,
        "system_step_expected_agents": [],
        "system_step_completed_agents": [],
        "current_agent": "agent_1",
        "current_turn_steps": 0,
        "turn_count": 0,
        # A single explicit handoff is useful; alternating Pass() calls are not.
        # Reset only after a real environment action, not after a memory lookup.
        "passes_since_env_action": 0,
        "communication_history": [],
        "message_queue": [],
        "success": False,
        "fail_reason": None,
        "failure_type": None,
        "token_usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "api_calls": 0,
        },
        "last_acting_agent": None,
        # Skill-memory library (mirrors RPent's resources/<env>/memory/): a
        # reviewed MEMORY.md index plus feedback_*.md leaf notes. The index is
        # embedded into the system prompt every turn; leaf notes are fetched
        # on demand via the ReadMemory(<file_name>) pseudo-action, which this
        # loop intercepts below (see the "memory_lookup" branch) instead of
        # forwarding it to the AI2-THOR environment.
        "memory_library": MemoryLibrary.for_env("ai2thor", agent_mode="dual") if bool((config.get("memory") or {}).get("enabled", True)) else None,
        "history_feedback": bool(config.get("dual_agent", {}).get("history_feedback", False)),
        "llm_history_feedback": bool(
            config.get("dual_agent", {}).get("llm_history_feedback", False)
        ),
        # Image downscale factor for VLM inputs (``image.scale``). 1.0 = no
        # scaling (original behavior); e.g. 0.5 resizes 800x600 -> 400x300.
        "image_scale": _resolve_image_scale(config),
        # Number of recent history steps kept at original resolution
        # (``image.recent_steps``). Older history images are downscaled to
        # ``image_scale``. The current observation is always full-resolution.
        "image_recent_steps": _resolve_image_recent_steps(config),
        # Partner-view injection: when enabled, each decision step also feeds the
        # model a fresh first-person image rendered from the partner body's
        # camera (env.get_observation_for_agent).  ``partner_view_scale`` controls
        # the resolution of that injected image (defaults to ``image_scale``).
        "partner_view": bool(config.get("dual_agent", {}).get("partner_view", False)),
        "partner_view_scale": float(
            (config.get("image", {}) or {}).get("partner_view_scale")
            or _resolve_image_scale(config)
        ),
        "agent_1": initialize_agent_state(
            "agent_1", agent_vlms["agent_1"], agent_1_observation, per_agent_steps
        ),
        "agent_2": initialize_agent_state(
            "agent_2", agent_vlms["agent_2"], agent_2_observation, per_agent_steps
        ),
    }

    from mllm_base_agent.llm.messages import AIMessage, HumanMessage, SystemMessage

    # LLM History Analyzer: one analyzer per agent, built from that agent's own
    # VLM ("use the current LLM").  Only instantiated when the feature is on;
    # the loop looks up ``state["history_analyzers"][current_agent_id]``.
    if state.get("llm_history_feedback"):
        state["history_analyzers"] = {
            aid: HistoryAnalyzerAgent(vlm) for aid, vlm in agent_vlms.items() if vlm is not None
        }
    else:
        state["history_analyzers"] = {}

    iterations = 0
    while state["global_step_count"] < state["max_global_steps"]:
        iterations += 1
        if iterations > iteration_cap:
            if not state.get("fail_reason"):
                state["fail_reason"] = f"Iteration cap reached ({iteration_cap})"
            print(f"🛑 Iteration cap reached ({iteration_cap}); stopping")
            break

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

        # When control is handed over, refresh the now-active agent's view from
        # its own body (a Pass step) so it reasons about what IT can see.
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

        print(
            f"\n{'=' * 60}\n🧠 {current_agent_id.upper()} "
            f"Step {current_agent['step_count'] + 1} "
            f"(global {state['global_step_count'] + 1}/{state['max_global_steps']})\n{'=' * 60}"
        )

        # --- Build messages -------------------------------------------------
        memory_library: Optional[MemoryLibrary] = state.get("memory_library")
        memory_index_block = memory_library.index_prompt_block() if memory_library else ""
        system_prompt = COLLABORATIVE_AGENT_SYSTEM_PROMPT.format(
            task_prompt=task_prompt,
            shared_context=build_shared_context(state, current_agent_id),
            partner_trajectory=build_partner_trajectory(state, current_agent_id),
            memory_index_block=memory_index_block,
        )
        messages = [SystemMessage(content=system_prompt)]

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
                    analyzer_annotations, analyzer_response = analyzer.analyze(
                        recent_history, task_prompt
                    )
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
                # Distance-based scaling: the last history entry is 1 step
                # ago, the oldest is N steps ago.  Entries within
                # ``image_recent_steps`` of the current step keep full
                # resolution (scale=1.0); older entries are downscaled to
                # ``image_scale`` to control request-body size.  This keeps
                # fine visual detail where the agent needs it most (recent
                # context) while still preventing HTTP 413 on long episodes.
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
                    "text": "**Messages from Partner:**\n"
                    + "\n".join(f"- {msg.get('message', '')}" for msg in pending_messages),
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
        # and the env actually has 2 embodied agents.  The partner image uses
        # ``partner_view_scale`` (defaults to ``image_scale``) since it is
        # auxiliary context, not the primary visual input for the action.
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

        current_content.append(
            # Current observation: always full resolution — it is the most
            # critical visual input for the agent's immediate decision.  When
            # image_scale < 1.0 only *historical* images are downscaled (and
            # even then only those older than image_recent_steps), never the
            # current frame.
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

        # --- Call VLM with API retries -------------------------------------
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
                response = current_agent["vlm"].invoke(messages)
                response_text = (
                    response.content if hasattr(response, "content") else str(response)
                )
                usage = _extract_token_usage_from_response(response)
                _accumulate_token_usage(state, usage)
                for key in step_token_usage:
                    step_token_usage[key] += usage.get(key, 0)
                break
            except Exception as api_error:
                err_str = str(api_error)
                is_api = any(
                    k in err_str.lower()
                    for k in [
                        "api",
                        "request failed",
                        "connection",
                        "timeout",
                        "timed out",
                        "http",
                        "429",
                        "500",
                        "400",
                    ]
                )
                if is_api and api_attempt < LOCAL_RETRY_CONFIG["api_max_retries"] - 1:
                    delay = (
                        LOCAL_RETRY_CONFIG["api_retry_delay"]
                        if "400" in err_str
                        else LOCAL_RETRY_CONFIG["retry_delay"]
                    )
                    print(
                        f"⚠️  API Error (attempt {api_attempt + 1}/"
                        f"{LOCAL_RETRY_CONFIG['api_max_retries']}): {err_str[:200]}"
                    )
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
                print(
                    f"⚠️  Parse Error (parse attempt {parse_attempt + 1}/"
                    f"{LOCAL_RETRY_CONFIG['max_retries']}): {e}"
                )
                if parse_attempt < LOCAL_RETRY_CONFIG["max_retries"] - 1:
                    print(
                        f"   Waiting {LOCAL_RETRY_CONFIG['retry_delay']}s before re-calling VLM..."
                    )
                    time.sleep(LOCAL_RETRY_CONFIG["retry_delay"])
                    try:
                        response = current_agent["vlm"].invoke(messages)
                        response_text = (
                            response.content if hasattr(response, "content") else str(response)
                        )
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
                # Hand off instead of killing the whole episode: the partner may
                # still make progress, and a single parse failure should not
                # abort a dual-agent run.
                handoff_agent_or_finish(
                    state,
                    current_agent_id,
                    "Missing ACTION after retries; hand off to partner",
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
            print(f"📨 Communication -> {other_agent_id}: {communication_text[:120]}...")

        action_dict = parsed["parsed_action"]
        action_string = parsed["action_string"]
        thinking_text = parsed["thinking_text"]

        print(f"✓ Thinking: {thinking_text[:200]}{'...' if len(thinking_text) > 200 else ''}")
        print(f"✓ Action String: {action_string}")
        print(f"✓ Parsed Action: {action_dict}")

        # --- Hard gate: force a real ReadMemory(...) after repeated failures --
        # Benchmark analysis showed the softer text nudges alone are necessary
        # but not sufficient: models will paraphrase the memory index's
        # one-line summaries in <THINK> ("According to the memory entry...")
        # without ever actually emitting ReadMemory(<file_name>) to read the
        # full, situation-specific entry. Once consecutive_failures reaches
        # FORCE_READ_MEMORY_FAILURE_THRESHOLD, any action OTHER than a real
        # ReadMemory(...) is discarded here -- not executed, no step consumed,
        # no handoff -- and the agent is re-prompted with an explicit
        # rejection message on its immediate next turn. Bounded by
        # FORCE_READ_MEMORY_MAX_REJECTIONS so a model that still refuses to
        # comply does not burn the whole iteration_cap on one stuck turn.
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
        # ReadMemory(<file_name>) never reaches the AI2-THOR environment and
        # does NOT increment current_agent["step_count"] / global_step_count
        # (mirrors RPent's "reading memory is free" design -- see
        # mllm_base_agent/tools/memory.py). The looked-up content is folded
        # into short_term_history as a synthetic "tool result" turn so the
        # SAME agent sees it on its immediate next call, then keeps its turn
        # (does not hand off to the partner) since no world-state action was
        # taken yet.
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
            # streak: mark it consulted (so _should_force_read_memory lets the
            # next real action through) and reset the rejection counter (so a
            # LATER, separate failure streak gets the full
            # FORCE_READ_MEMORY_MAX_REJECTIONS budget again instead of
            # inheriting an already-exhausted counter).
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
            is_pass = action_dict.get("action_name") == "Pass"
            bare_pass = is_pass and not communication_text.strip()
            repeated_pass = is_pass and state.get("passes_since_env_action", 0) >= 1
            rejected_pass = bare_pass or repeated_pass
            trajectory_entry["reward"] = 0.0
            if bare_pass:
                trajectory_entry["error_message"] = (
                    "Bare Pass rejected: include a concrete <COMMUNICATE> handoff, or take a real action."
                )
            elif repeated_pass:
                trajectory_entry["error_message"] = (
                    "Consecutive Pass rejected: the partner must take a real environment action before another handoff."
                )
            elif is_pass:
                trajectory_entry["error_message"] = "Pass accepted as one explicit partner handoff."
            else:
                trajectory_entry["error_message"] = None
            if is_pass:
                state["passes_since_env_action"] = state.get("passes_since_env_action", 0) + 1
                current_agent["last_error_message"] = trajectory_entry["error_message"]
                if rejected_pass:
                    current_agent["consecutive_failures"] += 1
                    current_agent["memory_consulted_for_streak"] = False
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
            if is_pass and not rejected_pass:
                state["current_agent"] = "agent_2" if current_agent_id == "agent_1" else "agent_1"
                state["current_turn_steps"] = 0
                state["turn_count"] += 1
                print(f"🔄 Pass handoff to {state['current_agent']}")
            elif is_pass:
                print(f"🚫 {trajectory_entry['error_message']}")
            else:
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
        state["passes_since_env_action"] = 0
        thor_agent_id = AGENT_TO_THOR_ID.get(current_agent_id, 0)
        try:
            if getattr(env, "agent_count", 1) > 1:
                # Pass the acting agent's last observation metadata so interaction
                # resolution (PickupObject etc.) uses THIS body's visible objects,
                # not the controller's last_event (which may belong to the partner).
                observation, error_message = env.step_with_action_dict(
                    action_dict,
                    thor_agent_id=thor_agent_id,
                    vision_metadata=getattr(observation, "metadata", None),
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

        if observation is not None:
            if getattr(env, "agent_count", 1) > 1:
                current_agent["observation"] = observation
            else:
                state["agent_1"]["observation"] = observation
                state["agent_2"]["observation"] = observation

        state["last_acting_agent"] = current_agent_id

        if error_message:
            current_agent["consecutive_failures"] += 1
            current_agent["last_error_message"] = error_message
            # A new failure within the streak means whatever memory entry was
            # read before (if any) did not resolve it -- re-arm the hard gate
            # in _should_force_read_memory so it can force a (possibly
            # different) lookup again once the threshold is re-reached.
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

    # Final terminal-state verification (covers step-budget exhaustion).
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
        if not state.get("failure_type"):
            state["failure_type"] = "model_error"
        print(f"❌ Final terminal-state evaluation failed (score={final_score:.2f})")

    return state


# ---------------------------------------------------------------------------
# Output serialization (kept compatible with run_benchmark.py consumers)
# ---------------------------------------------------------------------------

def _json_safe(value):
    """Convert runner state fragments to JSON-safe values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump())
    return str(value)


def _merge_trajectories(state: dict) -> List[dict]:
    trajectory = []
    for agent_id in ("agent_1", "agent_2"):
        for entry in state.get(agent_id, {}).get("structured_trajectory", []) or []:
            copied = dict(entry)
            copied["agent_id"] = agent_id
            trajectory.append(copied)
    trajectory.sort(key=lambda x: (x.get("global_step", 0), x.get("agent_id", "")))
    return trajectory


def save_dual_episode_log(final_state: dict, task_id: str, task_data: dict, output_dir: str) -> Path:
    """Persist the dual-agent result format expected by the benchmark wrapper."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scene = task_data.get("scene", "UnknownScene")
    safe_scene = str(scene).replace(" ", "_").replace("/", "_")[:80]
    safe_task = str(task_id).replace(" ", "_").replace("/", "_")[:80]
    path = Path(output_dir) / f"dual_episode_{safe_scene}_{safe_task}_{timestamp}.json"

    success = bool(final_state.get("success", False))
    fail_reason = final_state.get("fail_reason")
    failure_type = final_state.get("failure_type")
    if not success and not failure_type:
        failure_type = "model_error"

    trajectory = _merge_trajectories(final_state)
    global_step_count = int(final_state.get("global_step_count", 0) or 0)
    global_action_count = len(trajectory) or global_step_count
    agent_1_steps = int(final_state.get("agent_1", {}).get("step_count", 0) or 0)
    agent_2_steps = int(final_state.get("agent_2", {}).get("step_count", 0) or 0)

    env = final_state.get("env")
    action_sequence = None
    if env is not None and hasattr(env, "get_action_sequence"):
        try:
            action_sequence = env.get_action_sequence()
        except Exception:
            action_sequence = None

    episode = {
        "task_id": task_id,
        "task": task_data.get("instruction", ""),
        "scene": scene,
        "mode": "dual_agent",
        "success": success,
        "failure_type": failure_type,
        "fail_reason": fail_reason,
        "global_step_count": global_step_count,
        "global_action_count": global_action_count,
        "agent_1_steps": agent_1_steps,
        "agent_2_steps": agent_2_steps,
        "turn_count": int(final_state.get("turn_count", 0) or 0),
        "communication_history": _json_safe(final_state.get("communication_history", [])),
        "trajectory": _json_safe(trajectory),
        "action_sequence": action_sequence,
        "timestamp": datetime.now().isoformat(),
        "metadata": {
            "token_usage": _json_safe(final_state.get("token_usage", {})),
            "max_steps": final_state.get("agent_1", {}).get("max_steps"),
            "max_global_steps": final_state.get("max_global_steps"),
        },
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(episode, handle, ensure_ascii=False, indent=2)
    return path


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def main():
    _start_stall_watchdog()

    # Load .env from several candidate locations (repo root / dual_agent / local).
    load_dotenv()
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    load_dotenv(Path(__file__).resolve().parent / ".env")

    parser = argparse.ArgumentParser(
        description="AI2-THOR dual-agent main entry (true alternating collaboration)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single task
  python -m mllm_base_agent.dual_agent.ai2thor.main --task ai2thor05002

  # Multiple tasks
  python -m mllm_base_agent.dual_agent.ai2thor.main --task ai2thor05001 ai2thor05002

  # Override step budget and switch interval
  python -m mllm_base_agent.dual_agent.ai2thor.main --task ai2thor05002 --max-steps 40 --switch-interval 3

  # Per-agent models
  python -m mllm_base_agent.dual_agent.ai2thor.main --task ai2thor05002 \
      --config experiments/configs/ai2thor/dual/config_close_gpt-5.yaml \
      --agent1 experiments/configs/ai2thor/config_close_gpt-5.yaml \
      --agent2 experiments/configs/ai2thor/config_close_kimi-k25.yaml
        """,
    )

    parser.add_argument(
        "--task", type=str, nargs="+", default=None, help="Task ID(s), e.g. ai2thor05002"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        help="Alias for --task (accepted by run_csv_benchmark.py)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/ai2thor/dual/config_close_gpt-5.yaml",
        help="Dual-agent config (repo / abs / relative path all accepted)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=None, help="Per-agent max steps override"
    )
    parser.add_argument(
        "--switch-interval", type=int, default=None, help="Agent switch interval (actions)"
    )
    parser.add_argument(
        "--collaboration-mode",
        type=str,
        default=None,
        choices=["alternating", "parallel", "sequential"],
        help="Collaboration mode (accepted by run_csv_benchmark.py)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Force CloudRendering platform (accepted by run_csv_benchmark.py)",
    )
    parser.add_argument(
        "--agent1", type=str, default=None, help="Agent 1 single-agent config path"
    )
    parser.add_argument(
        "--agent2", type=str, default=None, help="Agent 2 single-agent config path"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Output dir (single task) or parent dir"
    )
    parser.add_argument(
        "--recursion-limit",
        type=int,
        default=None,
        help="Override the loop iteration safety cap (legacy graph flag)",
    )
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
            "regardless of this setting. e.g. --image-recent-steps 3 keeps the "
            "3 most-recent history images full-size and only downscales older "
            "ones. Default 0 = only the current observation is full-resolution; "
            "all history images are downscaled (prior behavior)."
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

    task_ids = list(args.task or []) + list(args.tasks or [])
    if not task_ids:
        parser.error("at least one of --task / --tasks is required")
    args.task = task_ids

    print(f"\n{'=' * 60}")
    print("🔧 Config")
    print(f"{'=' * 60}")
    print(f"  Config: {args.config}")

    config = load_config(args.config)
    config_dict = config.get_all() if hasattr(config, "get_all") else config.config
    config_dict = apply_agent_model_overrides(config_dict, args.agent1, args.agent2)
    dual_config = config_dict.get("dual_agent", {})

    if not dual_config.get("equal_collaboration", False):
        print("\n⚠️  Note: equal_collaboration is not true; running equal collaboration anyway")

    default_per_agent_max_steps = int(
        dual_config.get("per_agent_max_steps") or config_dict.get("max_steps", 30)
    )
    switch_interval = int(dual_config.get("switch_interval", 1))

    if args.max_steps:
        default_per_agent_max_steps = int(args.max_steps)
        print(f"✓ Per-agent max steps override: {default_per_agent_max_steps}")

    if args.switch_interval:
        switch_interval = int(args.switch_interval)
        config_dict.setdefault("dual_agent", {})["switch_interval"] = switch_interval
        print(f"✓ Switch interval: {switch_interval}")

    collaboration_mode = (
        args.collaboration_mode
        or dual_config.get("collaboration_mode")
        or "alternating"
    )
    config_dict.setdefault("dual_agent", {})["collaboration_mode"] = collaboration_mode

    if args.headless:
        config_dict.setdefault("env", {})["platform"] = "CloudRendering"
        print("✓ Headless: CloudRendering platform forced")

    config_dict.setdefault("dual_agent", {})["history_feedback"] = bool(args.history_feedback)
    if args.history_feedback:
        print("✓ History feedback: enabled (action + result injected into per-step history)")

    config_dict.setdefault("dual_agent", {})["llm_history_feedback"] = bool(
        args.llm_history_feedback
    )
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
    # resolution; older ones are downscaled to --image-scale. The current
    # observation is always full-resolution. Only meaningful when
    # --image-scale < 1.0.
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
    # the partner body's camera so it can observe the shared scene from a
    # different angle.  ``--partner-view-scale`` defaults to ``--image-scale``
    # (partner images are auxiliary, so they follow the same downscale policy as
    # historical images unless explicitly overridden).
    config_dict.setdefault("dual_agent", {})["partner_view"] = bool(args.partner_view)
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

    print("\n  Mode: Equal Collaboration")
    print(f"  Default per-agent max steps: {default_per_agent_max_steps}")
    print(f"  Switch interval: {switch_interval} actions")
    print(f"  Collaboration mode: {collaboration_mode}")

    # Build VLMs once (shared or per-agent).
    print(f"\n{'=' * 60}")
    print("🤖 VLM")
    print(f"{'=' * 60}")
    agent_vlms, separate_models = get_agent_vlms(config_dict)

    for task_id in args.task:
        print(f"\n{'=' * 60}")
        print(f"📋 Task: {task_id}")
        print(f"{'=' * 60}")

        try:
            task_data = load_task_info(task_id)
        except FileNotFoundError as e:
            print(f"❌ {e}")
            continue

        task_prompt = task_data.get("instruction", "")
        golden_action_steps = task_data.get("golden_action_steps")
        recommended_max_steps = task_data.get("recommended_max_steps")
        per_agent_max_steps = (
            args.max_steps or recommended_max_steps or default_per_agent_max_steps
        )
        global_max_steps = 2 * int(per_agent_max_steps)
        print(f"  Instruction: {task_prompt}")
        print(f"  Scene: {task_data['scene']}")
        print(f"  Init actions: {len(task_data['init_actions'])} steps")
        if golden_action_steps is not None:
            print(f"  golden_actions.steps: {golden_action_steps}")
            print(
                f"  Per-agent max steps (10+n): {per_agent_max_steps} | "
                f"Global cap: {global_max_steps}"
            )
        else:
            print(
                f"  Per-agent max steps: {per_agent_max_steps} | Global cap: {global_max_steps}"
            )

        iteration_cap = args.recursion_limit or compute_recursion_limit(per_agent_max_steps)
        print(f"  Iteration cap: {iteration_cap}")

        config_dict["task"] = task_data.get("task_info", {})
        config_dict.setdefault("dual_agent", {})
        config_dict["dual_agent"]["per_agent_max_steps"] = per_agent_max_steps
        config_dict["dual_agent"]["max_global_steps"] = global_max_steps

        print(f"\n{'=' * 60}")
        print("🌍 Environment")
        print(f"{'=' * 60}")

        from envs.ai2thor import AI2ThorEnvWrapper

        if args.output_dir:
            if len(args.task) == 1:
                output_dir = args.output_dir
            else:
                output_dir = str(Path(args.output_dir) / task_id)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"dual_agent/outputs/task_{task_id}_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(Path(output_dir) / "agent1", exist_ok=True)
        os.makedirs(Path(output_dir) / "agent2", exist_ok=True)

        env_config = config.get("env", {})
        config_dict.setdefault("env", env_config)
        # Ensure two embodied bodies (agentCount=2); YAML may omit env.agent_count.
        config_dict["env"].setdefault("agent_count", 2)

        env = AI2ThorEnvWrapper(
            scene=task_data["scene"],
            grid_size=env_config.get("grid_size", 0.25),
            width=env_config.get("width", 800),
            height=env_config.get("height", 600),
            output_dir=output_dir,
            config=config_dict,
        )

        print(f"✓ Scene ready: {task_data['scene']}")

        try:
            # Run init actions on agent 1 first (these set up object states).
            execute_init_actions(env, task_data["init_actions"])

            # task_config passed into the loop carries instruction + max_steps +
            # success conditions used by the terminal-state evaluator.
            task_config = deepcopy(task_data.get("task_info", {}) or {})
            task_config.setdefault("instruction", task_prompt)
            task_config["max_steps"] = per_agent_max_steps

            full_config = deepcopy(config_dict)
            full_config.setdefault("dual_agent", {})
            full_config["dual_agent"].setdefault("relocate_agent2_near_agent1", True)
            full_config["dual_agent"].setdefault("second_agent_spawn_offset_m", 0.75)

            print(f"\n{'=' * 60}")
            print("🚀 Dual-agent collaboration")
            print(f"{'=' * 60}")
            print("Agent 1 + Agent 2 equal collaboration")
            print(f"  Switch interval: {switch_interval} actions")
            print(
                f"  Per-agent max steps: {per_agent_max_steps} | Global cap: {global_max_steps}"
            )
            print(f"{'=' * 60}\n")

            final_state = run_dual_agent_loop(
                env=env,
                agent_vlms=agent_vlms,
                task_config=task_config,
                task_output_dir=output_dir,
                config=full_config,
                collaboration_mode=collaboration_mode,
                switch_interval=switch_interval,
                iteration_cap=iteration_cap,
            )
            final_state["env"] = env

            dual_episode_path = save_dual_episode_log(
                final_state, task_id, task_data, output_dir
            )
            print(f"✓ Dual episode saved: {dual_episode_path}")

            print(f"\n{'=' * 60}")
            print(f"📊 Result: {task_id}")
            print(f"{'=' * 60}")

            success = final_state.get("success", False)
            result_symbol = "✅" if success else "❌"
            result_text = "Success" if success else "Failure"
            print(f"{result_symbol} Result: {result_text}")
            print(f"  Global step count: {final_state.get('global_step_count', 0)}")
            print(
                f"  Global action count: {len(_merge_trajectories(final_state))}"
                f" / {global_max_steps}"
            )
            print(f"  Agent 1 steps: {final_state.get('agent_1', {}).get('step_count', 0)}")
            print(f"  Agent 2 steps: {final_state.get('agent_2', {}).get('step_count', 0)}")
            print(f"  Turn count: {final_state.get('turn_count', 0)}")
            print(
                f"  Communications: {len(final_state.get('communication_history', []))}"
            )

            if not success:
                fail_reason = final_state.get("fail_reason", "Unknown")
                print(f"  Fail reason: {fail_reason}")

            print(f"\n  Output dir: {output_dir}")
            print(f"{'=' * 60}\n")

        finally:
            env.close()


if __name__ == "__main__":
    main()
