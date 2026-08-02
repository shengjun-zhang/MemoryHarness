"""Internal Think -> Act -> Evaluate -> Final runner.

This replaces the previous external graph state machine with a plain Python loop while
keeping the public `.invoke()` and `.stream()` shape used by legacy scripts.
"""

from __future__ import annotations

import base64
import json
import os
import time
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, Iterable, Optional

from mllm_base_agent.agent.state import AgentState
from actions.response_parser import extract_tag_block, parse_vlm_response
from actions.max_steps import resolve_max_steps_from_task
from mllm_base_agent.llm.messages import AIMessage, HumanMessage, SystemMessage
from mllm_base_agent.prompts import get_system_prompt
import re

LOCAL_RETRY_CONFIG = {
    'max_retries': 3,
    'api_max_retries': 5,
    'retry_delay': 2,
    'api_retry_delay': 5,
}
MODEL_HISTORY_TURNS = 29

EXTERNAL_FAILURE_TYPES = {'api_error', 'env_error', 'external_error'}
MODEL_FAILURE_TYPES = {'parse_error', 'action_error', 'model_error'}


class GraphRecursionError(RuntimeError):
    """Compatibility exception for old graph error handling."""


class ParseRetryError(Exception):
    pass


class APIRetryError(Exception):
    pass


def _success_value_for_failure_type(failure_type: Optional[str]) -> Optional[bool]:
    if failure_type in EXTERNAL_FAILURE_TYPES:
        return None
    if failure_type:
        return False
    return None


def _normalize_token_usage(raw_usage: Optional[dict]) -> Dict[str, int]:
    usage = raw_usage or {}

    def to_int(value: Any) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    prompt_tokens = to_int(usage.get('prompt_tokens'))
    completion_tokens = to_int(usage.get('completion_tokens'))
    total_tokens = to_int(usage.get('total_tokens')) or prompt_tokens + completion_tokens
    api_calls = to_int(usage.get('api_calls')) or (1 if total_tokens else 0)
    return {
        'prompt_tokens': prompt_tokens,
        'completion_tokens': completion_tokens,
        'total_tokens': total_tokens,
        'api_calls': api_calls,
    }


def _extract_token_usage_from_response(response: Any) -> Dict[str, int]:
    if response is None:
        return _normalize_token_usage({})
    metadata = getattr(response, 'response_metadata', None) or {}
    if isinstance(metadata, dict) and metadata.get('token_usage'):
        return _normalize_token_usage(metadata['token_usage'])
    usage_metadata = getattr(response, 'usage_metadata', None)
    if isinstance(usage_metadata, dict):
        return _normalize_token_usage(usage_metadata)
    additional_kwargs = getattr(response, 'additional_kwargs', None) or {}
    if isinstance(additional_kwargs, dict):
        return _normalize_token_usage(additional_kwargs.get('token_usage') or additional_kwargs.get('usage'))
    return _normalize_token_usage({})


def _accumulate_token_usage(state: AgentState, token_usage: Dict[str, int]) -> None:
    if 'token_usage' not in state or not isinstance(state.get('token_usage'), dict):
        state['token_usage'] = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0, 'api_calls': 0}
    usage = state['token_usage']
    normalized = _normalize_token_usage(token_usage)
    for key in ('prompt_tokens', 'completion_tokens', 'total_tokens', 'api_calls'):
        usage[key] = int(usage.get(key, 0) or 0) + normalized[key]


def _resolve_image_scale(config: dict) -> float:
    """Return the image downscale factor from config (``image.scale``).

    Defaults to ``1.0`` (no scaling -> original raw bytes, zero behavior
    change). Values are clamped to ``(0, 1.0]`` so an accidental 0 / negative
    / >1 value disables scaling rather than crashing PIL or enlarging images.
    """
    try:
        scale = float((config.get('image', {}) or {}).get('scale', 1.0))
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
    """
    try:
        recent_steps = int((config.get('image', {}) or {}).get('recent_steps', 0))
    except (TypeError, ValueError):
        return 0
    if recent_steps < 0:
        return 0
    return recent_steps


def _read_image_as_data_url(
    image_path: str,
    max_retries: int = 1,
    retry_delay: int = 0,
    scale: float = 1.0,
) -> str:
    """Read an image file and return a ``data:image/...;base64,...`` URL.

    When ``scale`` is in ``(0, 1)`` the image is downscaled by that factor
    with PIL (LANCZOS) before base64-encoding. This substantially reduces the
    request body size on long multi-image episodes and helps avoid HTTP 413
    (Request Entity Too Large) errors from the API gateway.

    When ``scale`` is ``None``/``>= 1.0`` or PIL is unavailable, the original
    raw bytes are returned unchanged (fast path, zero behavior change).
    """
    last_error: Optional[BaseException] = None
    for attempt in range(max_retries):
        try:
            if not scale or scale >= 1.0:
                with open(image_path, 'rb') as handle:
                    image_data = base64.b64encode(handle.read()).decode('utf-8')
                return f'data:image/png;base64,{image_data}'
            try:
                from PIL import Image
            except ImportError:
                with open(image_path, 'rb') as handle:
                    image_data = base64.b64encode(handle.read()).decode('utf-8')
                return f'data:image/png;base64,{image_data}'
            try:
                resample = getattr(getattr(Image, 'Resampling', Image), 'LANCZOS', Image.LANCZOS)
                with Image.open(image_path) as img:
                    w, h = img.size
                    new_w = max(1, int(round(w * scale)))
                    new_h = max(1, int(round(h * scale)))
                    img = img.convert('RGB')
                    if (new_w, new_h) != (w, h):
                        img = img.resize((new_w, new_h), resample)
                    buf = BytesIO()
                    img.save(buf, format='PNG', optimize=True)
                    image_data = base64.b64encode(buf.getvalue()).decode('utf-8')
                return f'data:image/png;base64,{image_data}'
            except Exception:
                with open(image_path, 'rb') as handle:
                    image_data = base64.b64encode(handle.read()).decode('utf-8')
                return f'data:image/png;base64,{image_data}'
        except Exception as exc:
            last_error = exc
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
    raise OSError(f'Failed to read image after {max_retries} attempts: {last_error}')


# --- ReadMemory(<file_name>) pseudo-action -----------------------------
#
# Mirrors the dual-agent text-protocol handling in
# mllm_base_agent/dual_agent/ai2thor/main.py::_parse_read_memory_action:
# this is a free, no-env-step lookup into the skill-memory library (see
# mllm_base_agent/tools/memory.py::MemoryLibrary), intercepted here before
# the action string ever reaches actions.parser.parse_action_string / the
# environment. It lets the single-agent runner consult its on-disk memory
# library through the exact same <ACTION> grammar it already uses for
# everything else, without needing a full tool-calling loop.
_READ_MEMORY_PATTERN = re.compile(r"^read\s*memory\s*\(\s*(.*?)\s*\)$", re.IGNORECASE)


def _parse_read_memory_action(action_string: str) -> Optional[str]:
    """Return the requested memory file name if ``action_string`` is a
    ``ReadMemory(<file_name>)`` call, else ``None``.
    """
    match = _READ_MEMORY_PATTERN.match((action_string or '').strip())
    if not match:
        return None
    file_name = match.group(1).strip().strip("'\"")
    return file_name or 'MEMORY.md'


def _memory_index_prompt_block(state: AgentState) -> str:
    """Return the memory-library index block to splice into the system prompt.

    Returns "" (no-op) when no memory library is configured for this run,
    so environments/setups without a memory library behave identically to
    before this feature existed.
    """
    memory_library = state.get('memory_library')
    if memory_library is None:
        return ""
    try:
        return memory_library.index_prompt_block()
    except Exception:
        return ""


def _handle_read_memory_action(state: AgentState, file_name: str, response_text: str) -> str:
    """Look up ``file_name`` in the memory library and return a text block
    describing the result, to be shown to the model on its immediate next turn.

    Mirrors the dual-agent runner's ``memory_lookup`` handling: this never
    increments ``step_count`` and is not recorded as a real trajectory step,
    it is folded into ``short_term_history`` as a synthetic observation so
    the SAME turn's decision loop can immediately re-prompt the model with
    the looked-up content.
    """
    memory_library = state.get('memory_library')
    if memory_library is None:
        return f"[Memory] ReadMemory({file_name}) failed: memory library not configured"

    # Tracks real lookups this episode, purely to drive _build_memory_nudge_text's
    # "you have never opened the library" escalation below.
    state['memory_reads_used'] = int(state.get('memory_reads_used', 0) or 0) + 1

    lookup_result = memory_library.read_entry(file_name)
    if 'error' in lookup_result:
        text = f"[Memory] ReadMemory({file_name}) failed: {lookup_result['error']}"
        print(f"⚠️  {text}")
        return text

    content = lookup_result.get('content', '')
    print(f"📖 Memory lookup: {file_name} ({lookup_result.get('size', 0)} chars)")
    return f"[Memory] Contents of {file_name}:\n{content}"


def _build_memory_nudge_text(state: AgentState) -> Optional[str]:
    """Return an escalating, situation-specific nudge towards ``ReadMemory``.

    Single-agent counterpart of the dual-agent runners'
    ``_build_memory_nudge_text`` (see
    ``mllm_base_agent/dual_agent/ai2thor/main.py``). A generic one-line
    mention of the memory library in the system prompt is easy for the model
    to skim past; this produces a short, blunt reminder appended to the
    *current* turn whose urgency scales with the number of consecutive
    failures and whether the agent has ever opened the library at all this
    episode.
    """
    consecutive_failures = _count_consecutive_failures(state)
    if state.get("memory_library") is None:
        return None
    memory_reads_used = int(state.get('memory_reads_used', 0) or 0)

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


def _build_messages(state: AgentState, image_url: str) -> list:
    config = state.get('config') or {}
    env_type = str(config.get('env', {}).get('type', 'ai2thor')).lower()
    context_config = config.get('context_management', {}) or {}
    enable_summary = bool(context_config.get('enable_long_term_summary', False))
    task_cfg = config.get('task', {}) or {}
    actions_cfg = config.get('actions', {}) or {}
    executor_type = state.get('executor_type')
    input_modality = task_cfg.get('input_modality') or state.get('input_modality')
    navigation_mode = actions_cfg.get('navigation_mode', 'discrete')
    vh_objects = None
    env = state.get('env')
    if env_type == 'virtualhome' and env is not None and hasattr(env, 'get_scene_interactable_object_types'):
        try:
            vh_objects = env.get_scene_interactable_object_types()
        except Exception:
            vh_objects = None
    success_criteria_block = state.get(
        'success_criteria_block',
        'Complete the task according to the instruction. Use EndTask(DONE) only after confirming success.',
    )
    prompt = get_system_prompt(
        env_type,
        enable_summary=enable_summary,
        executor_type=executor_type,
        input_modality=input_modality,
        navigation_mode=navigation_mode,
        virtualhome_interactable_object_types=vh_objects,
    ).format(
        task_prompt=state.get('task_prompt', 'Complete the task.'),
        success_criteria_block=success_criteria_block,
    )
    memory_block = _memory_index_prompt_block(state)
    if memory_block:
        prompt = f"{prompt}\n\n{memory_block}"
    elif state.get("memory_library") is None:
        prompt = f"{prompt}\n\nThis run has no skill-memory library. ReadMemory is unavailable; never output it."
    messages = [SystemMessage(content=prompt)]
    long_term_summary = state.get('long_term_summary', '')
    history = (state.get('short_term_history', []) or [])[-MODEL_HISTORY_TURNS:]
    # Downscale older history images to keep the request body size bounded on
    # long episodes (avoids HTTP 413 Request Entity Too Large from the API
    # gateway). The ``image_recent_steps`` most-recent history entries (and
    # the current observation, appended below) always stay full-resolution.
    image_scale = _resolve_image_scale(config)
    image_recent_steps = _resolve_image_recent_steps(config)
    n_hist = len(history)
    for idx, entry in enumerate(history):
        content = []
        if idx == 0 and enable_summary and long_term_summary.strip():
            content.append({'type': 'text', 'text': f'**Previous Exploration Summary (Long-term Memory):**\n{long_term_summary}\n\n---\n'})
        step_text = f"Step {entry.get('step', '?')}"
        # A ReadMemory(...) lookup result (see _handle_read_memory_action):
        # shown as plain text with no image, since it never touched the
        # environment and has no associated observation frame.
        memory_lookup_result = entry.get('memory_lookup_result')
        if memory_lookup_result:
            content.append({'type': 'text', 'text': f"{step_text}\n{memory_lookup_result}"})
            messages.append(HumanMessage(content=content))
            messages.append(AIMessage(content=entry.get('raw_response', '')))
            continue
        content.append({'type': 'text', 'text': step_text})
        hist_image = entry.get('image_path')
        if hist_image and os.path.exists(hist_image):
            distance_from_end = n_hist - idx
            entry_scale = 1.0 if distance_from_end <= image_recent_steps else image_scale
            try:
                content.append({'type': 'image_url', 'image_url': {'url': _read_image_as_data_url(hist_image, 1, 0, scale=entry_scale)}})
            except Exception:
                content.append({'type': 'text', 'text': '[Image unavailable]'})
        messages.append(HumanMessage(content=content))
        messages.append(AIMessage(content=entry.get('raw_response', '')))
    current_content = []
    if not history and enable_summary and long_term_summary.strip():
        current_content.append({'type': 'text', 'text': f'**Previous Exploration Summary (Long-term Memory):**\n{long_term_summary}\n\n---\n'})
    goal_image_path = state.get('goal_image_path')
    if goal_image_path and state.get('step_count', 0) == 0:
        try:
            current_content.append({'type': 'text', 'text': '**Goal Image (your target destination):**'})
            current_content.append({'type': 'image_url', 'image_url': {'url': _read_image_as_data_url(goal_image_path, 1, 0)}})
        except Exception:
            pass
    memory_nudge = _build_memory_nudge_text(state)
    if memory_nudge:
        current_content.append({'type': 'text', 'text': memory_nudge})
    done_checklist_nudge = _build_done_checklist_nudge_text(state)
    if done_checklist_nudge:
        current_content.append({'type': 'text', 'text': done_checklist_nudge})
    current_content.append({'type': 'text', 'text': f"Step {state.get('step_count', 0)}"})
    current_content.append({'type': 'image_url', 'image_url': {'url': image_url}})
    messages.append(HumanMessage(content=current_content))
    return messages



# ============================================================================
# Lookahead decision mode: "act-then-choose" (预演所有候选动作后再决策)
#
# Standard mode (default, unchanged): think_node picks ONE action from the raw
# current observation, then act_node executes it.
#
# Lookahead mode (opt-in via state['decision_mode'] == 'lookahead'): at step t,
# BEFORE asking the model to decide, we enumerate a set of candidate discrete
# actions (navigation + interaction actions inferred from currently-visible
# objects), actually execute each candidate against the live environment to
# obtain its resulting observation/image, then roll the environment back to
# the pre-step snapshot (position/rotation/posture/camera pitch + internal
# step counters) so no candidate is "spent". All candidate observations are
# then shown to the VLM together, and the model chooses which action to
# actually commit. The chosen action is executed for real (advancing the
# environment + step_count), continuing into the existing act_node/evaluate_node
# pipeline unchanged.
#
# This is entirely opt-in and additive: when decision_mode != 'lookahead', none
# of the code below is invoked and behavior is 100% identical to before.
# ============================================================================

DEFAULT_LOOKAHEAD_NAV_ACTIONS = (
    'MoveAhead', 'MoveBack', 'MoveLeft', 'MoveRight', 'RotateLeft', 'RotateRight',
)

# (attribute_flag_in_metadata, action_name_if_true, action_name_if_false_or_none)
# None means "no alternate action for the false case" (e.g. Pickup has no false-branch).
_INTERACTION_ATTR_RULES = [
    ('openable', 'isOpen', 'CloseObject', 'OpenObject'),
    ('toggleable', 'isToggled', 'ToggleObjectOff', 'ToggleObjectOn'),
    ('pickupable', None, None, 'PickupObject'),
    ('sliceable', 'isSliced', None, 'SliceObject'),
]


def _resolve_lookahead_config(config: dict) -> Dict[str, Any]:
    """Read the ``decision.lookahead`` config block with safe defaults."""
    decision_cfg = (config.get('decision') or {}) if isinstance(config, dict) else {}
    lookahead_cfg = decision_cfg.get('lookahead') or {}
    nav_actions = lookahead_cfg.get('navigation_actions') or list(DEFAULT_LOOKAHEAD_NAV_ACTIONS)
    try:
        max_interaction_candidates = int(lookahead_cfg.get('max_interaction_candidates', 4))
    except (TypeError, ValueError):
        max_interaction_candidates = 4
    return {
        'navigation_actions': list(nav_actions),
        'include_interaction_candidates': bool(lookahead_cfg.get('include_interaction_candidates', True)),
        'max_interaction_candidates': max(0, max_interaction_candidates),
    }


def _lookahead_thor_agent_id(state: AgentState) -> Optional[int]:
    """Which embodied AI2-THOR/ProcTHOR agentId to snapshot/act on (None for single-agent envs)."""
    env = state.get('env')
    agent_count = getattr(env, 'agent_count', 1) or 1
    if agent_count <= 1:
        return None
    return int(state.get('thor_agent_id', 0) or 0)


def _snapshot_agent_pose(env: Any, thor_agent_id: Optional[int]) -> Optional[Dict[str, Any]]:
    """Capture enough state to fully restore the embodied agent after a probe action.

    Uses the AI2-THOR/ProcTHOR ``last_event`` metadata (position/rotation/standing/
    camera pitch) plus the wrapper's own step bookkeeping (``step_counter`` and
    ``action_sequence`` length) so a probe action leaves zero footprint once
    ``_restore_agent_pose`` is called.
    """
    controller = getattr(env, 'controller', None)
    if controller is None or getattr(controller, 'last_event', None) is None:
        return None
    metadata = controller.last_event.metadata
    agent_index = int(thor_agent_id or 0)
    agents = metadata.get('agents')
    if isinstance(agents, list) and len(agents) > agent_index:
        agent_meta = agents[agent_index]
    else:
        agent_meta = metadata.get('agent') or {}
    position = agent_meta.get('position') or {}
    rotation = agent_meta.get('rotation') or {}
    if not position or not rotation:
        return None
    return {
        'position': dict(position),
        'rotation': dict(rotation),
        'horizon': float(agent_meta.get('cameraHorizon', 0.0) or 0.0),
        'standing': bool(agent_meta.get('isStanding', True)),
        'thor_agent_id': thor_agent_id,
        'step_counter': int(getattr(env, 'step_counter', 0) or 0),
        'action_sequence_len': len(getattr(env, 'action_sequence', []) or []),
    }


def _restore_agent_pose(env: Any, snapshot: Optional[Dict[str, Any]]) -> bool:
    """Teleport the agent back to a previously captured snapshot and undo step bookkeeping."""
    if not snapshot:
        return False
    controller = getattr(env, 'controller', None)
    if controller is None:
        return False
    step_kwargs: Dict[str, Any] = dict(
        action='TeleportFull',
        position=snapshot['position'],
        rotation=snapshot['rotation'],
        horizon=snapshot['horizon'],
        standing=snapshot['standing'],
        forceAction=True,
    )
    if snapshot.get('thor_agent_id') is not None:
        step_kwargs['agentId'] = int(snapshot['thor_agent_id'])
    try:
        controller.step(**step_kwargs)
    except Exception:
        # Older ai2thor versions expect x/y/z instead of a position dict.
        try:
            pos = snapshot['position']
            step_kwargs.pop('position', None)
            step_kwargs.update(x=pos.get('x', 0.0), y=pos.get('y', 0.0), z=pos.get('z', 0.0))
            controller.step(**step_kwargs)
        except Exception:
            return False
    # Undo step bookkeeping so the probe action leaves no trace in logs/step counts.
    if hasattr(env, 'step_counter'):
        env.step_counter = snapshot.get('step_counter', env.step_counter)
    action_sequence = getattr(env, 'action_sequence', None)
    if isinstance(action_sequence, list):
        keep_len = snapshot.get('action_sequence_len', len(action_sequence))
        del action_sequence[keep_len:]
    return True


def _nearest_visible_object(objects: list, predicate) -> Optional[dict]:
    candidates = [obj for obj in objects if obj.get('visible') and predicate(obj)]
    if not candidates:
        return None
    return min(candidates, key=lambda obj: obj.get('distance', float('inf')))


def _build_lookahead_interaction_candidates(state: AgentState, lookahead_cfg: Dict[str, Any]) -> list:
    """Infer a small set of interaction candidates from currently-visible objects.

    Mirrors the same object attributes used by the environment wrappers'
    ``_find_interaction_target`` (openable/toggleable/pickupable/sliceable) but is
    read-only here: it only decides *which action names/object types* are worth
    probing, the actual environment target resolution still happens inside
    ``step_with_action_dict`` when the candidate is executed.
    """
    if not lookahead_cfg.get('include_interaction_candidates', True):
        return []
    observation = state.get('observation')
    metadata = getattr(observation, 'metadata', None) if observation else None
    if not isinstance(metadata, dict):
        return []
    objects = metadata.get('objects') or []
    is_holding_object = bool(metadata.get('inventoryObjects'))

    candidates = []
    seen_action_names = set()
    limit = lookahead_cfg.get('max_interaction_candidates', 4)

    for attr_flag, state_flag, action_if_true, action_if_false in _INTERACTION_ATTR_RULES:
        if limit and len(candidates) >= limit:
            break
        action_name = None
        obj = None
        if attr_flag == 'pickupable':
            if is_holding_object:
                continue
            obj = _nearest_visible_object(objects, lambda o: o.get('pickupable', False))
            action_name = action_if_false
        else:
            obj = _nearest_visible_object(objects, lambda o, f=attr_flag: o.get(f, False))
            if obj is not None and state_flag is not None:
                action_name = action_if_true if obj.get(state_flag, False) else action_if_false
            elif obj is not None:
                action_name = action_if_false
        if obj is None or not action_name or action_name in seen_action_names:
            continue
        object_type = obj.get('objectType')
        if not object_type:
            continue
        seen_action_names.add(action_name)
        candidates.append({
            'action_dict': {
                'action_type': 'interaction',
                'action_name': action_name,
                'object_type': object_type,
            },
            'action_string': f'{action_name}({object_type})',
        })

    # A visible receptacle only matters when the agent is already holding something.
    if is_holding_object and (not limit or len(candidates) < limit):
        receptacle = _nearest_visible_object(objects, lambda o: o.get('receptacle', False))
        if receptacle is not None and receptacle.get('objectType'):
            object_type = receptacle['objectType']
            candidates.append({
                'action_dict': {
                    'action_type': 'interaction',
                    'action_name': 'PutObject',
                    'object_type': object_type,
                },
                'action_string': f'PutObject({object_type})',
            })

    return candidates


def _build_lookahead_candidates(state: AgentState) -> list:
    """Build the full candidate action list (navigation + interaction) for step t."""
    config = state.get('config') or {}
    lookahead_cfg = _resolve_lookahead_config(config)
    candidates = []
    for action_name in lookahead_cfg['navigation_actions']:
        candidates.append({
            'action_dict': {'action_type': 'navigation', 'action_name': action_name},
            'action_string': action_name,
        })
    candidates.extend(_build_lookahead_interaction_candidates(state, lookahead_cfg))
    return candidates


def _rollout_lookahead_candidates(state: AgentState) -> list:
    """Execute every candidate action once, capture its observation, then roll back.

    Returns a list of dicts: ``{action_dict, action_string, image_path, error_message,
    text_state}``. Candidates whose probe execution raises are skipped (not shown to
    the model) rather than aborting the whole step.
    """
    env = state.get('env')
    candidates = _build_lookahead_candidates(state)
    thor_agent_id = _lookahead_thor_agent_id(state)
    rollouts = []
    for candidate in candidates:
        snapshot = _snapshot_agent_pose(env, thor_agent_id)
        try:
            probe_kwargs = {}
            if thor_agent_id is not None:
                probe_kwargs['thor_agent_id'] = thor_agent_id
            observation, error_message = env.step_with_action_dict(candidate['action_dict'], **probe_kwargs)
        except TypeError:
            # Environment wrapper does not accept thor_agent_id kwarg (single-agent envs).
            try:
                observation, error_message = env.step_with_action_dict(candidate['action_dict'])
            except Exception as exc:
                observation, error_message = None, str(exc)
        except Exception as exc:
            observation, error_message = None, str(exc)
        if observation is not None:
            rollouts.append({
                'action_dict': candidate['action_dict'],
                'action_string': candidate['action_string'],
                'image_path': getattr(observation, 'image_path', None),
                'error_message': error_message,
                'text_state': getattr(observation, 'text_state', ''),
            })
        if snapshot is not None:
            _restore_agent_pose(env, snapshot)
    return rollouts


def _resolve_lookahead_image_scale(config: dict) -> float:
    """Return the downscale factor used for lookahead candidate preview images.

    Candidate images are auxiliary (like history/partner-view images), and a
    single step can carry many of them (6 navigation + up to
    ``max_interaction_candidates`` interaction candidates). Left at full
    resolution, they dominate the request body size and are the main cause of
    HTTP 413 (Request Entity Too Large) errors in lookahead mode even when
    ``--image-scale``/``--image-recent-steps`` are set, because those settings
    previously only applied to *history* images, not lookahead candidates.

    Defaults to ``image.lookahead_scale`` if explicitly configured, otherwise
    falls back to the same ``image.scale`` used for history downscaling so a
    single ``--image-scale`` flag now also bounds candidate image size.
    """
    image_cfg = (config.get('image', {}) or {}) if isinstance(config, dict) else {}
    if 'lookahead_scale' in image_cfg:
        try:
            scale = float(image_cfg.get('lookahead_scale', 1.0))
        except (TypeError, ValueError):
            return _resolve_image_scale(config)
        if scale <= 0.0 or scale > 1.0:
            return 1.0
        return scale
    return _resolve_image_scale(config)


def _build_lookahead_messages(state: AgentState, image_url: str, rollouts: list) -> list:
    """Like ``_build_messages`` but appends one image per probed candidate action.

    Reuses the exact same system prompt / short-term-history construction as the
    standard mode (via ``_build_messages``) so history formatting is unaffected;
    it only augments the *current* step's human message with the candidate
    previews and an explicit instruction to choose one of them.
    """
    messages = _build_messages(state, image_url)
    if not rollouts:
        return messages

    config = state.get('config') or {}
    candidate_scale = _resolve_lookahead_image_scale(config)

    lookahead_content = [{
        'type': 'text',
        'text': (
            'Lookahead preview: each candidate action below has ALREADY been tentatively '
            'executed from the current position and then undone, so you can see its exact '
            'resulting view before deciding. Choose ONE action from this list (or DONE/FAIL) '
            'as your <ACTION> for this step; it will then be executed for real.'
        ),
    }]
    for idx, rollout in enumerate(rollouts, 1):
        label = f"Candidate {idx}: {rollout['action_string']}"
        if rollout.get('error_message'):
            label += f" (execution failed: {rollout['error_message']})"
        lookahead_content.append({'type': 'text', 'text': label})
        image_path = rollout.get('image_path')
        if image_path and os.path.exists(image_path):
            try:
                lookahead_content.append({
                    'type': 'image_url',
                    'image_url': {'url': _read_image_as_data_url(image_path, 1, 0, scale=candidate_scale)},
                })
            except Exception:
                lookahead_content.append({'type': 'text', 'text': '[Candidate image unavailable]'})
    messages.append(HumanMessage(content=lookahead_content))
    return messages


def lookahead_think_node(state: AgentState) -> AgentState:
    """Lookahead variant of ``think_node``: probe all candidates, then let the VLM pick one.

    Structurally mirrors ``think_node`` (same retry/failure-handling/trajectory-logging
    contract) so ``act_node``/``evaluate_node`` downstream need no changes. The only
    behavioral difference is that the prompt sent to the model additionally contains
    one preview image per candidate action, obtained by actually stepping the
    environment and rolling it back before the "real" step is taken.
    """
    observation = state['observation']
    vlm = state['vlm']
    state.setdefault('structured_trajectory', [])
    state.setdefault('conversation_history', [])
    state.setdefault('short_term_history', [])
    state.setdefault('long_term_summary', '')
    max_retries = LOCAL_RETRY_CONFIG['max_retries']
    api_max_retries = LOCAL_RETRY_CONFIG.get('api_max_retries', max_retries)
    retry_delay = LOCAL_RETRY_CONFIG['retry_delay']
    api_retry_delay = LOCAL_RETRY_CONFIG['api_retry_delay']
    config = state.get('config') or {}
    env_type = str(config.get('env', {}).get('type', 'ai2thor')).lower()
    enable_summary = bool((config.get('context_management') or {}).get('enable_long_term_summary', False))

    try:
        image_url = _read_image_as_data_url(observation.image_path, max_retries, retry_delay)
    except Exception as exc:
        state['failure_type'] = 'external_error'
        state['fail_reason'] = str(exc)
        state['should_continue'] = False
        state['success'] = None
        return state

    try:
        rollouts = _rollout_lookahead_candidates(state)
    except Exception as exc:
        # Lookahead probing itself must never corrupt the live environment state;
        # if it fails unexpectedly, fall back to a normal (non-lookahead) decision
        # for this step rather than aborting the whole episode.
        rollouts = []
        print(f"⚠️  Lookahead candidate rollout failed, falling back to standard decision for this step: {exc}")

    # Outer loop: see think_node's identical loop for the full rationale.
    # The candidate rollouts above are probed exactly ONCE per real decision
    # (they actually step + roll back the live environment), so a ReadMemory
    # lookup here only re-runs the (cheap) message-building + VLM call, not
    # the environment probing.
    max_memory_lookups_per_turn = 5
    memory_consulted_this_turn = False
    for _memory_lookup_round in range(max_memory_lookups_per_turn + 1):
        messages = _build_lookahead_messages(state, image_url, rollouts)
        last_error: Optional[BaseException] = None
        response_text: Optional[str] = None
        step_token_usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0, 'api_calls': 0}

        for api_attempt in range(api_max_retries):
            try:
                response = vlm.invoke(messages)
                response_text = getattr(response, 'content', str(response))
                usage = _extract_token_usage_from_response(response)
                _accumulate_token_usage(state, usage)
                for key in step_token_usage:
                    step_token_usage[key] += usage.get(key, 0)
                break
            except Exception as exc:
                last_error = exc
                if api_attempt < api_max_retries - 1:
                    time.sleep(api_retry_delay)
                else:
                    state['failure_type'] = 'api_error'
                    state['fail_reason'] = f'API error after {api_max_retries} attempts: {exc}'
                    state['should_continue'] = False
                    state['success'] = None
                    state['structured_trajectory'].append({
                        'step': state.get('step_count', 0),
                        'thinking': '',
                        'action_string': '',
                        'action': {},
                        'raw_response': '',
                        'llm_token_usage': dict(step_token_usage),
                        'parse_error': state['fail_reason'],
                        'failure_type': 'api_error',
                        'reward': None,
                        'image_path': observation.image_path,
                    })
                    return state

        raw_action_block = extract_tag_block(response_text or '', 'ACTION')
        memory_file_name = _parse_read_memory_action(raw_action_block or '')
        if memory_file_name is not None and state.get("memory_library") is not None and _memory_lookup_round < max_memory_lookups_per_turn:
            memory_text = _handle_read_memory_action(state, memory_file_name, response_text or '')
            state.setdefault('short_term_history', []).append({
                'step': state.get('step_count', 0),
                'action_string': f'ReadMemory({memory_file_name})',
                'reward': 0,
                'image_path': None,
                'raw_response': (response_text or '')[:2000],
                'error_message': None,
                'memory_lookup_result': memory_text,
            })
            context = (state.get('config') or {}).get('context_management') or {}
            configured_history = int(context.get('short_term_history_window_size', MODEL_HISTORY_TURNS) or MODEL_HISTORY_TURNS)
            max_history = min(MODEL_HISTORY_TURNS, max(0, configured_history))
            if len(state['short_term_history']) > max_history:
                state['short_term_history'] = state['short_term_history'][-max_history:]
            memory_consulted_this_turn = True
            continue

        # --- Hard gate: force a real ReadMemory(...) after repeated failures --
        # See FORCE_READ_MEMORY_FAILURE_THRESHOLD / _should_force_read_memory.
        # Discards this action (no step consumed) and re-prompts within the
        # same memory sub-loop, bounded by max_memory_lookups_per_turn.
        if (
            memory_file_name is None
            and _memory_lookup_round < max_memory_lookups_per_turn
            and _should_force_read_memory(state, memory_consulted_this_turn)
        ):
            rejection_text = _force_read_memory_rejection_text(state)
            state.setdefault('short_term_history', []).append({
                'step': state.get('step_count', 0),
                'action_string': '',
                'reward': 0,
                'image_path': None,
                'raw_response': (response_text or '')[:2000],
                'error_message': None,
                'memory_lookup_result': f'[Memory Gate] {rejection_text}',
            })
            context = (state.get('config') or {}).get('context_management') or {}
            configured_history = int(context.get('short_term_history_window_size', MODEL_HISTORY_TURNS) or MODEL_HISTORY_TURNS)
            max_history = min(MODEL_HISTORY_TURNS, max(0, configured_history))
            if len(state['short_term_history']) > max_history:
                state['short_term_history'] = state['short_term_history'][-max_history:]
            continue

        for parse_attempt in range(max_retries):
            try:
                parsed = parse_vlm_response(
                    response_text or '',
                    enable_summary=enable_summary,
                    env_type=env_type,
                    executor_type=state.get('executor_type'),
                )
                action = parsed['parsed_action']
                is_completion = action.get('action_type') == 'task_completion'
                state['next_action'] = action
                state['should_continue'] = not is_completion
                state['task_done_by_model'] = action.get('action_name') == 'DONE'
                state['task_fail_by_model'] = action.get('action_name') == 'FAIL'
                if enable_summary and parsed.get('updated_summary'):
                    state['long_term_summary'] = parsed['updated_summary']
                trajectory_step = {
                    'step': state.get('step_count', 0),
                    'thinking': parsed['thinking_text'],
                    'action_string': parsed['action_string'],
                    'action': action,
                    'updated_summary': parsed.get('updated_summary', ''),
                    'raw_response': (response_text or '')[:2000],
                    'llm_token_usage': dict(step_token_usage),
                    'parse_error': None,
                    'retry_count': parse_attempt,
                    'reward': None,
                    'observation_summary': None,
                    'image_path': observation.image_path,
                    'lookahead_candidates': [r['action_string'] for r in rollouts],
                }
                state['structured_trajectory'].append(trajectory_step)
                state['conversation_history'].append({
                    'step': state.get('step_count', 0),
                    'user_message': f"Step {state.get('step_count', 0)} (lookahead: {len(rollouts)} candidates previewed)",
                    'assistant_response': response_text or '',
                    'llm_token_usage': dict(step_token_usage),
                    'action_executed': '',
                    'reward': None,
                    'error_message': None,
                })
                state['failure_type'] = None
                return state
            except Exception as exc:
                last_error = exc
                if parse_attempt < max_retries - 1:
                    try:
                        response = vlm.invoke(messages)
                        response_text = getattr(response, 'content', str(response))
                        usage = _extract_token_usage_from_response(response)
                        _accumulate_token_usage(state, usage)
                        for key in step_token_usage:
                            step_token_usage[key] += usage.get(key, 0)
                    except Exception as api_exc:
                        last_error = api_exc
                    time.sleep(retry_delay)

        state['failure_type'] = 'parse_error'
        state['fail_reason'] = f"Step {state.get('step_count', 0)} parse failed after {max_retries} retries: {last_error}"
        state['should_continue'] = False
        state['success'] = False
        state['structured_trajectory'].append({
            'step': state.get('step_count', 0),
            'thinking': '',
            'action_string': '',
            'action': {},
            'raw_response': (response_text or '')[:2000],
            'llm_token_usage': dict(step_token_usage),
            'parse_error': state['fail_reason'],
            'failure_type': 'parse_error',
            'reward': None,
            'image_path': observation.image_path,
        })
        return state

    state['failure_type'] = 'parse_error'
    state['fail_reason'] = (
        f"Step {state.get('step_count', 0)} exceeded {max_memory_lookups_per_turn} "
        "consecutive ReadMemory lookups without committing to a real action"
    )
    state['should_continue'] = False
    state['success'] = False
    state['structured_trajectory'].append({
        'step': state.get('step_count', 0),
        'thinking': '',
        'action_string': '',
        'action': {},
        'raw_response': '',
        'llm_token_usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0, 'api_calls': 0},
        'parse_error': state['fail_reason'],
        'failure_type': 'parse_error',
        'reward': None,
        'image_path': observation.image_path,
    })
    return state


def think_node(state: AgentState) -> AgentState:
    observation = state['observation']
    vlm = state['vlm']
    state.setdefault('structured_trajectory', [])
    state.setdefault('conversation_history', [])
    state.setdefault('short_term_history', [])
    state.setdefault('long_term_summary', '')
    max_retries = LOCAL_RETRY_CONFIG['max_retries']
    api_max_retries = LOCAL_RETRY_CONFIG.get('api_max_retries', max_retries)
    retry_delay = LOCAL_RETRY_CONFIG['retry_delay']
    api_retry_delay = LOCAL_RETRY_CONFIG['api_retry_delay']
    config = state.get('config') or {}
    env_type = str(config.get('env', {}).get('type', 'ai2thor')).lower()
    enable_summary = bool((config.get('context_management') or {}).get('enable_long_term_summary', False))

    try:
        image_url = _read_image_as_data_url(observation.image_path, max_retries, retry_delay)
    except Exception as exc:
        state['failure_type'] = 'external_error'
        state['fail_reason'] = str(exc)
        state['should_continue'] = False
        state['success'] = None
        return state

    # Outer loop: normally runs exactly once. When the model's <ACTION> is a
    # ReadMemory(<file_name>) pseudo-action (see _parse_read_memory_action),
    # the lookup result is folded into short_term_history as a synthetic,
    # image-less turn and the loop re-prompts the SAME think_node call for a
    # real decision -- mirroring the dual-agent runner's "reading memory is
    # free" behavior (no step_count increment, no act_node/env involvement).
    # A generous but finite cap avoids an infinite loop if the model keeps
    # requesting memory files forever instead of ever acting.
    max_memory_lookups_per_turn = 5
    memory_consulted_this_turn = False
    for _memory_lookup_round in range(max_memory_lookups_per_turn + 1):
        messages = _build_messages(state, image_url)
        last_error: Optional[BaseException] = None
        response_text: Optional[str] = None
        step_token_usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0, 'api_calls': 0}

        for api_attempt in range(api_max_retries):
            try:
                response = vlm.invoke(messages)
                response_text = getattr(response, 'content', str(response))
                usage = _extract_token_usage_from_response(response)
                _accumulate_token_usage(state, usage)
                for key in step_token_usage:
                    step_token_usage[key] += usage.get(key, 0)
                break
            except Exception as exc:
                last_error = exc
                if api_attempt < api_max_retries - 1:
                    time.sleep(api_retry_delay)
                else:
                    state['failure_type'] = 'api_error'
                    state['fail_reason'] = f'API error after {api_max_retries} attempts: {exc}'
                    state['should_continue'] = False
                    state['success'] = None
                    state['structured_trajectory'].append({
                        'step': state.get('step_count', 0),
                        'thinking': '',
                        'action_string': '',
                        'action': {},
                        'raw_response': '',
                        'llm_token_usage': dict(step_token_usage),
                        'parse_error': state['fail_reason'],
                        'failure_type': 'api_error',
                        'reward': None,
                        'image_path': observation.image_path,
                    })
                    return state

        # Free memory-lookup interception: check the raw <ACTION> content
        # before handing it to parse_vlm_response/parse_action_string (which
        # do not know about this pseudo-action and would raise on it).
        raw_action_block = extract_tag_block(response_text or '', 'ACTION')
        memory_file_name = _parse_read_memory_action(raw_action_block or '')
        if memory_file_name is not None and state.get("memory_library") is not None and _memory_lookup_round < max_memory_lookups_per_turn:
            memory_text = _handle_read_memory_action(state, memory_file_name, response_text or '')
            state.setdefault('short_term_history', []).append({
                'step': state.get('step_count', 0),
                'action_string': f'ReadMemory({memory_file_name})',
                'reward': 0,
                'image_path': None,
                'raw_response': (response_text or '')[:2000],
                'error_message': None,
                'memory_lookup_result': memory_text,
            })
            context = (state.get('config') or {}).get('context_management') or {}
            configured_history = int(context.get('short_term_history_window_size', MODEL_HISTORY_TURNS) or MODEL_HISTORY_TURNS)
            max_history = min(MODEL_HISTORY_TURNS, max(0, configured_history))
            if len(state['short_term_history']) > max_history:
                state['short_term_history'] = state['short_term_history'][-max_history:]
            memory_consulted_this_turn = True
            # Same turn, no step consumed: loop back and re-prompt for a
            # real decision now that the memory content is in context.
            continue

        # --- Hard gate: force a real ReadMemory(...) after repeated failures --
        # See FORCE_READ_MEMORY_FAILURE_THRESHOLD / _should_force_read_memory.
        # Discards this action (no step consumed) and re-prompts within the
        # same memory sub-loop, bounded by max_memory_lookups_per_turn.
        if (
            memory_file_name is None
            and _memory_lookup_round < max_memory_lookups_per_turn
            and _should_force_read_memory(state, memory_consulted_this_turn)
        ):
            rejection_text = _force_read_memory_rejection_text(state)
            state.setdefault('short_term_history', []).append({
                'step': state.get('step_count', 0),
                'action_string': '',
                'reward': 0,
                'image_path': None,
                'raw_response': (response_text or '')[:2000],
                'error_message': None,
                'memory_lookup_result': f'[Memory Gate] {rejection_text}',
            })
            context = (state.get('config') or {}).get('context_management') or {}
            configured_history = int(context.get('short_term_history_window_size', MODEL_HISTORY_TURNS) or MODEL_HISTORY_TURNS)
            max_history = min(MODEL_HISTORY_TURNS, max(0, configured_history))
            if len(state['short_term_history']) > max_history:
                state['short_term_history'] = state['short_term_history'][-max_history:]
            continue

        for parse_attempt in range(max_retries):
            try:
                parsed = parse_vlm_response(
                    response_text or '',
                    enable_summary=enable_summary,
                    env_type=env_type,
                    executor_type=state.get('executor_type'),
                )
                action = parsed['parsed_action']
                is_completion = action.get('action_type') == 'task_completion'
                state['next_action'] = action
                state['should_continue'] = not is_completion
                state['task_done_by_model'] = action.get('action_name') == 'DONE'
                state['task_fail_by_model'] = action.get('action_name') == 'FAIL'
                if enable_summary and parsed.get('updated_summary'):
                    state['long_term_summary'] = parsed['updated_summary']
                trajectory_step = {
                    'step': state.get('step_count', 0),
                    'thinking': parsed['thinking_text'],
                    'action_string': parsed['action_string'],
                    'action': action,
                    'updated_summary': parsed.get('updated_summary', ''),
                    'raw_response': (response_text or '')[:2000],
                    'llm_token_usage': dict(step_token_usage),
                    'parse_error': None,
                    'retry_count': parse_attempt,
                    'reward': None,
                    'observation_summary': None,
                    'image_path': observation.image_path,
                }
                state['structured_trajectory'].append(trajectory_step)
                state['conversation_history'].append({
                    'step': state.get('step_count', 0),
                    'user_message': f"Step {state.get('step_count', 0)}",
                    'assistant_response': response_text or '',
                    'llm_token_usage': dict(step_token_usage),
                    'action_executed': '',
                    'reward': None,
                    'error_message': None,
                })
                state['failure_type'] = None
                return state
            except Exception as exc:
                last_error = exc
                if parse_attempt < max_retries - 1:
                    try:
                        response = vlm.invoke(messages)
                        response_text = getattr(response, 'content', str(response))
                        usage = _extract_token_usage_from_response(response)
                        _accumulate_token_usage(state, usage)
                        for key in step_token_usage:
                            step_token_usage[key] += usage.get(key, 0)
                    except Exception as api_exc:
                        last_error = api_exc
                    time.sleep(retry_delay)

        state['failure_type'] = 'parse_error'
        state['fail_reason'] = f"Step {state.get('step_count', 0)} parse failed after {max_retries} retries: {last_error}"
        state['should_continue'] = False
        state['success'] = False
        state['structured_trajectory'].append({
            'step': state.get('step_count', 0),
            'thinking': '',
            'action_string': '',
            'action': {},
            'raw_response': (response_text or '')[:2000],
            'llm_token_usage': dict(step_token_usage),
            'parse_error': state['fail_reason'],
            'failure_type': 'parse_error',
            'reward': None,
            'image_path': observation.image_path,
        })
        return state

    # Exhausted the memory-lookup round cap without the model ever
    # committing to a real action: treat as a parse error so the episode
    # fails cleanly instead of looping forever.
    state['failure_type'] = 'parse_error'
    state['fail_reason'] = (
        f"Step {state.get('step_count', 0)} exceeded {max_memory_lookups_per_turn} "
        "consecutive ReadMemory lookups without committing to a real action"
    )
    state['should_continue'] = False
    state['success'] = False
    state['structured_trajectory'].append({
        'step': state.get('step_count', 0),
        'thinking': '',
        'action_string': '',
        'action': {},
        'raw_response': '',
        'llm_token_usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0, 'api_calls': 0},
        'parse_error': state['fail_reason'],
        'failure_type': 'parse_error',
        'reward': None,
        'image_path': observation.image_path,
    })
    return state


def act_node(state: AgentState) -> AgentState:
    if state.get('think_failed') or state.get('failure_type') in {'api_error', 'parse_error', 'external_error'}:
        return state
    action = state.get('next_action')
    if not action:
        state['failure_type'] = 'action_error'
        state['fail_reason'] = 'No action available from think_node'
        state['should_continue'] = False
        state['success'] = False
        return state
    action_type = action.get('action_type')
    action_name = action.get('action_name')
    observation = state.get('observation')
    error_message = None
    if action_type == 'task_completion':
        state['step_count'] = int(state.get('step_count', 0) or 0) + 1
    else:
        try:
            observation, error_message = state['env'].step_with_action_dict(action)
            if observation is None:
                state['failure_type'] = 'action_error'
                state['fail_reason'] = f'Model output invalid action: {error_message}'
                state['should_continue'] = False
                state['success'] = False
                return state
            state['observation'] = observation
            state['step_count'] = int(state.get('step_count', 0) or 0) + 1
        except Exception as exc:
            state['failure_type'] = 'env_error'
            state['fail_reason'] = f'Environment exception: {exc}'
            state['should_continue'] = False
            state['success'] = None
            return state

    if state.get('structured_trajectory'):
        last_step = state['structured_trajectory'][-1]
        last_step['reward'] = 0 if action_type == 'task_completion' else getattr(observation, 'reward', 0)
        last_step['observation_summary'] = f'Task completion: {action_name}' if action_type == 'task_completion' else getattr(observation, 'text_state', '')
        last_step['error_message'] = error_message
    if state.get('conversation_history'):
        last_conv = state['conversation_history'][-1]
        last_conv['action_executed'] = action_name
        last_conv['reward'] = 0 if action_type == 'task_completion' else getattr(observation, 'reward', 0)
        last_conv['error_message'] = error_message

    context = (state.get('config') or {}).get('context_management') or {}
    configured_history = int(context.get('short_term_history_window_size', MODEL_HISTORY_TURNS) or MODEL_HISTORY_TURNS)
    max_history = min(MODEL_HISTORY_TURNS, max(0, configured_history))
    action_string = action_name
    if action.get('object_type'):
        action_string = f"{action_name}({action.get('object_type')})"
    state.setdefault('short_term_history', []).append({
        'step': int(state.get('step_count', 1) or 1) - 1,
        'action_string': action_string,
        'reward': 0 if action_type == 'task_completion' else getattr(observation, 'reward', 0),
        'image_path': getattr(observation, 'image_path', None),
        'raw_response': state.get('structured_trajectory', [{}])[-1].get('raw_response', ''),
        'error_message': error_message,
    })
    if len(state['short_term_history']) > max_history:
        state['short_term_history'] = state['short_term_history'][-max_history:]
    return state


def _count_consecutive_failures(state: AgentState) -> int:
    count = 0
    for step in reversed(state.get('structured_trajectory', [])):
        reward = step.get('reward')
        if reward is None or reward < 0.05:
            count += 1
        else:
            break
    return count


# Hard-gate threshold: mirrors the dual-agent runners'
# ``FORCE_READ_MEMORY_FAILURE_THRESHOLD`` (see
# ``mllm_base_agent/dual_agent/ai2thor/main.py``). Once the trajectory shows
# this many consecutive failed/near-zero-reward steps, the model's next
# submitted action (within the current turn's ReadMemory sub-loop) is
# REQUIRED to be ReadMemory(...); anything else is discarded and re-prompted.
# Exists because the softer text nudge (``_build_memory_nudge_text`` below)
# alone lets models paraphrase the memory index's summaries in <THINK>
# without ever emitting a real ReadMemory(<file_name>) action.
FORCE_READ_MEMORY_FAILURE_THRESHOLD = 2


def _should_force_read_memory(state: AgentState, memory_consulted_this_turn: bool) -> bool:
    """Whether the model's next action THIS turn is required to be ``ReadMemory(...)``.

    ``memory_consulted_this_turn`` tracks whether a real ReadMemory lookup has
    already happened within the current think_node call's memory sub-loop
    (see callers) -- once it has, the gate lets a real action through even if
    the trajectory-level consecutive-failure count hasn't changed yet.
    """
    if state.get("memory_library") is None:
        return False
    if memory_consulted_this_turn:
        return False
    return _count_consecutive_failures(state) >= FORCE_READ_MEMORY_FAILURE_THRESHOLD


def _force_read_memory_rejection_text(state: AgentState) -> str:
    """Feedback shown when a non-ReadMemory action is rejected by the hard gate."""
    consecutive_failures = _count_consecutive_failures(state)
    return (
        f"**Action rejected -- not executed.** You have {consecutive_failures} consecutive "
        "failed/low-reward actions, which requires consulting the memory library before "
        "any other action is accepted. Your submitted action was discarded (no step "
        "budget was spent). Your ONLY valid <ACTION> this turn is `ReadMemory(<file_name>)` "
        "-- pick the entry from the index above that matches your current error. Any "
        "other action will continue to be rejected."
    )


def _build_done_checklist_nudge_text(state: AgentState) -> Optional[str]:
    """Return a one-time reminder to re-read the DONE-verification memory entry.

    Single-agent counterpart of the dual-agent runners'
    ``_build_done_checklist_nudge_text``. Fires exactly once per episode, the
    first time the remaining step budget drops to (or below) 20% of
    ``max_steps`` -- the point where a model is most tempted to claim DONE
    prematurely to avoid running out of budget.
    """
    if state.get('done_checklist_nudged'):
        return None
    max_steps = int(state.get('max_steps', 0) or 0)
    step_count = int(state.get('step_count', 0) or 0)
    if max_steps <= 0:
        return None
    remaining_fraction = (max_steps - step_count) / max_steps
    if remaining_fraction > 0.2:
        return None
    state['done_checklist_nudged'] = True
    return (
        "**Budget check:** you are in the final ~20% of your step budget. "
        "Before outputting DONE, run `ReadMemory(feedback_done_verification.md)` "
        "(free lookup) and re-verify every subgoal yourself -- do not claim "
        "DONE based on assumption alone."
    )


def perform_final_evaluation(state: AgentState = None, *, env=None, task_config: dict = None, observation=None) -> tuple:
    try:
        from evaluation import create_evaluator_from_config
        if state is not None:
            config = state.get('config') or {}
            task_config = config.get('task') or {}
            env = state.get('env')
            observation = state.get('observation')
        if not task_config or observation is None or not getattr(observation, 'metadata', None):
            return False, 0.0
        evaluator = create_evaluator_from_config(task_config)
        score = evaluator.evaluate(env, observation.metadata)
        return score >= 1.0, score
    except Exception:
        return False, 0.0


def evaluate_node(state: AgentState) -> AgentState:
    if state.get('task_done_by_model'):
        success, _score = perform_final_evaluation(state)
        state['success'] = success
        state['fail_reason'] = None if success else 'Model claimed DONE but success conditions not met'
        state['should_continue'] = False
        return state
    if state.get('task_fail_by_model'):
        state['success'] = False
        state['fail_reason'] = 'Model determined task cannot be completed or refused to continue'
        state['should_continue'] = False
        return state
    if state.get('should_continue') is False:
        return state
    if int(state.get('step_count', 0) or 0) >= int(state.get('max_steps', 30) or 30):
        state['success'] = False
        state['fail_reason'] = f"Reached maximum step limit ({state.get('max_steps')} steps)"
        state['should_continue'] = False
        return state
    state['should_continue'] = True
    return state


def final_node(state: AgentState) -> AgentState:
    output_dir = state.get('run_output_dir')
    if not output_dir:
        return state
    os.makedirs(output_dir, exist_ok=True)
    env = state.get('env')
    observation = state.get('observation')
    scene_name = getattr(env, 'scene', 'UnknownScene')
    metadata = getattr(observation, 'metadata', None) or {}
    if isinstance(metadata, dict):
        scene_name = metadata.get('sceneName', scene_name)
    task_name = ((state.get('config') or {}).get('task') or {}).get('name', 'task') or 'task'
    safe_scene = str(scene_name).replace(' ', '_').replace('/', '_')[:80]
    safe_task = str(task_name).replace(' ', '_').replace('/', '_')[:80]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(output_dir, f'episode_{safe_scene}_{safe_task}_{timestamp}.json')
    episode = {
        'task': state.get('task_prompt', ''),
        'scene': scene_name,
        'success': state.get('success', False),
        'fail_reason': state.get('fail_reason'),
        'failure_type': state.get('failure_type'),
        'step_count': state.get('step_count', 0),
        'max_steps': state.get('max_steps', 0),
        'action_sequence': env.get_action_sequence() if hasattr(env, 'get_action_sequence') else '(no action records)',
        'trajectory': [
            {
                'step': item.get('step'),
                'thinking': item.get('thinking'),
                'action_string': item.get('action_string'),
                'llm_token_usage': item.get('llm_token_usage'),
                'reward': item.get('reward'),
                'error_message': item.get('error_message'),
            }
            for item in state.get('structured_trajectory', [])
        ],
        'timestamp': datetime.now().isoformat(),
        'metadata': {
            'total_reward': sum((item.get('reward') or 0) for item in state.get('structured_trajectory', [])),
            'parse_errors_count': sum(1 for item in state.get('structured_trajectory', []) if item.get('parse_error')),
            'token_usage': state.get('token_usage', {}),
        },
    }
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(episode, handle, ensure_ascii=False, indent=2)
    state['episode_log_path'] = path
    return state


class AgentRunner:
    def __init__(self, recursion_limit: int = 1000) -> None:
        self.recursion_limit = recursion_limit

    def stream(self, initial_state: AgentState, config: Optional[dict] = None) -> Iterable[Dict[str, AgentState]]:
        state = initial_state
        task_cfg = (state.get('config') or {}).get('task') or {}
        if state.get('max_steps_override') is not None:
            state['max_steps'] = int(state['max_steps_override'])
        elif task_cfg:
            state['max_steps'] = resolve_max_steps_from_task(task_cfg, int(state.get('max_steps', 30) or 30))
        limit = int((config or {}).get('recursion_limit', self.recursion_limit) or self.recursion_limit)
        # Decision mode is opt-in and defaults to 'standard' (identical to prior behavior).
        # Can be set via state['decision_mode'] or config['decision']['mode'] (state takes priority).
        decision_mode = state.get('decision_mode')
        if not decision_mode:
            decision_mode = ((state.get('config') or {}).get('decision') or {}).get('mode', 'standard')
        decision_mode = str(decision_mode or 'standard').lower()
        active_think_node = lookahead_think_node if decision_mode == 'lookahead' else think_node
        iterations = 0
        while True:
            if iterations >= limit:
                raise GraphRecursionError(f'Recursion limit reached: {limit}')
            iterations += 1
            state = active_think_node(state)
            yield {'think': state}
            state = act_node(state)
            yield {'act': state}
            state = evaluate_node(state)
            yield {'evaluate': state}
            if not state.get('should_continue', True):
                state = final_node(state)
                yield {'final': state}
                break
        self.last_state = state

    def invoke(self, initial_state: AgentState, config: Optional[dict] = None) -> AgentState:
        final_state = initial_state
        for chunk in self.stream(initial_state, config=config):
            for update in chunk.values():
                final_state = update
        return final_state


def create_agent_graph() -> AgentRunner:
    return AgentRunner()


_parse_action_string = None
try:
    from actions.parser import parse_action_string as _parse_action_string
except Exception:
    pass

parse_action_string = _parse_action_string
_perform_final_evaluation = lambda state: perform_final_evaluation(state)[0]
execute_action = lambda env, action_dict: (*env.step_with_action_dict(action_dict), False) if action_dict.get('action_type') != 'task_completion' else (None, None, True)
