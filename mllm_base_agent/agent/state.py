"""Agent state definitions for the unified runner."""

from typing import Any, Dict, List, Optional, TypedDict

from mllm_base_agent.llm.schemas import EnvObservation


class AgentState(TypedDict, total=False):
    task_prompt: str
    observation: Optional[EnvObservation]
    step_count: int
    max_steps: int
    success: Optional[bool]
    fail_reason: Optional[str]
    failure_type: Optional[str]
    short_term_history: List[Dict[str, Any]]
    long_term_summary: str
    structured_trajectory: List[Dict[str, Any]]
    conversation_history: List[Dict[str, Any]]
    next_action: Optional[dict]
    should_continue: bool
    task_done_by_model: bool
    task_fail_by_model: bool
    think_failed: bool
    token_usage: Dict[str, int]
    vlm: Any
    env: Any
    config: Optional[Dict[str, Any]]
    executor_type: Optional[str]
    goal_image_path: Optional[str]
    run_output_dir: Optional[str]
    # Decision mode: 'standard' (default, unchanged) or 'lookahead' (probe all
    # candidate actions first, then choose based on all resulting observations).
    # See mllm_base_agent/agent/runner.py::lookahead_think_node.
    decision_mode: Optional[str]
    thor_agent_id: Optional[int]
    # Skill-memory library (mirrors the dual-agent MEMORY.md/feedback_*.md
    # pattern, see mllm_base_agent/tools/memory.py::MemoryLibrary): a
    # reviewed index plus leaf notes distilled from past benchmark runs. The
    # index is embedded into the system prompt every turn; leaf notes are
    # fetched on demand via the ReadMemory(<file_name>) pseudo-action, which
    # think_node/lookahead_think_node intercept before it ever reaches the
    # environment (see runner.py's "memory_lookup" handling).
    memory_library: Any


def get_recent_trajectory(state: AgentState, n: int = 3) -> str:
    structured = state.get("structured_trajectory", [])
    if not structured:
        return "No action history"
    lines = []
    for step in structured[-n:]:
        reward = step.get("reward", 0) or 0
        result = "Success" if reward > 0 else "Failure"
        lines.append(f"- Step {step.get('step', '?')}: {step.get('action_string', 'Unknown')} [{result}]")
    return "\n".join(lines)


def build_short_term_context(short_term_history: List[Dict[str, Any]]) -> str:
    if not short_term_history:
        return "No short-term history"
    return " -> ".join(
        f"Step{entry.get('step', '?')}: {entry.get('action_string', 'Unknown')}"
        for entry in short_term_history
    )
