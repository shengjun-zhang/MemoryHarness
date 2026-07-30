"""A minimal, opt-in tool-calling loop wired to :class:`~mllm_base_agent.tools.toolkit.Toolkit`.

This is the smallest possible "agentic loop" as described in
``docs/agentic_planner_vs_single_step_analysis.md`` (section 四, step 3): the
model can autonomously call any registered tool (e.g. read/write/list/find
files) across multiple turns, until it explicitly calls ``finish``.

It is entirely additive and does not touch the existing Think -> Act ->
Evaluate embodied-action loop in ``mllm_base_agent/agent/runner.py``: that
loop still drives navigation/interaction actions in the AI2-THOR/ProcTHOR/etc.
environments unchanged. ``ToolCallingAgent`` is a separate, general-purpose
helper for tasks that only need "let an LLM read/write/find files and other
common tools", such as report writing, log triage, or config inspection
sub-tasks, and a template for wiring up richer environment-specific toolkits
later (mirroring RPent's ``ApiAgentLoop``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from mllm_base_agent.llm.messages import AIMessage, AssistantMessage, SystemMessage, ToolMessage, UserMessage
from mllm_base_agent.tools.toolkit import Toolkit


@dataclass
class ToolCallingAgent:
    """Drives a multi-turn "the model calls tools until it calls finish" loop.

    Args:
        vlm: Any object with an OpenAI-compatible ``.invoke(messages) -> ModelResponse``
            method, e.g. :class:`mllm_base_agent.llm.provider.OpenAICompatibleChatModel`.
        toolkit: The tool registry to expose to the model (defaults to a plain
            :class:`Toolkit`, i.e. just the common file/IO tools + ``finish``).
        max_turns: Hard cap on the number of model calls before giving up.
        system_prompt: System prompt prepended to the conversation. If left as
            ``None`` (the default), it is auto-built from
            ``toolkit.build_system_prompt()`` -- i.e. the tool usage guide in
            :data:`mllm_base_agent.tools.common.COMMON_TOOLS_USAGE_GUIDE` (plus
            any environment-specific guidance a Toolkit subclass adds) -- so
            callers get sensible tool-usage instructions without writing their
            own. Pass an explicit string to override, or ``""`` to send none.
    """

    vlm: Any
    toolkit: Toolkit = field(default_factory=Toolkit)
    max_turns: int = 20
    system_prompt: Optional[str] = None

    def __post_init__(self) -> None:
        if hasattr(self.vlm, "bind_tools"):
            self.vlm.bind_tools(self.toolkit.get_tools_spec())
        if self.system_prompt is None:
            self.system_prompt = self.toolkit.build_system_prompt()

    def run(self, task_prompt: str) -> Dict[str, Any]:
        """Run the loop for one task and return a summary dict.

        Returns a dict with keys: ``status`` ('success'/'failure'/'stuck'/
        'max_turns_reached'), ``summary`` (model-provided or synthesized),
        ``turns`` (number of model calls made), ``messages`` (full transcript,
        OpenAI-message-shaped dicts via the message objects used), and
        ``tool_calls`` (list of ``{name, arguments, result}`` for every tool
        call made, in order).
        """
        messages: List[Any] = []
        if self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt))
        messages.append(UserMessage(content=task_prompt))

        tool_call_log: List[Dict[str, Any]] = []

        for turn in range(1, self.max_turns + 1):
            response = self.vlm.invoke(messages)
            tool_calls = getattr(response, "tool_calls", None) or []

            if not tool_calls:
                # Plain-text response with no tool call: treat as an implicit
                # stop (mirrors "no more actions to take") rather than looping
                # forever, since the model didn't call `finish` explicitly.
                messages.append(AIMessage(content=response.content or ""))
                return {
                    "status": "stuck",
                    "summary": response.content or "Model returned no tool call and did not call finish.",
                    "turns": turn,
                    "messages": messages,
                    "tool_calls": tool_call_log,
                }

            messages.append(AssistantMessage(content=response.content or "", tool_calls=tool_calls))

            finish_result: Optional[Dict[str, Any]] = None
            for call in tool_calls:
                function = call.get("function") or {}
                name = function.get("name", "")
                try:
                    arguments = json.loads(function.get("arguments") or "{}")
                except (json.JSONDecodeError, TypeError):
                    arguments = {}
                tool_result = self.toolkit.execute_tool(name, arguments, call_id=call.get("id"))
                tool_call_log.append({"name": name, "arguments": arguments, "result": tool_result.result})
                messages.append(ToolMessage(content=tool_result.content, tool_call_id=tool_result.call_id or "", name=name))
                if tool_result.is_finish:
                    finish_result = tool_result.result

            if finish_result is not None:
                return {
                    "status": finish_result.get("status", "success"),
                    "summary": finish_result.get("summary", ""),
                    "turns": turn,
                    "messages": messages,
                    "tool_calls": tool_call_log,
                }

        return {
            "status": "max_turns_reached",
            "summary": f"Stopped after {self.max_turns} turns without the model calling finish.",
            "turns": self.max_turns,
            "messages": messages,
            "tool_calls": tool_call_log,
        }


__all__ = ["ToolCallingAgent"]
