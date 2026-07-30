"""Tool registry and dispatch for LLM-driven tool calling.

``Toolkit`` is the agent-facing tool container, modeled after RPent's
``rpent/tools/toolkit.py``: it registers a schema + handler per tool and lets
a planner/runner call tools by name through :meth:`Toolkit.get_tools_spec`
(what to tell the LLM about) and :meth:`Toolkit.execute_tool` (how to actually
run one).

The base class only registers the generic file/IO tools from
:mod:`mllm_base_agent.tools.common`. Environment-specific subclasses can
register additional tools on top via :meth:`add_tool` (e.g. a future
perception/segmentation tool), mirroring how RPent's ``LiberoToolkit``
extends its own ``Toolkit`` base class.
"""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple

ToolHandler = Callable[..., Dict[str, Any]]


@dataclass
class ToolResult:
    """Result of executing one tool call.

    Carries the raw result dict (for logging / ``finish``-signal detection)
    alongside OpenAI-shaped message content the LLM/runner can feed straight
    back into the conversation as a ``tool`` role message.
    """

    name: str
    result: Dict[str, Any]
    call_id: Optional[str] = None

    content: str = field(default="", init=False, repr=False)
    is_finish: bool = field(default=False, init=False)

    #: Max characters of the JSON text emitted in :attr:`content`.
    MAX_TEXT_CHARS_IN_RESULT: ClassVar[int] = 60000

    def __post_init__(self) -> None:
        self.content = self._build_content()
        self.is_finish = bool(isinstance(self.result, dict) and self.result.get("_finish"))

    def _build_content(self) -> str:
        result = self.result
        if not isinstance(result, dict):
            text = str(result)
        else:
            text = json.dumps(result, indent=2, default=str, ensure_ascii=False)
        if len(text) > self.MAX_TEXT_CHARS_IN_RESULT:
            text = text[: self.MAX_TEXT_CHARS_IN_RESULT] + "\n[truncated]"
        return text

    def to_tool_message(self) -> Dict[str, Any]:
        """Return an OpenAI-shaped ``{"role": "tool", ...}`` message dict."""
        message: Dict[str, Any] = {"role": "tool", "name": self.name, "content": self.content}
        if self.call_id:
            message["tool_call_id"] = self.call_id
        return message


class Toolkit:
    """Base toolkit: registers common tools and dispatches tool calls.

    Subclasses extend ``__init__`` (calling ``super().__init__()`` first) and
    register additional tools with :meth:`add_tool`. Override :meth:`close`
    to release any resources (servers, file handles, ...) at the end of a run.
    """

    def __init__(self) -> None:
        # name -> (spec, handler)
        self._tools: Dict[str, Tuple[Dict[str, Any], ToolHandler]] = {}
        self._register_common_tools()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def add_tool(self, name: str, spec: Dict[str, Any], handler: ToolHandler) -> None:
        """Register one tool under ``name`` with its OpenAI-shaped schema and handler.

        Args:
            name: Tool name as the LLM sees it (e.g. ``"read_text_file"``).
            spec: OpenAI-shaped tool schema dict, i.e.
                ``{"type": "function", "function": {"name", "description", "parameters"}}``.
            handler: Callable invoked with the tool's input kwargs; returns a
                JSON-able result dict.
        """
        self._tools[name] = (spec, handler)

    def remove_tool(self, name: str) -> None:
        """Unregister a previously-added tool, if present. No-op otherwise."""
        self._tools.pop(name, None)

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def _register_common_tools(self) -> None:
        """Register the generic file/IO tools shared by every toolkit instance."""
        from mllm_base_agent.tools import common

        for spec in common.TOOLS_SPEC:
            name = spec["function"]["name"]
            self.add_tool(name, spec, common.TOOL_HANDLERS[name])

    # ------------------------------------------------------------------
    # Model-facing prompt guidance
    # ------------------------------------------------------------------
    #
    # Per-tool ``description``/``parameters`` text (in get_tools_spec()) tells
    # the model *what* each tool does; this tells it *when/how* to use them as
    # a workflow. Kept as code (not a separate doc file) so it stays in sync
    # with whatever tools are actually registered.

    def usage_guide(self) -> str:
        """Return the usage guidance for this toolkit's tools.

        Base implementation returns the guidance for the common file/IO
        tools (:data:`mllm_base_agent.tools.common.COMMON_TOOLS_USAGE_GUIDE`).
        Subclasses that register additional tools should override this to
        append their own guidance, e.g.::

            def usage_guide(self) -> str:
                return super().usage_guide() + "\\n" + MY_ENV_TOOLS_USAGE_GUIDE
        """
        from mllm_base_agent.tools import common

        return common.COMMON_TOOLS_USAGE_GUIDE

    def build_system_prompt(self, task_instructions: str = "") -> str:
        """Compose a ready-to-use system prompt: tool usage guide + task instructions.

        This is what :class:`mllm_base_agent.tools.loop.ToolCallingAgent` uses
        by default when no explicit ``system_prompt`` is supplied, so callers
        get sensible tool-usage guidance without having to author it themselves.
        """
        guide = self.usage_guide().strip()
        if not task_instructions:
            return guide
        return f"{guide}\n\n{task_instructions.strip()}"

    # ------------------------------------------------------------------
    # Planner-facing API
    # ------------------------------------------------------------------

    def get_tools_spec(self) -> List[Dict[str, Any]]:
        """Return the OpenAI-shaped tool schemas the LLM sees (for the ``tools`` request field)."""
        return [spec for spec, _ in self._tools.values()]

    def get_tool_names(self) -> List[str]:
        return list(self._tools.keys())

    def execute_tool(self, name: str, input_dict: Optional[Dict[str, Any]] = None, call_id: Optional[str] = None) -> ToolResult:
        """Dispatch a tool call to its registered handler.

        Never raises: handler exceptions and unknown tool names are captured
        into an ``{"error": ...}`` result so a calling loop can always feed
        something back to the LLM instead of crashing.
        """
        input_dict = input_dict or {}
        entry = self._tools.get(name)
        if entry is None:
            return ToolResult(name=name, result={"error": f"unknown tool: {name}"}, call_id=call_id)
        handler = entry[1]
        try:
            result = handler(**input_dict)
            if not isinstance(result, dict):
                result = {"result": result}
        except TypeError as exc:
            result = {"error": f"bad arguments for {name}: {exc}", "got": input_dict}
        except Exception as exc:
            result = {"error": str(exc), "traceback": traceback.format_exc()}
        return ToolResult(name=name, result=result, call_id=call_id)

    # ------------------------------------------------------------------
    # Lifecycle hooks (overridden by env-specific toolkits)
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release any resources held by the toolkit at the end of a run. Default: no-op."""


__all__ = ["Toolkit", "ToolResult", "ToolHandler"]
