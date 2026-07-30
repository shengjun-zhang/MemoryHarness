"""Agent tool support: the ``@tool`` decorator plus the common tool registry.

Two layers live here:

* The legacy ``tool`` decorator below (used by e.g.
  ``mllm_base_agent/environments/ai2thor/utils.py``) is a lightweight,
  framework-free marker for standalone helper functions.
* :class:`~mllm_base_agent.tools.toolkit.Toolkit` and
  :mod:`~mllm_base_agent.tools.common` implement a real, LLM-facing
  tool-calling registry (read/write/list/find/search files + ``finish``),
  modeled after RPent's ``rpent/tools/`` package. See
  ``docs/agentic_planner_vs_single_step_analysis.md`` for the full context.
"""

from functools import wraps
from typing import Any, Callable, Optional

from mllm_base_agent.tools.toolkit import Toolkit, ToolResult
from mllm_base_agent.tools.loop import ToolCallingAgent
from mllm_base_agent.tools import common


def tool(func: Optional[Callable[..., Any]] = None, *decorator_args: Any, **decorator_kwargs: Any):
    """No-op replacement for the previous external tool decorator.

    It preserves the wrapped function and attaches small metadata fields used by
    simple tool registries.
    """
    def decorate(inner: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(inner)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return inner(*args, **kwargs)
        wrapper.name = decorator_kwargs.get('name') or getattr(inner, '__name__', 'tool')
        wrapper.description = decorator_kwargs.get('description') or getattr(inner, '__doc__', '')
        wrapper.is_spatialworld_tool = True
        return wrapper
    if callable(func):
        return decorate(func)
    return decorate

__all__ = ['tool', 'Toolkit', 'ToolResult', 'ToolCallingAgent', 'common']
