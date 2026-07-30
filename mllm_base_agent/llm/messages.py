"""Small message schema used by SpatialWorld providers and runners.

The classes intentionally mirror the tiny subset of legacy message objects that
this project used: a role, a content payload, and optional metadata on responses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

MessageContent = Union[str, List[Dict[str, Any]]]


@dataclass
class BaseMessage:
    content: MessageContent
    role: str


@dataclass
class SystemMessage(BaseMessage):
    def __init__(self, content: MessageContent):
        super().__init__(content=content, role="system")


@dataclass
class UserMessage(BaseMessage):
    def __init__(self, content: MessageContent):
        super().__init__(content=content, role="user")


class HumanMessage(UserMessage):
    """Backward-compatible alias for the old message class name."""


@dataclass
class AssistantMessage(BaseMessage):
    #: OpenAI-shaped ``tool_calls`` list, e.g.
    #: ``[{"id": "...", "type": "function", "function": {"name": ..., "arguments": "..."}}]``.
    #: Empty when the model responded with plain text (the common case).
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)

    def __init__(self, content: MessageContent, tool_calls: Optional[List[Dict[str, Any]]] = None):
        super().__init__(content=content, role="assistant")
        self.tool_calls = tool_calls or []


class AIMessage(AssistantMessage):
    """Backward-compatible alias for the old message class name."""


@dataclass
class ToolMessage(BaseMessage):
    """Result of a tool call, fed back to the model as a ``tool`` role message.

    Mirrors the OpenAI/RPent convention: ``tool_call_id`` links this result
    back to the ``tool_calls`` entry that requested it, and ``name`` records
    which tool produced ``content`` (see
    :meth:`mllm_base_agent.tools.toolkit.ToolResult.to_tool_message`).
    """

    tool_call_id: str = ""
    name: str = ""

    def __init__(self, content: MessageContent, tool_call_id: str = "", name: str = ""):
        super().__init__(content=content, role="tool")
        self.tool_call_id = tool_call_id
        self.name = name


@dataclass
class TextPart:
    text: str

    def to_payload(self) -> Dict[str, str]:
        return {"type": "text", "text": self.text}


@dataclass
class ImagePart:
    url: str

    def to_payload(self) -> Dict[str, Dict[str, str]]:
        return {"type": "image_url", "image_url": {"url": self.url}}


@dataclass
class ModelResponse:
    content: str
    response_metadata: Dict[str, Any] = field(default_factory=dict)
    usage_metadata: Dict[str, Any] = field(default_factory=dict)
    additional_kwargs: Dict[str, Any] = field(default_factory=dict)
    #: OpenAI-shaped ``tool_calls`` the model requested, if any (empty list
    #: when it responded with plain text). See ``OpenAICompatibleChatModel.invoke``.
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


def _normalize_content(content: Any) -> MessageContent:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        normalized: List[Dict[str, Any]] = []
        for item in content:
            if isinstance(item, TextPart):
                normalized.append(item.to_payload())
            elif isinstance(item, ImagePart):
                normalized.append(item.to_payload())
            elif isinstance(item, dict):
                normalized.append(item)
            else:
                normalized.append({"type": "text", "text": str(item)})
        return normalized
    return str(content)


def coerce_message(message: Any) -> BaseMessage:
    if isinstance(message, BaseMessage):
        return message
    role = getattr(message, "role", None)
    content = getattr(message, "content", None)
    if role == "tool":
        return ToolMessage(
            content=_normalize_content(content),
            tool_call_id=getattr(message, "tool_call_id", "") or "",
            name=getattr(message, "name", "") or "",
        )
    if role == "assistant":
        return AssistantMessage(content=_normalize_content(content), tool_calls=getattr(message, "tool_calls", None))
    if role in {"system", "user"}:
        return BaseMessage(content=_normalize_content(content), role=role)
    name = type(message).__name__.lower()
    if "system" in name:
        return SystemMessage(_normalize_content(content))
    if "tool" in name:
        return ToolMessage(
            content=_normalize_content(content),
            tool_call_id=getattr(message, "tool_call_id", "") or "",
            name=getattr(message, "name", "") or "",
        )
    if "ai" in name or "assistant" in name:
        return AssistantMessage(_normalize_content(content), tool_calls=getattr(message, "tool_calls", None))
    return UserMessage(_normalize_content(content))


def to_openai_messages(messages: Sequence[Any]) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for raw in messages:
        msg = coerce_message(raw)
        entry: Dict[str, Any] = {"role": msg.role, "content": _normalize_content(msg.content)}
        if isinstance(msg, AssistantMessage) and msg.tool_calls:
            entry["tool_calls"] = msg.tool_calls
            # OpenAI requires content to be null (not empty string) on
            # assistant messages that only carry tool_calls.
            if entry["content"] == "":
                entry["content"] = None
        if isinstance(msg, ToolMessage):
            entry["tool_call_id"] = msg.tool_call_id
            if msg.name:
                entry["name"] = msg.name
        payload.append(entry)
    return payload


__all__ = [
    "AIMessage",
    "AssistantMessage",
    "BaseMessage",
    "HumanMessage",
    "ImagePart",
    "MessageContent",
    "ModelResponse",
    "SystemMessage",
    "TextPart",
    "ToolMessage",
    "UserMessage",
    "coerce_message",
    "to_openai_messages",
]
