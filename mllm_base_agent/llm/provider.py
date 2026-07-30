"""LLM/VLM providers without external chain frameworks dependencies."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Sequence

import requests
from dotenv import load_dotenv

from mllm_base_agent.llm.messages import ModelResponse, to_openai_messages

load_dotenv()

GEMINI_RESPONSES_MODELS = {
    "Gemini 3-Pro-Preview",
    "Gemini-3-Flash-Preview",
    "Gemini-3.1-Pro-Preview",
}

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


API_RETRY_CONFIG = {
    "max_retries": _env_int("VLM_API_MAX_RETRIES", 5),
    "retry_delay": _env_int("VLM_API_RETRY_DELAY", 2),
    "retry_delay_400": _env_int("VLM_API_RETRY_DELAY_400", 5),
    "retry_delay_500": _env_int("VLM_API_RETRY_DELAY_500", 3),
    "exponential_backoff": True,
}


def _normalize_usage_dict(result: Dict[str, Any]) -> Dict[str, int]:
    usage = result.get("usage") or result.get("usage_metadata") or {}
    if not isinstance(usage, dict):
        if hasattr(usage, "model_dump"):
            usage = usage.model_dump()
        else:
            usage = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
    if "prompt_tokens" not in usage:
        usage["prompt_tokens"] = usage.get("promptTokenCount") or usage.get("prompt_token_count")
    if "completion_tokens" not in usage:
        usage["completion_tokens"] = usage.get("candidatesTokenCount") or usage.get("completion_token_count")
    if "total_tokens" not in usage:
        usage["total_tokens"] = usage.get("totalTokenCount") or usage.get("total_token_count")

    def to_int(value: Any) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    prompt_tokens = to_int(usage.get("prompt_tokens"))
    completion_tokens = to_int(usage.get("completion_tokens"))
    total_tokens = to_int(usage.get("total_tokens")) or prompt_tokens + completion_tokens
    normalized = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }
    return {k: v for k, v in normalized.items() if v}


def _extract_message(result: Dict[str, Any]) -> Dict[str, Any]:
    choices = result.get("choices") or []
    if not choices:
        raise ValueError(f"Unexpected API response without choices: {result}")
    return choices[0].get("message") or {}


def _extract_content(result: Dict[str, Any]) -> str:
    message = _extract_message(result)
    content = message.get("content", "")
    if isinstance(content, dict):
        return str(content.get("text", content))
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p)
    return str(content or "")


def _extract_tool_calls(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract OpenAI-shaped ``tool_calls`` from a chat completion response.

    Returns ``[]`` when the model responded with plain text (no tool calls
    requested), which is the default/legacy behavior for callers that never
    pass ``tools`` in the first place.
    """
    message = _extract_message(result)
    tool_calls = message.get("tool_calls") or []
    normalized: List[Dict[str, Any]] = []
    for call in tool_calls:
        if hasattr(call, "model_dump"):
            call = call.model_dump()
        if not isinstance(call, dict):
            continue
        function = call.get("function") or {}
        if hasattr(function, "model_dump"):
            function = function.model_dump()
        normalized.append({
            "id": call.get("id", ""),
            "type": call.get("type", "function"),
            "function": {
                "name": (function or {}).get("name", ""),
                "arguments": (function or {}).get("arguments", "{}"),
            },
        })
    return normalized


class OpenAICompatibleChatModel:
    """Small `.invoke(messages)` model for OpenAI-compatible chat APIs."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        top_p: Optional[float] = None,
        timeout: int = 300,
        client_max_retries: Optional[int] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not api_key:
            raise ValueError("API key is required for the selected provider")
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.timeout = timeout
        self.client_max_retries = client_max_retries
        self.model_kwargs = model_kwargs or {}
        self._client = None
        # Set via bind_tools(); when non-empty, invoke() automatically attaches
        # them to every request unless the caller passes tools=... explicitly.
        self._bound_tools: Optional[List[Dict[str, Any]]] = None
        self._bound_tool_choice: Optional[Any] = None

    @property
    def chat_url(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    def _get_openai_client(self) -> Any:
        if self._client is not None:
            return self._client or None
        try:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url.rstrip("/") + "/",
                timeout=self.timeout,
                max_retries=0,
            )
        except Exception:
            self._client = False
        return self._client or None

    def _request_with_openai_client(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        client = self._get_openai_client()
        if client is None:
            raise RuntimeError("openai client unavailable")
        response = client.chat.completions.create(**payload)
        choices = []
        for choice in getattr(response, "choices", []) or []:
            msg = getattr(choice, "message", None)
            if msg is not None:
                message: Dict[str, Any] = {"content": getattr(msg, "content", "")}
                tool_calls = getattr(msg, "tool_calls", None)
                if tool_calls:
                    message["tool_calls"] = [
                        tc.model_dump() if hasattr(tc, "model_dump") else tc for tc in tool_calls
                    ]
                choices.append({"message": message})
        usage = getattr(response, "usage", None)
        if hasattr(usage, "model_dump"):
            usage = usage.model_dump()
        return {
            "choices": choices,
            "usage": usage or {},
            "model": getattr(response, "model", payload.get("model")),
        }

    def _request_with_requests(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        response = requests.post(self.chat_url, headers=headers, data=json.dumps(payload), timeout=self.timeout)
        if response.status_code >= 400:
            body = response.text.strip()
            if len(body) > 1000:
                body = body[:1000] + "..."
            raise requests.HTTPError(
                f"{response.status_code} Error for url: {response.url}; body: {body}",
                response=response,
            )
        return response.json()

    def _make_api_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        max_retries = (
            self.client_max_retries
            if self.client_max_retries is not None
            else API_RETRY_CONFIG["max_retries"]
        )
        last_error: Optional[BaseException] = None
        for attempt in range(max_retries):
            try:
                try:
                    result = self._request_with_openai_client(payload)
                except Exception:
                    result = self._request_with_requests(payload)
                if not result.get("choices"):
                    raise ValueError("API returned empty choices")
                return result
            except Exception as exc:
                last_error = exc
                if attempt >= max_retries - 1:
                    break
                text = str(exc).lower()
                delay = API_RETRY_CONFIG["retry_delay"]
                if "400" in text:
                    delay = API_RETRY_CONFIG["retry_delay_400"]
                elif any(code in text for code in ("500", "502", "503", "504")):
                    delay = API_RETRY_CONFIG["retry_delay_500"]
                if API_RETRY_CONFIG.get("exponential_backoff"):
                    delay *= 2 ** attempt
                time.sleep(delay)
        raise ValueError(f"API request failed after {max_retries} attempts: {last_error}")

    def bind_tools(
        self,
        tools: Optional[List[Dict[str, Any]]],
        tool_choice: Optional[Any] = "auto",
    ) -> "OpenAICompatibleChatModel":
        """Attach an OpenAI-shaped ``tools`` spec (e.g. from ``Toolkit.get_tools_spec()``).

        Every subsequent ``invoke()`` call automatically includes ``tools``/
        ``tool_choice`` in the request unless the caller overrides them via
        keyword arguments. Passing ``tools=None`` clears the binding. Mutates
        and returns ``self`` so it can be used either as ``vlm.bind_tools(...)``
        or ``vlm = vlm.bind_tools(...)``.
        """
        self._bound_tools = list(tools) if tools else None
        self._bound_tool_choice = tool_choice if self._bound_tools else None
        return self

    def invoke(self, messages: Sequence[Any], **kwargs: Any) -> ModelResponse:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": to_openai_messages(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self._bound_tools:
            payload["tools"] = self._bound_tools
            payload["tool_choice"] = self._bound_tool_choice
        if self.model_kwargs:
            payload.update(self.model_kwargs)
        payload.update(kwargs)
        result = self._make_api_request(payload)
        usage = _normalize_usage_dict(result)
        metadata: Dict[str, Any] = {}
        if usage:
            metadata["token_usage"] = usage
        if result.get("model"):
            metadata["model_name"] = result["model"]
        return ModelResponse(
            content=_extract_content(result),
            response_metadata=metadata,
            usage_metadata=usage,
            tool_calls=_extract_tool_calls(result),
        )

def get_vlm(
    provider: str = "openai",
    model_name: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2000,
    top_p: Optional[float] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: Optional[int] = None,
    client_max_retries: Optional[int] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    **_: Any,
) -> OpenAICompatibleChatModel:
    provider = (provider or "openai").lower()
    openai_compatible_providers = {"openai", "openai_compatible", "vision_http"}
    if provider in openai_compatible_providers:
        key = api_key or os.getenv("OPENAI_API_KEY")
        model = model_name or os.getenv("VLM_MODEL", "gpt-4o")
        return OpenAICompatibleChatModel(
            model=model,
            api_key=key,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            base_url=base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            timeout=timeout if timeout is not None else _env_int("VLM_API_TIMEOUT", 300),
            client_max_retries=client_max_retries,
            model_kwargs=model_kwargs,
        )
    supported = "', '".join(sorted(openai_compatible_providers))
    raise ValueError(f"Unsupported provider: {provider}. Supported providers: '{supported}'")


create_vlm = get_vlm
