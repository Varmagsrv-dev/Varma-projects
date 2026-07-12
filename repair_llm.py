from pathlib import Path

files = {
    'src/agentforge/llm/base_provider.py': '''"""Shared base functionality for concrete LLM provider implementations.

``BaseLLMProvider`` is not itself referenced by the
:class:`agentforge.core.ports.llm_port.LLMProvider` protocol (which is
structural, not nominal), but every concrete provider in this package
subclasses it to avoid re-implementing common request validation, logging,
and token-counting delegation.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence

from agentforge.core.exceptions import LLMContextLengthExceededError
from agentforge.core.ports.llm_port import LLMMessage, LLMRequest, LLMResponse, LLMStreamEvent
from agentforge.llm.token_counter import LocalTokenCounter
from agentforge.logging import get_logger

__all__ = ["BaseLLMProvider"]

logger = get_logger(__name__)


class BaseLLMProvider(ABC):
    """Common scaffolding for concrete :class:`~agentforge.core.ports.llm_port.LLMProvider` implementations.

    Attributes:
        context_window_tokens: The maximum context window (input + output)
            the underlying model supports, used for local pre-flight
            validation before making a request.
    """

    def __init__(self, *, context_window_tokens: int = 200_000) -> None:
        """Initialize shared provider state.

        Args:
            context_window_tokens: The model's context window size in
                tokens, used by :meth:`_validate_request_length`.
        """
        self.context_window_tokens = context_window_tokens
        self._token_counter = LocalTokenCounter()

    def _validate_request_length(self, request: LLMRequest) -> None:
        """Perform a local, best-effort context-length pre-check.

        This is a defensive check only; the authoritative check is
        whatever the provider's API itself enforces. Catching obviously
        oversized requests locally avoids an unnecessary network round
        trip and gives a clearer error message.

        Args:
            request: The request to validate.

        Raises:
            agentforge.core.exceptions.LLMContextLengthExceededError: If
                the estimated token count of ``request.messages`` plus
                ``request.max_output_tokens`` exceeds
                :attr:`context_window_tokens`.
        """
        estimated_input_tokens = self._token_counter.count_messages(request.messages)
        total_estimated = estimated_input_tokens + request.max_output_tokens
        if total_estimated > self.context_window_tokens:
            raise LLMContextLengthExceededError(
                f"Estimated request size ({total_estimated} tokens) exceeds the "
                f"model's context window ({self.context_window_tokens} tokens)",
                context={
                    "estimated_input_tokens": estimated_input_tokens,
                    "max_output_tokens": request.max_output_tokens,
                    "context_window_tokens": self.context_window_tokens,
                    "model": request.model,
                },
            )

    def count_tokens(self, messages: Sequence[LLMMessage]) -> int:
        """Estimate the token count of a message sequence using the local counter.

        Args:
            messages: The candidate message sequence.

        Returns:
            The estimated token count. Concrete providers may override
            this with a provider-exact tokenizer if one is available.
        """
        return self._token_counter.count_messages(messages)

    def _log_request_start(self, request: LLMRequest) -> float:
        """Log the start of an LLM request and return a monotonic start time.

        Args:
            request: The request being sent.

        Returns:
            The :func:`time.monotonic` timestamp at call time, to be
            passed to :meth:`_log_request_end` for duration calculation.
        """
        logger.info(
            "llm_request_started",
            model=request.model,
            message_count=len(request.messages),
            tool_count=len(request.tools),
            max_output_tokens=request.max_output_tokens,
        )
        return time.monotonic()

    def _log_request_end(self, request: LLMRequest, response: LLMResponse, start_time: float) -> None:
        """Log the successful completion of an LLM request.

        Args:
            request: The original request.
            response: The resulting response.
            start_time: The value previously returned by
                :meth:`_log_request_start`.
        """
        duration_seconds = time.monotonic() - start_time
        logger.info(
            "llm_request_completed",
            model=response.model,
            stop_reason=response.stop_reason.value,
            prompt_tokens=response.token_usage.prompt_tokens,
            completion_tokens=response.token_usage.completion_tokens,
            duration_seconds=round(duration_seconds, 3),
            tool_calls_requested=len(response.tool_calls),
        )

    @abstractmethod
    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Perform a single, non-streamed completion request.

        Concrete providers must implement this by translating ``request``
        into their wire format, calling
        :meth:`_validate_request_length` first, and translating the
        provider's response back into an :class:`~agentforge.core.ports.llm_port.LLMResponse`.

        Args:
            request: The fully-specified request.

        Returns:
            The complete response.
        """
        raise NotImplementedError

    @abstractmethod
    def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
        """Perform a streamed completion request.

        Args:
            request: The fully-specified request.

        Returns:
            An async iterator of stream events.
        """
        raise NotImplementedError
''',
    'src/agentforge/llm/retry.py': '''"""Retry helpers for transient LLM provider failures."""

from __future__ import annotations

import asyncio
import functools
from collections.abc import Awaitable, Callable
from typing import TypeVar

from agentforge.core.exceptions import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMRateLimitError,
)

__all__ = ["with_llm_retry"]

T = TypeVar("T")


def with_llm_retry(
    *,
    max_attempts: int = 3,
    min_wait_seconds: float = 0.25,
    max_wait_seconds: float | None = None,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Retry transient LLM failures with a simple exponential backoff."""

    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: object, **kwargs: object) -> T:
            last_error: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except LLMAuthenticationError:
                    raise
                except (LLMConnectionError, LLMRateLimitError) as exc:
                    last_error = exc
                    if attempt >= max_attempts:
                        raise
                    wait_seconds = min_wait_seconds * (2 ** (attempt - 1))
                    if max_wait_seconds is not None:
                        wait_seconds = min(wait_seconds, max_wait_seconds)
                    if wait_seconds > 0:
                        await asyncio.sleep(wait_seconds)
            if last_error is not None:
                raise last_error
            raise RuntimeError("retry loop completed without a result")

        return wrapper

    return decorator
''',
    'src/agentforge/llm/token_counter.py': '''"""Lightweight local token counting utilities for LLM requests."""

from __future__ import annotations

from collections.abc import Sequence

from agentforge.core.ports.llm_port import LLMMessage, LLMMessageRole

__all__ = ["LocalTokenCounter"]


class LocalTokenCounter:
    """Estimate token counts without calling a remote tokenizer."""

    def __init__(self, *, encoding_name: str | None = None) -> None:
        self.encoding_name = encoding_name
        self._encoding = None
        try:
            import tiktoken

            self._encoding = tiktoken.get_encoding(encoding_name or "cl100k_base")
            self.encoding_name = encoding_name or "cl100k_base"
        except Exception:
            self.encoding_name = None

    def count_text(self, text: str) -> int:
        """Estimate the token count for a single text value."""
        if not text:
            return 0
        if self._encoding is not None:
            try:
                return len(self._encoding.encode(text))
            except Exception:
                pass
        return max(1, len(text.split()) + max(1, len(text) // 4))

    def count_messages(self, messages: Sequence[LLMMessage]) -> int:
        """Estimate the token count of a sequence of messages."""
        total = 0
        for message in messages:
            total += self.count_text(message.content)
            total += 4
            if message.role == LLMMessageRole.ASSISTANT and message.tool_calls:
                total += 8 * len(message.tool_calls)
            if message.role == LLMMessageRole.TOOL_RESULT and message.tool_call_id:
                total += 4
        return total
''',
    'src/agentforge/llm/anthropic_provider.py': '''"""Anthropic Messages API provider implementation."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from agentforge.core.exceptions import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMResponseParsingError,
    LLMTimeoutError,
)
from agentforge.core.ports.llm_port import (
    LLMMessage,
    LLMMessageRole,
    LLMRequest,
    LLMResponse,
    LLMStopReason,
    LLMStreamEvent,
    LLMStreamEventKind,
)
from agentforge.core.value_objects.token_usage import TokenUsage
from agentforge.llm.base_provider import BaseLLMProvider
from agentforge.llm.retry import with_llm_retry
from agentforge.logging import get_logger

__all__ = ["AnthropicProvider"]

logger = get_logger(__name__)

_DEFAULT_BASE_URL: str = "https://api.anthropic.com/v1"


class AnthropicProvider(BaseLLMProvider):
    """LLM provider implementation for Anthropic's Messages API."""

    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str | None = None,
        timeout_seconds: float = 120.0,
        context_window_tokens: int = 200_000,
    ) -> None:
        super().__init__(context_window_tokens=context_window_tokens)
        self._api_key = api_key
        self._base_url = base_url or _DEFAULT_BASE_URL
        self._timeout_seconds = timeout_seconds

    def _build_headers(self) -> dict[str, str]:
        if not self._api_key:
            raise LLMAuthenticationError("No API key configured for Anthropic provider")
        return {
            "x-api-key": self._api_key,
            "content-type": "application/json",
            "anthropic-version": "2023-06-01",
        }

    def _build_payload(self, request: LLMRequest, *, stream: bool) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        for message in request.messages:
            if message.role == LLMMessageRole.TOOL_RESULT:
                messages.append({"role": "user", "content": message.content})
            else:
                messages.append({"role": message.role.value, "content": message.content})

        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_output_tokens,
            "stream": stream,
        }
        if request.system_prompt:
            payload["system"] = request.system_prompt
        return payload

    def _parse_response(self, request: LLMRequest, raw: dict[str, Any]) -> LLMResponse:
        try:
            content_blocks = raw["content"]
            text_parts: list[str] = []
            for block in content_blocks:
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            content = "".join(text_parts)
            stop_reason_raw = raw.get("stop_reason") or "end_turn"
            stop_reason = (
                LLMStopReason.END_TURN
                if stop_reason_raw == "end_turn"
                else LLMStopReason.TOOL_USE
            )
            usage_raw = raw.get("usage", {})
            token_usage = TokenUsage(
                prompt_tokens=usage_raw.get("input_tokens", 0),
                completion_tokens=usage_raw.get("output_tokens", 0),
            )
            return LLMResponse(
                content=content,
                tool_calls=(),
                stop_reason=stop_reason,
                token_usage=token_usage,
                model=raw.get("model", request.model),
            )
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMResponseParsingError(
                f"Failed to parse Anthropic response: {exc}",
                context={"raw_keys": list(raw.keys())},
                cause=exc,
            ) from exc

    async def _post(self, payload: dict[str, Any]) -> httpx.Response:
        headers = self._build_headers()
        try:
            async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
                response = await client.post(f"{self._base_url}/messages", headers=headers, json=payload)
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(
                f"Anthropic request timed out after {self._timeout_seconds}s", cause=exc
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMConnectionError(f"Failed to connect to Anthropic API: {exc}", cause=exc) from exc

        if response.status_code == 401:
            raise LLMAuthenticationError("Anthropic API rejected the provided credentials")
        if response.status_code == 429:
            retry_after = response.headers.get("retry-after")
            raise LLMRateLimitError(
                "Anthropic API rate limit exceeded",
                retry_after_seconds=float(retry_after) if retry_after else None,
            )
        if response.status_code >= 500:
            raise LLMConnectionError(
                f"Anthropic API returned server error {response.status_code}",
                context={"status_code": response.status_code},
            )
        return response

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self._validate_request_length(request)
        start_time = self._log_request_start(request)
        payload = self._build_payload(request, stream=False)

        @with_llm_retry(max_attempts=3)
        async def _call() -> httpx.Response:
            return await self._post(payload)

        http_response = await _call()
        try:
            raw = http_response.json()
        except json.JSONDecodeError as exc:
            raise LLMResponseParsingError("Response body was not valid JSON", cause=exc) from exc
        response = self._parse_response(request, raw)
        self._log_request_end(request, response, start_time)
        return response

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
        response = await self.complete(request)
        yield LLMStreamEvent(kind=LLMStreamEventKind.MESSAGE_COMPLETE, final_response=response)
''',
    'src/agentforge/llm/openai_compatible_provider.py': '''"""An OpenAI Chat Completions-compatible LLM provider implementation.

Targets any backend exposing the widely-adopted OpenAI Chat Completions
wire format (OpenAI itself, and numerous compatible gateways/proxies), so
that AgentForge can be pointed at such a backend without a dedicated
per-vendor integration.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from agentforge.core.exceptions import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMResponseParsingError,
    LLMTimeoutError,
)
from agentforge.core.ports.llm_port import (
    LLMMessage,
    LLMMessageRole,
    LLMRequest,
    LLMResponse,
    LLMStopReason,
    LLMStreamEvent,
    LLMStreamEventKind,
    ToolCallRequest,
)
from agentforge.core.value_objects.token_usage import TokenUsage
from agentforge.llm.base_provider import BaseLLMProvider
from agentforge.llm.retry import with_llm_retry
from agentforge.logging import get_logger

__all__ = ["OpenAICompatibleProvider"]

logger = get_logger(__name__)

_DEFAULT_BASE_URL: str = "https://api.openai.com/v1"

_FINISH_REASON_MAP: dict[str, LLMStopReason] = {
    "stop": LLMStopReason.END_TURN,
    "length": LLMStopReason.MAX_TOKENS,
    "tool_calls": LLMStopReason.TOOL_USE,
    "function_call": LLMStopReason.TOOL_USE,
}


class OpenAICompatibleProvider(BaseLLMProvider):
    """LLM provider implementation for OpenAI Chat-Completions-compatible APIs."""

    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str | None = None,
        timeout_seconds: float = 120.0,
        context_window_tokens: int = 128_000,
    ) -> None:
        super().__init__(context_window_tokens=context_window_tokens)
        self._api_key = api_key
        self._base_url = base_url or _DEFAULT_BASE_URL
        self._timeout_seconds = timeout_seconds

    def _build_headers(self) -> dict[str, str]:
        if not self._api_key:
            raise LLMAuthenticationError("No API key configured for OpenAI-compatible provider")
        return {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}

    def _build_payload(self, request: LLMRequest, *, stream: bool) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        for message in request.messages:
            if message.role == LLMMessageRole.TOOL_RESULT:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": message.tool_call_id,
                        "content": message.content,
                    }
                )
            elif message.role == LLMMessageRole.ASSISTANT and message.tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": message.content or None,
                        "tool_calls": [
                            {
                                "id": tool_call.call_id,
                                "type": "function",
                                "function": {
                                    "name": tool_call.tool_name,
                                    "arguments": json.dumps(dict(tool_call.arguments)),
                                },
                            }
                            for tool_call in message.tool_calls
                        ],
                    }
                )
            else:
                messages.append({"role": message.role.value, "content": message.content})

        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_output_tokens,
            "temperature": request.temperature,
            "stream": stream,
        }
        if request.stop_sequences:
            payload["stop"] = list(request.stop_sequences)
        if request.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": dict(tool.parameters_json_schema),
                    },
                }
                for tool in request.tools
            ]
        return payload

    def _parse_response(self, request: LLMRequest, raw: dict[str, Any]) -> LLMResponse:
        try:
            choice = raw["choices"][0]
            message = choice["message"]
            content = message.get("content") or ""
            tool_calls: list[ToolCallRequest] = []
            for tool_call_raw in message.get("tool_calls") or []:
                function_raw = tool_call_raw["function"]
                arguments = json.loads(function_raw["arguments"]) if function_raw["arguments"] else {}
                tool_calls.append(
                    ToolCallRequest(
                        call_id=tool_call_raw["id"],
                        tool_name=function_raw["name"],
                        arguments=arguments,
                    )
                )
            finish_reason_raw = choice.get("finish_reason") or "stop"
            stop_reason = _FINISH_REASON_MAP.get(finish_reason_raw, LLMStopReason.END_TURN)
            usage_raw = raw.get("usage", {})
            token_usage = TokenUsage(
                prompt_tokens=usage_raw.get("prompt_tokens", 0),
                completion_tokens=usage_raw.get("completion_tokens", 0),
            )
            return LLMResponse(
                content=content,
                tool_calls=tuple(tool_calls),
                stop_reason=stop_reason,
                token_usage=token_usage,
                model=raw.get("model", request.model),
            )
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise LLMResponseParsingError(
                f"Failed to parse OpenAI-compatible response: {exc}",
                context={"raw_keys": list(raw.keys())},
                cause=exc,
            ) from exc

    async def _post(self, payload: dict[str, Any]) -> httpx.Response:
        headers = self._build_headers()
        try:
            async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
                response = await client.post(
                    f"{self._base_url}/chat/completions", headers=headers, json=payload
                )
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(
                f"OpenAI-compatible request timed out after {self._timeout_seconds}s", cause=exc
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMConnectionError(f"Failed to connect to LLM API: {exc}", cause=exc) from exc

        if response.status_code == 401:
            raise LLMAuthenticationError("LLM API rejected the provided credentials")
        if response.status_code == 429:
            retry_after = response.headers.get("retry-after")
            raise LLMRateLimitError(
                "LLM API rate limit exceeded",
                retry_after_seconds=float(retry_after) if retry_after else None,
            )
        if response.status_code >= 500:
            raise LLMConnectionError(
                f"LLM API returned server error {response.status_code}",
                context={"status_code": response.status_code},
            )
        return response

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self._validate_request_length(request)
        start_time = self._log_request_start(request)
        payload = self._build_payload(request, stream=False)

        @with_llm_retry(max_attempts=3)
        async def _call() -> httpx.Response:
            return await self._post(payload)

        http_response = await _call()
        try:
            raw = http_response.json()
        except json.JSONDecodeError as exc:
            raise LLMResponseParsingError("Response body was not valid JSON", cause=exc) from exc
        response = self._parse_response(request, raw)
        self._log_request_end(request, response, start_time)
        return response

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
        self._validate_request_length(request)
        payload = self._build_payload(request, stream=True)
        headers = self._build_headers()

        accumulated_text: list[str] = []
        model_name = request.model
        finish_reason_raw = "stop"

        try:
            async with httpx.AsyncClient(timeout=self._timeout_seconds) as client, client.stream(
                "POST", f"{self._base_url}/chat/completions", headers=headers, json=payload
            ) as http_response:
                if http_response.status_code == 401:
                    raise LLMAuthenticationError("LLM API rejected the provided credentials")
                if http_response.status_code == 429:
                    raise LLMRateLimitError("LLM API rate limit exceeded")
                async for line in http_response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[len("data: ") :].strip()
                    if not data_str or data_str == "[DONE]":
                        continue
                    chunk = json.loads(data_str)
                    model_name = chunk.get("model", model_name)
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    delta = choice.get("delta", {})
                    text = delta.get("content")
                    if text:
                        accumulated_text.append(text)
                        yield LLMStreamEvent(kind=LLMStreamEventKind.TEXT_DELTA, text_delta=text)
                    if choice.get("finish_reason"):
                        finish_reason_raw = choice["finish_reason"]
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(
                f"Streaming request timed out after {self._timeout_seconds}s", cause=exc
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMConnectionError(f"Failed to connect to LLM API for streaming: {exc}", cause=exc) from exc

        final_response = LLMResponse(
            content="".join(accumulated_text),
            tool_calls=(),
            stop_reason=_FINISH_REASON_MAP.get(finish_reason_raw, LLMStopReason.END_TURN),
            token_usage=TokenUsage.zero(),
            model=model_name,
        )
        yield LLMStreamEvent(kind=LLMStreamEventKind.MESSAGE_COMPLETE, final_response=final_response)
''',
    'src/agentforge/llm/local_provider.py': '''"""A local-model LLM provider implementation for self-hosted backends.

Targets OpenAI-Chat-Completions-compatible local inference servers
(``llama.cpp``'s server mode, ``Ollama``'s OpenAI-compatible endpoint,
``vLLM``, etc.), differing from
:class:`agentforge.llm.openai_compatible_provider.OpenAICompatibleProvider`
mainly in defaults appropriate for local use: no API key required by
default, localhost base URL, and a smaller default context window
reflecting typical locally-hosted model sizes. Implemented by composing
:class:`OpenAICompatibleProvider` rather than duplicating wire-format
logic, since local inference servers overwhelmingly standardize on the
same JSON shape.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

from agentforge.core.ports.llm_port import LLMMessage, LLMRequest, LLMResponse, LLMStreamEvent
from agentforge.llm.openai_compatible_provider import OpenAICompatibleProvider
from agentforge.logging import get_logger

__all__ = ["LocalProvider"]

logger = get_logger(__name__)

_DEFAULT_LOCAL_BASE_URL: str = "http://localhost:11434/v1"
_LOCAL_PLACEHOLDER_API_KEY: str = "local-no-auth-required"


class LocalProvider:
    """LLM provider implementation for locally self-hosted inference servers.

    Delegates all wire-format translation to an internally held
    :class:`~agentforge.llm.openai_compatible_provider.OpenAICompatibleProvider`,
    configured with sensible local-inference defaults. Satisfies
    :class:`agentforge.core.ports.llm_port.LLMProvider` structurally.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        timeout_seconds: float = 300.0,
        context_window_tokens: int = 32_768,
        api_key: str | None = None,
    ) -> None:
        self._delegate = OpenAICompatibleProvider(
            api_key=api_key or _LOCAL_PLACEHOLDER_API_KEY,
            base_url=base_url or _DEFAULT_LOCAL_BASE_URL,
            timeout_seconds=timeout_seconds,
            context_window_tokens=context_window_tokens,
        )
        logger.info(
            "local_provider_initialized",
            base_url=base_url or _DEFAULT_LOCAL_BASE_URL,
            context_window_tokens=context_window_tokens,
        )

    async def complete(self, request: LLMRequest) -> LLMResponse:
        return await self._delegate.complete(request)

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
        return self._delegate.stream(request)

    def count_tokens(self, messages: Sequence[LLMMessage]) -> int:
        return self._delegate.count_tokens(messages)
''',
}

for rel_path, content in files.items():
    path = Path(rel_path)
    path.write_text(content, encoding='utf-8')
    print(f'wrote {rel_path}')
