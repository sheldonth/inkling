"""Stdlib HTTP/SSE client for llama-server OpenAI & control endpoints."""

from __future__ import annotations

from dataclasses import dataclass, field
import http.client
import json
import socket
from typing import Any, Callable
from urllib.parse import urlparse


class ServerClientError(RuntimeError):
    """Raised on HTTP, network, or API errors communicating with llama-server."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


@dataclass
class ChatResult:
    content: str = ""
    reasoning: str = ""
    finish_reason: str | None = None
    timings: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)
    interrupted: bool = False


class ServerClient:
    """Thin client around llama-server's HTTP & SSE APIs."""

    def __init__(self, base_url: str, api_key: str = "") -> None:
        self.base_url = base_url.rstrip("/")
        parsed = urlparse(self.base_url)
        self.scheme = parsed.scheme or "http"
        self.host = parsed.hostname or "127.0.0.1"
        self.port = parsed.port or (443 if self.scheme == "https" else 80)
        self.base_path = parsed.path.rstrip("/")
        self.api_key = api_key

    def _connect(self, timeout: float | None) -> http.client.HTTPConnection:
        if self.scheme == "https":
            return http.client.HTTPSConnection(self.host, self.port, timeout=timeout)
        return http.client.HTTPConnection(self.host, self.port, timeout=timeout)

    def _headers(self, json_body: bool = False) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if json_body:
            headers["Content-Type"] = "application/json"
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _request_json(
        self,
        method: str,
        path: str,
        payload: Any = None,
        timeout: float = 10.0,
    ) -> tuple[int, Any]:
        conn = self._connect(timeout)
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        full_path = f"{self.base_path}{path}"
        try:
            conn.request(method, full_path, body=body, headers=self._headers(json_body=body is not None))
            resp = conn.getresponse()
            raw = resp.read()
            status = resp.status
        except (OSError, http.client.HTTPException) as exc:
            raise ServerClientError(f"Failed to connect to {self.base_url}{path}: {exc}") from exc
        finally:
            conn.close()

        if not raw:
            return status, None
        try:
            return status, json.loads(raw.decode("utf-8", errors="replace"))
        except json.JSONDecodeError:
            return status, raw.decode("utf-8", errors="replace")

    def health(self, timeout: float = 2.0) -> tuple[bool, str]:
        """Return (is_ready, state_description)."""
        try:
            status, data = self._request_json("GET", "/health", timeout=timeout)
        except ServerClientError as exc:
            return False, f"unreachable ({exc})"

        if status == 200:
            if isinstance(data, dict):
                return True, str(data.get("status", "ok"))
            return True, "ok"
        if status == 503:
            if isinstance(data, dict) and isinstance(data.get("error"), dict):
                return False, str(data["error"].get("message", "loading model"))
            return False, "loading model"
        return False, f"HTTP {status}"

    def props(self, timeout: float = 5.0) -> dict[str, Any]:
        status, data = self._request_json("GET", "/props", timeout=timeout)
        if status != 200 or not isinstance(data, dict):
            raise ServerClientError(f"GET /props failed (HTTP {status}): {data}", status=status)
        return data

    def models(self, timeout: float = 5.0) -> dict[str, Any]:
        status, data = self._request_json("GET", "/v1/models", timeout=timeout)
        if status != 200 or not isinstance(data, dict):
            raise ServerClientError(f"GET /v1/models failed (HTTP {status}): {data}", status=status)
        return data

    def get_lora_adapters(self, timeout: float = 5.0) -> list[dict[str, Any]]:
        status, data = self._request_json("GET", "/lora-adapters", timeout=timeout)
        if status != 200 or not isinstance(data, list):
            raise ServerClientError(f"GET /lora-adapters failed (HTTP {status}): {data}", status=status)
        return data

    def set_lora_scale(
        self,
        scale: float,
        adapter_id: int = 0,
        timeout: float = 10.0,
    ) -> list[dict[str, Any]]:
        payload = [{"id": int(adapter_id), "scale": float(scale)}]
        status, data = self._request_json("POST", "/lora-adapters", payload=payload, timeout=timeout)
        if status != 200:
            msg = _extract_error_message(data) or f"HTTP {status}"
            raise ServerClientError(f"POST /lora-adapters failed: {msg}", status=status)
        return self.get_lora_adapters(timeout=timeout)

    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        stream: bool = True,
        reasoning_effort: str | None = None,
        lora_scale: float | None = None,
        lora_id: int = 0,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        on_reasoning: Callable[[str], None] | None = None,
        on_content: Callable[[str], None] | None = None,
        timeout: float | None = None,
    ) -> ChatResult:
        """Send a /v1/chat/completions request with optional SSE streaming."""
        body: dict[str, Any] = {
            "messages": messages,
            "stream": stream,
            "timings_per_token": True,
        }
        if model:
            body["model"] = model
        if reasoning_effort and reasoning_effort != "default":
            body["reasoning_effort"] = reasoning_effort
        if lora_scale is not None:
            body["lora"] = [{"id": int(lora_id), "scale": float(lora_scale)}]
        if temperature is not None:
            body["temperature"] = float(temperature)
        if top_p is not None:
            body["top_p"] = float(top_p)
        if top_k is not None:
            body["top_k"] = int(top_k)
        if min_p is not None:
            body["min_p"] = float(min_p)
        if max_tokens is not None and max_tokens > 0:
            body["max_tokens"] = int(max_tokens)
        if seed is not None and seed >= 0:
            body["seed"] = int(seed)

        if not stream:
            return self._chat_non_stream(body, on_reasoning=on_reasoning, on_content=on_content, timeout=timeout)
        return self._chat_stream(body, on_reasoning=on_reasoning, on_content=on_content, timeout=timeout)

    def _chat_non_stream(
        self,
        body: dict[str, Any],
        *,
        on_reasoning: Callable[[str], None] | None,
        on_content: Callable[[str], None] | None,
        timeout: float | None,
    ) -> ChatResult:
        status, data = self._request_json(
            "POST",
            "/v1/chat/completions",
            payload=body,
            timeout=timeout or 600.0,
        )
        if status != 200 or not isinstance(data, dict):
            msg = _extract_error_message(data) or f"HTTP {status}"
            raise ServerClientError(f"Chat completion failed: {msg}", status=status)

        result = ChatResult()
        if isinstance(data.get("timings"), dict):
            result.timings = data["timings"]
        if isinstance(data.get("usage"), dict):
            result.usage = data["usage"]

        choices = data.get("choices") or []
        if choices and isinstance(choices[0], dict):
            choice = choices[0]
            result.finish_reason = choice.get("finish_reason")
            msg_obj = choice.get("message") or {}
            reasoning = msg_obj.get("reasoning_content") or ""
            content = msg_obj.get("content") or ""
            if reasoning:
                result.reasoning = reasoning
                if on_reasoning:
                    on_reasoning(reasoning)
            if content:
                result.content = content
                if on_content:
                    on_content(content)
        return result

    def _chat_stream(
        self,
        body: dict[str, Any],
        *,
        on_reasoning: Callable[[str], None] | None,
        on_content: Callable[[str], None] | None,
        timeout: float | None,
    ) -> ChatResult:
        conn = self._connect(timeout)
        headers = self._headers(json_body=True)
        headers["Accept"] = "text/event-stream"
        encoded = json.dumps(body).encode("utf-8")
        full_path = f"{self.base_path}/v1/chat/completions"

        result = ChatResult()
        reasoning_parts: list[str] = []
        content_parts: list[str] = []

        try:
            conn.request("POST", full_path, body=encoded, headers=headers)
            resp = conn.getresponse()
            if resp.status != 200:
                raw = resp.read().decode("utf-8", errors="replace")
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    parsed = raw
                msg = _extract_error_message(parsed) or raw or f"HTTP {resp.status}"
                raise ServerClientError(f"Chat completion failed: {msg}", status=resp.status)

            # Read SSE line by line using resp.readline() so chunks stream immediately.
            while True:
                raw_line = resp.readline()
                if not raw_line:
                    break
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or line.startswith(":"):
                    continue
                if not line.startswith("data:"):
                    continue
                payload_str = line[5:].strip()
                if payload_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload_str)
                except json.JSONDecodeError:
                    continue

                if "error" in chunk:
                    msg = _extract_error_message(chunk) or "Unknown streaming error"
                    raise ServerClientError(msg)

                if isinstance(chunk.get("timings"), dict):
                    result.timings = chunk["timings"]
                if isinstance(chunk.get("usage"), dict):
                    result.usage = chunk["usage"]

                choices = chunk.get("choices")
                if not choices or not isinstance(choices, list):
                    continue
                choice = choices[0]
                if choice.get("finish_reason") is not None:
                    result.finish_reason = choice.get("finish_reason")
                delta = choice.get("delta")
                if not isinstance(delta, dict):
                    continue

                r_text = delta.get("reasoning_content")
                if isinstance(r_text, str) and r_text:
                    reasoning_parts.append(r_text)
                    if on_reasoning:
                        on_reasoning(r_text)

                c_text = delta.get("content")
                if isinstance(c_text, str) and c_text:
                    content_parts.append(c_text)
                    if on_content:
                        on_content(c_text)

        except KeyboardInterrupt:
            result.interrupted = True
        except (OSError, socket.error, http.client.HTTPException) as exc:
            raise ServerClientError(f"Connection lost during streaming: {exc}") from exc
        finally:
            conn.close()

        result.reasoning = "".join(reasoning_parts)
        result.content = "".join(content_parts)
        return result


def _extract_error_message(data: Any) -> str | None:
    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, dict):
            msg = err.get("message")
            if msg:
                return str(msg)
        if isinstance(err, str):
            return err
        if "message" in data:
            return str(data["message"])
    if isinstance(data, str) and data.strip():
        return data.strip()
    return None
