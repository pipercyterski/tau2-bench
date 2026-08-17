"""Stdlib-only client for the per-run Odyssey proxy.

Why this exists rather than ``from dystopic.odyssey import proxy_call``: the
sandbox installs *tau2's* requirements, and adding the Dystopic SDK to them
would drag httpx/pydantic pins into a dependency set that litellm already
constrains tightly. The proxy contract is one POST, so we implement it against
``urllib`` and owe the environment nothing.

The contract (docs/reference/wire-contract, "Proxy call contract"):

    POST {proxy_url}/tools/{tool_name}
    Authorization: Bearer {run_token}
    Content-Type: application/json

    {"order_id": "4521"}            <- the tool's argument object, verbatim

    200 {"tool_name": ..., "response": <payload>, "source": ..., ...}

Only ``response`` is the tool's payload; ``tool_name`` / ``source`` /
``latency_ms`` are platform plumbing and must never reach the model's prompt,
so :meth:`OdysseyProxy.call` unwraps exactly that field (tolerating a legacy
body that *is* the payload).

Retries mirror the SDK's published policy exactly -- 429 and the three
retryable ``503`` error classes, 4 attempts, exponential backoff with +/-25%
jitter -- because a narrower policy would surface a sibling parallel call's
lock contention to the model as a tool failure, and a wider one would retry
genuinely terminal 4xx.
"""

from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable

DEFAULT_TIMEOUT_S = 120.0
MAX_ATTEMPTS = 4

# 503s the platform declares transient. Anything else -- including an
# unrecognised 503 -- is terminal.
RETRYABLE_503_CLASSES = frozenset(
    {"lock_contention", "context_store_unavailable", "trace_sequence_contended"}
)

# 401 classes that mean "this run has closed". Retrying or continuing to reason
# over further tool results is pointless once one of these lands.
STALE_TOKEN_CLASSES = frozenset(
    {"run_token_expired", "run_token_invalid", "run_token_rejected"}
)


class ProxyError(RuntimeError):
    """A tool call did not produce a payload.

    Branch on ``error_class`` (the platform's stable discriminator), never on
    the English message.
    """

    def __init__(
        self,
        message: str,
        *,
        tool_name: str,
        status_code: int | None = None,
        body: Any = None,
        error_class: str | None = None,
    ) -> None:
        super().__init__(message)
        self.tool_name = tool_name
        self.status_code = status_code
        self.body = body
        self.error_class = error_class


class StaleRunTokenError(ProxyError):
    """The run token is expired/invalid/rejected -- the run is over."""


def _error_class(body: Any) -> str | None:
    if isinstance(body, dict):
        detail = body.get("detail")
        if isinstance(detail, dict):
            klass = detail.get("error_class")
            if isinstance(klass, str):
                return klass
    return None


def _parse(raw: bytes) -> Any:
    text = raw.decode("utf-8", errors="replace")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _backoff(attempt: int) -> float:
    return min(0.5 * (2**attempt), 4.0) * (0.75 + random.random() * 0.5)


class OdysseyProxy:
    """One handle per run; ``call`` is the whole surface."""

    def __init__(
        self,
        proxy_url: str,
        run_token: str,
        *,
        timeout: float = DEFAULT_TIMEOUT_S,
        auto_retry: bool = True,
    ) -> None:
        self.base_url = proxy_url.rstrip("/")
        self.run_token = run_token
        self.timeout = timeout
        self.auto_retry = auto_retry
        self.n_calls = 0

    def url_for(self, tool_name: str) -> str:
        return f"{self.base_url}/tools/{urllib.parse.quote(tool_name, safe='')}"

    def call(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """POST one tool call and return its unwrapped payload.

        Raises :class:`ProxyError` (or :class:`StaleRunTokenError`) on every
        failure; the caller decides whether that becomes a tool-visible error
        payload or ends the turn.
        """
        self.n_calls += 1
        last: ProxyError | None = None
        for attempt in range(MAX_ATTEMPTS):
            try:
                return self._attempt(tool_name, arguments)
            except ProxyError as exc:
                last = exc
                if not self.auto_retry or not self._retryable(exc):
                    raise
                if attempt == MAX_ATTEMPTS - 1:
                    raise
                time.sleep(_backoff(attempt))
        raise last  # pragma: no cover - unreachable; the loop always exits above

    @staticmethod
    def _retryable(exc: ProxyError) -> bool:
        if exc.status_code == 429:
            return True
        return exc.status_code == 503 and exc.error_class in RETRYABLE_503_CLASSES

    def _attempt(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        request = urllib.request.Request(
            self.url_for(tool_name),
            data=json.dumps(arguments, default=str).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.run_token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = _parse(response.read())
        except urllib.error.HTTPError as exc:
            body = _parse(exc.read())
            klass = _error_class(body)
            status = exc.code
            cls = (
                StaleRunTokenError
                if status == 401 and klass in STALE_TOKEN_CLASSES
                else ProxyError
            )
            raise cls(
                f"{tool_name}: proxy returned HTTP {status}"
                + (f" ({klass})" if klass else ""),
                tool_name=tool_name,
                status_code=status,
                body=body,
                error_class=klass,
            ) from exc
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            # Transport-level: no status, and the SDK does not retry these.
            raise ProxyError(
                f"{tool_name}: proxy transport failure: {exc!r}",
                tool_name=tool_name,
            ) from exc

        if isinstance(body, dict) and "response" in body:
            return body["response"]
        return body  # legacy/bare payload

    def call_as_content(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        renderer: "Callable[[str, Any], tuple[Any, bool]] | None" = None,
    ) -> tuple[str, bool]:
        """``call`` shaped for a tau2 ``ToolMessage``: ``(content, is_error)``.

        A failed call becomes a structured error payload rather than an
        exception, so the model can recover or explain the failure -- exactly
        what tau2's own environment does when a tool raises. A stale run token
        is the one exception: it propagates, because the run has closed and
        every subsequent call would fail the same way.

        ``renderer`` is the domain's chance to restate a *successful* payload in
        the shape its own tools return, and returns ``(payload, is_error)``. It
        exists because a declared ``ledger_read`` projection always yields an
        object while several tau2 tools return a bare string -- see
        ``dystopic_entry.render_retail``. It is deliberately NOT applied to
        transport errors: those are ours, not the domain's.
        """
        try:
            payload = self.call(tool_name, arguments)
        except StaleRunTokenError:
            raise
        except ProxyError as exc:
            return (
                json.dumps(
                    {
                        "error": exc.error_class or "tool_call_failed",
                        "message": str(exc),
                    }
                ),
                True,
            )
        is_error = False
        if renderer is not None:
            payload, is_error = renderer(tool_name, payload)
        if isinstance(payload, str):
            return payload, is_error
        return json.dumps(payload, default=str), is_error
