"""Bounded, fail-open capture for synchronous and asynchronous client agents."""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
import queue
import random
import re
import threading
import time
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar, cast

import httpx

from litmusai.runtime.config import validate_url
from litmusai.runtime.models import (
    EventType,
    Message,
    Payload,
    RuntimeEvent,
    SessionEnd,
    ToolActivity,
    new_id,
)

P = ParamSpec("P")
R = TypeVar("R")


class RuntimeClient:
    """Queue capture without waiting for the network in an agent's execution path.

    Use as a sync or async context manager. ``emit`` means queued locally;
    ``flush`` waits for acknowledgement or exhausted attempts, and reports success.
    Inspect ``health`` for failures. A process crash can lose unacknowledged events.
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        project_id: str,
        *,
        capacity: int = 1000,
        timeout: float = 2,
        max_attempts: int = 3,
        max_event_bytes: int = 65536,
        allow_local_http: bool = False,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if capacity < 1 or not 0 < timeout <= 30 or not 1 <= max_attempts <= 10:
            raise ValueError("invalid capture queue or retry limits")
        self.endpoint = validate_url(endpoint, allow_local_http)
        self.project_id = project_id
        self.producer_id = new_id()
        self.max_event_bytes = max_event_bytes
        self._sequence = 0
        self._condition = threading.Condition()
        self._queue: queue.Queue[RuntimeEvent] = queue.Queue(maxsize=capacity)
        self._pending = 0
        self._closing = False
        self._started = False
        self._stats = {"queued": 0, "accepted": 0, "dropped": 0, "failed": 0}
        self._last_error: str | None = None
        self._attempts = max_attempts
        self._timeout = timeout
        self._http = httpx.Client(
            base_url=self.endpoint,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
            transport=transport,
            follow_redirects=False,
            trust_env=False,
        )
        self._thread = threading.Thread(target=self._send, name="litmus-capture", daemon=True)

    def __enter__(self) -> RuntimeClient:
        """Start the background sender once."""
        with self._condition:
            if self._closing or self._started:
                raise RuntimeError("runtime client cannot be entered twice")
            self._started = True
            self._thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        """Attempt a bounded graceful flush without replacing agent exceptions."""
        self.close()

    async def __aenter__(self) -> RuntimeClient:
        """Start capture for an asynchronous agent."""
        return self.__enter__()

    async def __aexit__(self, *args: object) -> None:
        """Flush without blocking the caller's event loop."""
        await asyncio.to_thread(self.close)

    @property
    def health(self) -> dict[str, Any]:
        """Return local capture counters; no payloads or credentials are included."""
        with self._condition:
            return {**self._stats, "pending": self._pending, "last_error": self._last_error}

    def _dropped(self, reason: str) -> None:
        with self._condition:
            self._stats["dropped"] += 1
            self._last_error = reason

    def emit(self, event: RuntimeEvent) -> bool:
        """Return whether the event was queued, not whether it was durably accepted."""
        try:
            event.validate_boundary()
            if event.project_id != self.project_id:
                raise ValueError("project mismatch")
            if len(event.model_dump_json().encode()) > self.max_event_bytes:
                raise ValueError("oversized event")
        except Exception:
            self._dropped("invalid_event")
            return False
        with self._condition:
            if not self._started or self._closing:
                self._dropped("client_not_running")
                return False
            try:
                self._queue.put_nowait(event.model_copy(deep=True))
            except queue.Full:
                self._dropped("queue_full")
                return False
            self._pending += 1
            self._stats["queued"] += 1
            return True

    def session(
        self,
        agent_id: str,
        session_id: str | None = None,
        *,
        deployment_id: str = "default",
        actor_id: str | None = None,
    ) -> RuntimeSession:
        """Create correlation context; no network operation is performed."""
        return RuntimeSession(self, agent_id, session_id or new_id(), deployment_id, actor_id)

    def flush(self, timeout: float = 5) -> bool:
        """Wait at most timeout seconds and return whether all captured events succeeded."""
        deadline = time.monotonic() + max(0, timeout)
        with self._condition:
            while self._pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return not (self._stats["failed"] or self._stats["dropped"])

    async def aflush(self, timeout: float = 5) -> bool:
        """Asynchronous version of flush."""
        return await asyncio.to_thread(self.flush, timeout)

    def close(self, timeout: float = 5) -> bool:
        """Stop accepting events and flush; the daemon finishes remaining bounded attempts."""
        with self._condition:
            self._closing = True
        if not self._started:
            self._http.close()
            return True
        return self.flush(timeout)

    def _send(self) -> None:
        try:
            while True:
                try:
                    event = self._queue.get(timeout=0.1)
                except queue.Empty:
                    if self._closing:
                        return
                    continue
                accepted = False
                for attempt in range(self._attempts):
                    try:
                        deadline = time.monotonic() + self._timeout
                        with self._http.stream(
                            "POST", "/v1/events", json=[event.model_dump(mode="json")]
                        ) as response:
                            if response.status_code == 200:
                                acknowledgement = bytearray()
                                for chunk in response.iter_bytes(chunk_size=8192):
                                    acknowledgement.extend(chunk)
                                    if len(acknowledgement) > 16384 or time.monotonic() > deadline:
                                        raise ValueError("acknowledgement limit exceeded")
                                result = json.loads(acknowledgement)["results"][0]
                                accepted = result["event_id"] == event.event_id and result[
                                    "status"
                                ] in {"accepted", "duplicate"}
                                if accepted or result["status"] != "busy":
                                    break
                            elif (
                                response.status_code not in {408, 429}
                                and response.status_code < 500
                            ):
                                break
                    except Exception:
                        # Transport errors and malformed acknowledgements remain capture failures.
                        pass
                    if attempt + 1 < self._attempts:
                        time.sleep(random.uniform(0.05, min(1, 0.1 * 2**attempt)))
                with self._condition:
                    self._stats["accepted" if accepted else "failed"] += 1
                    if not accepted:
                        self._last_error = "ingestion_failed"
                    self._pending -= 1
                    self._condition.notify_all()
                self._queue.task_done()
        finally:
            self._http.close()


class RuntimeSession:
    """Explicit instrumentation of the boundaries visible to the client application."""

    def __init__(
        self,
        client: RuntimeClient,
        agent_id: str,
        session_id: str,
        deployment_id: str,
        actor_id: str | None,
    ) -> None:
        self.client = client
        self.identity: dict[str, Any] = dict(
            agent_id=agent_id, session_id=session_id, deployment_id=deployment_id, actor_id=actor_id
        )

    def __enter__(self) -> RuntimeSession:
        """Use this session as a synchronous context manager."""
        return self

    def __exit__(self, *args: object) -> None:
        """Emit an optional end marker."""
        self.emit("session.ended", SessionEnd())

    async def __aenter__(self) -> RuntimeSession:
        """Use this session as an asynchronous context manager."""
        return self

    async def __aexit__(self, *args: object) -> None:
        """Emit an optional end marker without waiting for delivery."""
        self.__exit__()

    def emit(
        self,
        event_type: EventType,
        payload: Payload,
        *,
        tool_call_id: str | None = None,
        parent_event_id: str | None = None,
    ) -> bool:
        """Capture an explicit application boundary and return local queue status."""
        try:
            with self.client._condition:
                self.client._sequence += 1
                event = RuntimeEvent(
                    **self.identity,
                    project_id=self.client.project_id,
                    producer_id=self.client.producer_id,
                    sequence=self.client._sequence,
                    event_type=event_type,
                    payload=payload,
                    tool_call_id=tool_call_id,
                    parent_event_id=parent_event_id,
                )
                return self.client.emit(event)
        except Exception:
            self.client._dropped("capture_failed")
            return False

    def emit_message(self, text: str) -> bool:
        """Capture a user message without allowing it to modify policy."""
        return self._message("message.received", text, "user")

    def emit_context(self, text: str) -> bool:
        """Capture retrieved content before it is passed to the agent."""
        return self._message("context.received", text, "context")

    def emit_response(self, text: str, destination: str | None = None) -> bool:
        """Capture a complete response and its application-known destination."""
        return self._message("response.completed", text, "assistant", destination)

    def _message(
        self, kind: EventType, text: str, role: Any, destination: str | None = None
    ) -> bool:
        try:
            return self.emit(kind, Message(text=text, role=role, destination=destination))
        except Exception:
            self.client._dropped("capture_failed")
            return False

    def wrap_tool(
        self,
        function: Callable[P, R],
        *,
        name: str | None = None,
        destination_argument: str | None = None,
    ) -> Callable[P, R]:
        """Capture requested/completed/failed around exactly one underlying call.

        The returned function must be registered with the agent. Only JSON-compatible
        arguments/results are captured. Instrumentation failures never replace a tool result.
        """
        tool_name: str = name or str(getattr(function, "__name__", "tool"))
        if name is None:
            tool_name = re.sub(r"[^\w.:-]", "_", tool_name)[:128] or "tool"

        def capture(
            kind: EventType,
            call_id: str,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            result: Any = None,
            error: BaseException | None = None,
        ) -> None:
            try:
                bound = inspect.signature(function).bind(*args, **kwargs)
                bound.apply_defaults()
                # No default=str: executing arbitrary repr methods can have side effects.
                arguments = json.loads(json.dumps(dict(bound.arguments), allow_nan=False))
                destination = arguments.get(destination_argument) if destination_argument else None
                self.emit(
                    kind,
                    ToolActivity(
                        name=tool_name,
                        arguments=arguments,
                        destination=destination,
                        result=json.loads(json.dumps(result, allow_nan=False)),
                        error_type=type(error).__name__ if error else None,
                    ),
                    tool_call_id=call_id,
                )
            except Exception:
                self.client._dropped("tool_capture_failed")

        if inspect.iscoroutinefunction(function):

            @functools.wraps(function)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
                call_id = new_id()
                capture("tool.requested", call_id, args, kwargs)
                try:
                    result = await function(*args, **kwargs)
                except BaseException as error:
                    capture("tool.failed", call_id, args, kwargs, error=error)
                    raise
                capture("tool.completed", call_id, args, kwargs, result)
                return result

            return cast(Callable[P, R], async_wrapper)

        @functools.wraps(function)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            call_id = new_id()
            capture("tool.requested", call_id, args, kwargs)
            try:
                result = function(*args, **kwargs)
            except BaseException as error:
                capture("tool.failed", call_id, args, kwargs, error=error)
                raise
            capture("tool.completed", call_id, args, kwargs, result)
            return result

        return sync_wrapper
