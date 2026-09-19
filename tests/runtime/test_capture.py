"""Capture must never change the result, exception, or execution count of a real tool."""

import asyncio
import json
import threading

import httpx
import pytest

from litmusai.runtime import RuntimeClient


def receiver(events):
    def receive(request):
        batch = json.loads(request.content)
        events.extend(batch)
        return httpx.Response(
            200,
            json={
                "results": [{"event_id": item["event_id"], "status": "accepted"} for item in batch]
            },
        )

    return httpx.MockTransport(receive)


def client(transport, **kwargs):
    return RuntimeClient("https://collector.example", "test", "p", transport=transport, **kwargs)


def test_sync_tool_executes_once_and_correlates_arguments_defaults_result():
    events, calls = [], []

    def send(recipient, text="default"):
        calls.append((recipient, text))
        return {"sent": True}

    with client(receiver(events)) as monitor:
        with monitor.session("agent", "session") as session:
            assert session.emit_message("hello")
            wrapped = session.wrap_tool(send, destination_argument="recipient")
            assert wrapped("outside.example") == {"sent": True}
        assert monitor.flush()
        assert monitor.health["accepted"] == 4
    assert calls == [("outside.example", "default")]
    requested, completed = events[1:3]
    assert requested["tool_call_id"] == completed["tool_call_id"]
    assert requested["payload"]["destination"] == "outside.example"
    assert requested["payload"]["arguments"] == {"recipient": "outside.example", "text": "default"}
    assert completed["payload"]["result"] == {"sent": True}
    assert [event["sequence"] for event in events] == [1, 2, 3, 4]


async def test_async_success_failure_cancellation_and_sync_tool_on_async_agent():
    events, calls = [], []

    async def fail():
        calls.append("fail")
        raise ValueError("private exception text")

    async def cancel():
        calls.append("cancel")
        raise asyncio.CancelledError()

    async def success(value):
        calls.append("success")
        return value

    async with client(receiver(events)) as monitor:
        async with monitor.session("agent") as session:
            with pytest.raises(ValueError, match="private exception text"):
                await session.wrap_tool(fail)()
            with pytest.raises(asyncio.CancelledError):
                await session.wrap_tool(cancel)()
            assert await session.wrap_tool(success)(3) == 3
            assert session.wrap_tool(lambda: 4)() == 4
        assert await monitor.aflush()
    assert calls == ["fail", "cancel", "success"]
    assert "private exception text" not in json.dumps(events)
    assert sum(e["event_type"] == "tool.failed" for e in events) == 2


def test_outage_retries_do_not_reexecute_tool_and_are_visible():
    attempts, calls = [], []

    def unavailable(request):
        attempts.append(json.loads(request.content)[0]["event_id"])
        return httpx.Response(503)

    with client(httpx.MockTransport(unavailable), max_attempts=2) as monitor:
        session = monitor.session("agent")
        assert session.wrap_tool(lambda: calls.append(1))() is None
        assert monitor.flush() is False
        assert monitor.health["failed"] == 2
        assert monitor.health["last_error"] == "ingestion_failed"
    assert calls == [1]
    assert len(attempts) == 4 and attempts[0] == attempts[1] and attempts[2] == attempts[3]


def test_queue_overflow_is_bounded_visible_and_does_not_break_tool():
    entered, release = threading.Event(), threading.Event()

    def blocked(request):
        entered.set()
        assert release.wait(3)
        event = json.loads(request.content)[0]
        return httpx.Response(
            200, json={"results": [{"event_id": event["event_id"], "status": "accepted"}]}
        )

    with client(httpx.MockTransport(blocked), capacity=1) as monitor:
        session = monitor.session("agent")
        assert session.emit_message("one")
        assert entered.wait(1)
        assert session.emit_message("two")
        assert not session.emit_message("three")
        assert session.wrap_tool(lambda: "original result")() == "original result"
        assert monitor.health["dropped"] == 3
        assert monitor.flush(0.01) is False
        release.set()
        monitor.flush()
    assert monitor.health["accepted"] == 2


def test_invalid_payload_and_unsupported_result_do_not_replace_tool_behavior():
    events = []
    with client(receiver(events)) as monitor:
        session = monitor.session("agent")
        assert not session.emit_message("x" * 40000)
        value = object()
        assert session.wrap_tool(lambda: value)() is value
        assert not monitor.flush()
        assert monitor.health["dropped"] == 2
    assert len(events) == 1 and events[0]["event_type"] == "tool.requested"


def test_emit_copies_mutable_arguments_and_close_rejects_new_events(make_event):
    events = []
    with client(receiver(events)) as monitor:
        event = make_event()
        assert monitor.emit(event)
        event.payload.arguments["body"] = "changed"
        assert monitor.flush()
        assert monitor.close()
        assert not monitor.emit(make_event())
    assert events[0]["payload"]["arguments"]["body"] == "hello"


@pytest.mark.parametrize(
    "url",
    ["http://external.example", "https://user:pass@example.com", "https://example.com?key=secret"],
)
def test_client_rejects_insecure_or_embedded_credentials(url):
    with pytest.raises(ValueError):
        RuntimeClient(url, "key", "p")
