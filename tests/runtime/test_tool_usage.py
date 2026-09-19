"""Conversation limits use durable observations, not worker timing or retry counts."""

import json
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from litmusai.runtime.config import ToolUsagePolicy
from litmusai.runtime.engine import Engine
from litmusai.runtime.models import Message, ToolActivity
from litmusai.runtime.publishers import verify_webhook
from litmusai.runtime.redaction import Redactor
from litmusai.runtime.service import create_app
from litmusai.runtime.store import Store

BASE = datetime(2026, 9, 18, tzinfo=timezone.utc)


def configured(config, **changes):
    rule = ToolUsagePolicy(**{"policy_id": "tool-burst", "max_calls": 2, **changes})
    project = config.projects[0].model_copy(update={"usage_policies": [rule]})
    return config.model_copy(update={"projects": [project]})


def capture(store, config, make_event, second, **fields):
    fields.setdefault("payload", ToolActivity(name="send_email", destination="support@example.com"))
    event = make_event(**fields)
    item = (
        Redactor(config.projects[0].policy)
        .capture(event)
        .model_copy(
            update={"received_at": BASE + timedelta(seconds=second)},
        )
    )
    assert store.accept(item, config.projects[0]) == "accepted"
    return item


async def drain(store, config):
    engine = Engine(store, config)
    while await engine.process_one("p"):
        pass


def alerts(store):
    return [a for a in store.alerts("p") if a["category"] == "excessive_tool_usage"]


async def test_exact_limit_then_one_notification_per_crossing(config, store, make_event):
    config = configured(config)
    first = [capture(store, config, make_event, n) for n in range(2)]
    await drain(store, config)
    assert alerts(store) == []
    first.append(capture(store, config, make_event, 2))
    capture(store, config, make_event, 3)
    await drain(store, config)
    assert len(alerts(store)) == 1
    alert = alerts(store)[0]
    assert alert["schema_version"] == "1.1"
    assert alert["usage"]["observed_count"] == 3
    assert alert["usage"]["limit"] == 2
    assert alert["usage"]["window_seconds"] == 60
    assert alert["usage"]["counting_basis"] == "collector_received_at"
    assert alert["source_event_ids"] == [c.event.event_id for c in first]
    findings = [row["finding"] for row in store.findings("p")]
    assert [f["outcome"] for f in findings if f["detector"] == "tool_usage"].count("detected") == 2
    assert store.status("p")["suppressed"] == 1


async def test_retries_duplicate_call_ids_and_completions_do_not_inflate(config, store, make_event):
    config = configured(config)
    first = capture(store, config, make_event, 0)
    assert store.accept(first, config.projects[0]) == "duplicate"
    capture(store, config, make_event, 1, tool_call_id=first.event.tool_call_id)
    for n, kind in enumerate(["tool.completed", "tool.failed"], 2):
        capture(
            store, config, make_event, n, event_type=kind, tool_call_id=first.event.tool_call_id
        )
    capture(store, config, make_event, 4)
    await drain(store, config)
    assert alerts(store) == []
    capture(store, config, make_event, 5)
    await drain(store, config)
    assert alerts(store)[0]["usage"]["observed_count"] == 3


async def test_restart_delayed_workers_and_equal_timestamps_preserve_count(config, make_event):
    config = configured(config)
    first = Store(config.database)
    first.register_config(config)
    events = [capture(first, config, make_event, 0) for _ in range(4)]
    # Deliberately process newest first; no mutable worker counter may affect results.
    with first.db:
        for n, event in enumerate(events):
            first.db.execute("UPDATE jobs SET created=? WHERE event=?", (-n, event.event.event_id))
    first.close()
    restarted = Store(config.database)
    try:
        await drain(restarted, config)
        assert len(alerts(restarted)) == 1
        assert alerts(restarted)[0]["usage"]["observed_count"] == 3
        assert alerts(restarted)[0]["source_event_ids"] == [e.event.event_id for e in events[:3]]
    finally:
        restarted.close()


async def test_rolling_window_excludes_left_boundary_and_rearms(config, store, make_event):
    config = configured(config)
    for second in [0, 1, 2, 62, 63]:
        capture(store, config, make_event, second)
    await drain(store, config)
    assert len(alerts(store)) == 1
    capture(store, config, make_event, 64)
    await drain(store, config)
    assert len(alerts(store)) == 2
    assert len({a["alert_id"] for a in alerts(store)}) == 2


async def test_late_source_timestamps_do_not_rewrite_receipt_windows(config, store, make_event):
    config = configured(config)
    for n, observed in enumerate([BASE, BASE + timedelta(days=1), BASE - timedelta(days=1)]):
        capture(store, config, make_event, n, observed_at=observed)
    await drain(store, config)
    assert alerts(store)[0]["usage"]["observed_count"] == 3


@pytest.mark.parametrize("scope", ["session_id", "agent_id", "deployment_id", "project_id"])
async def test_scopes_cannot_contribute_to_other_conversations(config, store, make_event, scope):
    config = configured(config)
    for n in range(2):
        capture(store, config, make_event, n)
    other = config
    if scope == "project_id":
        other = config.model_copy(
            update={
                "projects": [config.projects[0].model_copy(update={"project_id": "other"})],
            }
        )
    for n in range(2):
        capture(store, other, make_event, n + 2, **{scope: "other"})
    await drain(store, config)
    assert alerts(store) == []


async def test_tool_and_agent_filters_and_multiple_policy_versions(config, store, make_event):
    config = configured(config, tools=["send_email"], agents=["agent"], deployments=["default"])
    rule = config.projects[0].usage_policies[0]
    additional = rule.model_copy(update={"policy_id": "all-tools", "max_calls": 3, "tools": []})
    config = config.model_copy(
        update={
            "projects": [
                config.projects[0].model_copy(
                    update={"usage_policies": [rule, additional]},
                )
            ]
        }
    )
    store.register_config(config)
    capture(store, config, make_event, 0, payload=ToolActivity(name="lookup"))
    for n in range(1, 4):
        capture(store, config, make_event, n)
    for n in range(4, 8):
        capture(store, config, make_event, n, agent_id="excluded")
    await drain(store, config)
    assert {a["policy_id"] for a in alerts(store)} == {"tool-burst", "all-tools"}
    assert len(alerts(store)) == 2
    changed = config.model_copy(
        update={
            "projects": [
                config.projects[0].model_copy(
                    update={
                        "usage_policies": [rule.model_copy(update={"max_calls": 4})],
                    }
                )
            ]
        }
    )
    with pytest.raises(ValueError, match="version"):
        store.register_config(changed)


async def test_non_tool_capture_gap_prevents_clear_but_not_proven_excess(config, store, make_event):
    config = configured(config)
    capture(
        store,
        config,
        make_event,
        0,
        sequence=2,
        event_type="message.received",
        tool_call_id=None,
        payload=Message(text="gap"),
    )
    capture(store, config, make_event, 1)
    await drain(store, config)
    finding = next(
        row["finding"] for row in store.findings("p") if row["finding"]["detector"] == "tool_usage"
    )
    assert finding["outcome"] == "insufficient_context" and finding["context_incomplete"]
    for n in [2, 3]:
        capture(store, config, make_event, n)
    await drain(store, config)
    assert alerts(store)[0]["context_incomplete"]


@pytest.mark.parametrize("initially_enabled", [True, False])
async def test_new_or_reenabled_policy_alerts_when_already_above_limit(
    config,
    store,
    make_event,
    initially_enabled,
):
    enabled = configured(config)
    capture(store, enabled if initially_enabled else config, make_event, 0)
    for n in range(1, 5):
        capture(store, config, make_event, n)
    capture(store, enabled, make_event, 5)
    capture(store, enabled, make_event, 6)
    await drain(store, enabled)
    assert len(alerts(store)) == 1
    assert alerts(store)[0]["usage"]["observed_count"] == 6


async def test_new_version_lower_limit_alerts_once_and_pending_jobs_keep_snapshot(
    config,
    store,
    make_event,
):
    first = configured(config)
    high = first.projects[0].usage_policies[0].model_copy(update={"max_calls": 100})
    first = first.model_copy(
        update={
            "projects": [
                first.projects[0].model_copy(
                    update={"usage_policies": [high]},
                )
            ]
        }
    )
    for n in range(5):
        capture(store, first, make_event, n)
    low = high.model_copy(update={"max_calls": 2, "version": "2"})
    second = first.model_copy(
        update={
            "projects": [
                first.projects[0].model_copy(
                    update={"usage_policies": [low]},
                )
            ]
        }
    )
    for n in [5, 6]:
        capture(store, second, make_event, n)
    await drain(store, second)
    assert len(alerts(store)) == 1
    assert alerts(store)[0]["policy_version"] == "2"
    assert alerts(store)[0]["usage"]["observed_count"] == 6


async def test_reenabling_after_previous_breach_emits_a_fresh_alert(config, store, make_event):
    enabled = configured(config)
    for n in range(3):
        capture(store, enabled, make_event, n)
    capture(store, config, make_event, 3)
    capture(store, enabled, make_event, 4)
    capture(store, enabled, make_event, 5)
    await drain(store, enabled)
    assert len(alerts(store)) == 2
    assert {a["usage"]["observed_count"] for a in alerts(store)} == {3, 5}


def test_http_capture_to_signed_loopback_webhook(config, store, make_event):
    config = configured(config)
    received = []
    notified = threading.Event()

    class Receiver(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            verified = verify_webhook(
                body,
                "synthetic-webhook-key-1234",
                self.headers["X-Litmus-Timestamp"],
                self.headers["X-Litmus-Signature"],
            )
            if verified:
                received.append(json.loads(body))
                notified.set()
            self.send_response(204 if verified else 401)
            self.end_headers()

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Receiver)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    destination = config.destinations[0].model_copy(
        update={
            "url": f"http://127.0.0.1:{server.server_port}/events",
                "allow_local_http": True,
                "version": "2",
        }
    )
    config = config.model_copy(update={"destinations": [destination]})
    try:
        with TestClient(create_app(config, store=store)) as client:
            events = [
                make_event(
                    payload=ToolActivity(
                        name="send_email",
                        destination="support@example.com",
                    )
                )
                for _ in range(4)
            ]
            response = client.post(
                "/v1/events",
                json=[e.model_dump(mode="json") for e in events],
                headers={"Authorization": "Bearer synthetic-project-key-1234"},
            )
            assert response.status_code == 200
            assert all(row["status"] == "accepted" for row in response.json()["results"])
            assert notified.wait(5), "no signed notification reached the receiver"
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                status = store.status("p")
                if status["jobs"] == {"done": 16} and all(
                    d["state"] == "acknowledged" for d in status["destinations"]
                ):
                    break
                time.sleep(0.01)
            else:
                pytest.fail("usage jobs or notification acknowledgement did not complete")
            assert len(received) == 1
            assert received[0]["data"]["usage"]["observed_count"] == 3
            assert received[0]["data"]["source_event_ids"] == [e.event_id for e in events[:3]]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


async def test_atomic_notification_retry_does_not_recount(config, store, make_event):
    config = configured(config)
    for n in range(3):
        capture(store, config, make_event, n)
    store.db.execute(
        "CREATE TRIGGER fail_usage BEFORE INSERT ON deliveries "
        "BEGIN SELECT RAISE(ABORT, 'test'); END"
    )
    with pytest.raises(sqlite3.IntegrityError):
        await drain(store, config)
    assert alerts(store) == []
    with store.db:
        store.db.execute("DROP TRIGGER fail_usage")
        store.db.execute("UPDATE jobs SET lease_until=0")
    await drain(store, config)
    assert len(alerts(store)) == 1
    assert alerts(store)[0]["usage"]["observed_count"] == 3


async def test_canonical_delivery_has_usage_and_legacy_alert_retains_null_actor(
    config,
    store,
    make_event,
):
    config = configured(config)
    for n in range(3):
        capture(store, config, make_event, n)
    capture(store, config, make_event, 4, payload=ToolActivity(name="not-allowed"))
    await drain(store, config)
    sent = []

    class Receiver:
        async def publish(self, body, delivery_id):
            sent.append(json.loads(body))

    engine = Engine(store, config, publisher_factory=lambda _: Receiver())
    while await engine.publish_one("webhook"):
        pass
    usage = next(e for e in sent if e["data"]["category"] == "excessive_tool_usage")
    assert usage["data"]["usage"]["observed_count"] == 3
    assert usage["data"]["session_id"] == "session"
    legacy = next(e for e in sent if e["data"]["category"] == "unauthorized_tool")
    assert legacy["data"]["schema_version"] == "1.0" and "usage" not in legacy["data"]
    assert legacy["data"]["actor_id"] is None


@pytest.mark.parametrize("values", [{"max_calls": 0}, {"max_calls": True}, {"window_seconds": 0}])
def test_invalid_limits_rejected(values):
    with pytest.raises(ValidationError):
        ToolUsagePolicy(**{"policy_id": "limit", "max_calls": 20, **values})


def test_duplicate_policy_ids_rejected(config):
    values = config.projects[0].model_dump()
    values["usage_policies"] = [{"policy_id": "default", "max_calls": 20}]
    with pytest.raises(ValidationError, match="unique"):
        type(config.projects[0]).model_validate(values)


async def test_bounded_evidence_does_not_truncate_count(config, store, make_event):
    config = configured(config, max_calls=60)
    events = [capture(store, config, make_event, 0) for _ in range(62)]
    await drain(store, config)
    assert len(alerts(store)) == 1
    alert = alerts(store)[0]
    assert alert["usage"]["observed_count"] == 61
    assert alert["usage"]["source_events_truncated"]
    assert not alert["context_incomplete"]
    assert alert["source_event_ids"] == [c.event.event_id for c in events[11:61]]


def test_retention_removes_observations_and_gaps(config, store, make_event):
    capture(store, configured(config), make_event, 0, sequence=2)
    with store.db:
        store.db.execute("UPDATE tool_calls SET received=0")
        store.db.execute("UPDATE usage_gaps SET received=0")
    store.cleanup(1)
    assert store.db.execute("SELECT COUNT(*) FROM tool_calls").fetchone()[0] == 0
    assert store.db.execute("SELECT COUNT(*) FROM usage_gaps").fetchone()[0] == 0
