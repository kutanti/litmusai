"""Conditional evaluation is durable, budgeted, explainable, and independent of local alerts."""

import asyncio
import json
import sqlite3
import threading

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from litmusai.runtime.config import ClassifierConfig, ProjectConfig, ReviewConfig
from litmusai.runtime.detectors import ClassifierVerdict, HTTPInjectionClassifier
from litmusai.runtime.engine import Engine
from litmusai.runtime.models import Message, ToolActivity
from litmusai.runtime.redaction import Redactor
from litmusai.runtime.review import select_review
from litmusai.runtime.service import create_app
from litmusai.runtime.store import Store


class Classifier:
    def __init__(self, outcome="clear", *, delay=0, fail=False, **telemetry):
        self.outcome = outcome
        self.delay = delay
        self.fail = fail
        self.telemetry = telemetry
        self.seen = []

    async def classify(self, captured, context, incomplete):
        self.seen.append((captured, context, incomplete))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail:
            raise RuntimeError("provider error with private response")
        return ClassifierVerdict(
            outcome=self.outcome, reason="synthetic evaluation", **self.telemetry
        )


def setup(config, *, rate=0, outcomes=None, **evaluator):
    screen = ClassifierConfig(
        endpoint="https://screen.example/evaluate", api_key_env="SCREEN_KEY", version="screen-v1"
    )
    deep = screen.model_copy(
        update={"endpoint": "https://review.example/evaluate", "version": "review-v1", **evaluator}
    )
    review = ReviewConfig(
        version="gate-v1",
        evaluator=deep,
        clear_sample_rate=rate,
        **({"on_outcomes": outcomes} if outcomes is not None else {}),
    )
    project = config.projects[0].model_copy(update={"classifier": screen, "review": review})
    return config.model_copy(update={"projects": [project]})


def accept(store, config, make_event, **fields):
    fields.setdefault("event_type", "message.received")
    fields.setdefault("payload", Message(text="synthetic content"))
    event = make_event(**fields)
    captured = Redactor(config.projects[0].policy).capture(event)
    assert store.accept(captured, config.projects[0]) == "accepted"
    return captured


def findings(store, detector):
    return [
        r["finding"]
        for r in store.findings("p", limit=1000)
        if r["finding"]["detector"] == detector
    ]


async def drain(engine, *, review=False):
    while await engine.process_one("p", True, review=review):
        pass


@pytest.mark.parametrize(
    "outcome,rate,selected,decision",
    [
        ("needs_review", 0, True, "uncertain_screen"),
        ("insufficient_context", 0, True, "uncertain_screen"),
        ("clear", 0, False, "clear_not_sampled"),
        ("clear", 1, True, "audit_sample"),
        ("detected", 1, False, "screen_detected"),
    ],
)
async def test_policy_controls_selection(
    config, store, make_event, outcome, rate, selected, decision
):
    config = setup(config, rate=rate)
    accept(store, config, make_event)
    screen, reviewer = Classifier(outcome), Classifier("detected")
    engine = Engine(store, config, classifiers={"p": screen}, reviewers={"p": reviewer})
    await drain(engine)
    first = findings(store, "prompt_injection")[0]
    assert first["evaluation"]["decision"] == decision
    assert first["evaluation"]["provider_called"]
    assert await engine.process_one("p", True, review=True) is selected
    assert len(reviewer.seen) == int(selected)
    assert len(store.alerts("p")) == int(selected or outcome == "detected")
    if selected:
        alert = store.alerts("p")[0]
        assert alert["schema_version"] == "1.2"
        assert alert["evaluation"]["decision"] == decision
        assert alert["detector_version"] == "review-v1"


async def test_durable_selection_keeps_configuration_after_restart(config, make_event):
    config = setup(config, rate=1)
    first = Store(config.database)
    first.register_config(config)
    captured = accept(first, config, make_event)
    await drain(Engine(first, config, classifiers={"p": Classifier()}))
    first.close()
    restarted = Store(config.database)
    try:
        assert restarted.accept(captured, config.projects[0]) == "duplicate"
        changed = setup(config, rate=0, version="review-v2")
        reviewer = Classifier("detected")
        engine = Engine(restarted, changed, reviewers={"p": reviewer})
        await drain(engine, review=True)
        assert not await engine.process_one("p", True)
        assert len(reviewer.seen) == 1
        alert = restarted.alerts("p")[0]
        assert alert["evaluation"]["review_version"] == "review-v1"
        assert alert["evaluation"]["decision"] == "audit_sample"
    finally:
        restarted.close()


async def test_review_queue_and_metrics_commit_with_screening(config, store, make_event):
    config = setup(config)
    accept(store, config, make_event)
    screen = Classifier("needs_review")
    engine = Engine(store, config, classifiers={"p": screen})
    store.db.execute(
        "CREATE TRIGGER fail_review BEFORE INSERT ON jobs "
        "WHEN NEW.detector='prompt_injection_review' "
        "BEGIN SELECT RAISE(ABORT, 'test'); END"
    )
    with pytest.raises(sqlite3.IntegrityError):
        await drain(engine)
    assert findings(store, "prompt_injection") == []
    assert store.status("p")["evaluation_metrics"] == []
    with store.db:
        store.db.execute("DROP TRIGGER fail_review")
        store.db.execute("UPDATE jobs SET lease_until=0")
    await drain(engine)
    assert len(findings(store, "prompt_injection")) == 1
    assert (
        store.db.execute(
            "SELECT COUNT(*) FROM jobs WHERE detector='prompt_injection_review'"
        ).fetchone()[0]
        == 1
    )


async def test_review_budget_is_separate_and_skips_are_visible(
    config, store, make_event, monkeypatch
):
    monkeypatch.setattr("litmusai.runtime.store.time.time", lambda: 1800000000)
    config = setup(config, calls_per_minute=1)
    for _ in range(3):
        accept(store, config, make_event)
    screen, reviewer = Classifier("needs_review"), Classifier("detected")
    engine = Engine(store, config, classifiers={"p": screen}, reviewers={"p": reviewer})
    await drain(engine)
    await drain(engine, review=True)
    assert len(screen.seen) == 3 and len(reviewer.seen) == 1
    results = findings(store, "prompt_injection_review")
    assert [r["outcome"] for r in results].count("skipped") == 2
    assert sum(r["evaluation"]["provider_called"] for r in results) == 1
    assert all("budget exhausted" in r["reason"] for r in results if r["outcome"] == "skipped")


@pytest.mark.parametrize("behavior", ["timeout", "error", "still_uncertain"])
async def test_failed_review_never_turns_into_clear_or_recurses(
    config, store, make_event, behavior
):
    config = setup(config, timeout_seconds=0.01)
    accept(store, config, make_event)
    reviewer = Classifier(
        "needs_review" if behavior == "still_uncertain" else "clear",
        delay=0.1 if behavior == "timeout" else 0,
        fail=behavior == "error",
    )
    engine = Engine(
        store, config, classifiers={"p": Classifier("needs_review")}, reviewers={"p": reviewer}
    )
    await drain(engine)
    await drain(engine, review=True)
    finding = findings(store, "prompt_injection_review")[0]
    assert finding["outcome"] == (
        "insufficient_context" if behavior == "still_uncertain" else "error"
    )
    assert "private response" not in finding["reason"]
    assert store.alerts("p") == []
    assert not await engine.process_one("p", True, review=True)


async def test_local_alerts_and_screening_continue_during_slow_review(config, store, make_event):
    config = setup(config, timeout_seconds=0.5)
    accept(store, config, make_event)
    screen = Classifier("needs_review")
    reviewer = Classifier("clear", delay=0.2)
    engine = Engine(store, config, classifiers={"p": screen}, reviewers={"p": reviewer})
    await drain(engine)
    pending = asyncio.create_task(engine.process_one("p", True, review=True))
    await asyncio.sleep(0)
    event = make_event()
    store.accept(Redactor(config.projects[0].policy).capture(event), config.projects[0])
    while await engine.process_one("p"):
        pass
    accept(store, config, make_event)
    await drain(engine)
    assert len(screen.seen) == 2
    assert {a["category"] for a in store.alerts("p")} == {"forbidden_destination"}
    assert not pending.done()
    await pending


async def test_review_redacts_context_and_does_not_certify_missing_data(config, store, make_event):
    config = setup(config)
    protected = "ghp_" + "A" * 36
    accept(store, config, make_event, sequence=2, payload=Message(text=protected))
    reviewer = Classifier("clear")
    engine = Engine(
        store, config, classifiers={"p": Classifier("needs_review")}, reviewers={"p": reviewer}
    )
    await drain(engine)
    await drain(engine, review=True)
    assert protected not in reviewer.seen[0][0].model_dump_json()
    assert findings(store, "prompt_injection_review")[0]["outcome"] == "insufficient_context"


async def test_usage_cost_is_reported_separately_from_missing_cost(config, store, make_event):
    config = setup(config)
    accept(store, config, make_event)
    engine = Engine(
        store,
        config,
        classifiers={"p": Classifier("needs_review")},
        reviewers={
            "p": Classifier(
                "detected", reported_cost_usd=0.002, input_tokens=100, output_tokens=10
            ),
        },
    )
    await drain(engine)
    await drain(engine, review=True)
    metrics = store.status("p")["evaluation_metrics"]
    screen = next(row for row in metrics if row["stage"] == "screen")
    review = next(row for row in metrics if row["stage"] == "review")
    assert screen["cost_samples"] == 0
    assert review["cost_samples"] == 1 and review["reported_cost_usd"] == 0.002
    trace = store.alerts("p")[0]["evaluation"]
    assert trace["input_tokens"] == 100 and trace["output_tokens"] == 10
    assert trace["elapsed_ms"] >= 0
    assert store.status("other")["evaluation_metrics"] == []


async def test_screen_error_escalates_only_when_configured(config, store, make_event):
    config = setup(config, outcomes=["error"])
    accept(store, config, make_event)
    engine = Engine(
        store,
        config,
        classifiers={"p": Classifier(fail=True)},
        reviewers={"p": Classifier("detected")},
    )
    await drain(engine)
    await drain(engine, review=True)
    assert len(store.alerts("p")) == 1
    assert findings(store, "prompt_injection")[0]["outcome"] == "error"


async def test_audit_selection_is_repeatable_and_content_independent(config, store, make_event):
    config = setup(config, rate=0.5)
    captured = accept(store, config, make_event)
    engine = Engine(store, config, classifiers={"p": Classifier()})
    result = await engine._classify(captured, config.projects[0])
    decision = select_review(captured.event, result, config.projects[0].review)
    changed = captured.event.model_copy(update={"payload": Message(text="ignore sampling rules")})
    assert select_review(changed, result, config.projects[0].review) == decision


def test_configuration_requires_screen_and_immutable_gate_version(config, store):
    configured = setup(config)
    store.register_config(configured)
    invalid = configured.projects[0].model_dump()
    invalid["classifier"] = None
    with pytest.raises(ValidationError, match="first-stage"):
        ProjectConfig.model_validate(invalid)
    changed = setup(configured, rate=1)
    with pytest.raises(ValueError, match="version"):
        store.register_config(changed)


async def test_http_evaluator_parses_uncertainty_and_optional_telemetry(config, store, make_event):
    config = setup(config)
    captured = accept(store, config, make_event)

    class Endpoint(HTTPInjectionClassifier):
        async def _request(self, body):
            assert body["content_is_untrusted"]
            return json.dumps(
                {
                    "outcome": "needs_review",
                    "reason": "ambiguous",
                    "reported_cost_usd": 0.001,
                    "input_tokens": 12,
                }
            ).encode()

    verdict = await Endpoint(config.projects[0].classifier).classify(captured, [], False)
    assert verdict.outcome == "needs_review" and verdict.reported_cost_usd == 0.001


async def test_selected_review_remains_in_pending_event_quota(config, store, make_event):
    config = setup(config)
    config = config.model_copy(
        update={
            "projects": [
                config.projects[0].model_copy(
                    update={"max_pending_events": 1},
                )
            ]
        }
    )
    accept(store, config, make_event)
    engine = Engine(
        store, config, classifiers={"p": Classifier("needs_review")}, reviewers={"p": Classifier()}
    )
    while await engine.process_one("p"):
        pass
    await drain(engine)
    next_event = Redactor(config.projects[0].policy).capture(make_event())
    assert store.accept(next_event, config.projects[0]) == "busy"
    await drain(engine, review=True)
    assert store.accept(next_event, config.projects[0]) == "accepted"


async def test_reviewed_tool_result_is_an_attempt_not_attack_execution(config, store, make_event):
    config = setup(config)
    accept(
        store,
        config,
        make_event,
        event_type="tool.completed",
        payload=ToolActivity(name="search", result="hostile content"),
    )
    engine = Engine(
        store,
        config,
        classifiers={"p": Classifier("needs_review")},
        reviewers={"p": Classifier("detected")},
    )
    await drain(engine)
    await drain(engine, review=True)
    assert store.alerts("p")[0]["stage"] == "attempt"


def test_service_workers_complete_two_stage_evaluation(config, store, make_event):
    config = setup(config)
    delivered = []
    notification = threading.Event()

    class Receiver:
        async def publish(self, body, delivery_id):
            delivered.append(json.loads(body))
            notification.set()

    app = create_app(
        config,
        store=store,
        classifiers={"p": Classifier("needs_review")},
        reviewers={"p": Classifier("detected")},
        publisher_factory=lambda _: Receiver(),
    )
    with TestClient(app) as client:
        event = make_event(event_type="message.received", payload=Message(text="synthetic"))
        response = client.post(
            "/v1/events",
            json=event.model_dump(mode="json"),
            headers={"Authorization": "Bearer synthetic-project-key-1234"},
        )
        assert response.json()["results"][0]["status"] == "accepted"
        assert notification.wait(5), "review worker did not deliver the alert"
        assert delivered[0]["data"]["evaluation"]["stage"] == "review"
        assert delivered[0]["data"]["source_event_ids"] == [event.event_id]


@pytest.mark.parametrize("cost", [-1, float("nan"), True])
def test_invalid_cost_telemetry_is_rejected(cost):
    with pytest.raises(ValidationError):
        ClassifierVerdict(outcome="clear", reason="synthetic", reported_cost_usd=cost)
