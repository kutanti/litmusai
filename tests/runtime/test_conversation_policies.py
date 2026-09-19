"""Policy orchestration tests, not claims of a model's detection accuracy."""

import asyncio
import json
import sqlite3
import threading

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from litmusai.runtime.config import ClassifierConfig, ConversationPolicy, ProjectConfig
from litmusai.runtime.engine import Engine
from litmusai.runtime.models import CloudEvent, Message, ToolActivity
from litmusai.runtime.policies import HTTPPolicyEvaluator, PolicyVerdict
from litmusai.runtime.redaction import Redactor
from litmusai.runtime.service import create_app
from litmusai.runtime.store import Store


def rule(category="business_policy", **overrides):
    values = dict(
        policy_id="customer-rules",
        version="1",
        category=category,
        rubric="Flag responses that promise a refund greater than 100 dollars.",
        event_types=["response.completed"],
        evaluator=ClassifierConfig(
            provider="http",
            endpoint="https://judge.example/policy",
            api_key_env="POLICY_KEY",
            version="judge-v1",
        ),
    )
    if category == "suspicious_pattern":
        values["min_context_events"] = 1
    if category == "ungrounded_response":
        values["grounding_tools"] = ["lookup_policy"]
    return ConversationPolicy(**{**values, **overrides})


def configured(config, *policies):
    return config.model_copy(
        update={
            "projects": [
                config.projects[0].model_copy(
                    update={"conversation_policies": list(policies or [rule()])},
                )
            ]
        }
    )


def accept(store, config, make_event, **overrides):
    event = make_event(
        **{
            "event_type": "response.completed",
            "tool_call_id": None,
            "payload": Message(text="Your refund is 500 dollars.", role="assistant"),
            **overrides,
        }
    )
    item = Redactor(config.projects[0].policy).capture(event)
    assert store.accept(item, config.projects[0]) == "accepted"
    return item


def source(store, config, make_event, **overrides):
    return accept(
        store,
        config,
        make_event,
        **{
            "event_type": "tool.completed",
            "tool_call_id": "lookup-call",
            "payload": ToolActivity(name="lookup_policy", result={"max_refund": 100}),
            **overrides,
        },
    )


class Evaluator:
    def __init__(self, outcome="detected", **response):
        self.outcome, self.response, self.seen = outcome, response, []

    async def evaluate(self, body):
        self.seen.append(body)
        return PolicyVerdict(
            **{
                "outcome": self.outcome,
                "reason": "synthetic policy verdict",
                "source_event_ids": [row["event_id"] for row in body["events"]],
                "evidence": ["The current response and captured policy disagree."],
                **self.response,
            }
        )


async def drain(engine):
    while await engine.process_one("p", policies=True):
        pass


def finding(store):
    return next(
        row["finding"]
        for row in store.findings("p")
        if row["finding"]["detector"].startswith("conversation_policy:")
    )


@pytest.mark.parametrize(
    "category",
    [
        "sensitive_data_request",
        "business_policy",
        "suspicious_pattern",
        "abuse",
        "out_of_scope",
        "ungrounded_response",
    ],
)
async def test_configured_categories_emit_cited_versioned_events(
    config, store, make_event, category
):
    config = configured(config, rule(category))
    source(store, config, make_event)
    current = accept(store, config, make_event)
    evaluator = Evaluator()
    engine = Engine(store, config, policy_evaluators={("p", "customer-rules"): evaluator})
    await drain(engine)
    alert = store.alerts("p")[0]
    assert alert["category"] == category and alert["schema_version"] == "1.3"
    assert alert["source_event_ids"][-1] == current.event.event_id
    assert alert["policy_version"] == "1" and alert["detector_version"] == "judge-v1"
    assert alert["evaluation"]["stage"] == "policy"
    assert "risk_score" not in alert
    assert (
        evaluator.seen[0]["policy"]["rubric"] == config.projects[0].conversation_policies[0].rubric
    )
    assert evaluator.seen[0]["content_is_untrusted"]


async def test_absent_grounding_prevents_provider_call_and_clear(config, store, make_event):
    config = configured(config, rule("ungrounded_response"))
    accept(store, config, make_event)
    evaluator = Evaluator("clear")
    await drain(Engine(store, config, policy_evaluators={("p", "customer-rules"): evaluator}))
    assert evaluator.seen == [] and store.alerts("p") == []
    assert finding(store)["outcome"] == "insufficient_context"
    assert finding(store)["evaluation"]["decision"] == "missing_context"
    assert store.db.execute("SELECT COUNT(*) FROM policy_budgets").fetchone()[0] == 0


@pytest.mark.parametrize("scope", ["agent_id", "deployment_id", "session_id", "project_id"])
async def test_grounding_never_leaks_across_scopes(config, store, make_event, scope):
    config = configured(config, rule("ungrounded_response"))
    other = config
    if scope == "project_id":
        other = config.model_copy(
            update={
                "projects": [
                    config.projects[0].model_copy(
                        update={"project_id": "other"},
                    )
                ]
            }
        )
    source(store, other, make_event, **{scope: "other"})
    accept(store, config, make_event)
    evaluator = Evaluator()
    await drain(Engine(store, config, policy_evaluators={("p", "customer-rules"): evaluator}))
    assert evaluator.seen == []
    assert finding(store)["outcome"] == "insufficient_context"


@pytest.mark.parametrize(
    "kind",
    [
        "unknown_reference",
        "no_current",
        "no_evidence",
        "pattern_only_current",
        "grounding_only_current",
    ],
)
async def test_positive_verdicts_require_valid_current_and_policy_evidence(
    config,
    store,
    make_event,
    kind,
):
    category = (
        "suspicious_pattern"
        if kind == "pattern_only_current"
        else ("ungrounded_response" if kind == "grounding_only_current" else "business_policy")
    )
    config = configured(config, rule(category))
    old = source(store, config, make_event)
    current = accept(store, config, make_event)
    values = {"source_event_ids": [current.event.event_id]}
    if kind == "unknown_reference":
        values["source_event_ids"] = [current.event.event_id, "invented"]
    if kind == "no_current":
        values["source_event_ids"] = [old.event.event_id]
    if kind == "no_evidence":
        values["evidence"] = []
    evaluator = Evaluator(**values)
    await drain(Engine(store, config, policy_evaluators={("p", "customer-rules"): evaluator}))
    assert store.alerts("p") == [] and finding(store)["outcome"] == "error"


@pytest.mark.parametrize(
    "score,expected", [(0.69, "clear"), (0.7, "detected"), (None, "insufficient_context")]
)
async def test_threshold_has_explicit_score_semantics(config, store, make_event, score, expected):
    config = configured(
        config,
        rule(
            threshold=0.7, score_semantics="customer violation score", score_version="customer-v1"
        ),
    )
    accept(store, config, make_event)
    evaluator = Evaluator(risk_score=score)
    await drain(Engine(store, config, policy_evaluators={("p", "customer-rules"): evaluator}))
    assert finding(store)["outcome"] == expected
    if expected == "detected":
        result = store.alerts("p")[0]["risk_score"]
        assert result == {
            "value": 0.7,
            "threshold": 0.7,
            "semantics": "customer violation score",
            "version": "customer-v1",
        }


async def test_incomplete_context_never_certifies_clear(config, store, make_event):
    config = configured(config)
    accept(store, config, make_event, sequence=2)
    evaluator = Evaluator("clear")
    await drain(Engine(store, config, policy_evaluators={("p", "customer-rules"): evaluator}))
    assert finding(store)["outcome"] == "insufficient_context"
    assert finding(store)["context_incomplete"]


async def test_truncation_cannot_masquerade_as_authoritative_grounding(config, store, make_event):
    limited = rule("ungrounded_response")
    limited = limited.model_copy(
        update={
            "evaluator": limited.evaluator.model_copy(
                update={"context_chars": 512},
            )
        }
    )
    config = configured(config, limited)
    source(store, config, make_event, payload=ToolActivity(name="lookup_policy", result="x" * 2000))
    accept(store, config, make_event)
    evaluator = Evaluator()
    await drain(Engine(store, config, policy_evaluators={("p", "customer-rules"): evaluator}))
    assert evaluator.seen == [] and finding(store)["context_incomplete"]


async def test_multiple_policies_have_independent_budgets_and_filters(
    config, store, make_event, monkeypatch
):
    monkeypatch.setattr("litmusai.runtime.store.time.time", lambda: 1800000000)
    first = rule(agents=["agent"], deployments=["default"])
    first = first.model_copy(
        update={"evaluator": first.evaluator.model_copy(update={"calls_per_minute": 1})}
    )
    second = first.model_copy(update={"policy_id": "second", "category": "out_of_scope"})
    config = configured(config, first, second)
    for _ in range(2):
        accept(store, config, make_event)
    accept(store, config, make_event, agent_id="excluded")
    accept(store, config, make_event, event_type="message.received")
    a, b = Evaluator(), Evaluator()
    engine = Engine(
        store, config, policy_evaluators={("p", "customer-rules"): a, ("p", "second"): b}
    )
    await drain(engine)
    assert len(a.seen) == len(b.seen) == 1
    results = [r["finding"] for r in store.findings("p")]
    assert [r["outcome"] for r in results].count("skipped") == 2
    assert {a["policy_id"] for a in store.alerts("p")} == {"customer-rules", "second"}


async def test_restart_deduplication_and_saved_rubric(config, make_event):
    config = configured(config)
    original = Store(config.database)
    original.register_config(config)
    captured = accept(original, config, make_event)
    original.close()
    restarted = Store(config.database)
    try:
        assert restarted.accept(captured, config.projects[0]) == "duplicate"
        newer = configured(config, rule(version="2", rubric="changed rule"))
        evaluator = Evaluator()
        await drain(
            Engine(restarted, newer, policy_evaluators={("p", "customer-rules"): evaluator})
        )
        assert len(evaluator.seen) == 1
        assert evaluator.seen[0]["policy"]["version"] == "1"
        assert evaluator.seen[0]["policy"]["rubric"] != "changed rule"
    finally:
        restarted.close()


async def test_redaction_applies_to_requests_and_evaluator_evidence(config, store, make_event):
    config = configured(config)
    secret = "ghp_" + "A" * 36
    accept(store, config, make_event, payload=Message(role="assistant", text=secret))
    evaluator = Evaluator(reason=secret, evidence=[secret])
    await drain(Engine(store, config, policy_evaluators={("p", "customer-rules"): evaluator}))
    assert secret not in json.dumps(evaluator.seen)
    assert secret not in "\n".join(store.db.iterdump())


@pytest.mark.parametrize("behavior", ["timeout", "error", "uncertain"])
async def test_evaluator_failure_remains_explicit(config, store, make_event, behavior):
    policy = rule()
    policy = policy.model_copy(
        update={"evaluator": policy.evaluator.model_copy(update={"timeout_seconds": 0.01})}
    )
    config = configured(config, policy)
    accept(store, config, make_event)

    class Failing(Evaluator):
        async def evaluate(self, body):
            if behavior == "timeout":
                await asyncio.sleep(0.1)
            if behavior == "error":
                raise ValueError("private provider failure")
            return PolicyVerdict(outcome="needs_review", reason="uncertain")

    await drain(Engine(store, config, policy_evaluators={("p", "customer-rules"): Failing()}))
    assert finding(store)["outcome"] == (
        "insufficient_context" if behavior == "uncertain" else "error"
    )
    assert store.alerts("p") == []


def test_worker_publishes_policy_alert_and_optional_score(config, store, make_event):
    config = configured(
        config, rule(threshold=0.7, score_semantics="violation score", score_version="1")
    )
    delivered, notification = [], threading.Event()

    class Receiver:
        async def publish(self, body, delivery_id):
            delivered.append(json.loads(body))
            notification.set()

    app = create_app(
        config,
        store=store,
        policy_evaluators={("p", "customer-rules"): Evaluator(risk_score=0.9)},
        publisher_factory=lambda _: Receiver(),
    )
    with TestClient(app) as client:
        event = make_event(event_type="response.completed", payload=Message(text="synthetic"))
        response = client.post(
            "/v1/events",
            json=event.model_dump(mode="json"),
            headers={"Authorization": "Bearer synthetic-project-key-1234"},
        )
        assert response.json()["results"][0]["status"] == "accepted"
        assert notification.wait(5)
        assert delivered[0]["data"]["risk_score"]["value"] == 0.9
        assert delivered[0]["data"]["source_event_ids"] == [event.event_id]
        assert CloudEvent.model_validate(delivered[0]).data.evaluation.stage == "policy"


@pytest.mark.parametrize(
    "overrides",
    [
        {"category": "ungrounded_response", "grounding_tools": []},
        {"category": "suspicious_pattern", "min_context_events": 0},
        {"threshold": 0.7},
        {"score_semantics": "probability"},
        {"event_types": ["session.ended"]},
        {"min_context_events": 50},
    ],
)
def test_incomplete_policy_definitions_rejected(overrides):
    with pytest.raises(ValidationError):
        rule(**overrides)


def test_policy_ids_and_versions_are_immutable(config, store):
    config = configured(config)
    store.register_config(config)
    with pytest.raises(ValueError, match="version"):
        store.register_config(configured(config, rule(rubric="changed without new version")))
    values = config.projects[0].model_dump()
    values["conversation_policies"][0]["policy_id"] = "default"
    with pytest.raises(ValidationError, match="unique"):
        ProjectConfig.model_validate(values)


async def test_http_policy_contract_rejects_provider_policy_override():
    class Endpoint(HTTPPolicyEvaluator):
        async def _request(self, body):
            return json.dumps(
                {"outcome": "detected", "reason": "unsafe", "category": "attacker-controlled"}
            ).encode()

    with pytest.raises(ValidationError):
        await Endpoint(rule().evaluator).evaluate({"task": "evaluate_conversation_policy"})


async def test_database_upgrade_preserves_old_snapshots_and_pending_work(config, make_event):
    first = Store(config.database)
    captured = accept(first, config, make_event)
    legacy = config.projects[0].model_dump(
        exclude={"usage_policies", "review", "conversation_policies"}
    )
    with first.db:
        first.db.execute("DELETE FROM schema_version")
        first.db.execute("INSERT INTO schema_version VALUES (1)")
        first.db.execute("DROP TABLE policy_budgets")
        first.db.execute("UPDATE jobs SET config=?", (json.dumps(legacy),))
    first.close()
    upgraded = Store(config.database)
    try:
        assert upgraded.db.execute("SELECT MAX(version) FROM schema_version").fetchone()[0] == 2
        assert upgraded.accept(captured, config.projects[0]) == "duplicate"
        engine = Engine(upgraded, config)
        while await engine.process_one("p"):
            pass
        assert upgraded.status("p")["jobs"]["done"] == 2
        assert upgraded.use_policy_budget("p", "new-policy", 1)
    finally:
        upgraded.close()


def test_unknown_database_version_rejected_before_schema_changes(tmp_path):
    path = str(tmp_path / "future.sqlite")
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE schema_version(version INTEGER)")
        db.execute("INSERT INTO schema_version VALUES (999)")
    with pytest.raises(ValueError, match="database version"):
        Store(path)
    with sqlite3.connect(path) as db:
        tables = [row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        assert tables == ["schema_version"]


def test_jobs_save_only_their_applicable_policy_snapshot(config, store, make_event):
    config = configured(config, rule(), rule(policy_id="other", rubric="unrelated policy"))
    captured = accept(store, config, make_event)
    jobs = store.db.execute(
        "SELECT detector,config FROM jobs WHERE event=?", (captured.event.event_id,)
    ).fetchall()
    for job in jobs:
        snapshot = ProjectConfig.model_validate_json(job["config"])
        expected = (
            job["detector"].removeprefix("conversation_policy:")
            if job["detector"].startswith("conversation_policy:")
            else None
        )
        assert [p.policy_id for p in snapshot.conversation_policies] == (
            [expected] if expected else []
        )
