"""Classifier scores, screen-only telemetry, estimated cost, and rate-limit skips."""

import json

import httpx
import pytest

from litmusai.runtime.config import ClassifierConfig, ConversationPolicy
from litmusai.runtime.detectors import (
    ClassifierVerdict,
    ProviderRateLimitError,
    new_client,
    post_json,
    retry_after_seconds,
)
from litmusai.runtime.engine import Engine
from litmusai.runtime.models import CloudEvent, Message, RiskScore
from litmusai.runtime.policies import PolicyVerdict
from litmusai.runtime.redaction import Redactor

SCORE = {"threshold": 0.7, "semantics": "calibrated injection probability", "version": "cal-1"}


class Screen:
    def __init__(self, verdict=None, error=None):
        self.verdict = verdict
        self.error = error

    async def classify(self, captured, context, incomplete):
        if self.error:
            raise self.error
        return self.verdict


def screen_only(config):
    classifier = ClassifierConfig(
        endpoint="https://screen.example/evaluate", api_key_env="SCREEN_KEY", version="screen-v1"
    )
    project = config.projects[0].model_copy(update={"classifier": classifier})
    return config.model_copy(update={"projects": [project]})


def accept(store, config, make_event, **fields):
    fields.setdefault("event_type", "message.received")
    fields.setdefault("payload", Message(text="synthetic content"))
    fields.setdefault("tool_call_id", None)
    captured = Redactor(config.projects[0].policy).capture(make_event(**fields))
    assert store.accept(captured, config.projects[0]) == "accepted"
    return captured


def injection_findings(store):
    return [
        row["finding"]
        for row in store.findings("p", limit=100)
        if row["finding"]["detector"] == "prompt_injection"
    ]


async def run(store, config, classifier):
    engine = Engine(store, config, classifiers={"p": classifier})
    while await engine.process_one("p", True):
        pass


async def test_score_version_and_estimate_reach_findings_alerts_and_status(
    config, store, make_event
):
    config = screen_only(config)
    store.register_config(config)
    accept(store, config, make_event)
    verdict = ClassifierVerdict(
        outcome="detected",
        reason="decision model probability met the detection threshold",
        risk_score=RiskScore(value=0.91, **SCORE),
        estimated_cost_usd=0.00002,
        input_tokens=400,
        output_tokens=0,
        detector_version="jev-1.13.0.q-0123abcd",
    )
    await run(store, config, Screen(verdict))
    finding = injection_findings(store)[0]
    assert finding["risk_score"]["value"] == 0.91
    assert finding["detector_version"] == "screen-v1+jev-1.13.0.q-0123abcd"
    trace = finding["evaluation"]
    assert trace["decision"] == "review_not_configured"
    assert "review_version" not in trace and "gate_version" not in trace
    assert trace["estimated_cost_usd"] == 0.00002
    assert "reported_cost_usd" in trace and trace["reported_cost_usd"] is None

    alert = store.alerts("p")[0]
    assert alert["schema_version"] == "1.4"
    assert alert["risk_score"]["threshold"] == 0.7
    assert alert["detector_version"] == "screen-v1+jev-1.13.0.q-0123abcd"
    envelope = json.loads(
        store.db.execute("SELECT content FROM revisions").fetchone()[0]
    )
    assert CloudEvent.model_validate(envelope).data.evaluation.review_version is None
    assert "review_version" not in envelope["data"]["evaluation"]

    row = store.status("p")["evaluation_metrics"][0]
    assert row["stage"] == "screen" and row["decision"] == "review_not_configured"
    assert row["cost_samples"] == 0 and row["reported_cost_usd"] == 0
    assert row["estimate_samples"] == 1 and row["estimated_cost_usd"] == 0.00002


@pytest.mark.parametrize(
    ("outcome", "value"),
    [("detected", 0.2), ("clear", 0.9), ("needs_review", 0.95)],
)
async def test_score_that_contradicts_outcome_is_an_error(
    config, store, make_event, outcome, value
):
    config = screen_only(config)
    store.register_config(config)
    accept(store, config, make_event)
    verdict = ClassifierVerdict(
        outcome=outcome, reason="synthetic", risk_score=RiskScore(value=value, **SCORE)
    )
    await run(store, config, Screen(verdict))
    finding = injection_findings(store)[0]
    assert finding["outcome"] == "error"
    assert "risk_score" not in finding
    assert store.alerts("p") == []


async def test_uncertain_score_is_kept_when_outcome_is_insufficient(config, store, make_event):
    config = screen_only(config)
    store.register_config(config)
    accept(store, config, make_event)
    verdict = ClassifierVerdict(
        outcome="insufficient_context",
        reason="a long tool result was not fully scanned",
        context_incomplete=True,
        risk_score=RiskScore(value=0.9, **SCORE),
    )
    await run(store, config, Screen(verdict))
    finding = injection_findings(store)[0]
    assert finding["outcome"] == "insufficient_context"
    assert finding["risk_score"]["value"] == 0.9


async def test_screen_only_findings_without_new_fields_keep_version(config, store, make_event):
    config = screen_only(config)
    store.register_config(config)
    accept(store, config, make_event)
    await run(store, config, Screen(ClassifierVerdict(outcome="clear", reason="no attack")))
    finding = injection_findings(store)[0]
    assert finding["detector_version"] == "screen-v1"
    assert finding["evaluation"]["provider_called"] is True
    assert "estimated_cost_usd" not in finding["evaluation"]


@pytest.mark.parametrize("called", [True, False])
async def test_rate_limit_becomes_explicit_skip(config, store, make_event, called):
    config = screen_only(config)
    store.register_config(config)
    accept(store, config, make_event)
    error = ProviderRateLimitError(5.0, provider_called=called)
    await run(store, config, Screen(error=error))
    finding = injection_findings(store)[0]
    assert finding["outcome"] == "skipped"
    assert "rate limited" in finding["reason"]
    assert finding["evaluation"]["provider_called"] is called


async def test_policy_rate_limit_estimate_and_version(config, store, make_event):
    rule = ConversationPolicy(
        policy_id="abuse",
        version="1",
        category="abuse",
        rubric="Flag abusive user messages.",
        evaluator=ClassifierConfig(
            endpoint="https://policy.example/evaluate",
            api_key_env="POLICY_KEY",
            version="policy-v1",
        ),
        event_types=["message.received"],
    )
    project = config.projects[0].model_copy(update={"conversation_policies": [rule]})
    config = config.model_copy(update={"projects": [project]})
    store.register_config(config)

    class Limited:
        async def evaluate(self, body):
            raise ProviderRateLimitError(None, provider_called=False)

    class Scored:
        async def evaluate(self, body):
            return PolicyVerdict(
                outcome="detected",
                reason="decision model probability met the policy threshold",
                source_event_ids=[body["current_event_id"]],
                evidence=["current event scored by a decision model"],
                estimated_cost_usd=0.00001,
                detector_version="laya-english.q-89abcdef",
            )

    first = accept(store, config, make_event)
    engine = Engine(store, config, policy_evaluators={("p", "abuse"): Limited()})
    while await engine.process_one("p", True, policies=True):
        pass
    policy_finding = next(
        r["finding"]
        for r in store.findings("p")
        if r["finding"]["event_id"] == first.event.event_id
        and r["finding"]["detector"] == "conversation_policy:abuse"
    )
    assert policy_finding["outcome"] == "skipped"
    assert policy_finding["evaluation"]["decision"] == "provider_rate_limited"
    assert policy_finding["evaluation"]["provider_called"] is False

    second = accept(store, config, make_event, sequence=2)
    engine = Engine(store, config, policy_evaluators={("p", "abuse"): Scored()})
    while await engine.process_one("p", True, policies=True):
        pass
    scored = next(
        r["finding"]
        for r in store.findings("p")
        if r["finding"]["event_id"] == second.event.event_id
        and r["finding"]["detector"] == "conversation_policy:abuse"
    )
    assert scored["outcome"] == "detected"
    assert scored["detector_version"] == "policy-v1+laya-english.q-89abcdef"
    assert scored["evaluation"]["estimated_cost_usd"] == 0.00001
    alert = next(a for a in store.alerts("p") if a["category"] == "abuse")
    assert alert["schema_version"] == "1.4"


def test_retry_after_parsing():
    assert retry_after_seconds("5") == 5.0
    assert retry_after_seconds("-3") == 0.0
    assert retry_after_seconds("Wed, 21 Oct 2015 07:28:00 GMT") == 0.0
    assert retry_after_seconds("soon") is None
    assert retry_after_seconds(None) is None
    assert retry_after_seconds("nan") is None


async def test_post_json_maps_rate_limits_only_when_requested(httpx_mock):
    httpx_mock.add_response(status_code=429, headers={"Retry-After": "7"})
    with pytest.raises(ProviderRateLimitError) as limited:
        await post_json(
            None, "https://provider.example/v1", {}, headers={}, timeout=1,
            rate_limited_statuses=frozenset({429}),
        )
    assert limited.value.retry_after_seconds == 7.0
    httpx_mock.add_response(status_code=429)
    with pytest.raises(httpx.HTTPStatusError):
        await post_json(None, "https://provider.example/v1", {}, headers={}, timeout=1)


async def test_shared_client_stores_no_cookies_and_bounds_responses(httpx_mock):
    httpx_mock.add_response(json={"ok": True}, headers={"Set-Cookie": "session=abc; Path=/"})
    httpx_mock.add_response(json={"ok": True})
    httpx_mock.add_response(content=b"x" * 20000)
    async with new_client() as client:
        for _ in range(2):
            await post_json(client, "https://provider.example/v1", {}, headers={}, timeout=1)
        with pytest.raises(ValueError):
            await post_json(client, "https://provider.example/v1", {}, headers={}, timeout=1)
    second = httpx_mock.get_requests()[1]
    assert "cookie" not in second.headers
    assert not client.cookies


def _finding(**values):
    from litmusai.runtime.models import DetectionResult

    return DetectionResult(
        event_id="e",
        detector=values.pop("detector", "prompt_injection"),
        policy_id="p",
        policy_version="1",
        outcome="detected",
        category="prompt_injection",
        reason="synthetic",
        **values,
    )


def test_schema_versions_and_payload_exclusions():
    from litmusai.runtime.models import EvaluationTrace, PolicyEvaluationTrace
    from litmusai.runtime.store import alert_schema_version, omitted_fields

    review = EvaluationTrace(
        stage="screen",
        decision="screen_detected",
        screen_version="s",
        review_version="r",
        gate_version="g",
    )
    screen_only = EvaluationTrace(
        stage="screen", decision="review_not_configured", screen_version="s"
    )
    policy = PolicyEvaluationTrace(decision="policy_evaluation", evaluator_version="v")
    score = RiskScore(value=0.9, **SCORE)
    cases = [
        (_finding(), "1.0"),
        (_finding(evaluation=review), "1.2"),
        (_finding(evaluation=policy, risk_score=score), "1.3"),
        (_finding(evaluation=screen_only), "1.4"),
        (_finding(evaluation=review, risk_score=score), "1.4"),
        (_finding(evaluation=review.model_copy(update={"estimated_cost_usd": 0.1})), "1.4"),
        (_finding(evaluation=policy.model_copy(update={"estimated_cost_usd": 0.1})), "1.4"),
    ]
    for finding, expected in cases:
        assert alert_schema_version(finding) == expected
    dumped = json.loads(_finding(evaluation=review).model_dump_json(
        exclude=omitted_fields(_finding(evaluation=review))
    ))
    assert set(dumped["evaluation"]) == {
        "stage", "decision", "screen_version", "review_version", "gate_version",
        "provider_called", "elapsed_ms", "reported_cost_usd", "input_tokens", "output_tokens",
    }
    assert "usage" not in dumped and "risk_score" not in dumped
