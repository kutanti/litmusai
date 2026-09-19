"""Evidence-constrained evaluation of trusted conversation rubrics."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from typing import Annotated, Protocol

from pydantic import Field

from litmusai.runtime.config import ConversationPolicy
from litmusai.runtime.detectors import ClassifierVerdict, HTTPInjectionClassifier, result
from litmusai.runtime.models import (
    CapturedEvent,
    DetectionResult,
    Identifier,
    PolicyEvaluationTrace,
    RiskScore,
    ToolActivity,
)


class PolicyVerdict(ClassifierVerdict):
    """Positive decisions must cite supplied evidence including the current activity."""

    source_event_ids: list[Identifier] = Field(default_factory=list, max_length=50)
    evidence: list[Annotated[str, Field(min_length=1, max_length=1000)]] = Field(
        default_factory=list,
        max_length=10,
    )
    risk_score: float | None = Field(default=None, ge=0, le=1, strict=True)


class PolicyEvaluator(Protocol):
    """Evaluate only the provided evidence; never execute or rerun an agent."""

    async def evaluate(self, body: dict[str, object]) -> PolicyVerdict:
        """Return a bounded structured decision with evidence references."""
        ...


class HTTPPolicyEvaluator(HTTPInjectionClassifier):
    """Reuse bounded HTTPS transport with the separate conversation-policy contract."""

    async def evaluate(self, body: dict[str, object]) -> PolicyVerdict:
        """Validate the entire response before treating it as evidence."""
        return PolicyVerdict.model_validate_json(await self._request(body))


async def evaluate_policy(
    captured: CapturedEvent,
    context: list[CapturedEvent],
    incomplete: bool,
    policy: ConversationPolicy,
    reserve_budget: Callable[[], bool],
    evaluator: PolicyEvaluator | None = None,
) -> DetectionResult:
    """Apply prerequisites and thresholds without treating missing evidence as a pass."""
    event = captured.event
    started = time.perf_counter()
    called = False
    verdict: PolicyVerdict | None = None
    decision = "policy_evaluation"
    sources: list[str] = [event.event_id]
    evidence: list[str] = []
    score: RiskScore | None = None

    def finish(outcome: str, reason: str) -> DetectionResult:
        return result(
            captured,
            policy,
            "conversation_policy:" + policy.policy_id,
            outcome=outcome,
            reason=reason,
            category=policy.category,
            detector_version=policy.evaluator.version,
            evidence=evidence,
        ).model_copy(
            update={
                "source_event_ids": sources,
                "context_incomplete": incomplete,
                "risk_score": score,
                "evaluation": PolicyEvaluationTrace(
                    decision=decision,
                    evaluator_version=policy.evaluator.version,
                    provider_called=called,
                    elapsed_ms=(time.perf_counter() - started) * 1000 if called else 0,
                    reported_cost_usd=verdict.reported_cost_usd if verdict else None,
                    input_tokens=verdict.input_tokens if verdict else None,
                    output_tokens=verdict.output_tokens if verdict else None,
                ),
            }
        )

    if len(context) < policy.min_context_events:
        incomplete, decision = True, "missing_context"
        return finish("insufficient_context", "required conversation history unavailable")

    items: list[dict[str, object]] = []
    available: dict[str, CapturedEvent] = {}
    remaining = policy.evaluator.context_chars
    for item in [captured, *reversed(context)]:
        payload = json.dumps(item.event.payload.model_dump(mode="json"), ensure_ascii=True)
        selected = payload[:remaining]
        if len(selected) != len(payload):
            incomplete = True
        # Only complete payloads can serve as authoritative grounding evidence.
        if selected:
            items.append(
                {
                    "event_id": item.event.event_id,
                    "event_type": item.event.event_type,
                    "observed_at": item.event.observed_at.isoformat(),
                    "received_at": item.received_at.isoformat(),
                    "content_is_untrusted": True,
                    "content": selected,
                }
            )
            if len(selected) == len(payload):
                available[item.event.event_id] = item
        remaining -= len(selected)
        if remaining <= 0:
            incomplete = True
            break
    grounding = {
        key
        for key, item in available.items()
        if item.event.event_type == "tool.completed"
        and isinstance(item.event.payload, ToolActivity)
        and item.event.payload.name in policy.grounding_tools
        and item.event.payload.result is not None
        and not item.event.payload.error_type
    }
    if (
        event.event_id not in available
        or (policy.category == "ungrounded_response" and not grounding)
        or (len(available) - int(event.event_id in available) < policy.min_context_events)
    ):
        incomplete, decision = True, "missing_context"
        return finish("insufficient_context", "required evidence does not fit available context")
    if not reserve_budget():
        decision = "budget_exhausted"
        return finish("skipped", "conversation policy budget exhausted; coverage degraded")
    body: dict[str, object] = {
        "schema_version": "1.0",
        "task": "evaluate_conversation_policy",
        "content_is_untrusted": True,
        "context_incomplete": incomplete,
        "current_event_id": event.event_id,
        "policy": {
            "id": policy.policy_id,
            "version": policy.version,
            "category": policy.category,
            "rubric": policy.rubric,
            "threshold": policy.threshold,
            "score_semantics": policy.score_semantics,
            "score_version": policy.score_version,
        },
        "grounding_event_ids": sorted(grounding),
        "events": list(reversed(items)),
    }
    try:
        called = True
        adapter = evaluator or HTTPPolicyEvaluator(policy.evaluator)
        verdict = await asyncio.wait_for(
            adapter.evaluate(body), timeout=policy.evaluator.timeout_seconds
        )
        # Revalidate injected adapters as well as the HTTP response.
        verdict = PolicyVerdict.model_validate(verdict.model_dump())
        incomplete = incomplete or verdict.context_incomplete
        outcome = verdict.outcome
        if outcome == "needs_review":
            outcome = "insufficient_context"
        if policy.threshold is not None and outcome in {"clear", "detected"}:
            if verdict.risk_score is None:
                return finish("insufficient_context", "evaluator omitted the required policy score")
            score = RiskScore(
                value=verdict.risk_score,
                threshold=policy.threshold,
                semantics=policy.score_semantics or "",
                version=policy.score_version or "",
            )
            outcome = "detected" if verdict.risk_score >= policy.threshold else "clear"
        cited = set(verdict.source_event_ids)
        if not cited.issubset(available):
            raise ValueError("unknown or incomplete evidence references")
        if outcome == "detected":
            if event.event_id not in cited or not verdict.evidence:
                raise ValueError("positive verdict lacks current evidence")
            if policy.category == "suspicious_pattern" and len(cited) < 2:
                raise ValueError("pattern verdict lacks cross-turn evidence")
            if policy.category == "ungrounded_response" and not cited.intersection(grounding):
                raise ValueError("grounding verdict lacks authoritative evidence")
        if outcome == "clear" and incomplete:
            outcome = "insufficient_context"
        sources = list(dict.fromkeys(verdict.source_event_ids)) or sources
        evidence = verdict.evidence
        return finish(outcome, verdict.reason)
    except Exception:
        return finish(
            "error", "conversation evaluator failed or returned invalid evidence; coverage degraded"
        )
