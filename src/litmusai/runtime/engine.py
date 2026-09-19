"""Independent local, semantic, and destination workers for live monitoring."""

from __future__ import annotations

import asyncio
import json
import random
import time
from collections.abc import Callable

from litmusai.runtime.config import (
    DESTINATION_ADAPTER,
    Destination,
    ProjectConfig,
    RuntimeConfig,
    ThreatPolicy,
    ToolUsagePolicy,
)
from litmusai.runtime.detectors import (
    ClassifierVerdict,
    HTTPInjectionClassifier,
    InjectionClassifier,
    LakeraInjectionClassifier,
    result,
    sensitive_data,
    tool_policy,
)
from litmusai.runtime.models import CapturedEvent, DetectionResult, EvaluationTrace
from litmusai.runtime.publishers import AlertPublisher, PublishError, publisher_for
from litmusai.runtime.redaction import Redactor
from litmusai.runtime.review import select_review
from litmusai.runtime.store import Store


class Engine:
    """Run single-instance workers with separate queues for every project and destination."""

    def __init__(
        self,
        store: Store,
        config: RuntimeConfig,
        *,
        classifiers: dict[str, InjectionClassifier] | None = None,
        reviewers: dict[str, InjectionClassifier] | None = None,
        publisher_factory: Callable[[Destination], AlertPublisher] = publisher_for,
    ) -> None:
        self.store = store
        self.config = config
        self.classifiers = classifiers or {}
        self.reviewers = reviewers or {}
        self.publisher_factory = publisher_factory
        self._publishers: dict[str, AlertPublisher] = {}
        self.tasks: list[asyncio.Task[None]] = []
        self.last_worker_error: str | None = None
        self.worker_errors: dict[str, str] = {}
        self.last_cleanup: float | None = None

    async def start(self) -> None:
        """Start independent workers so slow providers and destinations cannot block local rules."""
        if self.tasks:
            raise RuntimeError("engine is already running")
        self.store.register_config(self.config)
        for project in self.config.projects:
            for semantic in (False, True):
                self.tasks.append(asyncio.create_task(self._jobs(project.project_id, semantic)))
            self.tasks.append(
                asyncio.create_task(self._jobs(project.project_id, True, review=True))
            )
        destinations = set(self.store.destination_ids()) | {
            d.destination_id for d in self.config.destinations
        }
        for destination in destinations:
            self.tasks.append(asyncio.create_task(self._deliveries(destination)))
        self.tasks.append(asyncio.create_task(self._cleanup()))

    async def stop(self) -> None:
        """Stop workers; interrupted leases become recoverable after their bounded expiry."""
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        self.tasks.clear()

    async def _jobs(self, project: str, semantic: bool, *, review: bool = False) -> None:
        worker_id = f"jobs:{project}:{'review' if review else semantic}"
        while True:
            try:
                if await self.process_one(project, semantic, review=review):
                    self.worker_errors.pop(worker_id, None)
                    await asyncio.sleep(0)
                    continue
            except Exception:
                self.last_worker_error = "processing_failed"
                self.worker_errors[worker_id] = "processing_failed"
            await asyncio.sleep(self.config.poll_seconds)

    async def process_one(
        self,
        project: str,
        semantic: bool = False,
        *,
        review: bool = False,
    ) -> bool:
        """Process at most one durable job; useful for deterministic integration checks."""
        job = self.store.claim_job(project, semantic, review=review)
        if not job:
            return False
        captured = self.store.captured(project, job["event"])
        settings = ProjectConfig.model_validate_json(job["config"])
        policy: ThreatPolicy | ToolUsagePolicy = settings.policy
        if job["detector"].startswith("tool_usage:"):
            policy = next(
                p
                for p in settings.usage_policies
                if p.policy_id == job["detector"].removeprefix("tool_usage:")
            )
        try:
            if isinstance(policy, ToolUsagePolicy):
                findings = [self.store.tool_usage(captured, policy)]
            elif job["detector"] == "tool_policy":
                findings = tool_policy(captured, settings.policy)
            elif job["detector"] == "sensitive_data":
                findings = sensitive_data(captured, policy)
            else:
                findings = [await self._classify(captured, settings, review=review)]
        except Exception:
            findings = [
                result(
                    captured,
                    policy,
                    job["detector"],
                    outcome="error",
                    reason="detector failed; coverage degraded",
                    detector_version=(
                        settings.classifier.version
                        if job["detector"] == "prompt_injection" and settings.classifier
                        else "1"
                    ),
                )
            ]
        follow_up = False
        if job["detector"] == "prompt_injection" and settings.review:
            finding = findings[0]
            if finding.evaluation:
                decision = select_review(captured.event, finding, settings.review)
                findings = [
                    finding.model_copy(
                        update={
                            "evaluation": finding.evaluation.model_copy(
                                update={"decision": decision}
                            ),
                        }
                    )
                ]
                follow_up = decision in {"uncertain_screen", "audit_sample"}
        # An external adapter must never persist echoed protected values.
        redactor = Redactor(settings.policy)
        safe = [
            finding.model_copy(
                update={
                    "reason": redactor.text(finding.reason)[0],
                    "evidence": [redactor.text(evidence)[0] for evidence in finding.evidence],
                }
            )
            for finding in findings
        ]
        self.store.finish_job(
            job,
            safe,
            captured,
            policy,
            self.config.destinations,
            follow_up=follow_up,
        )
        return True

    async def _classify(
        self,
        captured: CapturedEvent,
        project: ProjectConfig,
        *,
        review: bool = False,
    ) -> DetectionResult:
        event = captured.event
        applicable = event.event_type in {"message.received", "context.received", "tool.completed"}
        settings = project.review.evaluator if review and project.review else project.classifier
        started = time.perf_counter()

        def finish(
            finding: DetectionResult,
            verdict: ClassifierVerdict | None = None,
            *,
            called: bool = False,
        ) -> DetectionResult:
            trace = None
            if project.review and project.classifier:
                trace = EvaluationTrace(
                    stage="review" if review else "screen",
                    decision=self.store.review_origin(event.project_id, event.event_id)
                    if review
                    else "pending_selection",
                    screen_version=project.classifier.version,
                    review_version=project.review.evaluator.version,
                    gate_version=project.review.version,
                    provider_called=called,
                    elapsed_ms=(time.perf_counter() - started) * 1000 if called else 0,
                    reported_cost_usd=verdict.reported_cost_usd if verdict else None,
                    input_tokens=verdict.input_tokens if verdict else None,
                    output_tokens=verdict.output_tokens if verdict else None,
                )
            return finding.model_copy(
                update={
                    "detector": "prompt_injection_review" if review else "prompt_injection",
                    "detector_version": settings.version if settings else "1",
                    "evaluation": trace,
                }
            )

        if not applicable or settings is None:
            return finish(
                result(
                    captured,
                    project.policy,
                    "prompt_injection",
                    outcome="skipped",
                    reason="not applicable" if not applicable else "classifier not configured",
                    detector_version=settings.version if settings else "1",
                )
            )
        if not self.store.use_budget(project.project_id, settings.calls_per_minute, review=review):
            return finish(
                result(
                    captured,
                    project.policy,
                    "prompt_injection",
                    outcome="skipped",
                    reason="classifier budget exhausted; coverage degraded",
                    detector_version=settings.version,
                )
            )
        context, incomplete = self.store.context(captured, settings.context_events)
        # Historical events may predate newly configured protected patterns.
        # Apply this job's policy again before sending any stored context externally.
        redactor = Redactor(project.policy)
        context = [
            item.model_copy(update={"event": redactor.capture(item.event).event})
            for item in context
        ]
        adapters = self.reviewers if review else self.classifiers
        classifier = adapters.get(project.project_id) or (
            LakeraInjectionClassifier(settings)
            if settings.provider == "lakera"
            else HTTPInjectionClassifier(settings)
        )
        verdict = None
        try:
            verdict = await asyncio.wait_for(
                classifier.classify(captured, context, incomplete),
                timeout=settings.timeout_seconds,
            )
            outcome = verdict.outcome
            incomplete = incomplete or verdict.context_incomplete
            # A negative verdict on incomplete context cannot certify the entire interaction.
            if incomplete and outcome == "clear":
                outcome = "insufficient_context"
            if review and outcome == "needs_review":
                outcome = "insufficient_context"
            finding = result(
                captured,
                project.policy,
                "prompt_injection",
                outcome=outcome,
                category="prompt_injection",
                reason=verdict.reason,
                evidence=["classifier examined captured untrusted content"],
            )
        except Exception:
            finding = result(
                captured,
                project.policy,
                "prompt_injection",
                outcome="error",
                category="prompt_injection",
                reason="classifier failed; coverage degraded",
            )
        return finish(
            finding.model_copy(update={"context_incomplete": incomplete}),
            verdict,
            called=True,
        )

    async def _deliveries(self, destination: str) -> None:
        worker_id = f"deliveries:{destination}"
        while True:
            try:
                if await self.publish_one(destination):
                    self.worker_errors.pop(worker_id, None)
                    await asyncio.sleep(0)
                    continue
            except Exception:
                self.last_worker_error = "delivery_worker_failed"
                self.worker_errors[worker_id] = "delivery_worker_failed"
            await asyncio.sleep(self.config.poll_seconds)

    async def publish_one(self, destination: str) -> bool:
        """Publish one saved outbox payload, retaining canonical IDs on every retry."""
        delivery = self.store.claim_delivery(destination)
        if not delivery:
            return False
        config = DESTINATION_ADAPTER.validate_json(delivery["config"])
        if delivery["attempts"] > config.max_attempts:
            self.store.finish_delivery(delivery, error="retry_budget_exhausted")
            return True
        error: str | None = None
        retryable = True
        try:
            key = json.dumps([config.destination_id, config.version])
            if key not in self._publishers:
                self._publishers[key] = self.publisher_factory(config)
            publisher = self._publishers[key]
            await asyncio.wait_for(
                publisher.publish(delivery["content"].encode(), delivery["id"]),
                timeout=config.timeout_seconds + 1,
            )
        except PublishError as failure:
            error, retryable = failure.code, failure.retryable
        except (ImportError, ValueError):
            error, retryable = "publisher_configuration_error", False
        except Exception:
            error = "publish_failed"
        retry = error is not None and retryable and delivery["attempts"] < config.max_attempts
        self.store.finish_delivery(
            delivery,
            error=error,
            retry=retry,
            delay=random.uniform(0.5, min(60, 2 ** delivery["attempts"])),
        )
        return True

    async def _cleanup(self) -> None:
        while True:
            try:
                self.store.cleanup(self.config.retention_days)
                self.last_cleanup = time.time()
                self.worker_errors.pop("retention", None)
            except Exception:
                self.last_worker_error = "retention_failed"
                self.worker_errors["retention"] = "retention_failed"
            await asyncio.sleep(30)
