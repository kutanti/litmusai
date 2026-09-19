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
)
from litmusai.runtime.detectors import (
    HTTPInjectionClassifier,
    InjectionClassifier,
    LakeraInjectionClassifier,
    result,
    sensitive_data,
    tool_policy,
)
from litmusai.runtime.models import CapturedEvent, DetectionResult
from litmusai.runtime.publishers import AlertPublisher, PublishError, publisher_for
from litmusai.runtime.redaction import Redactor
from litmusai.runtime.store import Store


class Engine:
    """Run single-instance workers with separate queues for every project and destination."""

    def __init__(
        self,
        store: Store,
        config: RuntimeConfig,
        *,
        classifiers: dict[str, InjectionClassifier] | None = None,
        publisher_factory: Callable[[Destination], AlertPublisher] = publisher_for,
    ) -> None:
        self.store = store
        self.config = config
        self.classifiers = classifiers or {}
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

    async def _jobs(self, project: str, semantic: bool) -> None:
        worker_id = f"jobs:{project}:{semantic}"
        while True:
            try:
                if await self.process_one(project, semantic):
                    self.worker_errors.pop(worker_id, None)
                    await asyncio.sleep(0)
                    continue
            except Exception:
                self.last_worker_error = "processing_failed"
                self.worker_errors[worker_id] = "processing_failed"
            await asyncio.sleep(self.config.poll_seconds)

    async def process_one(self, project: str, semantic: bool = False) -> bool:
        """Process at most one durable job; useful for deterministic integration checks."""
        job = self.store.claim_job(project, semantic)
        if not job:
            return False
        captured = self.store.captured(project, job["event"])
        settings = ProjectConfig.model_validate_json(job["config"])
        policy = settings.policy
        try:
            if job["detector"] == "tool_policy":
                findings = tool_policy(captured, policy)
            elif job["detector"] == "sensitive_data":
                findings = sensitive_data(captured, policy)
            else:
                findings = [await self._classify(captured, settings)]
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
        # An external adapter must never persist echoed protected values.
        redactor = Redactor(policy)
        safe = [
            finding.model_copy(
                update={
                    "reason": redactor.text(finding.reason)[0],
                    "evidence": [redactor.text(evidence)[0] for evidence in finding.evidence],
                }
            )
            for finding in findings
        ]
        self.store.finish_job(job, safe, captured, policy, self.config.destinations)
        return True

    async def _classify(self, captured: CapturedEvent, project: ProjectConfig) -> DetectionResult:
        event = captured.event
        applicable = event.event_type in {"message.received", "context.received", "tool.completed"}
        settings = project.classifier
        if not applicable or settings is None:
            return result(
                captured,
                project.policy,
                "prompt_injection",
                outcome="skipped",
                reason="not applicable" if not applicable else "classifier not configured",
                detector_version=settings.version if settings else "1",
            )
        if not self.store.use_budget(project.project_id, settings.calls_per_minute):
            return result(
                captured,
                project.policy,
                "prompt_injection",
                outcome="skipped",
                reason="classifier budget exhausted; coverage degraded",
                detector_version=settings.version,
            )
        context, incomplete = self.store.context(captured, settings.context_events)
        # Historical events may predate newly configured protected patterns.
        # Apply this job's policy again before sending any stored context externally.
        redactor = Redactor(project.policy)
        context = [
            item.model_copy(update={"event": redactor.capture(item.event).event})
            for item in context
        ]
        classifier = self.classifiers.get(project.project_id) or (
            LakeraInjectionClassifier(settings)
            if settings.provider == "lakera"
            else HTTPInjectionClassifier(settings)
        )
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
        return finding.model_copy(
            update={"context_incomplete": incomplete, "detector_version": settings.version}
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
