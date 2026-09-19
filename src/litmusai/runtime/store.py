"""Transactional SQLite event/jobs and per-destination delivery outbox."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

from litmusai.runtime.config import (
    ConversationPolicy,
    Destination,
    ProjectConfig,
    RuntimeConfig,
    ThreatPolicy,
    ToolUsagePolicy,
)
from litmusai.runtime.models import (
    CapturedEvent,
    CloudEvent,
    DetectionResult,
    ThreatAlert,
    ToolActivity,
    ToolUsageEvidence,
    new_id,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY);
INSERT OR IGNORE INTO schema_version VALUES (2);
CREATE TABLE IF NOT EXISTS config_versions (
  kind TEXT, project TEXT, name TEXT, version TEXT, content TEXT,
  PRIMARY KEY(kind, project, name, version));
CREATE TABLE IF NOT EXISTS events (
  project TEXT, id TEXT, agent TEXT, deployment TEXT, session TEXT, producer TEXT,
  sequence INTEGER, received REAL, content TEXT,
  PRIMARY KEY(project,id), UNIQUE(project,producer,sequence));
CREATE INDEX IF NOT EXISTS event_context ON events(project,agent,deployment,session,received);
CREATE TABLE IF NOT EXISTS tool_calls (
  project TEXT, agent TEXT, deployment TEXT, session TEXT, call_id TEXT,
  event TEXT, name TEXT, received REAL, position INTEGER,
  PRIMARY KEY(project,agent,deployment,session,call_id));
CREATE INDEX IF NOT EXISTS tool_call_window
  ON tool_calls(project,agent,deployment,session,received);
CREATE UNIQUE INDEX IF NOT EXISTS tool_call_event ON tool_calls(project,event);
CREATE INDEX IF NOT EXISTS tool_call_order
  ON tool_calls(project,agent,deployment,session,position);
CREATE TABLE IF NOT EXISTS usage_gaps (
  project TEXT, agent TEXT, deployment TEXT, session TEXT, received REAL, position INTEGER);
CREATE INDEX IF NOT EXISTS usage_gap_window
  ON usage_gaps(project,agent,deployment,session,received);
CREATE TABLE IF NOT EXISTS producers (
  project TEXT, id TEXT, sequence INTEGER, gaps INTEGER DEFAULT 0, updated REAL,
  PRIMARY KEY(project,id));
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY, project TEXT, event TEXT, detector TEXT, config TEXT,
  state TEXT DEFAULT 'pending', lease TEXT, lease_until REAL DEFAULT 0,
  attempts INTEGER DEFAULT 0, created REAL,
  UNIQUE(project,event,detector));
CREATE INDEX IF NOT EXISTS jobs_pending ON jobs(project,detector,state,lease_until);
CREATE TABLE IF NOT EXISTS findings (
  id TEXT PRIMARY KEY, project TEXT, job TEXT, content TEXT, suppressed INTEGER,
  created REAL, detector TEXT, outcome TEXT);
CREATE INDEX IF NOT EXISTS findings_outcomes ON findings(project,detector,outcome);
CREATE TABLE IF NOT EXISTS alerts (
  project TEXT, id TEXT, episode TEXT, revision INTEGER, content TEXT, updated REAL,
  PRIMARY KEY(project,id), UNIQUE(project,episode));
CREATE TABLE IF NOT EXISTS revisions (
  event_id TEXT PRIMARY KEY, project TEXT, alert TEXT, revision INTEGER,
  content TEXT, created REAL, received REAL);
CREATE TABLE IF NOT EXISTS deliveries (
  id TEXT PRIMARY KEY, project TEXT, destination TEXT, config TEXT, event TEXT,
  state TEXT DEFAULT 'pending', attempts INTEGER DEFAULT 0, next_attempt REAL DEFAULT 0,
  lease TEXT, lease_until REAL DEFAULT 0, last_error TEXT, acknowledged REAL, created REAL,
  UNIQUE(destination,event));
CREATE INDEX IF NOT EXISTS deliveries_pending ON deliveries(destination,state,next_attempt);
CREATE INDEX IF NOT EXISTS deliveries_project ON deliveries(project,state);
CREATE TABLE IF NOT EXISTS delivery_attempts (
  delivery TEXT, attempt INTEGER, outcome TEXT, created REAL);
CREATE TABLE IF NOT EXISTS budgets (
  project TEXT, minute INTEGER, calls INTEGER, PRIMARY KEY(project,minute));
CREATE TABLE IF NOT EXISTS review_budgets (
  project TEXT, minute INTEGER, calls INTEGER, PRIMARY KEY(project,minute));
CREATE TABLE IF NOT EXISTS policy_budgets (
  project TEXT, policy TEXT, minute INTEGER, calls INTEGER,
  PRIMARY KEY(project,policy,minute));
CREATE TABLE IF NOT EXISTS evaluation_metrics (
  project TEXT, stage TEXT, decision TEXT, outcome TEXT, observations INTEGER,
  provider_calls INTEGER, cost_samples INTEGER, reported_cost_usd REAL, elapsed_ms REAL,
  PRIMARY KEY(project,stage,decision,outcome));
CREATE TABLE IF NOT EXISTS metrics (
  project TEXT, name TEXT, value INTEGER, PRIMARY KEY(project,name));
"""


class Store:
    """Single-instance store. Public mutations commit atomically under a process lock."""

    def __init__(self, path: str) -> None:
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False, timeout=5)
        self.db.row_factory = sqlite3.Row
        existing = self.db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_version'",
        ).fetchone()
        if existing:
            version = self.db.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
            if version not in {1, 2}:
                self.db.close()
                raise ValueError("unsupported runtime database version")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript(_SCHEMA)

    def close(self) -> None:
        """Close the store after workers have stopped."""
        with self._lock:
            self.db.close()

    def healthy(self) -> bool:
        """Probe the connection without returning database details to unauthenticated callers."""
        with self._lock:
            try:
                return bool(self.db.execute("SELECT 1").fetchone()[0] == 1)
            except sqlite3.Error:
                return False

    def register_config(self, config: RuntimeConfig) -> None:
        """Prevent reusing immutable policy/destination versions with different contents."""
        versions = [
            (
                "policy",
                p.project_id,
                p.policy.policy_id,
                p.policy.version,
                p.policy.model_dump_json(),
            )
            for p in config.projects
        ] + [
            ("destination", d.project_id, d.destination_id, d.version, d.model_dump_json())
            for d in config.destinations
        ]
        versions.extend(
            (
                "conversation_policy",
                p.project_id,
                rule.policy_id,
                rule.version,
                rule.model_dump_json(),
            )
            for p in config.projects
            for rule in p.conversation_policies
        )
        versions.extend(
            ("usage_policy", p.project_id, rule.policy_id, rule.version, rule.model_dump_json())
            for p in config.projects
            for rule in p.usage_policies
        )
        versions.extend(
            (
                "review",
                p.project_id,
                "prompt_injection",
                p.review.version,
                p.review.model_dump_json(),
            )
            for p in config.projects
            if p.review is not None
        )
        versions.extend(
            (
                "classifier",
                p.project_id,
                "prompt_injection",
                p.classifier.version,
                p.classifier.model_dump_json(),
            )
            for p in config.projects
            if p.classifier is not None
        )
        with self._lock, self.db:
            for kind, project, name, version, content in versions:
                previous = self.db.execute(
                    "SELECT content FROM config_versions WHERE kind=? AND project=? AND name=? "
                    "AND version=?",
                    (kind, project, name, version),
                ).fetchone()
                if previous and previous[0] != content:
                    raise ValueError("configuration changed without incrementing its version")
                self.db.execute(
                    "INSERT OR IGNORE INTO config_versions VALUES (?,?,?,?,?)",
                    (kind, project, name, version, content),
                )

    def accept(self, captured: CapturedEvent, project: ProjectConfig) -> str:
        """Durably accept a sanitized event and jobs; return a safe per-event status."""
        event = captured.event
        if event.project_id != project.project_id:
            raise ValueError("event project differs from authenticated project")
        now = captured.received_at.timestamp()
        with self._lock, self.db:
            previous = self.db.execute(
                "SELECT content FROM events WHERE project=? AND id=?",
                (project.project_id, event.event_id),
            ).fetchone()
            if previous:
                old = CapturedEvent.model_validate_json(previous[0])
                return (
                    "duplicate"
                    if old.event == event and old.signals == captured.signals
                    else "conflict"
                )
            pending = self.db.execute(
                "SELECT COUNT(DISTINCT event) FROM jobs WHERE project=? AND state!='done'",
                (project.project_id,),
            ).fetchone()[0]
            if pending >= project.max_pending_events:
                return "busy"
            stored = self.db.execute(
                "SELECT COUNT(*) FROM events WHERE project=?",
                (project.project_id,),
            ).fetchone()[0]
            deliveries = self.db.execute(
                "SELECT COUNT(*) FROM deliveries WHERE project=? AND state!='acknowledged'",
                (project.project_id,),
            ).fetchone()[0]
            if stored >= project.max_stored_events or deliveries >= project.max_pending_deliveries:
                return "busy"
            producer = self.db.execute(
                "SELECT sequence FROM producers WHERE project=? AND id=?",
                (project.project_id, event.producer_id),
            ).fetchone()
            gap = event.sequence != (producer[0] + 1 if producer else 1)
            captured = captured.model_copy(update={"capture_gap": gap})
            try:
                inserted = self.db.execute(
                    "INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        project.project_id,
                        event.event_id,
                        event.agent_id,
                        event.deployment_id,
                        event.session_id,
                        event.producer_id,
                        event.sequence,
                        now,
                        captured.model_dump_json(),
                    ),
                )
            except sqlite3.IntegrityError:
                return "conflict"
            scope = (event.project_id, event.agent_id, event.deployment_id, event.session_id)
            if gap:
                self.db.execute(
                    "INSERT INTO usage_gaps VALUES (?,?,?,?,?,?)",
                    (*scope, now, inserted.lastrowid),
                )
            detectors = ["tool_policy", "sensitive_data", "prompt_injection"]
            detectors.extend(
                "conversation_policy:" + rule.policy_id
                for rule in project.conversation_policies
                if rule.applies(event)
            )
            if event.event_type == "tool.requested" and isinstance(event.payload, ToolActivity):
                added = self.db.execute(
                    "INSERT OR IGNORE INTO tool_calls VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        *scope,
                        event.tool_call_id,
                        event.event_id,
                        event.payload.name,
                        now,
                        inserted.lastrowid,
                    ),
                ).rowcount
                if added:
                    detectors.extend(
                        "tool_usage:" + rule.policy_id
                        for rule in project.usage_policies
                        if rule.applies(event)
                    )
            self.db.execute(
                "INSERT INTO producers VALUES (?,?,?,?,?) ON CONFLICT(project,id) DO UPDATE SET "
                "sequence=MAX(sequence,excluded.sequence), gaps=gaps+excluded.gaps, "
                "updated=excluded.updated",
                (project.project_id, event.producer_id, event.sequence, int(gap), now),
            )
            for detector in detectors:
                snapshot = project.model_copy(
                    update={
                        "usage_policies": [
                            p
                            for p in project.usage_policies
                            if detector == "tool_usage:" + p.policy_id
                        ],
                        "conversation_policies": [
                            p
                            for p in project.conversation_policies
                            if detector == "conversation_policy:" + p.policy_id
                        ],
                        "classifier": project.classifier
                        if detector == "prompt_injection"
                        else None,
                        "review": project.review if detector == "prompt_injection" else None,
                    }
                )
                self.db.execute(
                    "INSERT INTO jobs (id,project,event,detector,config,created) "
                    "VALUES (?,?,?,?,?,?)",
                    (
                        new_id(),
                        project.project_id,
                        event.event_id,
                        detector,
                        snapshot.model_dump_json(),
                        now,
                    ),
                )
            return "accepted"

    def claim_job(
        self,
        project: str,
        semantic: bool,
        *,
        review: bool = False,
        policies: bool = False,
    ) -> dict[str, Any] | None:
        """Lease one job; expired leases are recovered without rewriting previous findings."""
        now = time.time()
        lane = (
            "detector LIKE 'conversation_policy:%'"
            if policies
            else "detector='prompt_injection_review'"
            if review
            else "detector='prompt_injection'"
            if semantic
            else "detector NOT IN ('prompt_injection','prompt_injection_review') "
            "AND detector NOT LIKE 'conversation_policy:%'"
        )
        with self._lock, self.db:
            row = self.db.execute(
                "SELECT * FROM jobs WHERE project=? AND " + lane + " "
                "AND (state='pending' OR (state='leased' AND lease_until<?)) "
                "ORDER BY created,id LIMIT 1",
                (project, now),
            ).fetchone()
            if row is None:
                return None
            lease = new_id()
            self.db.execute(
                "UPDATE jobs SET state='leased',lease=?,lease_until=?,attempts=attempts+1 "
                "WHERE id=?",
                (lease, now + 60, row["id"]),
            )
            return {**dict(row), "lease": lease, "attempts": row["attempts"] + 1}

    def captured(self, project: str, event_id: str) -> CapturedEvent:
        """Read a sanitized event within the authenticated project."""
        with self._lock:
            row = self.db.execute(
                "SELECT content FROM events WHERE project=? AND id=?", (project, event_id)
            ).fetchone()
        if row is None:
            raise KeyError("event unavailable")
        return CapturedEvent.model_validate_json(row[0])

    def context(self, captured: CapturedEvent, limit: int) -> tuple[list[CapturedEvent], bool]:
        """Read only bounded prior-arriving context from the same agent/deployment/session."""
        event = captured.event
        with self._lock:
            rows = self.db.execute(
                "SELECT content FROM events WHERE project=? AND agent=? AND deployment=? "
                "AND session=? AND received<=? AND id!=? ORDER BY received DESC LIMIT ?",
                (
                    event.project_id,
                    event.agent_id,
                    event.deployment_id,
                    event.session_id,
                    captured.received_at.timestamp(),
                    event.event_id,
                    limit + 1,
                ),
            ).fetchall()
        context = [CapturedEvent.model_validate_json(r[0]) for r in reversed(rows[:limit])]
        incomplete = (
            len(rows) > limit or captured.capture_gap or any(c.capture_gap for c in context)
        )
        return context, incomplete

    def use_budget(self, project: str, maximum: int, *, review: bool = False) -> bool:
        """Reserve a persisted per-project classifier call within a UTC minute."""
        minute = int(time.time() // 60)
        table = "review_budgets" if review else "budgets"
        with self._lock, self.db:
            row = self.db.execute(
                f"SELECT calls FROM {table} WHERE project=? AND minute=?", (project, minute)
            ).fetchone()
            if row and row[0] >= maximum:
                return False
            self.db.execute(
                f"INSERT INTO {table} VALUES (?,?,1) ON CONFLICT(project,minute) "
                "DO UPDATE SET calls=calls+1",
                (project, minute),
            )
        return True

    def review_origin(self, project: str, event: str) -> str:
        """Read the selection committed atomically with the pending review job."""
        with self._lock:
            row = self.db.execute(
                "SELECT f.content FROM findings f JOIN jobs j ON f.job=j.id "
                "WHERE j.project=? AND j.event=? AND j.detector='prompt_injection' LIMIT 1",
                (project, event),
            ).fetchone()
        if row is None:
            raise ValueError("review selection unavailable")
        trace = DetectionResult.model_validate_json(row[0]).evaluation
        if trace is None or trace.decision not in {"uncertain_screen", "audit_sample"}:
            raise ValueError("event was not selected for review")
        return trace.decision

    def use_policy_budget(self, project: str, policy: str, maximum: int) -> bool:
        """Keep each rubric's provider allowance isolated and durable across restarts."""
        minute = int(time.time() // 60)
        with self._lock, self.db:
            row = self.db.execute(
                "SELECT calls FROM policy_budgets WHERE project=? AND policy=? AND minute=?",
                (project, policy, minute),
            ).fetchone()
            if row and row[0] >= maximum:
                return False
            self.db.execute(
                "INSERT INTO policy_budgets VALUES (?,?,?,1) "
                "ON CONFLICT(project,policy,minute) DO UPDATE SET calls=calls+1",
                (project, policy, minute),
            )
        return True

    def tool_usage(self, captured: CapturedEvent, policy: ToolUsagePolicy) -> DetectionResult:
        """Evaluate a receipt-time window from durable, deduplicated request observations."""
        event = captured.event
        scope = (event.project_id, event.agent_id, event.deployment_id, event.session_id)
        with self._lock:
            current = self.db.execute(
                "SELECT received,position FROM tool_calls WHERE project=? AND agent=? "
                "AND deployment=? AND session=? AND event=?",
                (*scope, event.event_id),
            ).fetchone()
            if current is None:
                raise ValueError("tool request observation unavailable")
            end, position = current
            start = end - policy.window_seconds
            where = (
                "project=? AND agent=? AND deployment=? AND session=? "
                "AND received>? AND received<=? AND position<=?"
            )
            params: tuple[Any, ...] = (*scope, start, end, position)
            incomplete = bool(
                self.db.execute(
                    "SELECT 1 FROM usage_gaps WHERE " + where + " LIMIT 1",
                    params,
                ).fetchone()
            )
            if policy.tools:
                where += " AND name IN (" + ",".join("?" for _ in policy.tools) + ")"
                params += tuple(policy.tools)
            count = self.db.execute(
                "SELECT COUNT(*) FROM tool_calls WHERE " + where,
                params,
            ).fetchone()[0]
            rows = self.db.execute(
                "SELECT event FROM tool_calls WHERE " + where + " ORDER BY position DESC LIMIT 50",
                params,
            ).fetchall()
            crossed = self._usage_crossed(scope, position, count, policy)
        exceeded = count > policy.max_calls
        usage = ToolUsageEvidence(
            observed_count=count,
            limit=policy.max_calls,
            window_seconds=policy.window_seconds,
            window_start=datetime.fromtimestamp(start, timezone.utc),
            window_end=datetime.fromtimestamp(end, timezone.utc),
            threshold_crossed=crossed,
            source_events_truncated=count > len(rows),
        )
        return DetectionResult(
            event_id=event.event_id,
            detector="tool_usage",
            policy_id=policy.policy_id,
            policy_version=policy.version,
            category="excessive_tool_usage",
            severity=policy.severity,
            stage="requested",
            outcome="detected" if exceeded else "insufficient_context" if incomplete else "clear",
            reason="conversation tool-call limit exceeded"
            if exceeded
            else "capture gaps; observed count is a lower bound"
            if incomplete
            else "observed tool-call count is within the configured limit",
            evidence=[
                f"distinct tool requests={count}; limit={policy.max_calls}; "
                f"window_seconds={policy.window_seconds}; basis=collector_received_at"
            ],
            source_event_ids=[row[0] for row in reversed(rows)],
            context_incomplete=incomplete,
            usage=usage,
        )

    def _usage_crossed(
        self,
        scope: tuple[str, str, str, str],
        position: int,
        count: int,
        policy: ToolUsagePolicy,
    ) -> bool:
        """Also notify when enabling/lowering a policy over an already busy conversation."""
        if count <= policy.max_calls:
            return False
        if count == policy.max_calls + 1:
            return True
        tool_filter = ""
        previous_params: tuple[Any, ...] = ("tool_usage:" + policy.policy_id, *scope, position)
        if policy.tools:
            tool_filter = " AND c.name IN (" + ",".join("?" for _ in policy.tools) + ")"
            previous_params += tuple(policy.tools)
        previous = self.db.execute(
            "SELECT c.received,c.position,j.config FROM tool_calls c LEFT JOIN jobs j "
            "ON c.project=j.project AND c.event=j.event AND j.detector=? "
            "WHERE c.project=? AND c.agent=? AND c.deployment=? AND c.session=? "
            "AND c.position<?" + tool_filter + " ORDER BY c.position DESC LIMIT 1",
            previous_params,
        ).fetchone()
        if previous is None or previous["config"] is None:
            return True
        settings = ProjectConfig.model_validate_json(previous["config"])
        if not any(
            p.policy_id == policy.policy_id and p.version == policy.version
            for p in settings.usage_policies
        ):
            return True
        where = (
            "project=? AND agent=? AND deployment=? AND session=? "
            "AND received>? AND received<=? AND position<=?"
        )
        params: tuple[Any, ...] = (
            *scope,
            previous["received"] - policy.window_seconds,
            previous["received"],
            previous["position"],
        )
        if policy.tools:
            where += " AND name IN (" + ",".join("?" for _ in policy.tools) + ")"
            params += tuple(policy.tools)
        prior_count = self.db.execute(
            "SELECT COUNT(*) FROM tool_calls WHERE " + where,
            params,
        ).fetchone()[0]
        return bool(prior_count <= policy.max_calls)

    def finish_job(
        self,
        job: dict[str, Any],
        results: list[DetectionResult],
        captured: CapturedEvent,
        policy: ThreatPolicy | ToolUsagePolicy | ConversationPolicy,
        destinations: list[Destination],
        *,
        follow_up: bool = False,
    ) -> bool:
        """Commit findings, alert revisions and matching destination outbox entries together."""
        now = time.time()
        with self._lock, self.db:
            owned = self.db.execute(
                "SELECT 1 FROM jobs WHERE id=? AND state='leased' AND lease=?",
                (job["id"], job["lease"]),
            ).fetchone()
            if not owned:
                return False
            for finding in results:
                suppressed = False
                if finding.outcome == "detected" and finding.category:
                    suppressed = self._alert(captured, finding, policy, destinations, now)
                self.db.execute(
                    "INSERT INTO findings VALUES (?,?,?,?,?,?,?,?)",
                    (
                        new_id(),
                        job["project"],
                        job["id"],
                        finding.model_dump_json(
                            exclude={
                                name
                                for name in ("usage", "evaluation", "risk_score")
                                if getattr(finding, name) is None
                            },
                        ),
                        int(suppressed),
                        now,
                        finding.detector,
                        finding.outcome,
                    ),
                )
                if finding.evaluation:
                    trace = finding.evaluation
                    self.db.execute(
                        "INSERT INTO evaluation_metrics VALUES (?,?,?,?,?,?,?,?,?) "
                        "ON CONFLICT(project,stage,decision,outcome) DO UPDATE SET "
                        "observations=observations+1,provider_calls=provider_calls+excluded.provider_calls,"
                        "cost_samples=cost_samples+excluded.cost_samples,"
                        "reported_cost_usd=reported_cost_usd+excluded.reported_cost_usd,"
                        "elapsed_ms=elapsed_ms+excluded.elapsed_ms",
                        (
                            job["project"],
                            trace.stage,
                            trace.decision,
                            finding.outcome,
                            1,
                            int(trace.provider_called),
                            int(trace.reported_cost_usd is not None),
                            trace.reported_cost_usd or 0,
                            trace.elapsed_ms,
                        ),
                    )
            if follow_up:
                self.db.execute(
                    "INSERT INTO jobs (id,project,event,detector,config,created) "
                    "VALUES (?,?,?,?,?,?)",
                    (
                        new_id(),
                        job["project"],
                        job["event"],
                        "prompt_injection_review",
                        job["config"],
                        job["created"],
                    ),
                )
            self.db.execute(
                "UPDATE jobs SET state='done',lease=NULL,lease_until=0 WHERE id=?", (job["id"],)
            )
        return True

    def _alert(
        self,
        captured: CapturedEvent,
        finding: DetectionResult,
        policy: ThreatPolicy | ToolUsagePolicy | ConversationPolicy,
        destinations: list[Destination],
        now: float,
    ) -> bool:
        assert finding.category is not None
        if finding.usage and not finding.usage.threshold_crossed:
            return True
        event = captured.event
        episode = json.dumps(
            [
                event.agent_id,
                event.deployment_id,
                event.session_id,
                policy.policy_id,
                policy.version,
                finding.detector,
                finding.category,
                event.tool_call_id or event.parent_event_id or event.event_id,
            ]
        )
        episode = hashlib.sha256(episode.encode()).hexdigest()
        row = self.db.execute(
            "SELECT * FROM alerts WHERE project=? AND episode=?", (event.project_id, episode)
        ).fetchone()
        previous = ThreatAlert.model_validate_json(row["content"]) if row else None
        if (
            previous
            and now - row["updated"] < policy.cooldown_seconds
            and previous.stage == finding.stage
            and previous.evidence == finding.evidence
            and previous.severity == finding.severity
        ):
            return True
        # Late requested events must not downgrade an already observed activity stage.
        rank = {"attempt": 0, "requested": 1, "observed": 2}
        stage = (
            previous.stage
            if previous and rank[previous.stage] > rank[finding.stage]
            else finding.stage
        )
        source_ids = list(
            dict.fromkeys(
                (previous.source_event_ids if previous else []) + finding.source_event_ids,
            )
        )[-50:]
        alert = ThreatAlert(
            schema_version="1.3"
            if finding.evaluation and finding.evaluation.stage == "policy"
            else "1.2"
            if finding.evaluation
            else "1.1"
            if finding.usage
            else "1.0",
            alert_id=previous.alert_id if previous else new_id(),
            revision=previous.revision + 1 if previous else 1,
            project_id=event.project_id,
            agent_id=event.agent_id,
            deployment_id=event.deployment_id,
            session_id=event.session_id,
            actor_id=event.actor_id,
            category=finding.category,
            severity=finding.severity,
            stage=stage,
            evidence=finding.evidence,
            reason=finding.reason,
            source_event_ids=source_ids,
            policy_id=policy.policy_id,
            policy_version=policy.version,
            detector=finding.detector,
            detector_version=finding.detector_version,
            observed_at=event.observed_at,
            context_incomplete=finding.context_incomplete,
            usage=finding.usage,
            evaluation=finding.evaluation,
            risk_score=finding.risk_score,
        )
        envelope = CloudEvent(
            source=f"urn:litmusai:project:{quote(event.project_id, safe='')}",
            subject=f"alerts/{alert.alert_id}",
            type="com.litmusai.threat.updated" if previous else "com.litmusai.threat.detected",
            data=alert,
        )
        self.db.execute(
            "INSERT INTO alerts VALUES (?,?,?,?,?,?) ON CONFLICT(project,id) DO UPDATE SET "
            "revision=excluded.revision,content=excluded.content,updated=excluded.updated",
            (
                event.project_id,
                alert.alert_id,
                episode,
                alert.revision,
                alert.model_dump_json(
                    exclude={
                        name
                        for name in ("usage", "evaluation", "risk_score")
                        if getattr(alert, name) is None
                    }
                ),
                now,
            ),
        )
        self.db.execute(
            "INSERT INTO revisions VALUES (?,?,?,?,?,?,?)",
            (
                envelope.id,
                event.project_id,
                alert.alert_id,
                alert.revision,
                envelope.model_dump_json(
                    exclude={
                        "data": {
                            name
                            for name in ("usage", "evaluation", "risk_score")
                            if getattr(finding, name) is None
                        }
                    },
                ),
                now,
                captured.received_at.timestamp(),
            ),
        )
        for destination in destinations:
            if destination.project_id != event.project_id:
                continue
            if any(
                choices and value not in choices
                for choices, value in (
                    (destination.categories, finding.category),
                    (destination.severities, finding.severity),
                    (destination.agents, event.agent_id),
                    (destination.deployments, event.deployment_id),
                )
            ):
                continue
            self.db.execute(
                "INSERT INTO deliveries (id,project,destination,config,event,created) "
                "VALUES (?,?,?,?,?,?)",
                (
                    new_id(),
                    event.project_id,
                    destination.destination_id,
                    destination.model_dump_json(),
                    envelope.id,
                    now,
                ),
            )
        return False

    def claim_delivery(self, destination: str) -> dict[str, Any] | None:
        """Lease one destination-specific delivery, including interrupted attempts."""
        now = time.time()
        with self._lock, self.db:
            row = self.db.execute(
                "SELECT d.*,r.content FROM deliveries d JOIN revisions r ON r.event_id=d.event "
                "WHERE destination=? AND ((state='pending' AND next_attempt<=?) OR "
                "(state='leased' AND lease_until<?)) ORDER BY d.created,d.id LIMIT 1",
                (destination, now, now),
            ).fetchone()
            if row is None:
                return None
            lease = new_id()
            self.db.execute(
                "UPDATE deliveries SET state='leased',lease=?,lease_until=?,"
                "attempts=attempts+1 WHERE id=?",
                (lease, now + 60, row["id"]),
            )
            return {**dict(row), "lease": lease, "attempts": row["attempts"] + 1}

    def finish_delivery(
        self,
        delivery: dict[str, Any],
        *,
        error: str | None = None,
        retry: bool = False,
        delay: float = 0,
    ) -> None:
        """Persist an acknowledgement or a bounded safe failure code (never exception text)."""
        now = time.time()
        state = "pending" if retry else "failed" if error else "acknowledged"
        with self._lock, self.db:
            changed = self.db.execute(
                "UPDATE deliveries SET state=?,last_error=?,next_attempt=?,"
                "lease=NULL,lease_until=0,"
                "acknowledged=? WHERE id=? AND state='leased' AND lease=?",
                (
                    state,
                    error,
                    now + delay,
                    None if error else now,
                    delivery["id"],
                    delivery["lease"],
                ),
            ).rowcount
            if changed:
                self.db.execute(
                    "INSERT INTO delivery_attempts VALUES (?,?,?,?)",
                    (delivery["id"], delivery["attempts"], error or state, now),
                )

    def destination_ids(self) -> list[str]:
        """Include pending snapshots even if removed from current configuration."""
        with self._lock:
            return [
                r[0]
                for r in self.db.execute(
                    "SELECT DISTINCT destination FROM deliveries "
                    "WHERE state IN ('pending','leased','failed')",
                )
            ]

    def alerts(self, project: str, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        """List latest redacted alert revisions within one project."""
        with self._lock:
            return [
                json.loads(r[0])
                for r in self.db.execute(
                    "SELECT content FROM alerts WHERE project=? "
                    "ORDER BY updated DESC LIMIT ? OFFSET ?",
                    (project, limit, offset),
                )
            ]

    def alert(self, project: str, alert_id: str) -> dict[str, Any] | None:
        """Read an alert with bounded publication state and attempt counts."""
        with self._lock:
            row = self.db.execute(
                "SELECT content FROM alerts WHERE project=? AND id=?", (project, alert_id)
            ).fetchone()
            if not row:
                return None
            deliveries = self.db.execute(
                "SELECT d.id,d.destination,d.event,d.state,d.attempts,d.last_error,d.acknowledged "
                "FROM deliveries d JOIN revisions r ON r.event_id=d.event "
                "WHERE d.project=? AND r.alert=? ORDER BY d.created DESC LIMIT 200",
                (project, alert_id),
            ).fetchall()
            return {"alert": json.loads(row[0]), "deliveries": [dict(r) for r in deliveries]}

    def findings(self, project: str, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        """Inspect missing context, errors, and notification suppression within a project."""
        with self._lock:
            return [
                {"finding": json.loads(row[0]), "suppressed": bool(row[1])}
                for row in self.db.execute(
                    "SELECT content,suppressed FROM findings WHERE project=? "
                    "ORDER BY created DESC LIMIT ? OFFSET ?",
                    (project, limit, offset),
                )
            ]

    def replay(self, project: str, delivery_id: str) -> bool:
        """Explicitly replay a failed delivery using its original destination and event IDs."""
        with self._lock, self.db:
            return bool(
                self.db.execute(
                    "UPDATE deliveries SET state='pending',next_attempt=0,"
                    "attempts=0,last_error=NULL "
                    "WHERE project=? AND id=? AND state='failed'",
                    (project, delivery_id),
                ).rowcount
            )

    def status(self, project: str) -> dict[str, Any]:
        """Report project coverage, queue lag, capture gaps, and publication states."""
        with self._lock:
            evaluations = [
                dict(row)
                for row in self.db.execute(
                    "SELECT stage,decision,outcome,observations,provider_calls,cost_samples,"
                    "reported_cost_usd,elapsed_ms FROM evaluation_metrics WHERE project=?",
                    (project,),
                )
            ]
            jobs = {
                r[0]: r[1]
                for r in self.db.execute(
                    "SELECT state,COUNT(*) FROM jobs WHERE project=? GROUP BY state",
                    (project,),
                )
            }
            destinations = [
                dict(r)
                for r in self.db.execute(
                    "SELECT destination,state,COUNT(*) AS count FROM deliveries WHERE project=? "
                    "GROUP BY destination,state",
                    (project,),
                )
            ]
            outcomes: dict[str, dict[str, int]] = {}
            for row in self.db.execute(
                "SELECT detector,outcome,COUNT(*) FROM findings WHERE project=? "
                "GROUP BY detector,outcome",
                (project,),
            ):
                outcomes.setdefault(row[0], {})[row[1]] = row[2]
            latency: dict[str, list[float]] = {}
            for row in self.db.execute(
                "SELECT d.destination,d.acknowledged-r.received FROM deliveries d "
                "JOIN revisions r ON d.event=r.event_id WHERE d.project=? "
                "AND d.state='acknowledged' ORDER BY d.acknowledged DESC LIMIT 1000",
                (project,),
            ):
                latency.setdefault(row[0], []).append(max(0, row[1]))
            oldest = self.db.execute(
                "SELECT MIN(created) FROM jobs WHERE project=? AND state!='done'",
                (project,),
            ).fetchone()[0]
            gaps = self.db.execute(
                "SELECT COALESCE(SUM(gaps),0) FROM producers WHERE project=?", (project,)
            ).fetchone()[0]
            suppressed = self.db.execute(
                "SELECT COUNT(*) FROM findings WHERE project=? AND suppressed=1",
                (project,),
            ).fetchone()[0]
            expired = {
                row[0]: row[1]
                for row in self.db.execute(
                    "SELECT name,value FROM metrics WHERE project=?",
                    (project,),
                )
            }
        return {
            "evaluation_metrics": evaluations,
            "jobs": jobs,
            "detector_outcomes": outcomes,
            "destinations": destinations,
            "capture_gaps": gaps,
            "suppressed": suppressed,
            "expired_work": expired,
            "oldest_job_seconds": max(0, time.time() - oldest) if oldest else 0,
            "publication_latency": {
                destination: {
                    "samples": len(values),
                    "p95_seconds": sorted(values)[math.ceil(len(values) * 0.95) - 1],
                }
                for destination, values in latency.items()
            },
            "latency_scope": (
                "collector receipt to publish ack; latest 1000 acknowledged deliveries"
            ),
        }

    def cleanup(self, retention_days: int) -> None:
        """Apply bounded retention, including expired unresolved work; expose before expiry."""
        cutoff = time.time() - retention_days * 86400
        with self._lock, self.db:
            for table, completed in (("jobs", "done"), ("deliveries", "acknowledged")):
                rows = self.db.execute(
                    f"SELECT project,COUNT(*) FROM {table} WHERE created<? AND state!=? "
                    "GROUP BY project",
                    (cutoff, completed),
                ).fetchall()
                for project, count in rows:
                    self.db.execute(
                        "INSERT INTO metrics VALUES (?,?,?) ON CONFLICT(project,name) "
                        "DO UPDATE SET value=value+excluded.value",
                        (project, "expired_" + table, count),
                    )
            for table in ("delivery_attempts", "deliveries", "findings", "jobs", "revisions"):
                self.db.execute(f"DELETE FROM {table} WHERE created<?", (cutoff,))
            self.db.execute("DELETE FROM alerts WHERE updated<?", (cutoff,))
            self.db.execute("DELETE FROM events WHERE received<?", (cutoff,))
            self.db.execute("DELETE FROM tool_calls WHERE received<?", (cutoff,))
            self.db.execute("DELETE FROM usage_gaps WHERE received<?", (cutoff,))
            self.db.execute("DELETE FROM producers WHERE updated<?", (cutoff,))
            self.db.execute("DELETE FROM budgets WHERE minute<?", (int(time.time() // 60) - 2,))
            self.db.execute(
                "DELETE FROM policy_budgets WHERE minute<?", (int(time.time() // 60) - 2,)
            )
            self.db.execute(
                "DELETE FROM review_budgets WHERE minute<?", (int(time.time() // 60) - 2,)
            )
