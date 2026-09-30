"""Local collector-to-HTTP-receiver experiment; no agents, vendors, or cloud topics invoked.

``--classifier-latency-ms`` injects a fake screen provider that waits for a fixed
time and answers clear, so queue delay under provider latency can be measured
without calling a real vendor. ``--provider-concurrency`` sets the number of
workers per provider lane.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import platform
import tempfile
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from litmusai.runtime.config import (
    ClassifierConfig,
    ProjectConfig,
    RuntimeConfig,
    ThreatPolicy,
    WebhookDestination,
)
from litmusai.runtime.detectors import ClassifierVerdict
from litmusai.runtime.models import CapturedEvent, Message, RuntimeEvent, ToolActivity, new_id
from litmusai.runtime.publishers import verify_webhook
from litmusai.runtime.service import create_app
from litmusai.runtime.store import Store


class FakeProvider:
    """Stand-in screen provider: fixed latency, always clear, records when each call starts."""

    def __init__(self, latency_seconds: float) -> None:
        self.latency_seconds = latency_seconds
        self.started: dict[str, float] = {}

    async def classify(
        self, captured: CapturedEvent, context: list[CapturedEvent], incomplete: bool
    ) -> ClassifierVerdict:
        self.started[captured.event.event_id] = time.time()
        await asyncio.sleep(self.latency_seconds)
        return ClassifierVerdict(outcome="clear", reason="fake provider")


def nearest_rank(values: list[float], fraction: float) -> float | None:
    """Nearest-rank percentile, matching the receipt percentile below."""
    ordered = sorted(values)
    return ordered[math.ceil(len(ordered) * fraction) - 1] if ordered else None


def screen_timings(store: Store, project: str, started: dict[str, float]) -> dict[str, object]:
    """Queue delay is job creation to provider call start; completion adds provider latency."""
    with store._lock:
        rows = store.db.execute(
            "SELECT j.event, j.created, f.created, f.outcome FROM jobs j "
            "JOIN findings f ON f.job = j.id "
            "WHERE j.project=? AND j.detector='prompt_injection'",
            (project,),
        ).fetchall()
        pending = store.db.execute(
            "SELECT COUNT(*) FROM jobs WHERE project=? AND detector='prompt_injection' "
            "AND state!='done'",
            (project,),
        ).fetchone()[0]
    queue = [started[row[0]] - row[1] for row in rows if row[0] in started]
    completion = [row[2] - row[1] for row in rows if row[0] in started]
    outcomes: dict[str, int] = {}
    for row in rows:
        outcomes[row[3]] = outcomes.get(row[3], 0) + 1
    return {
        "provider_calls": len(started),
        "screen_jobs_pending": pending,
        "screen_outcomes": outcomes,
        "queue_delay_p50_seconds": nearest_rank(queue, 0.50),
        "queue_delay_p95_seconds": nearest_rank(queue, 0.95),
        "queue_delay_max_seconds": max(queue) if queue else None,
        "screen_completion_p95_seconds": nearest_rank(completion, 0.95),
    }


def main() -> None:
    """Measure local delivery latency while including unsuccessful events in the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=int, default=200)
    parser.add_argument("--rate", type=float, default=10)
    parser.add_argument("--sessions", type=int, default=20)
    parser.add_argument("--output", default=".litmus/runtime-benchmark.json")
    parser.add_argument(
        "--classifier-latency-ms",
        type=float,
        default=0,
        help="inject a fake screen provider with this latency; 0 disables screening",
    )
    parser.add_argument("--provider-concurrency", type=int, default=1)
    args = parser.parse_args()
    if min(args.events, args.rate, args.sessions) <= 0:
        parser.error("events, rate, and sessions must be positive")
    if not 0 <= args.classifier_latency_ms <= 30000:
        parser.error("classifier latency must be between 0 and 30000 ms")
    if not 1 <= args.provider_concurrency <= 16:
        parser.error("provider concurrency must be between 1 and 16")
    provider = (
        FakeProvider(args.classifier_latency_ms / 1000) if args.classifier_latency_ms else None
    )
    api_key, signing_key = uuid4().hex, uuid4().hex
    os.environ["LITMUS_BENCHMARK_API_KEY"] = api_key
    os.environ["LITMUS_BENCHMARK_SIGNING_KEY"] = signing_key
    receipts: dict[str, float] = {}

    class Receiver(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers["Content-Length"]))
            verified = verify_webhook(
                body,
                signing_key,
                self.headers["X-Litmus-Timestamp"],
                self.headers["X-Litmus-Signature"],
            )
            if verified:
                event = json.loads(body)
                observed = datetime.fromisoformat(
                    event["data"]["observed_at"].replace("Z", "+00:00")
                )
                receipts[event["id"]] = time.time() - observed.timestamp()
            self.send_response(204 if verified else 401)
            self.end_headers()

        def log_message(self, format: str, *values: object) -> None:
            pass

    receiver = ThreadingHTTPServer(("127.0.0.1", 0), Receiver)
    receiver_thread = threading.Thread(target=receiver.serve_forever, daemon=True)
    receiver_thread.start()
    accepted = rejected = expected = 0
    sizes = []
    try:
        with tempfile.TemporaryDirectory(prefix="litmus-runtime-benchmark-") as directory:
            config = RuntimeConfig(
                database=str(Path(directory) / "runtime.sqlite"),
                projects=[
                    ProjectConfig(
                        project_id="benchmark",
                        api_key_env="LITMUS_BENCHMARK_API_KEY",
                        policy=ThreatPolicy(
                            allowed_tools=["send_email"], allowed_destinations=["allowed.example"]
                        ),
                        classifier=ClassifierConfig(
                            endpoint="https://fake-provider.invalid/screen",
                            api_key_env="LITMUS_BENCHMARK_API_KEY",
                            version="fake-provider",
                            timeout_seconds=30,
                            calls_per_minute=10000,
                        )
                        if provider
                        else None,
                    )
                ],
                provider_concurrency=args.provider_concurrency,
                destinations=[
                    WebhookDestination(
                        destination_id="receiver",
                        project_id="benchmark",
                        url=f"http://127.0.0.1:{receiver.server_port}/events",
                        allow_local_http=True,
                        signing_secret_env="LITMUS_BENCHMARK_SIGNING_KEY",
                    )
                ],
            )
            store = Store(config.database)
            try:
                classifiers = {"benchmark": provider} if provider else None
                with TestClient(create_app(config, store=store, classifiers=classifiers)) as api:
                    started = time.monotonic()
                    for index in range(args.events):
                        time.sleep(max(0, started + index / args.rate - time.monotonic()))
                        kind = index % 4
                        payload = (
                            Message(text="benign content " + "x" * 1024)
                            if kind == 0
                            else ToolActivity(
                                name="send_email",
                                destination=("allowed.example" if kind == 1 else "outside.example"),
                                arguments={
                                    "body": "x" * 1024 + ("ghp_" + "A" * 36 if kind == 3 else "")
                                },
                            )
                        )
                        event = RuntimeEvent(
                            project_id="benchmark",
                            agent_id="support",
                            session_id=f"session-{index % args.sessions}",
                            producer_id="benchmark",
                            sequence=index + 1,
                            event_type="message.received" if kind == 0 else "tool.requested",
                            tool_call_id=None if kind == 0 else new_id(),
                            payload=payload,
                        )
                        sizes.append(len(event.model_dump_json().encode()))
                        response = api.post(
                            "/v1/events",
                            json=event.model_dump(mode="json"),
                            headers={"Authorization": "Bearer " + api_key},
                        )
                        ok = (
                            response.status_code == 200
                            and response.json()["results"][0]["status"] == "accepted"
                        )
                        accepted += int(ok)
                        rejected += int(not ok)
                        expected += (2 if kind == 3 else 1 if kind == 2 else 0) * int(ok)
                    deadline = time.monotonic() + 10
                    while len(receipts) < expected and time.monotonic() < deadline:
                        time.sleep(0.05)
                    if provider:
                        calls = sum(1 for index in range(args.events) if index % 4 == 0)
                        drain = calls * provider.latency_seconds / args.provider_concurrency
                        deadline = time.monotonic() + 10 + 2 * drain
                        while time.monotonic() < deadline:
                            with store._lock:
                                pending = store.db.execute(
                                    "SELECT COUNT(*) FROM jobs WHERE project='benchmark' "
                                    "AND detector='prompt_injection' AND state!='done'"
                                ).fetchone()[0]
                            if not pending:
                                break
                            time.sleep(0.05)
                    # The receiver observes the event just before the publisher records its ack.
                    deadline = time.monotonic() + 2
                    while time.monotonic() < deadline:
                        status = store.status("benchmark")
                        if all(d["state"] == "acknowledged" for d in status["destinations"]):
                            break
                        time.sleep(0.05)
                    elapsed = time.monotonic() - started
                values = sorted(receipts.values())
                report = {
                    "platform": platform.platform(),
                    "python": platform.python_version(),
                    "processor": platform.processor(),
                    "logical_cpus": os.cpu_count(),
                    "storage": "temporary SQLite WAL, synchronous FULL",
                    "transport": "loopback HTTPS-exempt webhook",
                    "ingestion": "in-process ASGI TestClient; receiver uses real loopback HTTP",
                    "classifier": (
                        f"fake provider, {args.classifier_latency_ms:g} ms, always clear; "
                        "quality not measured"
                        if provider
                        else "disabled; semantic latency/quality not measured"
                    ),
                    "provider_concurrency": args.provider_concurrency,
                    "event_count": args.events,
                    "offered_events_per_second": args.rate,
                    "interleaved_sessions": args.sessions,
                    "duration_seconds": elapsed,
                    "payload_mix": (
                        "25% message, 25% allowed tool, "
                        "25% forbidden destination, 25% protected-data tool"
                    ),
                    "event_bytes_min_max": [min(sizes), max(sizes)],
                    "accepted": accepted,
                    "rejected": rejected,
                    "expected_notifications": expected,
                    "unique_received": len(values),
                    "missing_notifications": expected - len(values),
                    "receipt_p95_seconds": values[math.ceil(len(values) * 0.95) - 1]
                    if values
                    else None,
                    "receipt_max_seconds": max(values) if values else None,
                    "status": status,
                }
                if provider:
                    report.update(screen_timings(store, "benchmark", provider.started))
                output = Path(args.output)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(report, indent=2), encoding="utf-8")
                print(json.dumps(report, indent=2))
                if rejected or len(values) != expected:
                    raise SystemExit(1)
            finally:
                store.close()
    finally:
        receiver.shutdown()
        receiver.server_close()
        receiver_thread.join(timeout=2)


if __name__ == "__main__":
    main()
