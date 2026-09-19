"""Local collector-to-HTTP-receiver experiment; no agents, vendors, or cloud topics invoked."""

from __future__ import annotations

import argparse
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

from litmusai.runtime.config import ProjectConfig, RuntimeConfig, ThreatPolicy, WebhookDestination
from litmusai.runtime.models import Message, RuntimeEvent, ToolActivity, new_id
from litmusai.runtime.publishers import verify_webhook
from litmusai.runtime.service import create_app
from litmusai.runtime.store import Store


def main() -> None:
    """Measure local delivery latency while including unsuccessful events in the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=int, default=200)
    parser.add_argument("--rate", type=float, default=10)
    parser.add_argument("--sessions", type=int, default=20)
    parser.add_argument("--output", default=".litmus/runtime-benchmark.json")
    args = parser.parse_args()
    if min(args.events, args.rate, args.sessions) <= 0:
        parser.error("events, rate, and sessions must be positive")
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
                    )
                ],
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
                with TestClient(create_app(config, store=store)) as api:
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
                    "classifier": "disabled; semantic latency/quality not measured",
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
