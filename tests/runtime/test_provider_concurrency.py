"""Bounded provider concurrency per lane with one shared HTTP client."""

import asyncio

import httpx
import pytest

from litmusai.runtime.config import ClassifierConfig, RuntimeConfig
from litmusai.runtime.detectors import ClassifierVerdict, HTTPInjectionClassifier
from litmusai.runtime.engine import Engine
from litmusai.runtime.models import Message
from litmusai.runtime.redaction import Redactor


class Rendezvous:
    """Hold every call until ``expected`` calls are in flight, then release them together."""

    def __init__(self, expected):
        self.expected = expected
        self.active = 0
        self.peak = 0
        self.ready = asyncio.Event()
        self.client = None
        self.bound = []

    def bind_client(self, client):
        self.bound.append(client)
        self.client = client

    async def classify(self, captured, context, incomplete):
        self.active += 1
        self.peak = max(self.peak, self.active)
        if self.active >= self.expected:
            self.ready.set()
        await asyncio.wait_for(self.ready.wait(), timeout=5)
        self.active -= 1
        return ClassifierVerdict(outcome="detected", reason="synthetic detection")


def screened(config, concurrency):
    classifier = ClassifierConfig(
        endpoint="https://screen.example/evaluate", api_key_env="SCREEN_KEY", version="screen-v1"
    )
    project = config.projects[0].model_copy(update={"classifier": classifier})
    return config.model_copy(
        update={"projects": [project], "provider_concurrency": concurrency}
    )


def accept(store, config, make_event, **fields):
    fields.setdefault("event_type", "message.received")
    fields.setdefault("payload", Message(text="synthetic content"))
    fields.setdefault("tool_call_id", None)
    captured = Redactor(config.projects[0].policy).capture(make_event(**fields))
    assert store.accept(captured, config.projects[0]) == "accepted"
    return captured


def test_provider_concurrency_is_bounded(config):
    fields = config.model_dump()
    assert config.provider_concurrency == 1
    for invalid in (0, 17):
        with pytest.raises(ValueError, match="provider_concurrency"):
            RuntimeConfig.model_validate({**fields, "provider_concurrency": invalid})
    assert RuntimeConfig.model_validate({**fields, "provider_concurrency": 16})


async def test_simultaneous_results_group_into_one_alert(config, store, make_event):
    config = screened(config, 2)
    store.register_config(config)
    for sequence in (1, 2):
        accept(store, config, make_event, sequence=sequence, parent_event_id="turn-1")
    classifier = Rendezvous(expected=2)
    engine = Engine(store, config, classifiers={"p": classifier})
    processed = await asyncio.gather(engine.process_one("p", True), engine.process_one("p", True))
    assert processed == [True, True]
    assert classifier.peak == 2
    alerts = [a for a in store.alerts("p") if a["detector"] == "prompt_injection"]
    assert len(alerts) == 1
    assert alerts[0]["revision"] == 1
    rows = [r for r in store.findings("p") if r["finding"]["detector"] == "prompt_injection"]
    assert sorted(r["suppressed"] for r in rows) == [False, True]


class Delivered:
    def __init__(self):
        self.bodies = []

    async def publish(self, body, delivery_id):
        self.bodies.append(body)


async def test_workers_per_lane_share_one_bound_client(config, store, make_event):
    config = screened(config, 3)
    classifier = Rendezvous(expected=3)
    publisher = Delivered()
    engine = Engine(
        store, config, classifiers={"p": classifier}, publisher_factory=lambda _: publisher
    )
    await engine.start()
    try:
        client = engine.http_client
        assert isinstance(client, httpx.AsyncClient)
        assert classifier.client is client
        names = {task.get_coro().cr_code.co_name for task in engine.tasks}
        assert "_jobs" in names
        job_tasks = [t for t in engine.tasks if t.get_coro().cr_code.co_name == "_jobs"]
        # One local worker plus three workers for each of the screen, review and policy lanes.
        assert len(job_tasks) == 1 + 3 * 3
        for sequence in (1, 2, 3):
            accept(store, config, make_event, sequence=sequence)
        await asyncio.wait_for(classifier.ready.wait(), timeout=5)
        for _ in range(100):
            rows = [
                r for r in store.findings("p") if r["finding"]["detector"] == "prompt_injection"
            ]
            if len(rows) == 3:
                break
            await asyncio.sleep(0.02)
        assert classifier.peak == 3
        assert not engine.worker_errors
    finally:
        await engine.stop()
    assert engine.http_client is None
    assert classifier.client is None
    assert client.is_closed


async def test_worker_ids_keep_first_worker_name(config, store):
    engine = Engine(store, screened(config, 2), classifiers={})

    async def fail(*args, **kwargs):
        raise RuntimeError("synthetic")

    engine.process_one = fail
    engine.config = engine.config.model_copy(update={"poll_seconds": 0.01})
    tasks = [
        asyncio.create_task(engine._jobs("p", True)),
        asyncio.create_task(engine._jobs("p", True, worker=1)),
        asyncio.create_task(engine._jobs("p", True, review=True, worker=1)),
    ]
    for _ in range(50):
        if len(engine.worker_errors) == 3:
            break
        await asyncio.sleep(0.02)
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    assert set(engine.worker_errors) == {"jobs:p:True", "jobs:p:True#2", "jobs:p:review#2"}


async def test_built_in_adapter_uses_engine_client(config, store, make_event, monkeypatch):
    monkeypatch.setenv("SCREEN_KEY", "synthetic-screen-key-1234")
    config = screened(config, 1)
    store.register_config(config)
    accept(store, config, make_event)
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"outcome": "clear", "reason": "synthetic"})

    engine = Engine(store, config)
    engine.http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    original = HTTPInjectionClassifier.__init__
    clients = []

    def spy(self, settings, *, client=None):
        clients.append(client)
        original(self, settings, client=client)

    monkeypatch.setattr(HTTPInjectionClassifier, "__init__", spy)
    try:
        assert await engine.process_one("p", True)
    finally:
        await engine.http_client.aclose()
    assert clients == [engine.http_client]
    assert len(seen) == 1


async def test_benchmark_fake_provider_records_call_start(config, store, make_event):
    import runpy
    from pathlib import Path

    script = Path(__file__).resolve().parents[2] / "scripts" / "benchmark_runtime.py"
    module = runpy.run_path(str(script))
    assert module["nearest_rank"]([0.4, 0.1, 0.3, 0.2], 0.95) == 0.4
    assert module["nearest_rank"]([], 0.95) is None
    config = screened(config, 1)
    store.register_config(config)
    captured = accept(store, config, make_event)
    provider = module["FakeProvider"](0)
    engine = Engine(store, config, classifiers={"p": provider})
    assert await engine.process_one("p", True)
    timings = module["screen_timings"](store, "p", provider.started)
    assert timings["provider_calls"] == 1
    assert timings["screen_jobs_pending"] == 0
    assert timings["screen_outcomes"] == {"clear": 1}
    assert captured.event.event_id in provider.started
    assert timings["queue_delay_p95_seconds"] >= 0
