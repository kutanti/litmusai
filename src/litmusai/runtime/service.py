"""Optional authenticated FastAPI collector and project-scoped inspection API."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import sqlite3
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from litmusai.runtime.config import Destination, ProjectConfig, RuntimeConfig, secret
from litmusai.runtime.detectors import InjectionClassifier
from litmusai.runtime.engine import Engine
from litmusai.runtime.models import RuntimeEvent
from litmusai.runtime.policies import PolicyEvaluator
from litmusai.runtime.publishers import AlertPublisher, publisher_for
from litmusai.runtime.redaction import Redactor
from litmusai.runtime.store import Store


def create_app(
    config: RuntimeConfig,
    *,
    store: Store | None = None,
    classifiers: dict[str, InjectionClassifier] | None = None,
    reviewers: dict[str, InjectionClassifier] | None = None,
    policy_evaluators: dict[tuple[str, str], PolicyEvaluator] | None = None,
    publisher_factory: Callable[[Destination], AlertPublisher] = publisher_for,
    run_workers: bool = True,
) -> FastAPI:
    """Build a self-hosted single-instance service; secrets are resolved only at startup.

    Terminate public TLS at a trusted reverse proxy. Do not expose this service on
    an unencrypted public endpoint. ``run_workers=False`` supports deterministic tests.
    """
    credentials: list[tuple[bytes, ProjectConfig]] = []
    for project in config.projects:
        key = secret(project.api_key_env)
        if not 16 <= len(key) <= 4000:
            raise ValueError("runtime API keys require 16 to 4000 characters")
        digest = hashlib.sha256(key.encode()).digest()
        if any(hmac.compare_digest(digest, existing) for existing, _ in credentials):
            raise ValueError("runtime API keys must be unique across projects")
        credentials.append((digest, project))
    database = store or Store(config.database)
    database.register_config(config)
    engine = Engine(
        database,
        config,
        classifiers=classifiers,
        reviewers=reviewers,
        policy_evaluators=policy_evaluators,
        publisher_factory=publisher_factory,
    )
    redactors = {p.project_id: Redactor(p.policy) for p in config.projects}
    counters: dict[str, dict[str, int]] = {p.project_id: {} for p in config.projects}

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if run_workers:
            await engine.start()
        try:
            yield
        finally:
            await engine.stop()
            if store is None:
                database.close()

    app = FastAPI(
        title="LitmusAI runtime",
        version="1.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.store = database
    app.state.engine = engine

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, error: Exception) -> JSONResponse:
        return JSONResponse({"detail": "invalid request"}, status_code=422)

    @app.exception_handler(sqlite3.Error)
    async def storage_error(request: Request, error: Exception) -> JSONResponse:
        return JSONResponse({"detail": "storage unavailable"}, status_code=503)

    def authenticate(request: Request) -> ProjectConfig:
        header = request.headers.get("authorization", "")
        if not header.startswith("Bearer ") or len(header) > 4096:
            raise HTTPException(401, "authentication required")
        digest = hashlib.sha256(header[7:].encode()).digest()
        matched = None
        for key_digest, project in credentials:
            if hmac.compare_digest(digest, key_digest):
                matched = project
        if matched is None:
            raise HTTPException(401, "authentication required")
        return matched

    async def read_body(request: Request) -> bytes:
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > config.max_request_bytes:
                raise HTTPException(413, "request exceeds limit")
        return bytes(body)

    @app.post("/v1/events")
    async def ingest(request: Request) -> dict[str, Any]:
        project = authenticate(request)
        try:
            raw = await asyncio.wait_for(read_body(request), timeout=10)
            payload = json.loads(raw)
        except (ValueError, RecursionError):
            raise HTTPException(400, "invalid JSON") from None
        except asyncio.TimeoutError:
            raise HTTPException(408, "request timeout") from None
        items = payload if isinstance(payload, list) else [payload]
        if not 1 <= len(items) <= config.batch_size:
            raise HTTPException(413, "batch exceeds limit")
        results: list[dict[str, Any]] = []
        for item in items:
            event_id = None
            try:
                if len(json.dumps(item).encode()) > config.max_event_bytes:
                    raise ValueError("event exceeds limit")
                event = RuntimeEvent.model_validate(item)
                if event.project_id != project.project_id:
                    raise ValueError("project mismatch")
                captured = redactors[project.project_id].capture(event)
                event_id = event.event_id
                status = database.accept(captured, project)
            except (ValueError, TypeError, RecursionError):
                status = "rejected"
            counts = counters[project.project_id]
            counts[status] = counts.get(status, 0) + 1
            results.append({"event_id": event_id, "status": status})
        return {"results": results}

    @app.get("/v1/alerts")
    async def alerts(request: Request, limit: int = 100, offset: int = 0) -> dict[str, Any]:
        project = authenticate(request)
        if not 1 <= limit <= 200 or not 0 <= offset <= 1000000:
            raise HTTPException(400, "invalid pagination")
        return {"alerts": database.alerts(project.project_id, limit, offset)}

    @app.get("/v1/alerts/{alert_id}")
    async def alert(request: Request, alert_id: str) -> dict[str, Any]:
        project = authenticate(request)
        item = database.alert(project.project_id, alert_id)
        if item is None:
            raise HTTPException(404, "alert not found")
        return item

    @app.get("/v1/status")
    async def status(request: Request) -> dict[str, Any]:
        project = authenticate(request)
        return {
            **database.status(project.project_id),
            "ingestion": counters[project.project_id],
            "ingestion_counter_scope": "current_process",
            "worker_error": engine.last_worker_error,
            "last_retention_cleanup": engine.last_cleanup,
        }

    @app.get("/v1/findings")
    async def findings(request: Request, limit: int = 100, offset: int = 0) -> dict[str, Any]:
        project = authenticate(request)
        if not 1 <= limit <= 200 or not 0 <= offset <= 1000000:
            raise HTTPException(400, "invalid pagination")
        return {"findings": database.findings(project.project_id, limit, offset)}

    @app.post("/v1/deliveries/{delivery_id}/replay")
    async def replay(request: Request, delivery_id: str) -> dict[str, str]:
        project = authenticate(request)
        if not database.replay(project.project_id, delivery_id):
            raise HTTPException(404, "failed delivery not found")
        return {"status": "pending"}

    @app.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "live"}

    @app.get("/health/ready")
    async def ready() -> JSONResponse:
        workers_ok = not run_workers or (
            bool(engine.tasks)
            and not any(t.done() for t in engine.tasks)
            and not engine.worker_errors
        )
        workers_ok = workers_ok and database.healthy()
        return JSONResponse(
            {"status": "ready" if workers_ok else "unavailable"},
            status_code=200 if workers_ok else 503,
        )

    return app
