"""Authenticated HTTP ingestion and a live client-to-publisher demonstration."""

import json
import subprocess
import sys
import time

import httpx
import pytest
from click.testing import CliRunner

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from litmusai.cli.main import cli
from litmusai.runtime.client import RuntimeClient
from litmusai.runtime.config import ProjectConfig
from litmusai.runtime.service import create_app

HEADERS = {"Authorization": "Bearer synthetic-project-key-1234"}


def test_http_auth_project_scope_rejected_events_and_size_limits(
    config, store, make_event, monkeypatch
):
    monkeypatch.setenv("OTHER_RUNTIME_KEY", "synthetic-other-key-1234")
    config = config.model_copy(
        update={
            "projects": [
                *config.projects,
                ProjectConfig(project_id="other", api_key_env="OTHER_RUNTIME_KEY"),
            ],
            "max_request_bytes": 2048,
            "batch_size": 2,
        }
    )
    with TestClient(create_app(config, store=store, run_workers=False)) as api:
        event = make_event().model_dump(mode="json")
        assert api.post("/v1/events", json=event).status_code == 401
        wrong = {**event, "project_id": "other"}
        response = api.post("/v1/events", headers=HEADERS, json=[wrong, event])
        assert [r["status"] for r in response.json()["results"]] == ["rejected", "accepted"]
        assert (
            api.post("/v1/events", headers=HEADERS, json=[event]).json()["results"][0]["status"]
            == "duplicate"
        )
        assert api.post("/v1/events", headers=HEADERS, json=[event] * 3).status_code == 413
        assert api.post("/v1/events", headers=HEADERS, content=b"x" * 2049).status_code == 413
        assert api.post("/v1/events", headers=HEADERS, content=b"{").status_code == 400
        assert api.get("/v1/alerts?limit=1000", headers=HEADERS).status_code == 400
        assert api.get("/v1/status").status_code == 401
        assert api.get("/v1/alerts/nonexistent", headers=HEADERS).status_code == 404
        assert api.post("/v1/deliveries/nonexistent/replay", headers=HEADERS).status_code == 404
        assert api.get("/health/live").status_code == 200


def test_validation_does_not_echo_secrets_or_accept_payload_policy(config, store, make_event):
    with TestClient(create_app(config, store=store, run_workers=False)) as api:
        event = make_event().model_dump(mode="json")
        protected = "ghp_" + "X" * 36
        event["policy"] = {"allowed_tools": [protected]}
        response = api.post("/v1/events", json=event, headers=HEADERS)
        assert response.json()["results"][0]["status"] == "rejected"
        assert protected not in response.text
        event.pop("policy")
        event["event_type"] = "message.received"
        assert (
            api.post("/v1/events", json=event, headers=HEADERS).json()["results"][0]["status"]
            == "rejected"
        )


def test_live_client_alert_is_published_before_session_ends(config, store):
    received, executions = [], []

    class Publisher:
        async def publish(self, body, delivery_id):
            received.append(json.loads(body))

    app = create_app(config, store=store, publisher_factory=lambda c: Publisher())
    with TestClient(app) as api:

        def transport(request):
            response = api.post(
                "/v1/events", content=request.content, headers=dict(request.headers)
            )
            return httpx.Response(response.status_code, json=response.json())

        def send_email(recipient, body):
            executions.append((recipient, body))
            return {"accepted": True}

        with RuntimeClient(
            "https://collector.example",
            HEADERS["Authorization"][7:],
            "p",
            transport=httpx.MockTransport(transport),
        ) as monitor:
            with monitor.session("support", "live-session") as session:
                wrapped = session.wrap_tool(send_email, destination_argument="recipient")
                assert wrapped("outside.example", "hello") == {"accepted": True}
                assert monitor.flush()
                deadline = time.monotonic() + 3
                while not received and time.monotonic() < deadline:
                    time.sleep(0.01)
                assert received, api.get("/v1/status", headers=HEADERS).text
                assert received[0]["data"]["category"] == "forbidden_destination"
                assert received[0]["data"]["session_id"] == "live-session"
                assert (
                    store.db.execute(
                        "SELECT COUNT(*) FROM events WHERE content LIKE '%session.ended%'"
                    ).fetchone()[0]
                    == 0
                )
        assert len(executions) == 1


def test_credentials_cannot_be_shared_between_projects(config, monkeypatch):
    monkeypatch.setenv("OTHER_RUNTIME_KEY", "synthetic-project-key-1234")
    config = config.model_copy(
        update={
            "projects": [
                *config.projects,
                ProjectConfig(project_id="other", api_key_env="OTHER_RUNTIME_KEY"),
            ]
        }
    )
    with pytest.raises(ValueError, match="unique"):
        create_app(config)


def test_cli_runtime_and_base_import_without_optional_modules(tmp_path, config):
    import yaml

    config_path = tmp_path / "runtime.yaml"
    config_path.write_text(yaml.safe_dump(config.model_dump(mode="json")), encoding="utf-8")
    result = CliRunner().invoke(cli, ["runtime", "validate", "--config", str(config_path)])
    assert result.exit_code == 0 and "1 projects, 1 destinations" in result.output
    code = """
import sys
class BlockOptional:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'fastapi','uvicorn','confluent_kafka','azure'}:
            raise ImportError('optional imports blocked')
sys.meta_path.insert(0, BlockOptional())
import litmusai
from litmusai.runtime import RuntimeClient
from litmusai.cli.main import cli
assert 'runtime' in cli.commands
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
