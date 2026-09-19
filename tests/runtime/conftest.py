"""Runtime integration fixtures using synthetic payloads and isolated storage."""

import pytest

from litmusai.runtime.config import ProjectConfig, RuntimeConfig, ThreatPolicy, WebhookDestination
from litmusai.runtime.models import RuntimeEvent, ToolActivity, new_id
from litmusai.runtime.store import Store


@pytest.fixture
def policy():
    return ThreatPolicy(
        allowed_tools=["send_email"],
        allowed_destinations=["support@example.com"],
        destination_tools=["send_email"],
    )


@pytest.fixture
def config(tmp_path, monkeypatch, policy):
    monkeypatch.setenv("TEST_RUNTIME_KEY", "synthetic-project-key-1234")
    monkeypatch.setenv("TEST_WEBHOOK_KEY", "synthetic-webhook-key-1234")
    return RuntimeConfig(
        database=str(tmp_path / "runtime.sqlite"),
        projects=[ProjectConfig(project_id="p", api_key_env="TEST_RUNTIME_KEY", policy=policy)],
        destinations=[
            WebhookDestination(
                destination_id="webhook",
                project_id="p",
                url="https://receiver.example/events",
                signing_secret_env="TEST_WEBHOOK_KEY",
            )
        ],
    )


@pytest.fixture
def store(config):
    instance = Store(config.database)
    instance.register_config(config)
    yield instance
    instance.close()


@pytest.fixture
def make_event():
    def make(**values):
        fields = dict(
            project_id="p",
            agent_id="agent",
            session_id="session",
            producer_id=new_id(),
            sequence=1,
            event_type="tool.requested",
            tool_call_id=new_id(),
            payload=ToolActivity(
                name="send_email", arguments={"body": "hello"}, destination="attacker.example"
            ),
        )
        fields.update(values)
        return RuntimeEvent(**fields)

    return make
