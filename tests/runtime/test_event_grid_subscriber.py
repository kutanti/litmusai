"""CloudEvents subscription validation and authenticated example delivery."""

import json
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from litmusai.runtime.engine import Engine
from litmusai.runtime.redaction import Redactor


@pytest.fixture
def subscriber(tmp_path, monkeypatch):
    # Examples are repository files, not part of the installed distribution.
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))
    from examples.runtime.event_grid_subscriber import app

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("EVENT_GRID_SUBSCRIPTION_TOKEN", "synthetic-delivery-token")
    with TestClient(app) as client:
        yield client


@pytest.mark.parametrize("origin", ["eventgrid.azure.net", "untrusted.example", None])
def test_cloudevents_options_validation_does_not_require_delivery_authentication(
    subscriber, monkeypatch, origin
):
    monkeypatch.delenv("EVENT_GRID_SUBSCRIPTION_TOKEN")
    headers = {"WebHook-Request-Rate": "120"}
    if origin is not None:
        headers["WebHook-Request-Origin"] = origin
    response = subscriber.options("/events", headers=headers)
    if origin == "eventgrid.azure.net":
        assert response.status_code == 200
        assert response.headers["WebHook-Allowed-Origin"] == origin
        assert response.headers["WebHook-Allowed-Rate"] == "*"
        assert "POST" in response.headers["Allow"]
    else:
        assert response.status_code == 403
        assert "WebHook-Allowed-Origin" not in response.headers


@pytest.mark.parametrize("batch", [False, True])
async def test_validated_subscription_still_authenticates_and_deduplicates_deliveries(
    subscriber, config, store, make_event, tmp_path, batch
):
    store.accept(Redactor(config.projects[0].policy).capture(make_event()), config.projects[0])
    engine = Engine(store, config)
    while await engine.process_one("p"):
        pass
    event = json.loads(store.db.execute("SELECT content FROM revisions").fetchone()[0])
    payload = [event] if batch else event
    assert subscriber.options(
        "/events", headers={"WebHook-Request-Origin": "eventgrid.azure.net"}
    ).status_code == 200
    for token in (None, "wrong-token"):
        headers = {"Origin": "eventgrid.azure.net"}
        if token:
            headers["X-Litmus-Subscription-Token"] = token
        assert subscriber.post("/events", json=payload, headers=headers).status_code == 401
    database = tmp_path / ".litmus/event-grid-consumer.sqlite"
    assert not database.exists()
    for _ in range(2):
        response = subscriber.post(
            "/events",
            json=payload,
            headers={"X-Litmus-Subscription-Token": "synthetic-delivery-token"},
        )
        assert response.status_code == 200
        assert response.json() == {"accepted": True}
    with sqlite3.connect(database) as db:
        assert db.execute("SELECT COUNT(*) FROM seen").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM latest").fetchone()[0] == 1


@pytest.mark.parametrize("authenticated", [False, True])
def test_native_event_grid_schema_is_not_a_cloudevents_authentication_bypass(
    subscriber, authenticated
):
    headers = {"aeg-event-type": "SubscriptionValidation"}
    if authenticated:
        headers["X-Litmus-Subscription-Token"] = "synthetic-delivery-token"
    response = subscriber.post(
        "/events",
        headers=headers,
        json=[
            {
                "eventType": "Microsoft.EventGrid.SubscriptionValidationEvent",
                "data": {"validationCode": "synthetic-validation-code"},
            }
        ],
    )
    assert response.status_code == (400 if authenticated else 401)
    assert "validationResponse" not in response.json()
