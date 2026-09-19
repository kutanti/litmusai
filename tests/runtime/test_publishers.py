"""Publisher contracts exercise real SDK call shapes without external credentials."""

import json

import pytest

from litmusai.runtime.config import EventGridDestination, KafkaDestination
from litmusai.runtime.publishers import (
    EventGridPublisher,
    KafkaPublisher,
    PublishError,
    WebhookPublisher,
    sign_webhook,
    verify_webhook,
)


@pytest.mark.parametrize(
    "status,retryable", [(302, False), (401, False), (400, False), (429, True), (503, True)]
)
async def test_webhook_acknowledgement_and_failure_classification(
    config,
    httpx_mock,
    status,
    retryable,
):
    httpx_mock.add_response(
        url=config.destinations[0].url,
        status_code=status,
        headers={"Location": "https://elsewhere.example"},
    )
    with pytest.raises(PublishError) as caught:
        await WebhookPublisher(config.destinations[0]).publish(b'{"id":"event"}', "delivery")
    assert caught.value.retryable is retryable
    assert len(httpx_mock.get_requests()) == 1


async def test_webhook_exact_body_signature_delivery_id_and_timestamp(config, httpx_mock):
    body = b'{"id":"event", "data":{"revision":1}}'
    httpx_mock.add_response(status_code=204)
    await WebhookPublisher(config.destinations[0]).publish(body, "delivery")
    request = httpx_mock.get_request()
    assert request.content == body
    assert request.headers["content-type"] == "application/cloudevents+json"
    assert request.headers["x-litmus-delivery-id"] == "delivery"
    timestamp = request.headers["x-litmus-timestamp"]
    signature = request.headers["x-litmus-signature"]
    assert verify_webhook(body, "synthetic-webhook-key-1234", timestamp, signature)
    assert not verify_webhook(body + b" ", "synthetic-webhook-key-1234", timestamp, signature)
    assert not verify_webhook(body, "wrong", timestamp, signature)
    assert not verify_webhook(
        body, "synthetic-webhook-key-1234", timestamp, signature, now=int(timestamp) + 301
    )
    assert not verify_webhook(body, "key", "invalid", signature)
    assert sign_webhook(body, "key", "100") == sign_webhook(body, "key", "100")


async def test_kafka_uses_broker_ack_idempotent_options_partition_key_and_unchanged_body(
    monkeypatch,
):
    kafka = pytest.importorskip("confluent_kafka")
    captured = {}

    class Producer:
        def __init__(self, options):
            captured["options"] = options

        def produce(self, topic, value, key, on_delivery):
            captured.update(topic=topic, value=value, key=key)
            self.callback = on_delivery

        def poll(self, timeout):
            self.callback(None, object())

    monkeypatch.setattr(kafka, "Producer", Producer)
    config = KafkaDestination(
        destination_id="k", project_id="p", bootstrap_servers="broker:9093", topic="alerts"
    )
    body = b'{"id":"stable-event","data":{"project_id":"p","session_id":"s"}}'
    await KafkaPublisher(config).publish(body, "delivery")
    assert captured["value"] == body
    assert json.loads(captured["key"]) == ["p", "s"]
    assert captured["options"]["enable.idempotence"] and captured["options"]["acks"] == "all"
    assert captured["options"]["security.protocol"] == "SSL"


async def test_kafka_asynchronous_auth_failure_is_permanent(monkeypatch):
    kafka = pytest.importorskip("confluent_kafka")

    class Producer:
        def __init__(self, options):
            pass

        def produce(self, topic, **kwargs):
            self.callback = kwargs["on_delivery"]

        def poll(self, timeout):
            self.callback(kafka.KafkaError(kafka.KafkaError.TOPIC_AUTHORIZATION_FAILED), None)

    monkeypatch.setattr(kafka, "Producer", Producer)
    config = KafkaDestination(
        destination_id="k", project_id="p", bootstrap_servers="broker:9093", topic="alerts"
    )
    with pytest.raises(PublishError) as caught:
        await KafkaPublisher(config).publish(b'{"data":{"project_id":"p","session_id":"s"}}', "d")
    assert not caught.value.retryable


@pytest.mark.parametrize("mechanism", ["PLAIN", "SCRAM-SHA-256", "SCRAM-SHA-512"])
@pytest.mark.parametrize("failed_first_attempt", [False, True])
async def test_kafka_cached_producer_uses_rotated_sasl_credentials(
    monkeypatch, mechanism, failed_first_attempt
):
    kafka = pytest.importorskip("confluent_kafka")
    monkeypatch.setenv("TEST_KAFKA_USER", "original-user")
    monkeypatch.setenv("TEST_KAFKA_PASSWORD", "original-password")
    broker_credentials = (
        "original-user",
        "expired-password" if failed_first_attempt else "original-password",
    )
    producers = []
    bodies = []

    class Producer:
        def __init__(self, options):
            self.credentials = options["sasl.username"], options["sasl.password"]
            assert options["sasl.mechanism"] == mechanism
            producers.append(self)

        def set_sasl_credentials(self, username, password):
            self.credentials = username, password

        def produce(self, topic, value, key, on_delivery):
            bodies.append(value)
            self.callback = on_delivery

        def poll(self, timeout):
            error = None
            if self.credentials != broker_credentials:
                error = kafka.KafkaError(kafka.KafkaError.SASL_AUTHENTICATION_FAILED)
            self.callback(error, None)

    monkeypatch.setattr(kafka, "Producer", Producer)
    config = KafkaDestination(
        destination_id="k",
        project_id="p",
        bootstrap_servers="broker:9093",
        topic="alerts",
        security_protocol="SASL_SSL",
        sasl_mechanism=mechanism,
        username_env="TEST_KAFKA_USER",
        password_env="TEST_KAFKA_PASSWORD",
    )
    publisher = KafkaPublisher(config)
    body = b'{"id":"stable-event","data":{"project_id":"p","session_id":"s"}}'
    if failed_first_attempt:
        with pytest.raises(PublishError) as caught:
            await publisher.publish(body, "delivery")
        assert not caught.value.retryable
    else:
        await publisher.publish(body, "delivery")
    broker_credentials = "rotated-user", "rotated-password"
    monkeypatch.setenv("TEST_KAFKA_USER", broker_credentials[0])
    monkeypatch.setenv("TEST_KAFKA_PASSWORD", broker_credentials[1])
    await publisher.publish(body, "delivery")
    assert bodies == [body, body]
    assert len(producers) == 1  # Rotation preserves the producer's idempotence state.

    # Missing references must fail instead of silently continuing with cached secrets.
    monkeypatch.delenv("TEST_KAFKA_PASSWORD")
    with pytest.raises(ValueError, match="required runtime secret is missing"):
        await publisher.publish(body, "delivery")
    assert bodies == [body, body]


async def test_kafka_network_roundtrip_with_librdkafka_protocol_mock():
    """Exercise producer callbacks and consumer bytes using a loopback broker mock."""
    import time
    from uuid import uuid4

    kafka = pytest.importorskip("confluent_kafka")
    owner = kafka.Producer({"test.mock.num.brokers": 1, "log_level": 0})
    broker = next(iter(owner.list_topics(timeout=3).brokers.values()))
    bootstrap = f"{broker.host}:{broker.port}"
    topic = "litmus-test-" + uuid4().hex
    config = KafkaDestination(
        destination_id="mock",
        project_id="p",
        bootstrap_servers=bootstrap,
        topic=topic,
        security_protocol="PLAINTEXT",
        allow_insecure_development=True,
    )
    body = b'{"id":"stable-event","data":{"project_id":"p","session_id":"s"}}'
    await KafkaPublisher(config).publish(body, "delivery")
    consumer = kafka.Consumer(
        {
            "bootstrap.servers": bootstrap,
            "group.id": uuid4().hex,
            "enable.auto.commit": False,
            "log_level": 0,
        }
    )
    try:
        metadata = consumer.list_topics(topic, timeout=3)
        consumer.assign(
            [
                kafka.TopicPartition(topic, partition, kafka.OFFSET_BEGINNING)
                for partition in metadata.topics[topic].partitions
            ]
        )
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            message = consumer.poll(0.1)
            if message is not None and not message.error():
                assert message.value() == body
                return
        pytest.fail("protocol mock did not receive published bytes")
    finally:
        consumer.close()
        owner.flush(1)


async def test_event_grid_preserves_cloudevent_and_disables_nested_retries(monkeypatch):
    sdk = pytest.importorskip("azure.eventgrid.aio")
    captured = {}
    monkeypatch.setenv("TEST_GRID_KEY", "synthetic-key")

    class Client:
        def __init__(self, endpoint, credential, **options):
            captured.update(endpoint=endpoint, options=options)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def send(self, events):
            captured["event"] = events[0]

    monkeypatch.setattr(sdk, "EventGridPublisherClient", Client)
    config = EventGridDestination(
        destination_id="grid",
        project_id="p",
        endpoint="https://topic.example/api/events",
        access_key_env="TEST_GRID_KEY",
    )
    body = {
        "specversion": "1.0",
        "id": "stable",
        "source": "urn:litmusai:project:p",
        "type": "com.litmusai.threat.detected",
        "subject": "alerts/a",
        "time": "2026-09-18T00:00:00Z",
        "datacontenttype": "application/json",
        "data": {"alert_id": "a", "revision": 1},
    }
    await EventGridPublisher(config).publish(json.dumps(body).encode(), "delivery")
    assert captured["event"].id == "stable" and captured["event"].data == body["data"]
    assert captured["event"].type == body["type"]
    assert captured["options"]["retry_total"] == 0


@pytest.mark.parametrize("status,retryable", [(403, False), (429, True), (503, True)])
async def test_event_grid_classifies_errors_without_leaking_provider_body(
    monkeypatch, status, retryable
):
    sdk = pytest.importorskip("azure.eventgrid.aio")
    monkeypatch.setenv("TEST_GRID_KEY", "synthetic-key")

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def send(self, events):
            error = RuntimeError("private response data")
            error.status_code = status
            raise error

    monkeypatch.setattr(sdk, "EventGridPublisherClient", Client)
    config = EventGridDestination(
        destination_id="grid",
        project_id="p",
        endpoint="https://topic.example",
        access_key_env="TEST_GRID_KEY",
    )
    body = b'{"specversion":"1.0","id":"a","source":"urn:test","type":"test","data":{}}'
    with pytest.raises(PublishError) as caught:
        await EventGridPublisher(config).publish(body, "delivery")
    assert caught.value.retryable is retryable
    assert "private" not in str(caught.value)


async def test_generic_classifier_bounded_redacted_request_and_strict_response(
    make_event,
    policy,
    monkeypatch,
    httpx_mock,
):
    from litmusai.runtime.config import ClassifierConfig
    from litmusai.runtime.detectors import HTTPInjectionClassifier
    from litmusai.runtime.models import Message
    from litmusai.runtime.redaction import Redactor

    monkeypatch.setenv("TEST_CLASSIFIER_KEY", "synthetic-key")
    settings = ClassifierConfig(
        endpoint="https://classifier.example",
        api_key_env="TEST_CLASSIFIER_KEY",
        version="1",
        context_chars=512,
    )
    captured = Redactor(policy).capture(
        make_event(
            event_type="message.received",
            tool_call_id=None,
            payload=Message(text="ghp_" + "A" * 36 + "x" * 1000),
        )
    )
    httpx_mock.add_response(json={"outcome": "detected", "reason": "suspicious instructions"})
    verdict = await HTTPInjectionClassifier(settings).classify(captured, [], False)
    assert verdict.outcome == "detected"
    request = json.loads(httpx_mock.get_request().content)
    assert request["context_incomplete"]
    assert "ghp_" not in json.dumps(request)
    assert len(request["events"][0]["content"]) == 512


def test_transport_config_rejects_plaintext_and_embedded_secrets(config):
    from pydantic import ValidationError

    from litmusai.runtime.config import WebhookDestination

    with pytest.raises(ValidationError):
        KafkaDestination(
            destination_id="k",
            project_id="p",
            bootstrap_servers="localhost:9092",
            topic="alerts",
            security_protocol="PLAINTEXT",
        )
    with pytest.raises(ValidationError):
        WebhookDestination(
            destination_id="w",
            project_id="p",
            url="https://user:secret@receiver.example",
            signing_secret_env="KEY",
        )
    with pytest.raises(ValidationError):
        KafkaDestination(
            destination_id="k",
            project_id="p",
            bootstrap_servers="broker:9092",
            topic="alerts",
            security_protocol="SASL_SSL",
        )
