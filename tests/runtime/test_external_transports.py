"""Opt-in smoke tests for disposable infrastructure, never production agent actions."""

import json
import os
import time
from uuid import uuid4

import pytest

from litmusai.runtime.config import EventGridDestination, KafkaDestination
from litmusai.runtime.publishers import EventGridPublisher, KafkaPublisher


@pytest.mark.skipif(
    not os.getenv("LITMUS_TEST_KAFKA_BOOTSTRAP"), reason="test Kafka not configured"
)
async def test_disposable_kafka_publish_and_consume():
    kafka = pytest.importorskip("confluent_kafka")
    admin_module = pytest.importorskip("confluent_kafka.admin")
    bootstrap = os.environ["LITMUS_TEST_KAFKA_BOOTSTRAP"]
    topic = "litmus-test-" + uuid4().hex
    admin = admin_module.AdminClient({"bootstrap.servers": bootstrap})
    for future in admin.create_topics([admin_module.NewTopic(topic, 1, 1)]).values():
        future.result(timeout=20)
    consumer = kafka.Consumer(
        {
            "bootstrap.servers": bootstrap,
            "group.id": topic,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }
    )
    try:
        config = KafkaDestination(
            destination_id="test-kafka",
            project_id="p",
            bootstrap_servers=bootstrap,
            topic=topic,
            security_protocol="PLAINTEXT",
            allow_insecure_development=True,
            timeout_seconds=10,
        )
        body = json.dumps(
            {
                "specversion": "1.0",
                "id": uuid4().hex,
                "type": "com.litmusai.test",
                "source": "urn:litmusai:integration-test",
                "data": {"project_id": "p", "session_id": "synthetic"},
            }
        ).encode()
        await KafkaPublisher(config).publish(body, "test-delivery")
        consumer.subscribe([topic])
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            message = consumer.poll(0.1)
            if message is not None and not message.error():
                assert message.value() == body
                return
        pytest.fail("broker acknowledged the event but the consumer did not receive it")
    finally:
        consumer.close()
        for future in admin.delete_topics([topic]).values():
            future.result(timeout=20)


@pytest.mark.skipif(
    not os.getenv("LITMUS_TEST_EVENT_GRID_ENDPOINT"), reason="test Event Grid topic not configured"
)
async def test_event_grid_topic_accepts_synthetic_cloudevent():
    pytest.importorskip("azure.eventgrid.aio")
    config = EventGridDestination(
        destination_id="test-grid",
        project_id="p",
        endpoint=os.environ.get("LITMUS_TEST_EVENT_GRID_ENDPOINT", ""),
        access_key_env="LITMUS_TEST_EVENT_GRID_KEY",
    )
    body = json.dumps(
        {
            "specversion": "1.0",
            "id": uuid4().hex,
            "type": "com.litmusai.test",
            "source": "urn:litmusai:integration-test",
            "data": {"purpose": "synthetic publisher smoke test"},
        }
    ).encode()
    await EventGridPublisher(config).publish(body, "test-delivery")
