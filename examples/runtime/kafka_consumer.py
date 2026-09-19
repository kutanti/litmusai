"""Run: python -m examples.runtime.kafka_consumer; install litmuseval[runtime-kafka]."""

import os

from confluent_kafka import Consumer

from examples.runtime.consumer import ConsumerState


def main() -> None:
    """Consume with TLS and commit offsets only after durable, idempotent processing."""
    config = {
        "bootstrap.servers": os.environ["KAFKA_BOOTSTRAP_SERVERS"],
        "group.id": "litmus-threat-response",
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False,
        "security.protocol": "SSL",
    }
    if "KAFKA_USERNAME" in os.environ:
        config.update(
            {
                "security.protocol": "SASL_SSL",
                "sasl.mechanism": "PLAIN",
                "sasl.username": os.environ["KAFKA_USERNAME"],
                "sasl.password": os.environ["KAFKA_PASSWORD"],
            }
        )
    consumer = Consumer(config)
    state = ConsumerState(".litmus/kafka-consumer.sqlite")
    consumer.subscribe([os.environ.get("KAFKA_TOPIC", "agent-threats")])
    try:
        while True:
            message = consumer.poll(1)
            if message is None:
                continue
            if message.error():
                raise RuntimeError("Kafka receive failed; inspect broker health")
            state.consume(message.value())
            consumer.commit(message=message, asynchronous=False)
    finally:
        consumer.close()


if __name__ == "__main__":
    main()
