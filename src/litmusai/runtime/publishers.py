"""Independent CloudEvents publishers with bounded attempts and optional SDK imports."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import importlib
import json
import time
from typing import Any, Protocol

import httpx

from litmusai.runtime.config import (
    Destination,
    EventGridDestination,
    KafkaDestination,
    WebhookDestination,
    secret,
)


class PublishError(Exception):
    """A safe error code and retry classification, with no provider response text."""

    def __init__(self, code: str, retryable: bool = True) -> None:
        super().__init__(code)
        self.code = code
        self.retryable = retryable


class AlertPublisher(Protocol):
    """A successful publish acknowledges transport acceptance, not consumer execution."""

    async def publish(self, body: bytes, delivery_id: str) -> None:
        """Publish the exact canonical CloudEvent body or raise PublishError."""
        ...


def sign_webhook(body: bytes, key: str, timestamp: str) -> str:
    """Sign the timestamp and exact body so receivers can reject stale replays."""
    digest = hmac.new(key.encode(), timestamp.encode() + b"." + body, hashlib.sha256).hexdigest()
    return f"v1={digest}"


def verify_webhook(
    body: bytes,
    key: str,
    timestamp: str,
    signature: str,
    *,
    tolerance: float = 300,
    now: float | None = None,
) -> bool:
    """Verify signature and timestamp; receivers must additionally deduplicate event IDs."""
    try:
        if abs((time.time() if now is None else now) - int(timestamp)) > tolerance:
            return False
        return hmac.compare_digest(sign_webhook(body, key, timestamp), signature)
    except (ValueError, OverflowError, TypeError):
        return False


class WebhookPublisher:
    """Publish structured CloudEvents to an explicitly configured HTTPS endpoint."""

    def __init__(self, config: WebhookDestination) -> None:
        self.config = config

    async def publish(self, body: bytes, delivery_id: str) -> None:
        """Accept only 2xx; do not follow redirects or retain a receiver's response body."""
        timestamp = str(int(time.time()))
        headers = {
            "Content-Type": "application/cloudevents+json",
            "X-Litmus-Timestamp": timestamp,
            "X-Litmus-Delivery-ID": delivery_id,
            "X-Litmus-Signature": sign_webhook(
                body, secret(self.config.signing_secret_env), timestamp
            ),
        }
        if self.config.bearer_token_env:
            headers["Authorization"] = "Bearer " + secret(self.config.bearer_token_env)
        async with httpx.AsyncClient(
            timeout=self.config.timeout_seconds,
            follow_redirects=False,
            trust_env=False,
        ) as client:
            async with client.stream(
                "POST", self.config.url, content=body, headers=headers
            ) as response:
                status = response.status_code
                if not 200 <= status < 300:
                    raise PublishError(f"http_{status}", status in {408, 429} or status >= 500)


class KafkaPublisher:
    """A bounded producer with idempotence inside a process and stable IDs across restarts."""

    def __init__(self, config: KafkaDestination) -> None:
        self.config = config
        self._producer: Any = None

    def _produce(self, body: bytes) -> None:
        kafka = importlib.import_module("confluent_kafka")
        config = self.config
        if self._producer is None:
            options: dict[str, Any] = {
                "bootstrap.servers": config.bootstrap_servers,
                "security.protocol": config.security_protocol,
                "enable.idempotence": True,
                "acks": "all",
                "message.timeout.ms": max(1000, int(config.timeout_seconds * 1000)),
                "socket.timeout.ms": max(1000, int(config.timeout_seconds * 1000)),
                "queue.buffering.max.messages": 100,
                "queue.buffering.max.kbytes": 1024,
                # librdkafka logs can contain connection details; expose safe counters instead.
                "log_level": 0,
            }
            if config.security_protocol == "SASL_SSL":
                options.update(
                    {
                        "sasl.mechanism": config.sasl_mechanism,
                        "sasl.username": secret(config.username_env or ""),
                        "sasl.password": secret(config.password_env or ""),
                    }
                )
            for key, value in (
                ("ssl.ca.location", config.ca_location),
                ("ssl.certificate.location", config.certificate_location),
                ("ssl.key.location", config.key_location),
            ):
                if value:
                    options[key] = value
            if config.key_password_env:
                options["ssl.key.password"] = secret(config.key_password_env)
            self._producer = kafka.Producer(options)
        receipt: list[Any] = []

        def acknowledged(error: Any, message: Any) -> None:
            receipt.append(error)

        event = json.loads(body)
        partition_key = json.dumps(
            [event["data"]["project_id"], event["data"]["session_id"]]
        ).encode()
        self._producer.produce(
            config.topic, value=body, key=partition_key, on_delivery=acknowledged
        )
        deadline = time.monotonic() + config.timeout_seconds + 0.5
        while not receipt and time.monotonic() < deadline:
            self._producer.poll(min(0.1, max(0, deadline - time.monotonic())))
        if not receipt:
            raise PublishError("kafka_timeout")
        if receipt[0] is not None:
            error = receipt[0]
            permanent_codes = {
                kafka.KafkaError.TOPIC_AUTHORIZATION_FAILED,
                kafka.KafkaError.CLUSTER_AUTHORIZATION_FAILED,
                kafka.KafkaError.SASL_AUTHENTICATION_FAILED,
                kafka.KafkaError.MSG_SIZE_TOO_LARGE,
            }
            raise PublishError("kafka_delivery_failed", error.code() not in permanent_codes)

    async def publish(self, body: bytes, delivery_id: str) -> None:
        """Wait for broker acknowledgement off the event loop; consumer completion is separate."""
        await asyncio.to_thread(self._produce, body)


class EventGridPublisher:
    """Publish CloudEvents using Azure's asynchronous SDK with one attempt per outbox attempt."""

    def __init__(self, config: EventGridDestination) -> None:
        self.config = config

    async def publish(self, body: bytes, delivery_id: str) -> None:
        """Preserve event identity and acknowledge only successful topic acceptance."""
        eventgrid = importlib.import_module("azure.eventgrid.aio")
        messaging = importlib.import_module("azure.core.messaging")
        credentials = importlib.import_module("azure.core.credentials")
        config = self.config
        identity = None
        if config.access_key_env:
            credential = credentials.AzureKeyCredential(secret(config.access_key_env))
        else:
            identity_module = importlib.import_module("azure.identity.aio")
            identity = identity_module.DefaultAzureCredential(
                managed_identity_client_id=config.managed_identity_client_id,
                exclude_interactive_browser_credential=True,
            )
            credential = identity
        try:
            async with eventgrid.EventGridPublisherClient(
                config.endpoint,
                credential,
                retry_total=0,
                connection_timeout=config.timeout_seconds,
                read_timeout=config.timeout_seconds,
                logging_enable=False,
            ) as client:
                await client.send([messaging.CloudEvent.from_dict(json.loads(body))])
        except Exception as error:
            status = getattr(error, "status_code", None)
            raise PublishError(
                "event_grid_publish_failed", status is None or status in {408, 429} or status >= 500
            ) from None
        finally:
            if identity:
                await identity.close()


def publisher_for(config: Destination) -> AlertPublisher:
    """Construct an adapter without importing optional Kafka/Azure libraries until publication."""
    if isinstance(config, WebhookDestination):
        return WebhookPublisher(config)
    if isinstance(config, KafkaDestination):
        return KafkaPublisher(config)
    return EventGridPublisher(config)
