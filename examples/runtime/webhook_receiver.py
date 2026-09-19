"""Run: uvicorn examples.runtime.webhook_receiver:app --port 8766 --no-access-log."""

import os

from fastapi import FastAPI, HTTPException, Request

from examples.runtime.consumer import ConsumerState
from litmusai.runtime.publishers import verify_webhook

app = FastAPI()


@app.post("/events")
async def events(request: Request) -> dict[str, bool]:
    """Verify exact bytes and replay window before accepting the event."""
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > 262144:
            raise HTTPException(413, "event exceeds limit")
    if not verify_webhook(
        bytes(body),
        os.environ["LITMUS_WEBHOOK_SECRET"],
        request.headers.get("X-Litmus-Timestamp", ""),
        request.headers.get("X-Litmus-Signature", ""),
    ):
        raise HTTPException(401, "invalid signature")
    state = ConsumerState(os.environ.get("LITMUS_CONSUMER_DB", ".litmus/webhook-consumer.sqlite"))
    try:
        state.consume(bytes(body))
    except ValueError:
        raise HTTPException(400, "invalid event") from None
    return {"accepted": True}
