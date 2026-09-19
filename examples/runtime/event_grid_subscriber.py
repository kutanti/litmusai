"""CloudEvents subscriber; deploy behind HTTPS with an Event Grid delivery secret header."""

import hmac
import json
import os

from fastapi import FastAPI, HTTPException, Request, Response

from examples.runtime.consumer import ConsumerState

app = FastAPI()


@app.options("/events")
async def validate(request: Request) -> Response:
    """Handle CloudEvents webhook validation for Azure Event Grid."""
    if request.headers.get("WebHook-Request-Origin") != "eventgrid.azure.net":
        raise HTTPException(403, "unknown origin")
    return Response(headers={"WebHook-Allowed-Origin": "eventgrid.azure.net"})


@app.post("/events")
async def events(request: Request) -> dict[str, bool]:
    """Authenticate the subscription's configured secret delivery header and store revisions."""
    supplied = request.headers.get("X-Litmus-Subscription-Token", "")
    if not hmac.compare_digest(
        supplied.encode(), os.environ["EVENT_GRID_SUBSCRIPTION_TOKEN"].encode()
    ):
        raise HTTPException(401, "invalid subscription token")
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > 262144:
            raise HTTPException(413, "event exceeds limit")
    state = ConsumerState(".litmus/event-grid-consumer.sqlite")
    try:
        payload = json.loads(body)
        for event in payload if isinstance(payload, list) else [payload]:
            state.consume(json.dumps(event).encode())
    except ValueError:
        raise HTTPException(400, "invalid event") from None
    return {"accepted": True}
