"""WebSocket endpoint for case timeline updates.

This module provides a reusable registration function for the FastAPI app.
It extracts authentication handling and event forwarding into helper
functions, making the logic testable and reusable.
"""

import asyncio
from contextlib import suppress

from fastapi import FastAPI, WebSocket, Query
from fastapi import Depends
from api.auth import AuthError, TokenExpiredError, InvalidTokenError, verify_token as _verify_token
from core.timeline_payloads import TimelineEventPayload
from services.timeline_realtime import timeline_realtime_bus, TimelineRealtimeBus
from typing import Optional


def parse_auth_from_websocket(websocket: WebSocket, token: Optional[str] = None) -> Optional[str]:
    """Extract the auth token from either the query parameter or the Sec-WebSocket-Protocol header.

    The logic mirrors the original implementation in ``api/main.py``.
    Returns ``None`` if no token is found.
    """
    auth_token = token
    requested_protocols = []

    if "sec-websocket-protocol" in websocket.headers:
        header_val = websocket.headers["sec-websocket-protocol"]
        requested_protocols = [p.strip() for p in header_val.split(",")]
        if "access_token" in requested_protocols:
            idx = requested_protocols.index("access_token")
            if idx + 1 < len(requested_protocols):
                auth_token = requested_protocols[idx + 1]
    return auth_token


async def forward_timeline_events(websocket: WebSocket, case_id: int, bus: TimelineRealtimeBus) -> None:
    """Subscribe to the realtime bus and forward events to the websocket.

    Sends an initial ``subscribed`` message, then loops awaiting messages
    from the bus and forwards them as JSON objects.
    """
    await websocket.send_json({"type": "subscribed", "case_id": case_id})
    queue = await bus.subscribe(case_id)
    disconnect_task = asyncio.create_task(websocket.receive())
    try:
        while True:
            queue_task = asyncio.create_task(queue.get())
            done, pending = await asyncio.wait(
                {queue_task, disconnect_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if disconnect_task in done:
                queue_task.cancel()
                with suppress(asyncio.CancelledError):
                    await queue_task
                break

            payload_obj = TimelineEventPayload.model_validate(queue_task.result())
            await websocket.send_json(payload_obj.model_dump(mode="json"))
    finally:
        disconnect_task.cancel()
        with suppress(asyncio.CancelledError, RuntimeError):
            await disconnect_task
        await bus.unsubscribe(case_id, queue)


def register_case_timeline_endpoint(app: FastAPI) -> None:
    """Register the ``/ws/cases/{case_id}/timeline`` endpoint on the given app.
    """

    @app.websocket("/ws/cases/{case_id}/timeline")
    async def websocket_case_timeline_endpoint(
        websocket: WebSocket,
        case_id: int,
        token: Optional[str] = Query(None),
    ):
        # Authentication
        auth_token = parse_auth_from_websocket(websocket, token)
        if not auth_token:
            await websocket.close(code=4001, reason="Authentication required")
            return
        try:
            payload = _verify_token(auth_token)
            user_id = payload.get("sub")
            if not user_id:
                await websocket.close(code=4003, reason="Invalid token")
                return
        except (TokenExpiredError, InvalidTokenError, AuthError):
            await websocket.close(code=4001, reason="Invalid or expired token")
            return

        # Subprotocol handling
        requested_protocols = []
        if "sec-websocket-protocol" in websocket.headers:
            header_val = websocket.headers["sec-websocket-protocol"]
            requested_protocols = [p.strip() for p in header_val.split(",")]
        subprotocol = "access_token" if "access_token" in requested_protocols else None
        await websocket.accept(subprotocol=subprotocol)

        # Forward events
        await forward_timeline_events(websocket, case_id, timeline_realtime_bus)
