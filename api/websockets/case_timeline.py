"""WebSocket endpoint for case timeline updates.

This module provides a reusable registration function for the FastAPI app.
It extracts authentication handling and event forwarding into helper
functions, making the logic testable and reusable.
"""

import asyncio
from contextlib import suppress
from typing import Optional

import structlog
from fastapi import FastAPI, WebSocket
from fastapi import Depends
from api.config import get_settings
from api.jwt_auth import verify_token, InvalidTokenError as JWTInvalidTokenError, TokenExpiredError as JWTTokenExpiredError
from api.limiter import enforce_rate_limit, RateLimitExceeded
from core.timeline_payloads import TimelineEventPayload, TimelineSubscribedPayload
from db.models.cases import Case
from db.session import get_db, apply_rls_context, clear_rls_context, _is_postgres
from services.timeline_realtime import timeline_realtime_bus, TimelineRealtimeBus
from sqlalchemy.orm import Session

logger = structlog.get_logger(__name__)


settings = get_settings()


def parse_auth_from_websocket(websocket: WebSocket) -> Optional[str]:
    """Extract the auth token from the Sec-WebSocket-Protocol header.

    Returns ``None`` if no token is found.
    """
    if "sec-websocket-protocol" in websocket.headers:
        header_val = websocket.headers["sec-websocket-protocol"]
        requested_protocols = [p.strip() for p in header_val.split(",")]
        if "access_token" in requested_protocols:
            idx = requested_protocols.index("access_token")
            if idx + 1 < len(requested_protocols):
                return requested_protocols[idx + 1]
    return None


async def forward_timeline_events(websocket: WebSocket, case_id: int, user_id: str, bus: TimelineRealtimeBus, db: Session) -> None:
    """Subscribe to the realtime bus and forward events to the websocket.

    Sends an initial ``subscribed`` message, then loops awaiting messages
    from the bus and forwards them as JSON objects.  Every 60 seconds
    the function revalidates that the user still owns the case; if
    ownership was revoked the connection is closed.
    """
    REVALIDATION_INTERVAL = 60  # seconds

    await websocket.send_json(TimelineSubscribedPayload(case_id=case_id).model_dump(mode="json"))
    queue = await bus.subscribe(case_id)
    disconnect_task = asyncio.create_task(websocket.receive())
    try:
        while True:
            queue_task = asyncio.create_task(queue.get())
            done, pending = await asyncio.wait(
                {queue_task, disconnect_task},
                return_when=asyncio.FIRST_COMPLETED,
                timeout=REVALIDATION_INTERVAL,
            )

            if disconnect_task in done:
                queue_task.cancel()
                with suppress(asyncio.CancelledError):
                    await queue_task
                break

            if queue_task in done:
                payload_obj = TimelineEventPayload.model_validate(queue_task.result())
                await websocket.send_json(payload_obj.model_dump(mode="json"))
            else:
                # Timeout — revalidate authorization
                queue_task.cancel()
                with suppress(asyncio.CancelledError):
                    await queue_task
                if not _require_owned_case(case_id, user_id, db):
                    await websocket.close(code=1008, reason="Access revoked")
                    break
    finally:
        disconnect_task.cancel()
        with suppress(asyncio.CancelledError, RuntimeError):
            await disconnect_task
        await bus.unsubscribe(case_id, queue)


def _require_owned_case(case_id: int, user_id: str, db: Session) -> bool:
    try:
        owner_id = int(user_id)
    except (TypeError, ValueError):
        return False

    case = db.query(Case).filter(Case.id == case_id, Case.user_id == owner_id).first()
    return case is not None


def register_case_timeline_endpoint(app: FastAPI) -> None:
    """Register the ``/ws/cases/{case_id}/timeline`` endpoint on the given app.
    """

    @app.websocket("/ws/cases/{case_id}/timeline")
    async def websocket_case_timeline_endpoint(
        websocket: WebSocket,
        case_id: int,
        db: Session = Depends(get_db),
    ):
        # Origin validation — reject connections without an Origin header
        origin = websocket.headers.get("origin")
        if not origin:
            await websocket.close(code=4001, reason="Origin header required")
            return
        allowed = settings.CORS_ORIGINS + [f"https://{h}" for h in settings.ALLOWED_HOSTS] + [f"http://{h}" for h in settings.ALLOWED_HOSTS]
        if origin not in allowed and "*" not in settings.CORS_ORIGINS:
            await websocket.close(code=4001, reason="Origin not allowed")
            return

        # Authentication
        auth_token = parse_auth_from_websocket(websocket)
        if not auth_token:
            await websocket.close(code=4001, reason="Authentication required")
            return
        try:
            payload = verify_token(auth_token)
            user_id = payload.get("sub")
            if not user_id:
                await websocket.close(code=4003, reason="Invalid token")
                return
        except (JWTTokenExpiredError, JWTInvalidTokenError):
            await websocket.close(code=4001, reason="Invalid or expired token")
            return

        if not _require_owned_case(case_id, user_id, db):
            await websocket.close(code=1008, reason="Forbidden: You do not own this case")
            return

        # Apply RLS context so all DB queries in this WebSocket session are
        # scoped to the authenticated user at the database level.
        if _is_postgres:
            try:
                apply_rls_context(db, int(user_id))
            except (TypeError, ValueError):
                pass

        identifier = f"user:{user_id}"
        if websocket.client and websocket.client.host:
            identifier = f"{identifier}|ip:{websocket.client.host}"

        try:
            await enforce_rate_limit(
                identifier=identifier,
                endpoint=f"WS /ws/cases/{case_id}/timeline",
                limit=settings.WEBSOCKET_RATE_LIMIT_REQUESTS,
                window_seconds=settings.WEBSOCKET_RATE_LIMIT_WINDOW,
            )
        except RateLimitExceeded as exc:
            await websocket.close(code=1013, reason=exc.detail["message"])
            return

        # Subprotocol handling
        requested_protocols = []
        if "sec-websocket-protocol" in websocket.headers:
            header_val = websocket.headers["sec-websocket-protocol"]
            requested_protocols = [p.strip() for p in header_val.split(",")]
        subprotocol = "access_token" if "access_token" in requested_protocols else None
        await websocket.accept(subprotocol=subprotocol)

        # Forward events with periodic authorization revalidation
        await forward_timeline_events(websocket, case_id, user_id, timeline_realtime_bus, db)
