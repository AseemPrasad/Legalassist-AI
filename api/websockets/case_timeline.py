"""WebSocket endpoint for case timeline updates.

This module provides a reusable registration function for the FastAPI app.
It extracts authentication handling and event forwarding into helper
functions, making the logic testable and reusable.
"""

import asyncio
from contextlib import suppress
from typing import Optional

import jwt
import structlog
from fastapi import FastAPI, WebSocket
from fastapi import Depends
from api.config import get_settings
from api.limiter import enforce_rate_limit, RateLimitExceeded
from core.timeline_payloads import TimelineEventPayload, TimelineSubscribedPayload
from db.models.cases import Case
from db.session import get_db, apply_rls_context, clear_rls_context, _is_postgres
from services.timeline_realtime import timeline_realtime_bus, TimelineRealtimeBus
from sqlalchemy.orm import Session

logger = structlog.get_logger(__name__)


settings = get_settings()


class TokenExpiredError(Exception):
    pass


class InvalidTokenError(Exception):
    pass


class AuthError(Exception):
    pass


def _verify_token(token: str) -> dict:
    secrets_to_try = [settings.JWT_SECRET_KEY, settings.JWT_SECRET_KEY_PREVIOUS]
    secrets_to_try = [s for s in secrets_to_try if s and len(s.strip()) >= 16]

    payload = None
    last_error = None
    for secret in secrets_to_try:
        try:
            payload = jwt.decode(
                token,
                secret,
                algorithms=[settings.JWT_ALGORITHM],
                issuer=settings.JWT_ISSUER,
                audience=settings.JWT_AUDIENCE,
                options={"require": ["exp", "iat", "nbf", "iss", "aud", "jti", "type"], "verify_nbf": True},
            )
            break
        except jwt.ExpiredSignatureError as exc:
            last_error = exc
        except jwt.InvalidTokenError as exc:
            last_error = exc
            continue

    if payload is None:
        if isinstance(last_error, jwt.ExpiredSignatureError):
            raise TokenExpiredError("Token has expired") from last_error
        raise InvalidTokenError(str(last_error) if last_error else "Invalid token")

    if payload.get("type") != "access":
        raise InvalidTokenError("Invalid token type")

    jti = payload.get("jti")
    if jti:
        from api.jwt_auth import _is_token_revoked_cached
        if _is_token_revoked_cached(jti):
            logger.warning("websocket_token_revoked", jti=jti)
            raise InvalidTokenError("Token has been revoked")
    return payload


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


async def forward_timeline_events(websocket: WebSocket, case_id: int, bus: TimelineRealtimeBus) -> None:
    """Subscribe to the realtime bus and forward events to the websocket.

    Sends an initial ``subscribed`` message, then loops awaiting messages
    from the bus and forwards them as JSON objects.
    """
    await websocket.send_json(TimelineSubscribedPayload(case_id=case_id).model_dump(mode="json"))
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
        # Authentication
        auth_token = parse_auth_from_websocket(websocket)
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

        # Forward events
        await forward_timeline_events(websocket, case_id, timeline_realtime_bus)
