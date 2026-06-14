"""WebSocket endpoint for real-time document analysis job progress."""

import asyncio
from contextlib import suppress
from typing import Optional

import structlog
from fastapi import FastAPI, WebSocket
from fastapi import Depends
from sqlalchemy.orm import Session

from api.config import get_settings
from api.jwt_auth import verify_token, InvalidTokenError as JWTInvalidTokenError, TokenExpiredError as JWTTokenExpiredError
from api.limiter import enforce_rate_limit, RateLimitExceeded
from api.websockets.case_timeline import parse_auth_from_websocket
from services.job_realtime import job_realtime_bus, JobRealtimeBus
from db.session import get_db, apply_rls_context, clear_rls_context, _is_postgres
from api.job_registry import get_job_owner

logger = structlog.get_logger(__name__)

settings = get_settings()


async def forward_job_events(websocket: WebSocket, job_id: str, bus: JobRealtimeBus) -> None:
    """Subscribe to the job bus and forward events to the websocket.

    Sends an initial ``subscribed`` message, then loops awaiting messages
    from the bus.  Closes gracefully when the job reaches a terminal state.
    """
    await asyncio.wait_for(
        websocket.send_json({
            "event": "subscribed",
            "job_id": job_id,
            "message": "Listening for job progress events"
        }),
        timeout=5.0
    )
    queue = await bus.subscribe(job_id)
    try:
        while True:
            try:
                payload = await asyncio.wait_for(queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                # Send keepalive ping so client knows connection is alive
                await asyncio.wait_for(
                    websocket.send_json({"event": "ping", "job_id": job_id}),
                    timeout=5.0
                )
                continue

            await asyncio.wait_for(websocket.send_json(payload), timeout=5.0)

            # Terminal states close the connection naturally
            if payload.get("event") in ("completed", "failed", "cancelled"):
                break
    finally:
        await bus.unsubscribe(job_id, queue)


def register_job_progress_endpoint(app: FastAPI) -> None:
    """Register the ``/ws/jobs/{job_id}/progress`` endpoint on the given app."""

    @app.websocket("/ws/jobs/{job_id}/progress")
    async def websocket_job_progress_endpoint(
        websocket: WebSocket,
        job_id: str,
        db: Session = Depends(get_db),
    ):
        # Origin validation
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
            auth_token = websocket.query_params.get("token")
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

        # Authorization — verify user owns the job
        owner_id = get_job_owner(job_id)
        if owner_id is not None and owner_id != int(user_id):
            await websocket.close(code=1008, reason="Forbidden: You do not own this job")
            return

        # Rate limiting
        identifier = f"user:{user_id}"
        if websocket.client and websocket.client.host:
            identifier = f"{identifier}|ip:{websocket.client.host}"

        try:
            await enforce_rate_limit(
                identifier=identifier,
                endpoint=f"WS /ws/jobs/{job_id}/progress",
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

        # Apply RLS context for any DB queries in this session
        if _is_postgres:
            try:
                apply_rls_context(db, int(user_id))
            except (TypeError, ValueError):
                pass

        # Extract and propagate trace context from WebSocket headers
        from observability.instrumentation import use_extracted_trace_context, get_current_trace_headers
        incoming_trace = {
            key.lower(): value
            for key, value in websocket.headers.items()
            if key.lower() in {"traceparent", "tracestate", "baggage"}
        }
        
        # Forward events until job completes or client disconnects
        try:
            with use_extracted_trace_context(incoming_trace):
                await forward_job_events(websocket, job_id, job_realtime_bus)
        except Exception as e:
            logger.warning("websocket_job_progress_error", job_id=job_id, error=str(e))
        finally:
            if _is_postgres:
                with suppress(Exception):
                    clear_rls_context(db)