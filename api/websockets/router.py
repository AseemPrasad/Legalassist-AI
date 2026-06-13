"""
High-Availability WebSocket Router
=====================================
Central routing layer that connects:

  * Inbound WebSocket connections (FastAPI endpoints)
  * The :class:`~api.websockets.connection_manager.ConnectionManager` (per-process
    in-memory registry)
  * The :class:`~services.timeline_realtime.TimelineRealtimeBus` (case-scoped
    in-process pub/sub)
  * Optional Redis pub/sub bridge for **multi-instance routing** — when
    ``REDIS_URL`` is set the router subscribes to a Redis channel and
    relays messages to all locally connected WebSocket clients.  This
    enables sticky sessions across horizontally-scaled server instances.

Typical flow (single-instance)
-------------------------------
1. Client opens ``/ws/ha/cases/{case_id}`` with a valid auth or sticky token.
2. ``HAWebSocketRouter.handle_connection()`` is called.
3. Router registers the connection in :data:`connection_manager` and
   subscribes to ``case:{case_id}`` on the ``TimelineRealtimeBus``.
4. Messages published via ``timeline_realtime_bus.publish()`` are fanned-out
   to all subscribers of that case — no O(N²) message duplication.
5. On disconnect a ``ws_reconnect_required`` message is sent with a new
   sticky-session token and a back-off hint.

Multi-instance flow (Redis bridge)
------------------------------------
* On startup :meth:`HAWebSocketRouter.start_redis_bridge` subscribes to
  ``legalassist:ws:broadcast`` on Redis.
* Publishers (e.g. Celery workers) push JSON-serialised payloads to that
  channel.
* The bridge task receives the payload and calls
  :meth:`ConnectionManager.broadcast` so every *local* WebSocket subscriber
  receives the event without knowing which server instance produced it.

This replaces the naive O(N²) fan-out that would occur if every server
independently queried the database and pushed updates.
"""
from __future__ import annotations

import asyncio
import json
import os
from contextlib import suppress
from typing import Any, Dict, Optional

import structlog
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi import Depends
from sqlalchemy.orm import Session

from api.websockets.connection_manager import (
    ConnectionManager,
    ConnectionLimitExceeded,
    get_connection_manager,
)
from api.websockets.reconnection import (
    BackoffConfig,
    ReconnectBudget,
    build_reconnect_context,
    verify_sticky_token,
)
from db.session import get_db
from services.timeline_realtime import timeline_realtime_bus

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Helpers pulled from the existing case_timeline module (DRY)
# ---------------------------------------------------------------------------
from api.websockets.case_timeline import (
    parse_auth_from_websocket,
    _verify_token as _verify_auth_token,
    _require_owned_case,
    TokenExpiredError,
    InvalidTokenError,
)


# ---------------------------------------------------------------------------
# HAWebSocketRouter
# ---------------------------------------------------------------------------

class HAWebSocketRouter:
    """
    High-availability WebSocket routing engine.

    Parameters
    ----------
    manager:
        The :class:`ConnectionManager` singleton (injected for testability).
    jwt_secret:
        Secret used to sign/verify sticky-session tokens.  Defaults to
        ``settings.JWT_SECRET_KEY``.
    backoff_config:
        Optional custom backoff configuration for reconnect hints.
    """

    def __init__(
        self,
        manager: ConnectionManager,
        jwt_secret: str,
        backoff_config: Optional[BackoffConfig] = None,
    ) -> None:
        self._manager = manager
        self._jwt_secret = jwt_secret
        self._backoff_config = backoff_config or BackoffConfig()
        self._redis_bridge_task: Optional[asyncio.Task[None]] = None

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def handle_connection(
        self,
        *,
        websocket: WebSocket,
        case_id: int,
        db: Session,
    ) -> None:
        """
        Full lifecycle handler for a single HA WebSocket connection.

        1. Authenticate (auth JWT *or* sticky-session token)
        2. Authorise case ownership
        3. Register with :class:`ConnectionManager`
        4. Forward realtime events
        5. On disconnect — emit reconnect context to client
        """
        tenant_id, user_id = await self._authenticate(websocket)
        if tenant_id is None:
            return  # already closed with error code

        if not _require_owned_case(case_id, user_id, db):
            await websocket.close(code=1008, reason="Forbidden: You do not own this case")
            return

        channel_key = f"case:{case_id}"

        # ----------------------------------------------------------------
        # Register connection
        # ----------------------------------------------------------------
        try:
            queue = await self._manager.subscribe(
                tenant_id=tenant_id,
                channel_key=channel_key,
            )
        except ConnectionLimitExceeded as exc:
            logger.warning(
                "ws_ha_connection_limit",
                tenant_id=tenant_id,
                limit=exc.limit,
            )
            await websocket.close(
                code=1013,
                reason=f"Connection limit reached ({exc.limit}). Try again later.",
            )
            return

        # Accept *after* all validation passes to avoid half-open sockets.
        subprotocol = self._negotiate_subprotocol(websocket)
        await websocket.accept(subprotocol=subprotocol)

        # Send initial "subscribed" confirmation
        await websocket.send_json({
            "type": "subscribed",
            "channel": channel_key,
            "tenant_id": tenant_id,
            "max_connections": self._manager._max_connections_per_tenant,
            "tenant_connection_count": self._manager.tenant_count(tenant_id),
        })

        budget = ReconnectBudget(self._backoff_config)

        # Subscribe to the in-process realtime bus as well
        bus_queue = await timeline_realtime_bus.subscribe(case_id)
        bus_forward_task = asyncio.create_task(
            self._bridge_bus_to_manager(bus_queue, channel_key),
            name=f"bus_bridge:{channel_key}",
        )

        try:
            await self._forward_events(
                websocket=websocket,
                queue=queue,
                channel_key=channel_key,
            )
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            logger.error("ws_ha_error", channel_key=channel_key, error=str(exc))
        finally:
            bus_forward_task.cancel()
            with suppress(asyncio.CancelledError):
                await bus_forward_task
            await timeline_realtime_bus.unsubscribe(case_id, bus_queue)
            await self._manager.unsubscribe(
                tenant_id=tenant_id,
                channel_key=channel_key,
                queue=queue,
            )
            # Send reconnect advice before final close
            await self._send_reconnect_context(
                websocket=websocket,
                tenant_id=tenant_id,
                channel_key=channel_key,
                budget=budget,
            )

    # ------------------------------------------------------------------
    # Redis multi-instance bridge
    # ------------------------------------------------------------------

    async def start_redis_bridge(self, redis_url: str) -> None:
        """
        Start a background task that subscribes to Redis and relays
        messages to locally connected WebSocket clients.

        Messages on the ``legalassist:ws:broadcast`` channel must be
        JSON objects with at least:
        ``{"channel_key": "case:42", ...payload...}``
        """
        if self._redis_bridge_task is not None:
            return  # already running

        self._redis_bridge_task = asyncio.create_task(
            self._redis_bridge_loop(redis_url),
            name="ws_ha_redis_bridge",
        )
        logger.info("ws_ha_redis_bridge_started", redis_url=redis_url)

    async def stop_redis_bridge(self) -> None:
        """Cancel the Redis bridge task if running."""
        if self._redis_bridge_task is not None:
            self._redis_bridge_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._redis_bridge_task
            self._redis_bridge_task = None
            logger.info("ws_ha_redis_bridge_stopped")

    async def _redis_bridge_loop(self, redis_url: str) -> None:
        """Long-running coroutine that listens on Redis and fans out locally."""
        CHANNEL = "legalassist:ws:broadcast"
        try:
            import redis.asyncio as aioredis  # optional dependency
        except ImportError:
            logger.warning(
                "ws_ha_redis_bridge_unavailable",
                reason="redis[asyncio] not installed; multi-instance routing disabled",
            )
            return

        backoff = BackoffConfig(base_seconds=1, max_backoff_seconds=60, max_attempts=100)
        budget = ReconnectBudget(backoff)

        while budget.should_retry():
            try:
                client = aioredis.from_url(redis_url, decode_responses=True)
                pubsub = client.pubsub()
                await pubsub.subscribe(CHANNEL)
                logger.info("ws_ha_redis_subscribed", channel=CHANNEL)
                budget.reset()

                async for raw_message in pubsub.listen():
                    if raw_message["type"] != "message":
                        continue
                    try:
                        payload: Dict[str, Any] = json.loads(raw_message["data"])
                    except (json.JSONDecodeError, TypeError):
                        continue

                    channel_key = payload.get("channel_key")
                    if not channel_key:
                        continue

                    delivered = await self._manager.broadcast(channel_key, payload)
                    logger.debug(
                        "ws_ha_redis_relay",
                        channel_key=channel_key,
                        delivered=delivered,
                    )

            except asyncio.CancelledError:
                return
            except Exception as exc:
                delay = budget.next_delay()
                logger.warning(
                    "ws_ha_redis_bridge_reconnecting",
                    error=str(exc),
                    delay_seconds=round(delay, 2),
                )
                await asyncio.sleep(delay)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _authenticate(
        self, websocket: WebSocket
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Try sticky-session token first, then fall back to full auth JWT.

        Returns ``(tenant_id, user_id)`` or ``(None, None)`` on failure
        (the WebSocket will be closed before returning None).
        """
        # Check for sticky token in query params
        sticky_raw = websocket.query_params.get("sticky_token")
        if sticky_raw:
            decoded = verify_sticky_token(sticky_raw, secret=self._jwt_secret)
            if decoded:
                return decoded["tid"], decoded["tid"]
            # Invalid/expired sticky token → fall through to full auth
            logger.debug("ws_ha_sticky_token_invalid_fallback_to_auth")

        # Full auth token (Sec-WebSocket-Protocol header)
        auth_token = parse_auth_from_websocket(websocket)
        if not auth_token:
            await websocket.close(code=4001, reason="Authentication required")
            return None, None

        try:
            payload = _verify_auth_token(auth_token)
            user_id = payload.get("sub")
            if not user_id:
                await websocket.close(code=4003, reason="Invalid token: missing subject")
                return None, None
            return str(user_id), str(user_id)
        except (TokenExpiredError, InvalidTokenError):
            await websocket.close(code=4001, reason="Invalid or expired token")
            return None, None

    @staticmethod
    def _negotiate_subprotocol(websocket: WebSocket) -> Optional[str]:
        if "sec-websocket-protocol" in websocket.headers:
            protocols = [
                p.strip()
                for p in websocket.headers["sec-websocket-protocol"].split(",")
            ]
            if "access_token" in protocols:
                return "access_token"
        return None

    async def _forward_events(
        self,
        *,
        websocket: WebSocket,
        queue: "asyncio.Queue[Dict[str, Any]]",
        channel_key: str,
    ) -> None:
        """
        Pull messages from *queue* and push them to *websocket*.

        Listens for client disconnect by racing a ``receive()`` task
        against the queue.  Forwards SLOW_CONSUMER backpressure signals
        transparently so clients can self-throttle.
        """
        disconnect_task = asyncio.create_task(websocket.receive(), name="ws_disconnect_sentinel")
        try:
            while True:
                queue_task = asyncio.create_task(queue.get(), name="ws_queue_get")
                done, pending = await asyncio.wait(
                    {queue_task, disconnect_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if disconnect_task in done:
                    queue_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await queue_task
                    return

                msg = queue_task.result()
                await websocket.send_json(msg)
        finally:
            disconnect_task.cancel()
            with suppress(asyncio.CancelledError, RuntimeError):
                await disconnect_task

    async def _bridge_bus_to_manager(
        self,
        bus_queue: "asyncio.Queue[Dict[str, Any]]",
        channel_key: str,
    ) -> None:
        """Forward events from the ``TimelineRealtimeBus`` to the ``ConnectionManager``."""
        while True:
            try:
                payload = await bus_queue.get()
                await self._manager.broadcast(channel_key, payload)
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.warning("ws_ha_bus_bridge_error", channel_key=channel_key, error=str(exc))

    async def _send_reconnect_context(
        self,
        *,
        websocket: WebSocket,
        tenant_id: str,
        channel_key: str,
        budget: ReconnectBudget,
    ) -> None:
        ctx = build_reconnect_context(
            tenant_id=tenant_id,
            channel_key=channel_key,
            budget=budget,
            jwt_secret=self._jwt_secret,
            reason="connection_closed",
        )
        with suppress(Exception):
            await websocket.send_json({
                "type": "ws_reconnect_required",
                "should_reconnect": ctx.should_reconnect,
                "delay_hint_seconds": round(ctx.delay_hint_seconds, 3),
                "sticky_token": ctx.sticky_token,
                "reason": ctx.reason,
                "attempt": ctx.attempt,
            })


# ---------------------------------------------------------------------------
# Registration helper
# ---------------------------------------------------------------------------

def register_ha_websocket_endpoints(app: FastAPI) -> None:
    """
    Register HA WebSocket endpoints on *app*.

    Endpoints
    ---------
    ``/ws/ha/cases/{case_id}``
        High-availability real-time case updates with multiplexing,
        per-tenant connection limits, backpressure signals, and
        reconnect-token issuance.
    ``/ws/ha/stats``
        (Debug / ops) Returns connection manager statistics as JSON and
        immediately closes.  Gate behind a separate auth check in production.
    """
    from api.config import get_settings

    settings = get_settings()
    manager = get_connection_manager()
    router = HAWebSocketRouter(
        manager=manager,
        jwt_secret=settings.JWT_SECRET_KEY,
        backoff_config=BackoffConfig(
            base_seconds=float(os.getenv("WS_BACKOFF_BASE_SECONDS", "0.5")),
            max_backoff_seconds=float(os.getenv("WS_BACKOFF_MAX_SECONDS", "30")),
            max_attempts=int(os.getenv("WS_BACKOFF_MAX_ATTEMPTS", "20")),
        ),
    )

    # ---- Start Redis bridge if configured ----
    redis_url = os.getenv("REDIS_URL", "")

    @app.on_event("startup")
    async def _start_redis_bridge() -> None:
        if redis_url:
            await router.start_redis_bridge(redis_url)

    @app.on_event("shutdown")
    async def _stop_redis_bridge() -> None:
        await router.stop_redis_bridge()

    # ---- HA case endpoint ----
    @app.websocket("/ws/ha/cases/{case_id}")
    async def ha_case_websocket(
        websocket: WebSocket,
        case_id: int,
        db: Session = Depends(get_db),
    ) -> None:
        """
        HA WebSocket endpoint for real-time case timeline updates.

        Authentication
        --------------
        Pass a valid JWT in the ``Sec-WebSocket-Protocol`` header
        (``access_token, <token>``) **or** a short-lived sticky-session
        token in the ``?sticky_token=<token>`` query parameter.

        Backpressure
        ------------
        If the server-side queue for this connection fills beyond the
        configured threshold the client will receive a ``SLOW_CONSUMER``
        frame advising it to reduce consumption rate.

        Reconnection
        ------------
        On disconnect the server sends a ``ws_reconnect_required`` frame
        containing a ``sticky_token`` and a ``delay_hint_seconds`` value.
        The client should wait ``delay_hint_seconds`` before reconnecting
        using the sticky token.
        """
        await router.handle_connection(
            websocket=websocket,
            case_id=case_id,
            db=db,
        )

    # ---- Stats / ops endpoint ----
    @app.websocket("/ws/ha/stats")
    async def ha_stats_websocket(websocket: WebSocket) -> None:
        """
        WebSocket endpoint that streams connection-manager statistics.

        Returns a single JSON snapshot then closes. Intended for ops dashboards.
        """
        await websocket.accept()
        try:
            await websocket.send_json({
                "type": "ws_ha_stats",
                "total_dropped_messages": manager.total_dropped_messages,
                "total_slow_consumer_events": manager.total_slow_consumer_events,
            })
        finally:
            await websocket.close()
