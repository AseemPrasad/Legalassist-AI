"""
High-Availability WebSocket Connection Manager
===============================================
Provides tenant-aware, per-channel multiplexed connection management with:

  * Per-tenant connection limits to prevent N^2 fan-out
  * Per-queue backpressure signals (SLOW_CONSUMER / QUEUE_FULL events)
  * Drop-oldest-keep-newest policy when queues overflow
  * Thread-safe counters for observability

Used by the HA WebSocket router to manage all active subscriptions.
"""
from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set

import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Defaults (overridden by env / config in ConnectionManager.__init__)
# ---------------------------------------------------------------------------
DEFAULT_MAX_CONNECTIONS_PER_TENANT: int = 50
DEFAULT_QUEUE_MAX_SIZE: int = 128
DEFAULT_SLOW_CONSUMER_THRESHOLD: int = 80  # % of queue full → warn


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _WebSocketSubscriber:
    """Immutable subscriber record — one per active WebSocket connection."""
    tenant_id: str
    channel_key: str          # e.g. "case:42" or "notifications:user:7"
    queue: "asyncio.Queue[Dict[str, Any]]"
    connected_at: float = field(default_factory=time.monotonic)
    loop: asyncio.AbstractEventLoop = field(default_factory=asyncio.get_event_loop)
    thread_id: int = field(default_factory=threading.get_ident)

    def __hash__(self) -> int:  # frozen + explicit hash so we can use Sets
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other


@dataclass
class _Channel:
    """A broadcast channel for a single (tenant, channel_key) pair."""
    subscribers: Set[_WebSocketSubscriber] = field(default_factory=set)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # backpressure metrics
    dropped_total: int = 0
    slow_consumer_events: int = 0


# ---------------------------------------------------------------------------
# ConnectionManager
# ---------------------------------------------------------------------------

class ConnectionManager:
    """
    Thread-safe, multi-tenant WebSocket connection registry.

    Responsibilities
    ----------------
    * ``subscribe()``   — register a new WebSocket connection and return its queue
    * ``unsubscribe()`` — deregister a connection and clean up empty channels
    * ``broadcast()``   — fan-out a message to all subscribers of a channel
    * ``tenant_count()``— return the number of active connections for a tenant

    Backpressure
    ------------
    When a subscriber's queue exceeds ``slow_consumer_threshold``% capacity a
    ``SLOW_CONSUMER`` warning is injected into the queue.  When the queue is
    completely full the oldest item is evicted (drop-oldest-keep-newest policy)
    and a drop counter is incremented.
    """

    def __init__(
        self,
        max_connections_per_tenant: int = DEFAULT_MAX_CONNECTIONS_PER_TENANT,
        queue_max_size: int = DEFAULT_QUEUE_MAX_SIZE,
        slow_consumer_threshold: int = DEFAULT_SLOW_CONSUMER_THRESHOLD,
    ) -> None:
        self._max_connections_per_tenant = max(1, max_connections_per_tenant)
        self._queue_max_size = max(4, queue_max_size)
        self._slow_consumer_threshold = max(0, min(100, slow_consumer_threshold))

        # channel_key → _Channel
        self._channels: Dict[str, _Channel] = {}
        self._global_lock = asyncio.Lock()

        # tenant_id → connection count
        self._tenant_counts: Dict[str, int] = {}
        self._count_lock = threading.Lock()

        # global drop / slow-consumer totals
        self._total_dropped: int = 0
        self._total_slow_consumer_events: int = 0
        self._metric_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def subscribe(
        self,
        *,
        tenant_id: str,
        channel_key: str,
    ) -> asyncio.Queue[Dict[str, Any]]:
        """
        Register a new subscriber for *channel_key* under *tenant_id*.

        Raises
        ------
        ConnectionLimitExceeded
            When the tenant has reached ``max_connections_per_tenant``.
        """
        with self._count_lock:
            current = self._tenant_counts.get(tenant_id, 0)
            if current >= self._max_connections_per_tenant:
                raise ConnectionLimitExceeded(
                    tenant_id=tenant_id,
                    limit=self._max_connections_per_tenant,
                )
            self._tenant_counts[tenant_id] = current + 1

        channel = await self._get_or_create_channel(channel_key)
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=self._queue_max_size)
        subscriber = _WebSocketSubscriber(
            tenant_id=tenant_id,
            channel_key=channel_key,
            queue=queue,
            connected_at=time.monotonic(),
            loop=loop,
            thread_id=threading.get_ident(),
        )
        async with channel.lock:
            channel.subscribers.add(subscriber)

        logger.info(
            "ws_connected",
            tenant_id=tenant_id,
            channel_key=channel_key,
            tenant_connection_count=self._tenant_counts[tenant_id],
        )
        return queue

    async def unsubscribe(
        self,
        *,
        tenant_id: str,
        channel_key: str,
        queue: "asyncio.Queue[Dict[str, Any]]",
    ) -> None:
        """Deregister a subscriber and clean up empty channels."""
        async with self._global_lock:
            channel = self._channels.get(channel_key)

        if channel is None:
            return

        async with channel.lock:
            channel.subscribers = {s for s in channel.subscribers if s.queue is not queue}
            is_empty = len(channel.subscribers) == 0

        if is_empty:
            async with self._global_lock:
                if self._channels.get(channel_key) is channel and len(channel.subscribers) == 0:
                    del self._channels[channel_key]

        with self._count_lock:
            count = self._tenant_counts.get(tenant_id, 0)
            if count > 0:
                self._tenant_counts[tenant_id] = count - 1
            if self._tenant_counts.get(tenant_id, 0) == 0:
                self._tenant_counts.pop(tenant_id, None)

        logger.info(
            "ws_disconnected",
            tenant_id=tenant_id,
            channel_key=channel_key,
        )

    async def broadcast(
        self,
        channel_key: str,
        message: Dict[str, Any],
    ) -> int:
        """
        Fan-out *message* to all active subscribers on *channel_key*.

        Returns the number of subscribers the message was delivered to
        (excluding dropped / back-pressured connections).
        """
        async with self._global_lock:
            channel = self._channels.get(channel_key)

        if channel is None:
            return 0

        current_loop = asyncio.get_running_loop()

        async with channel.lock:
            targets = list(channel.subscribers)
            # remove dead loops proactively
            dead = [s for s in targets if s.loop.is_closed()]
            if dead:
                channel.subscribers.difference_update(dead)
                targets = [s for s in targets if not s.loop.is_closed()]

        delivered = 0
        threshold_size = int(self._queue_max_size * self._slow_consumer_threshold / 100)

        for subscriber in targets:
            q = subscriber.queue
            loop = subscriber.loop

            deliver = lambda q=q, ch=channel: self._put_with_backpressure(
                queue=q, message=message, channel=ch, threshold_size=threshold_size
            )

            if loop is current_loop:
                result = deliver()
                if result:
                    delivered += 1
            else:
                if not loop.is_closed():
                    try:
                        loop.call_soon_threadsafe(deliver)
                        delivered += 1
                    except RuntimeError:
                        logger.warning("ws_dead_loop_detected", channel_key=channel_key)

        return delivered

    def tenant_count(self, tenant_id: str) -> int:
        """Return the number of active connections for *tenant_id*."""
        with self._count_lock:
            return self._tenant_counts.get(tenant_id, 0)

    def channel_subscriber_count(self, channel_key: str) -> int:
        """Return the number of subscribers currently on *channel_key*."""
        channel = self._channels.get(channel_key)
        if channel is None:
            return 0
        return len(channel.subscribers)

    @property
    def total_dropped_messages(self) -> int:
        with self._metric_lock:
            return self._total_dropped

    @property
    def total_slow_consumer_events(self) -> int:
        with self._metric_lock:
            return self._total_slow_consumer_events

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _get_or_create_channel(self, channel_key: str) -> _Channel:
        async with self._global_lock:
            if channel_key not in self._channels:
                self._channels[channel_key] = _Channel()
            return self._channels[channel_key]

    def _put_with_backpressure(
        self,
        *,
        queue: "asyncio.Queue[Dict[str, Any]]",
        message: Dict[str, Any],
        channel: _Channel,
        threshold_size: int,
    ) -> bool:
        """
        Attempt to place *message* on *queue*.

        Behaviour
        ---------
        * If queue size >= threshold → inject SLOW_CONSUMER signal (once)
        * If queue is full → evict oldest, increment drop counter, put newest
        Returns True when message was placed (possibly after eviction).
        """
        qsize = queue.qsize()

        if qsize >= threshold_size:
            # Inject backpressure signal if not already in queue
            try:
                queue.put_nowait({"type": "SLOW_CONSUMER", "queue_size": qsize, "max": self._queue_max_size})
                channel.slow_consumer_events += 1
                with self._metric_lock:
                    self._total_slow_consumer_events += 1
            except asyncio.QueueFull:
                pass  # will be handled below

        if queue.full():
            try:
                queue.get_nowait()  # evict oldest
            except asyncio.QueueEmpty:
                pass
            else:
                channel.dropped_total += 1
                with self._metric_lock:
                    self._total_dropped += 1
                logger.warning(
                    "ws_queue_overflow_drop",
                    policy="drop_oldest_keep_newest",
                    channel_key=channel,
                )

        try:
            queue.put_nowait(message)
            return True
        except asyncio.QueueFull:
            # Race between eviction and put — give up on this message
            with self._metric_lock:
                self._total_dropped += 1
            return False


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ConnectionLimitExceeded(Exception):
    """Raised when a tenant exceeds the per-tenant connection limit."""

    def __init__(self, *, tenant_id: str, limit: int) -> None:
        self.tenant_id = tenant_id
        self.limit = limit
        super().__init__(f"Tenant '{tenant_id}' has reached the connection limit of {limit}.")


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

import os

_connection_manager: Optional[ConnectionManager] = None
_init_lock = threading.Lock()


def get_connection_manager() -> ConnectionManager:
    """Return the process-level singleton :class:`ConnectionManager`."""
    global _connection_manager
    if _connection_manager is None:
        with _init_lock:
            if _connection_manager is None:
                _connection_manager = ConnectionManager(
                    max_connections_per_tenant=int(
                        os.getenv("WS_MAX_CONNECTIONS_PER_TENANT", str(DEFAULT_MAX_CONNECTIONS_PER_TENANT))
                    ),
                    queue_max_size=int(
                        os.getenv("WS_QUEUE_MAX_SIZE", str(DEFAULT_QUEUE_MAX_SIZE))
                    ),
                    slow_consumer_threshold=int(
                        os.getenv("WS_SLOW_CONSUMER_THRESHOLD_PCT", str(DEFAULT_SLOW_CONSUMER_THRESHOLD))
                    ),
                )
    return _connection_manager
