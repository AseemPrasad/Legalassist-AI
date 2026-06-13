"""
Integration tests for HA WebSocket routing (Issue 2)
======================================================
Tests cover:

* :class:`~api.websockets.connection_manager.ConnectionManager`
  - Subscribe / unsubscribe lifecycle
  - Per-tenant connection limits
  - Backpressure: slow-consumer detection and drop-oldest eviction
  - Fan-out to many simulated clients

* :class:`~api.websockets.reconnection`
  - Exponential backoff delay properties
  - Sticky-session token issuance and verification
  - ReconnectBudget lifecycle

* :class:`~api.websockets.router.HAWebSocketRouter`
  - Simulated failover: connection drops, reconnect with sticky token
  - Multi-client fan-out without N^2 duplication

All tests are pure asyncio unit/integration tests and require no running
server instance.
"""
from __future__ import annotations

import asyncio
import math
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from api.websockets.connection_manager import (
    ConnectionLimitExceeded,
    ConnectionManager,
)
from api.websockets.reconnection import (
    BackoffConfig,
    ReconnectBudget,
    build_reconnect_context,
    compute_backoff_delay,
    issue_sticky_token,
    verify_sticky_token,
)


# ============================================================================
# Helpers
# ============================================================================

_SECRET = "test-secret-key-at-least-sixteen-chars"


def _make_manager(
    max_connections: int = 10,
    queue_max: int = 8,
    slow_threshold: int = 75,
) -> ConnectionManager:
    return ConnectionManager(
        max_connections_per_tenant=max_connections,
        queue_max_size=queue_max,
        slow_consumer_threshold=slow_threshold,
    )


# ============================================================================
# ConnectionManager tests
# ============================================================================

class TestConnectionManagerSubscribeUnsubscribe:
    @pytest.mark.asyncio
    async def test_subscribe_returns_queue(self):
        mgr = _make_manager()
        q = await mgr.subscribe(tenant_id="t1", channel_key="case:1")
        assert q is not None
        assert mgr.tenant_count("t1") == 1
        assert mgr.channel_subscriber_count("case:1") == 1

    @pytest.mark.asyncio
    async def test_unsubscribe_decrements_count(self):
        mgr = _make_manager()
        q = await mgr.subscribe(tenant_id="t1", channel_key="case:1")
        await mgr.unsubscribe(tenant_id="t1", channel_key="case:1", queue=q)
        assert mgr.tenant_count("t1") == 0
        assert mgr.channel_subscriber_count("case:1") == 0

    @pytest.mark.asyncio
    async def test_multiple_subscriptions_same_channel(self):
        mgr = _make_manager()
        q1 = await mgr.subscribe(tenant_id="t1", channel_key="case:5")
        q2 = await mgr.subscribe(tenant_id="t1", channel_key="case:5")
        assert q1 is not q2
        assert mgr.channel_subscriber_count("case:5") == 2
        assert mgr.tenant_count("t1") == 2

    @pytest.mark.asyncio
    async def test_channel_cleaned_up_when_empty(self):
        mgr = _make_manager()
        q = await mgr.subscribe(tenant_id="t1", channel_key="case:9")
        await mgr.unsubscribe(tenant_id="t1", channel_key="case:9", queue=q)
        assert "case:9" not in mgr._channels

    @pytest.mark.asyncio
    async def test_unsubscribe_nonexistent_channel_is_noop(self):
        mgr = _make_manager()
        q: asyncio.Queue = asyncio.Queue()
        # Should not raise
        await mgr.unsubscribe(tenant_id="t1", channel_key="nonexistent:99", queue=q)


class TestConnectionManagerLimits:
    @pytest.mark.asyncio
    async def test_connection_limit_raises(self):
        mgr = _make_manager(max_connections=2)
        await mgr.subscribe(tenant_id="t1", channel_key="case:1")
        await mgr.subscribe(tenant_id="t1", channel_key="case:2")
        with pytest.raises(ConnectionLimitExceeded) as exc_info:
            await mgr.subscribe(tenant_id="t1", channel_key="case:3")
        assert exc_info.value.limit == 2
        assert exc_info.value.tenant_id == "t1"

    @pytest.mark.asyncio
    async def test_connection_limit_per_tenant_independent(self):
        mgr = _make_manager(max_connections=2)
        await mgr.subscribe(tenant_id="t1", channel_key="case:1")
        await mgr.subscribe(tenant_id="t1", channel_key="case:2")
        # t2 should be independent
        q = await mgr.subscribe(tenant_id="t2", channel_key="case:1")
        assert mgr.tenant_count("t2") == 1


class TestConnectionManagerBroadcast:
    @pytest.mark.asyncio
    async def test_broadcast_delivers_to_all_subscribers(self):
        mgr = _make_manager()
        queues = []
        n = 5
        for i in range(n):
            q = await mgr.subscribe(tenant_id=f"t{i}", channel_key="case:42")
            queues.append(q)

        delivered = await mgr.broadcast("case:42", {"type": "test", "data": "hello"})
        assert delivered == n

        for q in queues:
            msg = q.get_nowait()
            assert msg["data"] == "hello"

    @pytest.mark.asyncio
    async def test_broadcast_to_unknown_channel_returns_zero(self):
        mgr = _make_manager()
        delivered = await mgr.broadcast("case:unknown", {"type": "test"})
        assert delivered == 0

    @pytest.mark.asyncio
    async def test_backpressure_drop_oldest(self):
        """When queue is full, oldest message is dropped and newest is kept."""
        mgr = _make_manager(queue_max=4, slow_threshold=100)  # threshold at 100% → no SLOW_CONSUMER
        q = await mgr.subscribe(tenant_id="t1", channel_key="case:1")

        # Fill queue
        for i in range(4):
            await mgr.broadcast("case:1", {"seq": i})

        # One more — should evict seq:0 and insert seq:4
        await mgr.broadcast("case:1", {"seq": 4})

        messages = []
        while not q.empty():
            messages.append(q.get_nowait())

        seqs = [m["seq"] for m in messages if "seq" in m]
        assert 0 not in seqs, "seq:0 should have been evicted (drop-oldest)"
        assert 4 in seqs, "seq:4 should be present (keep-newest)"
        assert mgr.total_dropped_messages >= 1

    @pytest.mark.asyncio
    async def test_slow_consumer_signal_injected(self):
        """SLOW_CONSUMER frame should appear when queue exceeds threshold."""
        mgr = _make_manager(queue_max=8, slow_threshold=50)  # threshold at 50% → 4 items
        q = await mgr.subscribe(tenant_id="t1", channel_key="case:2")

        # Send 5 messages — 5th should trigger SLOW_CONSUMER (> 50% of 8)
        for i in range(5):
            await mgr.broadcast("case:2", {"seq": i})

        messages = []
        while not q.empty():
            messages.append(q.get_nowait())

        types = [m.get("type") for m in messages]
        assert "SLOW_CONSUMER" in types


class TestManyClients:
    @pytest.mark.asyncio
    async def test_fan_out_many_clients_no_n_squared(self):
        """Ensure linear fan-out: each client gets exactly one copy."""
        n_clients = 30
        mgr = _make_manager(max_connections=n_clients + 1)
        queues = []
        for i in range(n_clients):
            q = await mgr.subscribe(tenant_id=f"user_{i}", channel_key="case:100")
            queues.append(q)

        payload = {"type": "event", "value": 42}
        delivered = await mgr.broadcast("case:100", payload)
        assert delivered == n_clients

        # Each client gets exactly one copy
        for q in queues:
            assert q.qsize() == 1
            msg = q.get_nowait()
            assert msg["value"] == 42


# ============================================================================
# Reconnection tests
# ============================================================================

class TestBackoffDelays:
    def test_delay_within_bounds(self):
        config = BackoffConfig(base_seconds=1.0, max_backoff_seconds=30.0, jitter=False)
        for attempt in range(10):
            delay = compute_backoff_delay(attempt, config)
            assert delay <= 30.0
            assert delay >= 0

    def test_delay_never_exceeds_max(self):
        config = BackoffConfig(base_seconds=2.0, max_backoff_seconds=10.0, jitter=False)
        # At attempt=10, uncapped = 2 * 2^10 = 2048 → capped at 10
        delay = compute_backoff_delay(10, config)
        assert delay == 10.0

    def test_jitter_produces_different_values(self):
        config = BackoffConfig(base_seconds=1.0, max_backoff_seconds=30.0, jitter=True)
        delays = {compute_backoff_delay(5, config) for _ in range(20)}
        # With full jitter, the probability that all 20 calls produce the same value is negligible
        assert len(delays) > 1

    def test_budget_should_retry_false_after_exhaustion(self):
        config = BackoffConfig(max_attempts=3)
        budget = ReconnectBudget(config)
        for _ in range(3):
            assert budget.should_retry()
            budget.next_delay()
        assert not budget.should_retry()

    def test_budget_reset_restores_retry(self):
        config = BackoffConfig(max_attempts=2)
        budget = ReconnectBudget(config)
        budget.next_delay()
        budget.next_delay()
        assert not budget.should_retry()
        budget.reset()
        assert budget.should_retry()
        assert budget.attempts == 0


# Skip all JWT-dependent tests if PyJWT is not installed
from api.websockets.reconnection import _jwt_available
_jwt_present = _jwt_available()

_skip_jwt = pytest.mark.skipif(
    not _jwt_present,
    reason="PyJWT not installed (pip install PyJWT>=2.8.0 to run these tests)",
)


class TestStickySessionTokens:
    @_skip_jwt
    def test_round_trip(self):
        token = issue_sticky_token(
            tenant_id="user:99",
            channel_key="case:7",
            secret=_SECRET,
        )
        decoded = verify_sticky_token(token, secret=_SECRET)
        assert decoded is not None
        assert decoded["tid"] == "user:99"
        assert decoded["ch"] == "case:7"
        assert decoded["type"] == "ws_sticky"

    @_skip_jwt
    def test_expired_token_returns_none(self):
        import datetime
        pyjwt = __import__("jwt")
        payload = {
            "type": "ws_sticky",
            "ch": "case:1",
            "tid": "user:1",
            "iat": datetime.datetime.now(tz=datetime.timezone.utc) - datetime.timedelta(seconds=600),
            "exp": datetime.datetime.now(tz=datetime.timezone.utc) - datetime.timedelta(seconds=300),
        }
        token = pyjwt.encode(payload, _SECRET, algorithm="HS256")
        result = verify_sticky_token(token, secret=_SECRET)
        assert result is None

    @_skip_jwt
    def test_wrong_secret_returns_none(self):
        token = issue_sticky_token(
            tenant_id="user:1",
            channel_key="case:1",
            secret=_SECRET,
        )
        result = verify_sticky_token(token, secret="completely-wrong-secret-key-abc")
        assert result is None

    @_skip_jwt
    def test_wrong_token_type_returns_none(self):
        import datetime
        pyjwt = __import__("jwt")
        payload = {
            "type": "access",  # wrong type
            "ch": "case:1",
            "tid": "user:1",
            "iat": datetime.datetime.now(tz=datetime.timezone.utc),
            "exp": datetime.datetime.now(tz=datetime.timezone.utc) + datetime.timedelta(minutes=5),
        }
        token = pyjwt.encode(payload, _SECRET, algorithm="HS256")
        result = verify_sticky_token(token, secret=_SECRET)
        assert result is None


class TestReconnectContext:
    @_skip_jwt
    def test_reconnect_context_has_sticky_token(self):
        budget = ReconnectBudget(BackoffConfig(max_attempts=5))
        ctx = build_reconnect_context(
            tenant_id="user:1",
            channel_key="case:1",
            budget=budget,
            jwt_secret=_SECRET,
        )
        assert ctx.should_reconnect is True
        assert ctx.sticky_token is not None
        assert ctx.delay_hint_seconds >= 0

    def test_reconnect_context_after_exhaustion(self):
        config = BackoffConfig(max_attempts=2)
        budget = ReconnectBudget(config)
        budget.next_delay()
        budget.next_delay()
        ctx = build_reconnect_context(
            tenant_id="user:1",
            channel_key="case:1",
            budget=budget,
            jwt_secret=_SECRET,
        )
        assert ctx.should_reconnect is False
        assert ctx.reason == "max_reconnect_attempts_exceeded"

    @_skip_jwt
    def test_reconnect_sticky_token_verifiable(self):
        budget = ReconnectBudget(BackoffConfig(max_attempts=5))
        ctx = build_reconnect_context(
            tenant_id="user:42",
            channel_key="case:99",
            budget=budget,
            jwt_secret=_SECRET,
        )
        decoded = verify_sticky_token(ctx.sticky_token, secret=_SECRET)
        assert decoded["tid"] == "user:42"
        assert decoded["ch"] == "case:99"



# ============================================================================
# Instance-failover simulation
# ============================================================================

class TestInstanceFailoverSimulation:
    """
    Simulates what happens when a WebSocket server restarts:
    - Client obtains a sticky token
    - Server goes away (connection dropped)
    - Client reconnects to a *new* instance using the sticky token
    - New instance verifies the sticky token and lets the client in immediately
    """

    @pytest.mark.asyncio
    @_skip_jwt
    async def test_sticky_token_allows_fast_reconnect_on_new_instance(self):
        # "Old" server issues a sticky token
        old_secret = _SECRET
        token = issue_sticky_token(
            tenant_id="user:7",
            channel_key="case:13",
            secret=old_secret,
        )

        # Simulate "old" server dying — new server starts up
        # New server also knows the shared JWT secret (same config)
        new_secret = _SECRET

        decoded = verify_sticky_token(token, secret=new_secret)
        assert decoded is not None, "New server must accept the sticky token"

        # New server can reconstruct the channel key and tenant_id directly
        assert decoded["ch"] == "case:13"
        assert decoded["tid"] == "user:7"

        # New server now subscribes the client to the correct channel
        new_manager = _make_manager()
        q = await new_manager.subscribe(
            tenant_id=decoded["tid"],
            channel_key=decoded["ch"],
        )
        assert q is not None
        assert new_manager.tenant_count("user:7") == 1

    @pytest.mark.asyncio
    async def test_reconnect_budget_correctly_tracks_attempts_across_failover(self):
        budget = ReconnectBudget(BackoffConfig(base_seconds=0.1, max_backoff_seconds=2.0, max_attempts=5))

        delays = []
        while budget.should_retry():
            delays.append(budget.next_delay())

        assert len(delays) == 5
        assert not budget.should_retry()

        # After reset (simulating successful reconnect), budget is fresh
        budget.reset()
        assert budget.should_retry()
        assert budget.attempts == 0
