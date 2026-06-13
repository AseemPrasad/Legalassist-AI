"""
WebSocket Reconnection Strategies
===================================
Implements robust reconnection logic for HA WebSocket clients:

  * Exponential backoff with jitter (full jitter algorithm)
  * Sticky-session tokens — opaque tokens that let a client re-attach
    to the same logical channel across server restarts or failover events
  * Reconnect budget tracking (max attempts before giving up)

The sticky-session token is a signed, time-limited JWT that encodes the
``channel_key`` and ``tenant_id``.  When a client re-connects with a
valid token the server skips the full authorisation round-trip and
immediately places the client back on the correct channel.

Design Decisions
----------------
* Tokens are short-lived (default 5 min).  A client must reconnect within
  that window; after expiry the client must present a full auth token.
* The signing secret is the same ``JWT_SECRET_KEY`` already used in the
  application — no new secret is introduced.
* The jitter algorithm is *full jitter*: sleep = random(0, cap) where cap
  doubles each retry up to ``max_backoff_seconds``.  This avoids
  thundering-herd on large fleets.
"""
from __future__ import annotations

import asyncio
import math
import random
import time
from dataclasses import dataclass, field
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Backoff
# ---------------------------------------------------------------------------

@dataclass
class BackoffConfig:
    """Configuration for exponential-backoff-with-jitter reconnect delays."""
    base_seconds: float = 0.5      # initial delay before first retry
    max_backoff_seconds: float = 30.0
    max_attempts: int = 20
    jitter: bool = True            # full-jitter by default


def compute_backoff_delay(
    attempt: int,
    config: BackoffConfig,
) -> float:
    """
    Return the delay in seconds for *attempt* (0-indexed).

    Uses *full-jitter* exponential backoff::

        cap   = min(max_backoff, base * 2^attempt)
        delay = random(0, cap)    # jitter=True (default)
              | cap               # jitter=False (deterministic)

    References
    ----------
    AWS Architecture Blog — "Exponential Backoff and Jitter"
    https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/
    """
    cap = min(config.max_backoff_seconds, config.base_seconds * math.pow(2, attempt))
    if config.jitter:
        return random.uniform(0, cap)
    return cap


class ReconnectBudget:
    """
    Tracks reconnect attempts for a single connection session and enforces
    ``BackoffConfig.max_attempts``.

    Usage::

        budget = ReconnectBudget(config)
        while budget.should_retry():
            delay = budget.next_delay()
            await asyncio.sleep(delay)
            # try to reconnect …
    """

    def __init__(self, config: BackoffConfig | None = None) -> None:
        self._config = config or BackoffConfig()
        self._attempt = 0
        self._last_connect: float = time.monotonic()

    @property
    def attempts(self) -> int:
        return self._attempt

    def should_retry(self) -> bool:
        return self._attempt < self._config.max_attempts

    def next_delay(self) -> float:
        delay = compute_backoff_delay(self._attempt, self._config)
        self._attempt += 1
        logger.debug(
            "ws_reconnect_scheduled",
            attempt=self._attempt,
            delay_seconds=round(delay, 3),
            max_attempts=self._config.max_attempts,
        )
        return delay

    def reset(self) -> None:
        """Call after a successful connection to reset the budget."""
        self._attempt = 0
        self._last_connect = time.monotonic()


# ---------------------------------------------------------------------------
# Sticky-session tokens
# ---------------------------------------------------------------------------

_DEFAULT_TTL_SECONDS: int = 300  # 5 minutes


def _get_jwt_module():
    """
    Return the PyJWT module, trying common import names.

    PyJWT installs as the ``jwt`` package on Python 3.x.  This helper
    tries both ``jwt`` and ``PyJWT`` so the code works regardless of
    how the package is installed.

    Raises RuntimeError if the library is not available.
    """
    import importlib
    for name in ("jwt", "PyJWT"):
        try:
            mod = importlib.import_module(name)
            # Confirm it's actually PyJWT (has encode/decode attributes)
            if hasattr(mod, "encode") and hasattr(mod, "decode"):
                return mod
        except ImportError:
            continue
    raise RuntimeError(
        "PyJWT is required for sticky-session tokens. "
        "Install it with: pip install PyJWT>=2.8.0"
    )


def _jwt_available() -> bool:
    """Return True if PyJWT is importable in the current environment."""
    try:
        _get_jwt_module()
        return True
    except RuntimeError:
        return False


def issue_sticky_token(
    *,
    tenant_id: str,
    channel_key: str,
    secret: str,
    algorithm: str = "HS256",
    ttl_seconds: int = _DEFAULT_TTL_SECONDS,
) -> str:
    """
    Issue a signed sticky-session token for *channel_key* / *tenant_id*.

    The token is a compact JWT with the following custom claims:

    ``type``
        Always ``"ws_sticky"`` — distinguishes this from auth access tokens.
    ``ch``
        The channel key (e.g. ``"case:42"``).
    ``tid``
        The tenant / user ID.

    Returns the encoded token string.

    Raises
    ------
    RuntimeError
        If the PyJWT package is not installed.
    """
    import datetime

    pyjwt = _get_jwt_module()
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    payload = {
        "type": "ws_sticky",
        "ch": channel_key,
        "tid": tenant_id,
        "iat": now,
        "exp": now + datetime.timedelta(seconds=ttl_seconds),
    }
    return pyjwt.encode(payload, secret, algorithm=algorithm)


def verify_sticky_token(
    token: str,
    *,
    secret: str,
    algorithm: str = "HS256",
) -> Optional[dict]:
    """
    Verify and decode a sticky-session token.

    Returns the decoded payload dict on success, or ``None`` if the token
    is invalid, expired, or the JWT library is unavailable.
    """
    try:
        pyjwt = _get_jwt_module()
    except RuntimeError:
        return None

    try:
        payload = pyjwt.decode(
            token,
            secret,
            algorithms=[algorithm],
            options={"require": ["exp", "iat", "type", "ch", "tid"]},
        )
        if payload.get("type") != "ws_sticky":
            return None
        return payload
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Server-side reconnection helper
# ---------------------------------------------------------------------------

@dataclass
class ReconnectContext:
    """
    Carries state about an in-progress or recent reconnect cycle.

    Produced by :func:`build_reconnect_context` and embedded in the
    ``ws_reconnect_required`` message sent to clients so they know how long
    to wait before their next attempt.
    """
    should_reconnect: bool
    delay_hint_seconds: float          # suggested client wait time
    sticky_token: Optional[str]        # None if token could not be issued
    reason: str = ""
    attempt: int = 0


def build_reconnect_context(
    *,
    tenant_id: str,
    channel_key: str,
    budget: ReconnectBudget,
    jwt_secret: str,
    reason: str = "server_disconnected",
) -> ReconnectContext:
    """
    Build a :class:`ReconnectContext` for the *current* attempt.

    Issues a fresh sticky-session token so the client can re-attach quickly.
    """
    if not budget.should_retry():
        return ReconnectContext(
            should_reconnect=False,
            delay_hint_seconds=0.0,
            sticky_token=None,
            reason="max_reconnect_attempts_exceeded",
            attempt=budget.attempts,
        )

    delay = budget.next_delay()

    try:
        token = issue_sticky_token(
            tenant_id=tenant_id,
            channel_key=channel_key,
            secret=jwt_secret,
        )
    except Exception:  # noqa: BLE001
        token = None

    return ReconnectContext(
        should_reconnect=True,
        delay_hint_seconds=delay,
        sticky_token=token,
        reason=reason,
        attempt=budget.attempts,
    )
