"""
OTP Service — thread-safe Redis-backed rate limiting and OTP lifecycle helpers.

Race-condition fix
------------------
The original `_get_otp_rate_limit_script` initialisation relied on a bare
``if _otp_rate_limit_script is None`` check that is not safe under concurrent
load.  Under a multi-threaded ASGI/WSGI server, multiple workers can enter
that block simultaneously, each registering the Lua script against a freshly
constructed Redis client.  This causes:

  * Duplicate Lua script registrations (wasted round-trips / SHA collisions)
  * Connection-pool fragmentation (each worker gets its own pool)
  * Intermittent ``NOSCRIPT`` errors when a worker later tries to use a handle
    registered by a now-garbage-collected client

The fix uses the canonical double-checked locking idiom with a
``threading.Lock``.  The outer ``if`` avoids acquiring the lock on every hot
call; the inner ``if`` ensures that only one thread completes the
initialisation even when several threads pass the outer check simultaneously.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import threading
from typing import Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

try:
    import redis as _redis_module
except ImportError:  # pragma: no cover – handled at runtime
    _redis_module = None  # type: ignore[assignment]

from config import Config
from db.models import OTPVerification, RevokedToken, User

# ---------------------------------------------------------------------------
# Redis-backed OTP rate-limit script (Lua, registered once per process)
# ---------------------------------------------------------------------------

_OTP_RATE_LIMIT_WINDOW_SECONDS = 60 * 60  # 1 hour

_OTP_RATE_LIMIT_SCRIPT = """
local current = redis.call('INCR', KEYS[1])
if current == 1 then
    redis.call('EXPIRE', KEYS[1], ARGV[1])
end
return current
"""

# Module-level singletons – written once, read many times.
_otp_rate_limit_client: object | None = None
_otp_rate_limit_script: object | None = None

# threading.Lock() guards the one-time initialisation block.
# Using a plain Lock (not RLock) is intentional: re-entrant acquisition
# inside _get_otp_rate_limit_script would indicate a logic error.
_otp_init_lock: threading.Lock = threading.Lock()

# In-process sliding-window event store (supplements Redis when unavailable)
_OTP_RATE_LIMIT_LOCK: threading.RLock = threading.RLock()
_OTP_RATE_LIMIT_EVENTS: dict[str, list[dt.datetime]] = {}


def _get_otp_rate_limit_script() -> object:
    """Return the registered Redis Lua script handle, initialising once.

    Thread safety
    ~~~~~~~~~~~~~
    Uses double-checked locking so the ``threading.Lock`` is only acquired
    during the single initialisation pass.  All subsequent calls read the
    already-set module global without contention.

    Raises
    ------
    RuntimeError
        If the ``redis`` package is not installed.
    """
    global _otp_rate_limit_client, _otp_rate_limit_script

    # Fast path – already initialised (no lock needed for a read of an
    # immutable reference once Python's GIL guarantees the write is atomic).
    if _otp_rate_limit_script is not None:
        return _otp_rate_limit_script

    # Slow path – acquire the lock and check again inside the critical section.
    with _otp_init_lock:
        # A second thread may have completed initialisation while we waited.
        if _otp_rate_limit_script is None:
            if _redis_module is None:
                raise RuntimeError(
                    "Redis is required for OTP rate limiting but is not installed."
                )

            redis_url = getattr(Config, "REDIS_URL", "redis://localhost:6379/0")
            _otp_rate_limit_client = _redis_module.from_url(
                redis_url, decode_responses=True
            )
            _otp_rate_limit_script = _otp_rate_limit_client.register_script(  # type: ignore[attr-defined]
                _OTP_RATE_LIMIT_SCRIPT
            )

    return _otp_rate_limit_script


def _reset_otp_rate_limit_connection() -> None:
    """Reset Redis connection state to allow self-healing after disconnection."""
    global _otp_rate_limit_client, _otp_rate_limit_script
    with _otp_init_lock:
        _otp_rate_limit_client = None
        _otp_rate_limit_script = None


def _otp_rate_limit_key(identifier: str) -> str:
    """Return a stable, normalised Redis key for an OTP rate-limit counter."""
    normalised = str(identifier).strip().lower()
    digest = hashlib.sha256(normalised.encode("utf-8")).hexdigest()
    return f"otp:rate:{digest}"


def _reserve_otp_rate_limit_slot(
    db: Session,
    email: str,
    max_requests_per_hour: int,
    label: str = "email",
    requester_ip: Optional[str] = None,
) -> bool:
    """Atomically reserve an OTP slot; raise ValueError if limit exceeded.

    The check-then-act sequence is wrapped in ``_OTP_RATE_LIMIT_LOCK`` so that
    concurrent threads cannot both pass the count check before either of them
    has appended to the event log.
    """
    if max_requests_per_hour <= 0:
        raise ValueError("Too many OTP requests. Please try again later.")

    normalised_email = str(email).strip().lower()
    if not normalised_email:
        raise ValueError(f"{label} is required for OTP rate limiting")

    now = dt.datetime.now(dt.timezone.utc)
    window_start = now - dt.timedelta(hours=1)

    # Best-effort Redis increment (non-fatal if Redis is unavailable)
    try:
        script = _get_otp_rate_limit_script()
        script(  # type: ignore[call-arg]
            keys=[_otp_rate_limit_key(f"email:{normalised_email}")],
            args=[_OTP_RATE_LIMIT_WINDOW_SECONDS],
        )
    except Exception:
        pass  # Fall back to in-process + DB enforcement below

    with _OTP_RATE_LIMIT_LOCK:
        # --- DB-backed enforcement ---
        recent_db = (
            db.query(OTPVerification)
            .filter(
                func.lower(OTPVerification.email) == normalised_email,
                OTPVerification.created_at >= window_start,
            )
            .count()
        )
        if recent_db >= max_requests_per_hour:
            raise ValueError("Too many OTP requests. Please try again later.")

        # --- In-process sliding window ---
        email_key = _otp_rate_limit_key(f"email:{normalised_email}")
        events = _OTP_RATE_LIMIT_EVENTS.setdefault(email_key, [])
        events[:] = [ts for ts in events if ts >= window_start]
        if len(events) >= max_requests_per_hour:
            raise ValueError("Too many OTP requests. Please try again later.")
        events.append(now)

        if requester_ip:
            normalised_ip = str(requester_ip).strip().lower()
            if normalised_ip:
                ip_key = _otp_rate_limit_key(f"ip:{normalised_ip}")
                ip_events = _OTP_RATE_LIMIT_EVENTS.setdefault(ip_key, [])
                ip_events[:] = [ts for ts in ip_events if ts >= window_start]
                ip_events.append(now)

    return True


# ---------------------------------------------------------------------------
# User helpers
# ---------------------------------------------------------------------------


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """Return the User record for the given email, or None."""
    return db.query(User).filter(User.email == email).first()


def create_user(db: Session, email: str) -> User:
    """Persist and return a new User."""
    user = User(email=email)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def update_user_last_login(db: Session, user_id: int) -> Optional[User]:
    """Stamp last_login for the given user and return the updated record."""
    user = db.query(User).filter(User.id == user_id).first()
    if user:
        user.last_login = dt.datetime.now(dt.timezone.utc)
        db.commit()
        db.refresh(user)
    return user


# ---------------------------------------------------------------------------
# OTP lifecycle
# ---------------------------------------------------------------------------


def create_otp_verification(
    db: Session,
    email: str,
    otp_hash: str,
    expires_at: dt.datetime,
    max_requests_per_hour: int = 5,
    requester_ip: Optional[str] = None,
) -> OTPVerification:
    """Create a rate-limited OTP verification record.

    The slot reservation and DB insert are wrapped in ``_OTP_RATE_LIMIT_LOCK``
    to prevent TOCTOU races where two concurrent requests both pass the rate-
    limit check before either has written its OTP row.
    """
    with _OTP_RATE_LIMIT_LOCK:
        _reserve_otp_rate_limit_slot(
            db, email, max_requests_per_hour, requester_ip=requester_ip
        )
        otp = OTPVerification(email=email, otp_hash=otp_hash, expires_at=expires_at)
        db.add(otp)
        db.commit()
        db.refresh(otp)
        return otp


def get_pending_otp(db: Session, email: str) -> Optional[OTPVerification]:
    """Return the most-recent valid, unused OTP for the given email."""
    now = dt.datetime.now(dt.timezone.utc)
    return (
        db.query(OTPVerification)
        .filter(
            OTPVerification.email == email,
            OTPVerification.is_used == False,  # noqa: E712
            OTPVerification.expires_at > now,
        )
        .order_by(OTPVerification.created_at.desc())
        .first()
    )


def mark_otp_as_used(db: Session, otp_id: int) -> bool:
    """Atomically mark an OTP as consumed; return False if already used."""
    updated = (
        db.query(OTPVerification)
        .filter(OTPVerification.id == otp_id, OTPVerification.is_used == False)  # noqa: E712
        .update({"is_used": True}, synchronize_session=False)
    )
    db.commit()
    return updated > 0


def cleanup_expired_otps(db: Session) -> int:
    """Delete all expired OTP records; return the number removed."""
    now = dt.datetime.now(dt.timezone.utc)
    deleted = db.query(OTPVerification).filter(OTPVerification.expires_at < now).delete()
    db.commit()
    return deleted


def record_otp_failed_attempt(
    db: Session,
    otp_id: int,
    lockout_duration_minutes: int = 15,
    max_failed_attempts: int = 5,
) -> Optional[OTPVerification]:
    """Increment failed-attempt counter; lock OTP when threshold is reached."""
    otp = db.query(OTPVerification).filter(OTPVerification.id == otp_id).first()
    if not otp:
        return None
    otp.failed_attempts = (otp.failed_attempts or 0) + 1
    if otp.failed_attempts >= max_failed_attempts:
        otp.locked_until = dt.datetime.now(dt.timezone.utc) + dt.timedelta(
            minutes=lockout_duration_minutes
        )
    db.commit()
    db.refresh(otp)
    return otp


def reset_otp_failed_attempts(db: Session, otp_id: int) -> Optional[OTPVerification]:
    """Clear failed-attempt counter and remove any lockout on the OTP."""
    otp = db.query(OTPVerification).filter(OTPVerification.id == otp_id).first()
    if not otp:
        return None
    otp.failed_attempts = 0
    otp.locked_until = None
    db.commit()
    db.refresh(otp)
    return otp


# ---------------------------------------------------------------------------
# JWT revocation helpers
# ---------------------------------------------------------------------------


def revoke_token(db: Session, jti: str, expires_at: dt.datetime) -> RevokedToken:
    """Persist a JWT revocation record."""
    token = RevokedToken(jti=jti, expires_at=expires_at)
    db.add(token)
    db.commit()
    db.refresh(token)
    return token


def is_token_revoked(db: Session, jti: str) -> bool:
    """Return True if the given JWT identifier has been revoked."""
    return (
        db.query(RevokedToken).filter(RevokedToken.jti == jti).first() is not None
    )


def cleanup_expired_revoked_tokens(db: Session) -> int:
    """Delete expired JWT revocation records; return the number removed."""
    now = dt.datetime.now(dt.timezone.utc)
    deleted = (
        db.query(RevokedToken).filter(RevokedToken.expires_at < now).delete()
    )
    db.commit()
    return deleted
