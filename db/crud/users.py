"""CRUD operations for User management."""

import datetime as dt
import threading
from typing import Optional

from sqlalchemy.orm import Session

from db.models import OTPVerification, User


_OTP_RATE_LIMIT_LOCK = threading.RLock()
_OTP_RATE_LIMIT_EVENTS: dict[str, list[dt.datetime]] = {}


def _otp_rate_limit_key(identifier: str) -> str:
    normalized = str(identifier).strip().lower().replace("@", "")
    if not normalized:
        raise ValueError("OTP request identifier is required")
    return f"otp:rate:{normalized}"


def _reserve_otp_rate_limit_slot(
    db: Session,
    identifier: str,
    max_per_hour: int,
    requester_ip: Optional[str] = None,
) -> None:
    """Reserve a slot in the per-identifier OTP rate limiter.
    
    Raises ValueError if the rate limit has been exceeded.
    """
    key = _otp_rate_limit_key(identifier)
    now = dt.datetime.now(dt.timezone.utc)
    cutoff = now - dt.timedelta(hours=1)

    with _OTP_RATE_LIMIT_LOCK:
        events = _OTP_RATE_LIMIT_EVENTS.get(key, [])
        _OTP_RATE_LIMIT_EVENTS[key] = [e for e in events if e > cutoff]

        if len(_OTP_RATE_LIMIT_EVENTS[key]) >= max_per_hour:
            raise ValueError(
                f"OTP rate limit exceeded for {identifier}. "
                f"Max {max_per_hour} requests per hour."
            )

        _OTP_RATE_LIMIT_EVENTS[key].append(now)


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """Get a user by email address."""
    return db.query(User).filter(User.email == email).first()


def create_user(db: Session, email: str, **kwargs) -> User:
    """Create a new user."""
    user = User(email=email, **kwargs)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def update_user_last_login(db: Session, user_id: int) -> Optional[User]:
    """Update last login timestamp for a user."""
    user = db.query(User).filter(User.id == user_id).first()
    if user:
        user.last_login = dt.datetime.now(dt.timezone.utc)
        db.commit()
        db.refresh(user)
    return user


def create_otp_verification(
    db: Session,
    email: str,
    otp_hash: str,
    expires_at: dt.datetime,
    max_requests_per_hour: int = 5,
    requester_ip: Optional[str] = None,
) -> OTPVerification:
    """Create a new OTP verification record."""
    with _OTP_RATE_LIMIT_LOCK:
        _reserve_otp_rate_limit_slot(db, email, max_requests_per_hour, requester_ip=requester_ip)
        otp = OTPVerification(email=email, otp_hash=otp_hash, expires_at=expires_at)
        db.add(otp)
        db.commit()
        db.refresh(otp)
        return otp


def get_pending_otp(db: Session, email: str) -> Optional[OTPVerification]:
    """Get the latest unused, non-expired OTP for an email."""
    now = dt.datetime.now(dt.timezone.utc)
    return db.query(OTPVerification).filter(
        OTPVerification.email == email,
        OTPVerification.is_used == False,
        OTPVerification.expires_at > now
    ).order_by(OTPVerification.created_at.desc()).first()


def mark_otp_as_used(db: Session, otp_id: int) -> bool:
    """Atomically mark OTP as used. Returns True only if OTP was not already used."""
    try:
        result = db.query(OTPVerification).filter(
            OTPVerification.id == otp_id,
            OTPVerification.is_used == False,
        ).update({"is_used": True}, synchronize_session=False)
        db.commit()
        return result > 0
    except Exception:
        db.rollback()
        return False


def is_email_locked_out(db: Session, email: str) -> Optional[dt.datetime]:
    """Check if email is currently locked out. Returns locked_until if locked, None otherwise."""
    lockout = db.query(OTPVerification).filter(
        OTPVerification.email == email,
        OTPVerification.locked_until != None,
        OTPVerification.locked_until > dt.datetime.now(dt.timezone.utc)
    ).order_by(OTPVerification.locked_until.desc()).first()
    return lockout.locked_until if lockout else None


def record_otp_failed_attempt(db: Session, email: str) -> None:
    """Record a failed OTP attempt for rate limiting."""
    key = _otp_rate_limit_key(email)
    now = dt.datetime.now(dt.timezone.utc)
    with _OTP_RATE_LIMIT_LOCK:
        if key not in _OTP_RATE_LIMIT_EVENTS:
            _OTP_RATE_LIMIT_EVENTS[key] = []
        _OTP_RATE_LIMIT_EVENTS[key].append(now)


def reset_otp_failed_attempts(db: Session, email: str) -> None:
    """Reset failed OTP attempts for an email."""
    key = _otp_rate_limit_key(email)
    with _OTP_RATE_LIMIT_LOCK:
        _OTP_RATE_LIMIT_EVENTS.pop(key, None)


def cleanup_expired_otps(db: Session) -> int:
    """Remove expired OTP records."""
    now = dt.datetime.now(dt.timezone.utc)
    deleted = db.query(OTPVerification).filter(
        OTPVerification.expires_at < now,
        OTPVerification.is_used == True,
    ).delete(synchronize_session=False)
    db.commit()
    return deleted


def create_or_update_user_preference(
    db: Session,
    user_id: int,
    key: str,
    value: str,
) -> None:
    """Create or update a user preference."""
    from db.models import UserPreference
    
    pref = db.query(UserPreference).filter(
        UserPreference.user_id == user_id,
        UserPreference.pref_key == key,
    ).first()
    
    if pref:
        pref.pref_value = value
    else:
        pref = UserPreference(user_id=user_id, pref_key=key, pref_value=value)
        db.add(pref)
    
    db.commit()