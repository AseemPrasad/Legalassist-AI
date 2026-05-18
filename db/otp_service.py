from __future__ import annotations

import datetime as dt
import threading
from typing import Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from db.models import OTPVerification, RevokedToken, User


_OTP_RATE_LIMIT_LOCK = threading.RLock()
_OTP_RATE_LIMIT_EVENTS: dict[str, list[dt.datetime]] = {}


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """Get a user by email address"""
    return db.query(User).filter(User.email == email).first()


def create_user(db: Session, email: str) -> User:
    """Create a new user"""
    user = User(email=email)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def update_user_last_login(db: Session, user_id: int) -> Optional[User]:
    """Update last login timestamp for a user"""
    user = db.query(User).filter(User.id == user_id).first()
    if user:
        user.last_login = dt.datetime.now(dt.timezone.utc)
        db.commit()
        db.refresh(user)
    return user


def _otp_rate_limit_key(identifier: str) -> str:
    normalized = str(identifier).strip().lower().replace("@", "")
    if not normalized:
        raise ValueError("OTP request identifier is required")
    return f"otp:rate:{normalized}"


def _get_otp_rate_limit_script():
    """Return a small in-process counter hook for compatibility/tests.

    The legacy codebase expects a Redis-backed atomic counter in production.
    In this service module, we still expose the hook so tests can stub it, but
    the actual enforcement below is backed by the database and an in-process
    timestamp window so OTP throttling works even without Redis.
    """

    def _script(*, keys, args):
        return 0

    return _script


def _reserve_otp_rate_limit_slot(
    db: Session,
    email: str,
    max_requests_per_hour: int,
    requester_ip: Optional[str] = None,
) -> bool:
    """Reserve an OTP request slot for the email, with optional IP tracking."""
    if max_requests_per_hour <= 0:
        raise ValueError("Too many OTP requests. Please try again later.")

    normalized_email = str(email).strip().lower()
    if not normalized_email:
        raise ValueError("OTP request email is required")

    now = dt.datetime.now(dt.timezone.utc)
    window_start = now - dt.timedelta(hours=1)

    recent_email_requests = db.query(OTPVerification).filter(
        func.lower(OTPVerification.email) == normalized_email,
        OTPVerification.created_at >= window_start,
    ).count()
    if recent_email_requests >= max_requests_per_hour:
        raise ValueError("Too many OTP requests. Please try again later.")

    script = _get_otp_rate_limit_script()
    script(keys=[_otp_rate_limit_key(f"email:{normalized_email}")], args=[3600])

    if requester_ip:
        normalized_ip = str(requester_ip).strip().lower()
        if normalized_ip:
            script(keys=[_otp_rate_limit_key(f"ip:{normalized_ip}")], args=[3600])

    with _OTP_RATE_LIMIT_LOCK:
        email_key = _otp_rate_limit_key(f"email:{normalized_email}")
        email_events = _OTP_RATE_LIMIT_EVENTS.setdefault(email_key, [])
        email_events[:] = [timestamp for timestamp in email_events if timestamp >= window_start]
        if len(email_events) >= max_requests_per_hour:
            raise ValueError("Too many OTP requests. Please try again later.")

        email_events.append(now)

        if requester_ip:
            normalized_ip = str(requester_ip).strip().lower()
            if normalized_ip:
                ip_key = _otp_rate_limit_key(f"ip:{normalized_ip}")
                ip_events = _OTP_RATE_LIMIT_EVENTS.setdefault(ip_key, [])
                ip_events[:] = [timestamp for timestamp in ip_events if timestamp >= window_start]
                ip_events.append(now)

    return True


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
    """Get the latest unused, non-expired OTP for an email"""
    now = dt.datetime.now(dt.timezone.utc)
    return db.query(OTPVerification).filter(
        OTPVerification.email == email,
        OTPVerification.is_used == False,
        OTPVerification.expires_at > now,
    ).order_by(OTPVerification.created_at.desc()).first()


def mark_otp_as_used(db: Session, otp_id: int) -> None:
    """Mark an OTP as used"""
    otp = db.query(OTPVerification).filter(OTPVerification.id == otp_id).first()
    if otp:
        otp.is_used = True
        db.commit()


def cleanup_expired_otps(db: Session) -> int:
    """Delete expired OTP records"""
    now = dt.datetime.now(dt.timezone.utc)
    deleted = db.query(OTPVerification).filter(OTPVerification.expires_at < now).delete()
    db.commit()
    return deleted


def revoke_token(db: Session, jti: str, expires_at: dt.datetime) -> RevokedToken:
    """Persist a JWT revocation record."""
    token = RevokedToken(jti=jti, expires_at=expires_at)
    db.add(token)
    db.commit()
    db.refresh(token)
    return token


def is_token_revoked(db: Session, jti: str) -> bool:
    """Check whether a JWT has already been revoked."""
    return db.query(RevokedToken).filter(RevokedToken.jti == jti).first() is not None


def cleanup_expired_revoked_tokens(db: Session) -> int:
    """Delete expired JWT revocation records."""
    now = dt.datetime.now(dt.timezone.utc)
    deleted = db.query(RevokedToken).filter(RevokedToken.expires_at < now).delete()
    db.commit()
    return deleted


def record_otp_failed_attempt(
    db: Session,
    otp_id: int,
    lockout_duration_minutes: int = 15,
    max_failed_attempts: int = 5,
) -> Optional[OTPVerification]:
    """Increment failed attempts for an OTP and lock it if needed."""
    otp = db.query(OTPVerification).filter(OTPVerification.id == otp_id).first()
    if not otp:
        return None

    otp.failed_attempts = (otp.failed_attempts or 0) + 1
    if otp.failed_attempts >= max_failed_attempts:
        otp.locked_until = dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=lockout_duration_minutes)

    db.commit()
    db.refresh(otp)
    return otp


def reset_otp_failed_attempts(db: Session, otp_id: int) -> Optional[OTPVerification]:
    """Reset failed attempts and clear any OTP lockout."""
    otp = db.query(OTPVerification).filter(OTPVerification.id == otp_id).first()
    if not otp:
        return None

    otp.failed_attempts = 0
    otp.locked_until = None
    db.commit()
    db.refresh(otp)
    return otp
