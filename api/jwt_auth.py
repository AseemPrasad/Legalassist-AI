"""Shared JWT utilities for the API.

This module centralizes JWT creation, verification, and revocation logic
so both `api.auth` and the legacy `auth` module can depend on a single
implementation and avoid security drift.
"""
from __future__ import annotations

import threading
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List

import jwt
import structlog
from api.config import get_settings
from database import SessionLocal, is_token_revoked, revoke_token

logger = structlog.get_logger(__name__)


class AuthError(Exception):
    pass


class TokenExpiredError(AuthError):
    pass


class InvalidTokenError(AuthError):
    pass


settings = get_settings()

_REVOCATION_CACHE: dict[str, tuple[bool, float]] = {}
_REVOCATION_CACHE_TTL: int = 300  # 5 minutes
_REVOCATION_CACHE_MAX_SIZE: int = 10000
_REVOCATION_CACHE_LOCK = threading.Lock()


def _prune_revocation_cache() -> None:
    """Remove expired entries and trim cache to max size.

    Caller must hold _REVOCATION_CACHE_LOCK.
    """
    now = datetime.now(timezone.utc).timestamp()
    cutoff = now - _REVOCATION_CACHE_TTL

    expired = [k for k, (_, ts) in _REVOCATION_CACHE.items() if ts < cutoff]
    for k in expired:
        del _REVOCATION_CACHE[k]

    if len(_REVOCATION_CACHE) > _REVOCATION_CACHE_MAX_SIZE:
        sorted_keys = sorted(
            _REVOCATION_CACHE.keys(),
            key=lambda k: _REVOCATION_CACHE[k][1],
        )
        excess = len(_REVOCATION_CACHE) - _REVOCATION_CACHE_MAX_SIZE
        for k in sorted_keys[:excess]:
            del _REVOCATION_CACHE[k]


def _is_token_revoked_cached(jti: str) -> bool:
    now = datetime.now(timezone.utc).timestamp()

    with _REVOCATION_CACHE_LOCK:
        cached = _REVOCATION_CACHE.get(jti)
        if cached is not None and (now - cached[1]) < _REVOCATION_CACHE_TTL:
            return cached[0]

    from database import SessionLocal, is_token_revoked
    with SessionLocal() as db:
        revoked = is_token_revoked(db, jti)

    with _REVOCATION_CACHE_LOCK:
        _REVOCATION_CACHE[jti] = (revoked, now)
        if len(_REVOCATION_CACHE) > _REVOCATION_CACHE_MAX_SIZE:
            _prune_revocation_cache()

    return revoked


def _get_jwt_secrets_to_try() -> List[str]:
    secrets_to_try = [settings.JWT_SECRET_KEY, settings.JWT_SECRET_KEY_PREVIOUS]
    stripped = (s.strip() for s in secrets_to_try if s and s.strip())
    return [s for s in dict.fromkeys(stripped) if len(s) >= 16]


def create_access_token(data: Dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()

    if "sub" not in to_encode and "user_id" in to_encode:
        to_encode["sub"] = str(to_encode.pop("user_id"))
    elif "sub" not in to_encode:
        raise ValueError("Token data must include 'sub' or 'user_id' claim")

    if not settings.JWT_SECRET_KEY or not settings.JWT_SECRET_KEY.strip():
        raise RuntimeError(
            "JWT_SECRET_KEY is not configured. Set JWT_SECRET (or JWT_SECRET_KEY) "
            "environment variable before issuing tokens."
        )

    to_encode.setdefault("jti", str(uuid.uuid4()))
    issued_at = datetime.now(timezone.utc)
    to_encode.setdefault("iat", issued_at)
    to_encode.setdefault("nbf", issued_at)
    to_encode.setdefault("iss", settings.JWT_ISSUER)
    to_encode.setdefault("aud", settings.JWT_AUDIENCE)
    to_encode.setdefault("type", "access")

    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.JWT_ACCESS_TOKEN_MINUTES)

    to_encode.update({"exp": expire})

    encoded_jwt = jwt.encode(
        to_encode,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )
    return encoded_jwt


def verify_token(token: str) -> Dict:
    try:
        payload = None
        last_error = None
        for secret in _get_jwt_secrets_to_try():
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
                continue
            except jwt.InvalidIssuerError as exc:
                last_error = exc
                raise InvalidTokenError("Invalid token issuer")
            except jwt.InvalidAudienceError as exc:
                last_error = exc
                raise InvalidTokenError("Invalid token audience")
            except jwt.InvalidTokenError as exc:
                last_error = exc
                continue

        if payload is None:
            if isinstance(last_error, jwt.ExpiredSignatureError):
                raise TokenExpiredError("Token has expired")
            raise InvalidTokenError(str(last_error) if last_error else "Invalid token")
        if payload.get("type") != "access":
            raise InvalidTokenError("Invalid token type")

        jti = payload.get("jti")
        if jti:
            if _is_token_revoked_cached(jti):
                raise InvalidTokenError("Token has been revoked")
        return payload
    except jwt.ExpiredSignatureError:
        raise TokenExpiredError("Token has expired")
    except jwt.InvalidIssuerError:
        raise InvalidTokenError("Invalid token issuer")
    except jwt.InvalidAudienceError:
        raise InvalidTokenError("Invalid token audience")
    except jwt.InvalidTokenError:
        raise InvalidTokenError("Invalid token")


def revoke_jwt_token(token: str) -> bool:
    if not token:
        return False

    try:
        payload = None
        last_error = None
        for secret in _get_jwt_secrets_to_try():
            try:
                payload = jwt.decode(
                    token,
                    secret,
                    algorithms=[settings.JWT_ALGORITHM],
                    issuer=settings.JWT_ISSUER,
                    audience=settings.JWT_AUDIENCE,
                    options={"verify_exp": False, "verify_signature": True, "verify_nbf": True, "require": ["exp", "iat", "nbf", "iss", "aud", "jti", "type"]},
                )
                break
            except jwt.InvalidTokenError as exc:
                last_error = exc
                continue

        if payload is None:
            return False

        token_type = payload.get("type")
        if token_type != "access":
            return False

        jti = payload.get("jti")
        exp = payload.get("exp")
        if not jti or not exp:
            return False

        expires_at = datetime.fromtimestamp(exp, tz=timezone.utc) if isinstance(exp, (int, float)) else exp

        with SessionLocal() as db:
            if not is_token_revoked(db, jti):
                revoke_token(db, jti, expires_at)
                # Ensure in-memory cache marks this JTI as revoked immediately
                now = datetime.now(timezone.utc).timestamp()
                _REVOCATION_CACHE[jti] = (True, now)
        return True
    except Exception as exc:
        logger.error("revoke_jwt_token_failed", error=type(exc).__name__)
        return False
