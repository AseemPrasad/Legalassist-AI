"""Shared JWT utilities for the API.

This module centralizes JWT creation, verification, and revocation logic
so both `api.auth` and the legacy `auth` module can depend on a single
implementation and avoid security drift.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List

import jwt
from api.config import get_settings
from database import SessionLocal, is_token_revoked, revoke_token


class AuthError(Exception):
    pass


class TokenExpiredError(AuthError):
    pass


class InvalidTokenError(AuthError):
    pass


settings = get_settings()


def _get_jwt_secrets_to_try() -> List[str]:
    secrets_to_try = [settings.JWT_SECRET_KEY, settings.JWT_SECRET_KEY_PREVIOUS]
    return [secret for secret in dict.fromkeys(secret.strip() for secret in secrets_to_try if secret and secret.strip())]


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
    to_encode.setdefault("iat", datetime.now(timezone.utc))
    to_encode.setdefault("iss", settings.JWT_ISSUER)
    to_encode.setdefault("aud", settings.JWT_AUDIENCE)
    to_encode.setdefault("type", "access")

    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(hours=settings.JWT_EXPIRATION_HOURS)

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
                    options={"require": ["exp", "iat", "iss", "aud", "jti", "type"]},
                )
                break
            except jwt.ExpiredSignatureError as exc:
                last_error = exc
                raise TokenExpiredError("Token has expired")
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
            raise InvalidTokenError(str(last_error) if last_error else "Invalid token")
        if payload.get("type") != "access":
            raise InvalidTokenError("Invalid token type")

        jti = payload.get("jti")
        if jti:
            with SessionLocal() as db:
                if is_token_revoked(db, jti):
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
        try:
            unverified = jwt.decode(token, options={"verify_signature": False, "verify_exp": False})
            jti = unverified.get("jti")
            exp = unverified.get("exp")
            if not jti or not exp:
                return False
        except Exception:
            return False

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
                    options={"verify_exp": False, "verify_signature": True, "require": ["exp", "iat", "iss", "aud", "jti", "type"]},
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
        return True
    except Exception:
        return False
