"""
Authentication and Authorization
"""
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, OAuth2PasswordBearer, APIKeyHeader
import secrets
import hashlib
from sqlalchemy.orm import Session

from api.config import get_settings
from api.errors import StructuredAPIError
from database import SessionLocal
from db.models import APIKey, User

# Import canonical JWT utilities from shared module
from api.jwt_auth import (
    AuthError,
    TokenExpiredError,
    InvalidTokenError,
    create_access_token,
    verify_token,
    revoke_jwt_token,
)


class AuthError(Exception):
    """Base authentication error"""
    pass


class TokenExpiredError(AuthError):
    """Token has expired"""
    pass


class InvalidTokenError(AuthError):
    """Token is invalid"""
    pass


settings = get_settings()
security = HTTPBearer(auto_error=False)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _get_jwt_secrets_to_try() -> list[str]:
    secrets_to_try = [settings.JWT_SECRET_KEY, settings.JWT_SECRET_KEY_PREVIOUS]
    return [secret for secret in dict.fromkeys(secret.strip() for secret in secrets_to_try if secret and secret.strip())]


# JWT token functions delegated to `api.jwt_auth`


def revoke_jwt_token(token: str) -> bool:
    """Revoke a JWT token by adding its JTI to the revocation table.

    This verifies the token signature (but not expiration) against
    current/previous secrets and enforces issuer/audience/type checks.
    Returns True on success, False otherwise.
    """
    if not token:
        return False

    try:
        # Fast path - extract without signature verification to get jti/exp
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

        # convert exp to datetime
        expires_at = datetime.fromtimestamp(exp, tz=timezone.utc) if isinstance(exp, (int, float)) else exp

        # Persist revocation
        with SessionLocal() as db:
            # db-level revoke_token is available via database shim
            from database import revoke_token, is_token_revoked

            if not is_token_revoked(db, jti):
                revoke_token(db, jti, expires_at)
        return True
    except Exception:
        return False


# ============================================================================
# API Key Management
# ============================================================================

def generate_api_key() -> str:
    """Generate a new API key"""
    return secrets.token_urlsafe(32)


def hash_api_key(key: str, salt: str) -> str:
    """Hash API key for storage with salt"""
    return hashlib.sha256((salt + key).encode()).hexdigest()


def verify_api_key(key: str, salt: str, key_hash: str) -> bool:
    """Verify API key against salt and hash"""
    return secrets.compare_digest(hash_api_key(key, salt), key_hash)


def create_api_key_record(
    db: Session,
    name: str,
    expires_in_days: Optional[int] = None,
    user_id: Optional[int] = None
) -> tuple[str, APIKey]:
    """Create a new API key and its storage record.

    Returns the one-time secret combined with key_id for display,
    and saves the APIKey record with the hashed secret and user association.
    """
    secret = generate_api_key()
    salt = secrets.token_hex(16)
    key_hash = hash_api_key(secret, salt)
    created_at = datetime.now(timezone.utc)
    expires_at = None

    if expires_in_days:
        expires_at = created_at + timedelta(days=expires_in_days)

    key_id = f"key_{secrets.token_hex(8)}"
    key_record = APIKey(
        key_id=key_id,
        name=name,
        key_hash=key_hash,
        key_salt=salt,
        user_id=user_id,
        created_at=created_at,
        expires_at=expires_at,
    )

    db.add(key_record)
    db.commit()
    db.refresh(key_record)

    # Return the combined API key for display/use
    combined_key = f"{key_id}.{secret}"
    return combined_key, key_record


# ============================================================================
# OAuth2 & API Key Authentication
# ============================================================================

class CurrentUser:
    """Current authenticated user with RBAC role"""
    def __init__(self, user_id: int, email: str, role: str = "client"):
        self.user_id = int(user_id)
        self.email = email
        self.role = role


def _resolve_api_key_user(api_key: str, db: Session) -> CurrentUser:
    """Validate a combined API key and return the associated user context."""

    if "." not in api_key:
        raise StructuredAPIError(
            status_code=status.HTTP_401_UNAUTHORIZED,
            error_code="INVALID_API_KEY_FORMAT",
            message="Invalid API key format",
        )

    key_id, secret = api_key.split(".", 1)
    key_record = db.query(APIKey).filter(APIKey.key_id == key_id).first()

    if not key_record:
        raise StructuredAPIError(
            status_code=status.HTTP_401_UNAUTHORIZED,
            error_code="INVALID_API_KEY",
            message="Invalid API key",
        )

    if not key_record.is_valid():
        raise StructuredAPIError(
            status_code=status.HTTP_401_UNAUTHORIZED,
            error_code="API_KEY_EXPIRED",
            message="API key has expired",
        )

    if not verify_api_key(secret, key_record.key_salt, key_record.key_hash):
        raise StructuredAPIError(
            status_code=status.HTTP_401_UNAUTHORIZED,
            error_code="INVALID_API_KEY",
            message="Invalid API key",
        )

    if key_record.user_id:
        user = db.query(User).filter(User.id == key_record.user_id).first()
        if user:
            return CurrentUser(
                user_id=user.id,
                email=user.email,
                role="admin" if getattr(user, "is_admin", False) else "user",
            )

    return CurrentUser(
        user_id=0,
        email="api_user",
        role="api",
    )


async def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    http_auth: Optional[HTTPAuthorizationCredentials] = Depends(security),
    x_api_key: Optional[str] = Depends(api_key_header),
) -> CurrentUser:
    """Get current authenticated user"""
    
    # Try JWT token first
    if token and not token.startswith("key_"):
        try:
            payload = verify_token(token)
        except TokenExpiredError:
            raise StructuredAPIError(status_code=status.HTTP_401_UNAUTHORIZED, error_code="TOKEN_EXPIRED", message="Token has expired")
        except InvalidTokenError:
            raise StructuredAPIError(status_code=status.HTTP_401_UNAUTHORIZED, error_code="INVALID_TOKEN", message="Invalid token")

        user_id = payload.get("sub")
        token_email = payload.get("email")

        if not user_id:
            raise StructuredAPIError(status_code=status.HTTP_401_UNAUTHORIZED, error_code="INVALID_TOKEN_PAYLOAD", message="Invalid token payload")

        db = SessionLocal()
        try:
            user = db.query(User).filter(User.id == int(user_id)).first()
            if not user:
                raise StructuredAPIError(status_code=status.HTTP_401_UNAUTHORIZED, error_code="USER_NOT_FOUND", message="User not found")
            return CurrentUser(user.id, user.email, "admin" if getattr(user, "is_admin", False) else "user")
        finally:
            db.close()
    
    # Try API Key from header — look up in database only.
    # Never treat API keys as JWTs; they are opaque secrets validated by hash.
    api_key = None
    if http_auth:
        api_key = http_auth.credentials
    elif x_api_key:
        api_key = x_api_key

    if api_key:
        
        if "." not in api_key:
            raise StructuredAPIError(status_code=status.HTTP_401_UNAUTHORIZED, error_code="INVALID_API_KEY_FORMAT", message="Invalid API key format")

        key_id, secret = api_key.split(".", 1)

        db = SessionLocal()
        try:
            key_record = db.query(APIKey).filter(
                APIKey.key_id == key_id
            ).first()

            if not key_record:
                raise StructuredAPIError(status_code=status.HTTP_401_UNAUTHORIZED, error_code="INVALID_API_KEY", message="Invalid API key")

            if not key_record.is_valid():
                raise StructuredAPIError(status_code=status.HTTP_401_UNAUTHORIZED, error_code="API_KEY_EXPIRED", message="API key has expired")

            if not verify_api_key(secret, key_record.key_salt, key_record.key_hash):
                raise StructuredAPIError(status_code=status.HTTP_401_UNAUTHORIZED, error_code="INVALID_API_KEY", message="Invalid API key")

            # Check if linked to a database user
            if key_record.user_id:
                user = db.query(User).filter(User.id == key_record.user_id).first()
                if user:
                    return CurrentUser(
                        user_id=user.id,
                        email=user.email,
                        role="admin" if getattr(user, "is_admin", False) else "user"
                    )

            # Fallback to default API user
            return CurrentUser(
                user_id=0,
                email="api_user",
                role="api"
            )
        finally:
            db.close()

    # Try X-API-Key header
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated"
    )


async def get_current_user_optional(
    token: Optional[str] = Depends(oauth2_scheme),
    http_auth: Optional[HTTPAuthorizationCredentials] = Depends(security),
    x_api_key: Optional[str] = Depends(api_key_header),
) -> Optional[CurrentUser]:
    """Get current user without raising on missing credentials.

    Returns the authenticated CurrentUser when valid credentials are present,
    or None when the request is truly anonymous. Invalid, expired, revoked,
    or insufficient credentials still raise HTTPException so callers do not
    accidentally treat failed authentication as unauthenticated access.
    """
    if not token and not http_auth and not x_api_key:
        return None

    try:
        return await get_current_user(token=token, http_auth=http_auth, x_api_key=x_api_key)
    except HTTPException:
        raise


async def get_admin_user(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    """Verify user is admin"""
    if user.role != "admin":
        raise StructuredAPIError(status_code=status.HTTP_403_FORBIDDEN, error_code="ADMIN_ACCESS_REQUIRED", message="Admin access required")
    return user


async def get_attorney_user(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    """Verify user is attorney or admin"""
    if user.role not in ["attorney", "admin"]:
        raise StructuredAPIError(status_code=status.HTTP_403_FORBIDDEN, error_code="ATTORNEY_ACCESS_REQUIRED", message="Attorney access required")
    return user
