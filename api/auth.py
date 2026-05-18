"""
Authentication and Authorization
"""
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, OAuth2PasswordBearer
import secrets
import hashlib

from api.config import get_settings
from database import SessionLocal, is_token_revoked
from db.models import APIKey, User


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


def _get_jwt_secrets_to_try() -> list[str]:
    secrets_to_try = [settings.JWT_SECRET_KEY, settings.JWT_SECRET_KEY_PREVIOUS]
    return [secret for secret in dict.fromkeys(secret.strip() for secret in secrets_to_try if secret and secret.strip())]


# ============================================================================
# JWT Token Management
# ============================================================================

def create_access_token(data: Dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
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
        algorithm=settings.JWT_ALGORITHM
    )
    return encoded_jwt


def verify_token(token: str) -> Dict:
    """Verify JWT token - raises domain-specific auth errors"""
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
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type",
            )

        jti = payload.get("jti")
        if jti:
            db = SessionLocal()
            try:
                if is_token_revoked(db, jti):
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Token has been revoked",
                    )
            finally:
                db.close()
        return payload
    except jwt.ExpiredSignatureError:
        raise TokenExpiredError("Token has expired")
    except jwt.InvalidIssuerError:
        raise InvalidTokenError("Invalid token issuer")
    except jwt.InvalidAudienceError:
        raise InvalidTokenError("Invalid token audience")
    except jwt.InvalidTokenError:
        raise InvalidTokenError("Invalid token")


# ============================================================================
# API Key Management
# ============================================================================

class APIKey:
    """API Key model"""
    def __init__(self, key_id: str, name: str, key_hash: str, key_salt: str, created_at: datetime, 
                 expires_at: Optional[datetime] = None):
        self.key_id = key_id
        self.name = name
        self.key_hash = key_hash
        self.key_salt = key_salt
        self.created_at = created_at
        self.expires_at = expires_at
    
    def is_valid(self) -> bool:
        """Check if API key is valid"""
        if self.expires_at and datetime.now(timezone.utc) > self.expires_at:
            return False
        return True


def generate_api_key() -> str:
    """Generate a new API key"""
    return secrets.token_urlsafe(32)


def hash_api_key(key: str, salt: str) -> str:
    """Hash API key for storage with salt"""
    return hashlib.sha256((salt + key).encode()).hexdigest()


def verify_api_key(key: str, salt: str, key_hash: str) -> bool:
    """Verify API key against salt and hash"""
    return hash_api_key(key, salt) == key_hash


def create_api_key_record(name: str, expires_in_days: Optional[int] = None) -> tuple[str, APIKey]:
    """Create a new API key and its storage record.

    Returns the one-time secret for immediate display plus an APIKey record
    that contains only the hashed value for persistence.
    """
    key = generate_api_key()
    salt = secrets.token_hex(16)
    key_hash = hash_api_key(key, salt)
    created_at = datetime.now(timezone.utc)
    expires_at = None

    if expires_in_days:
        expires_at = created_at + timedelta(days=expires_in_days)

    key_record = APIKey(
        key_id=f"key_{secrets.token_hex(8)}",
        name=name,
        key_hash=key_hash,
        key_salt=salt,
        created_at=created_at,
        expires_at=expires_at,
    )

    return key, key_record


# ============================================================================
# OAuth2 & API Key Authentication
# ============================================================================

class CurrentUser:
    """Current authenticated user"""
    def __init__(self, user_id: int, email: str, role: str = "user"):
        self.user_id = int(user_id)
        self.email = email
        self.role = role


async def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    http_auth: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> CurrentUser:
    """Get current authenticated user"""
    
    # Try JWT token first
    if token:
        try:
            payload = verify_token(token)
        except TokenExpiredError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )

        user_id = payload.get("sub")
        token_email = payload.get("email")

        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload"
            )

        db = SessionLocal()
        try:
            user = db.query(User).filter(User.id == int(user_id)).first()
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found"
                )
            return CurrentUser(user.id, user.email, "admin" if user.is_verified else "user")
        finally:
            db.close()
    
    # Try API Key from header — look up in database only.
    # Never treat API keys as JWTs; they are opaque secrets validated by hash.
    if http_auth:
        api_key = http_auth.credentials
        key_prefix = "key_"

        if not api_key.startswith(key_prefix):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key format"
            )

        db = SessionLocal()
        try:
            key_record = db.query(APIKey).filter(
                APIKey.key_id == api_key
            ).first()

            if not key_record:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid API key"
                )

            if not key_record.is_valid():
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="API key has expired"
                )

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
) -> Optional[CurrentUser]:
    """Get current user without raising on missing credentials.

    Returns the authenticated CurrentUser when valid credentials are present,
    or None when the request is truly anonymous. Invalid, expired, revoked,
    or insufficient credentials still raise HTTPException so callers do not
    accidentally treat failed authentication as unauthenticated access.
    """
    if not token and not http_auth:
        return None

    try:
        return await get_current_user(token=token, http_auth=http_auth)
    except HTTPException:
        raise


async def get_admin_user(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    """Verify user is admin"""
    if user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return user


async def get_attorney_user(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    """Verify user is attorney or admin"""
    if user.role not in ["attorney", "admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Attorney access required"
        )
    return user
