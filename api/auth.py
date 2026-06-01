"""
Authentication and Authorization
"""
import os
from datetime import datetime, timedelta
from typing import Optional, Dict
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, OAuth2PasswordBearer
import secrets
import hashlib

from api.config import get_settings

# Import shared JWT exception hierarchy and utilities from the canonical module.
# Do NOT redefine AuthError, TokenExpiredError, or InvalidTokenError here —
# redefining them would shadow these imports and break exception handling because
# verify_token() raises the jwt_auth classes, not any locally defined ones.
from api.jwt_auth import (
    AuthError,
    TokenExpiredError,
    InvalidTokenError,
    create_access_token,
    verify_token,
)

settings = get_settings()
security = HTTPBearer()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)


# ============================================================================
# JWT Token Management — delegated to api.jwt_auth
# ============================================================================
# create_access_token and verify_token are re-exported from api.jwt_auth above.
# They raise TokenExpiredError / InvalidTokenError (both subclasses of AuthError)
# so callers can catch typed exceptions instead of bare HTTPException.


# ============================================================================
# API Key Management
# ============================================================================

class APIKey:
    """API Key model"""
    def __init__(self, key_id: str, name: str, key_hash: str, created_at: datetime, 
                 expires_at: Optional[datetime] = None):
        self.key_id = key_id
        self.name = name
        self.key_hash = key_hash
        self.created_at = created_at
        self.expires_at = expires_at
    
    def is_valid(self) -> bool:
        """Check if API key is valid"""
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        return True


def generate_api_key() -> str:
    """Generate a new API key"""
    return secrets.token_urlsafe(32)


def hash_api_key(key: str) -> str:
    """Hash API key for storage"""
    return hashlib.sha256(key.encode()).hexdigest()


def create_api_key_record(name: str, expires_in_days: Optional[int] = None) -> tuple:
    """Create new API key record
    Returns: (key_to_display, key_hash_for_storage)
    """
    key = generate_api_key()
    key_hash = hash_api_key(key)
    expires_at = None
    
    if expires_in_days:
        expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
    
    return key, key_hash, expires_at


# ============================================================================
# OAuth2 & API Key Authentication
# ============================================================================

class CurrentUser:
    """Current authenticated user"""
    def __init__(self, user_id: str, email: str, role: str = "user"):
        self.user_id = user_id
        self.email = email
        self.role = role


async def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    http_auth: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> CurrentUser:
    """Get current authenticated user"""
    
    if not settings.AUTH_ENABLED:
        # Development mode - no auth required
        return CurrentUser("dev-user", "dev@example.com", "admin")
    
    # Try JWT token first
    if token:
        try:
            payload = verify_token(token)
            user_id = payload.get("sub")
            email = payload.get("email")
            role = payload.get("role", "user")

            if not user_id:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token payload"
                )

            return CurrentUser(user_id, email, role)
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

    # Try API Key from header — validate as a signed JWT.
    # Arbitrary or unsigned tokens are rejected by verify_token with a 401.
    if http_auth:
        api_key = http_auth.credentials
        try:
            payload = verify_token(api_key)
            user_id = payload.get("sub")
            email = payload.get("email", "api@example.com")
            role = payload.get("role", "user")

            if not user_id:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid API key payload"
                )

            return CurrentUser(user_id, email, role)
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
    
    # Try X-API-Key header
    # This would typically be validated against database
    
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
    or None for unauthenticated requests.  Use this dependency wherever the
    caller must handle anonymous traffic gracefully (e.g. rate-limit key
    generation) rather than enforcing authentication.
    """
    try:
        return await get_current_user(token=token, http_auth=http_auth)
    except HTTPException:
        return None


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
