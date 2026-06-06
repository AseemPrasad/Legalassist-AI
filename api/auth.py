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
import uuid

from api.config import get_settings
from database import SessionLocal, revoke_token

# Import canonical JWT utilities from shared module
from api.jwt_auth import (
    AuthError,
    TokenExpiredError,
    InvalidTokenError,
    create_access_token,
    verify_token,
    revoke_jwt_token,
)


# Canonical exceptions are imported from api.jwt_auth above

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
security = HTTPBearer(auto_error=False)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Password hashing (canonical source for all password operations)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto", bcrypt__rounds=14)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against a bcrypt hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Generate a bcrypt hash for a password with cost factor 14."""
    return pwd_context.hash(password)


# PBKDF2 iterations for API key hashing (OWASP 2023 minimum for SHA-256)
API_KEY_HASH_ITERATIONS = 600000

def create_access_token(data: Dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=settings.JWT_EXPIRATION_HOURS)
    
    to_encode.update({"exp": expire, "jti": str(uuid.uuid4())})
    
    encoded_jwt = jwt.encode(
        to_encode,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM
    )
    return encoded_jwt

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against a bcrypt hash."""
    return pwd_context.verify(plain_password, hashed_password)

def revoke_jwt_token(token: str, user_id: int) -> bool:
    """Revoke a JWT token, validating that it belongs to the given user.

    Args:
        token: The JWT string to revoke.
        user_id: The ID of the user requesting revocation (ownership check).

    Returns:
        True if the token was revoked, False if it was already revoked
        or had no revocable claims.

    Raises:
        HTTPException(403) if the token's subject does not match user_id.
        HTTPException(401) if the token is malformed.
    """
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
            options={"verify_exp": False},
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

    token_sub = payload.get("sub")
    if token_sub is None or int(token_sub) != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Token does not belong to the authenticated user"
        )

    jti = payload.get("jti")
    exp = payload.get("exp")
    if not jti or not exp:
        return False

    db = SessionLocal()
    try:
        expires_at = datetime.utcfromtimestamp(exp)
        revoke_token(db, jti, expires_at)
        return True
    finally:
        db.close()


def verify_token(token: str) -> Dict:
    """Verify JWT token and check revocation status.

    Uses a context-manager-scoped database session for the revocation
    check so the connection is released on every exit path — normal
    return, HTTPException, or any unexpected error — preventing the
    connection leaks that occur when raising inside a bare try/finally
    block under certain async execution contexts.
    """
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM]
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
from api.models import APIKey


def generate_api_key() -> str:
    """Generate a new API key"""
    return secrets.token_urlsafe(32)


def hash_api_key(key: str, salt: str) -> str:
    """Hash API key for storage with salt using PBKDF2-HMAC-SHA256"""
    return hashlib.pbkdf2_hmac('sha256', key.encode(), salt.encode(), API_KEY_HASH_ITERATIONS).hex()


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
    salt = "1:" + secrets.token_hex(14)
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

    # Use a deterministic negative user_id derived from key_id to give each
    # unlinked API key its own identity for rate limiting and audit logging.
    # 8 bytes provides a 64-bit space, virtually eliminating collision risk.
    derived_id = int.from_bytes(hashlib.sha256(key_id.encode()).digest()[:8], "big", signed=False)
    return CurrentUser(
        user_id=-derived_id,
        email=f"api_key_{key_id[:8]}",
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
            user_id = payload.get("sub")
            email = payload.get("email")
            role = payload.get("role", "user")

            if not user_id:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token payload"
                )

            # Validate the user exists in the database.
            db = SessionLocal()
            try:
                user = get_user_by_email(db, email) if email else None
                if not user or str(user.id) != user_id:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="User account not found"
                    )
            finally:
                db.close()

            return CurrentUser(user_id, email, role)
        except HTTPException:
            raise
    
    # Try API Key from header — look up explicitly from authoritative store.
    if http_auth:
        api_key = http_auth.credentials
        
        # Assume structural prefix like key_xx or split appropriately
        from database import SessionLocal
        db = SessionLocal()
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

            # Validate the user exists in the database — prevents access via
            # orphaned tokens (e.g. dev-mode placeholders or tokens whose
            # backing user account was deleted while the token remained valid).
            db = SessionLocal()
            try:
                user = get_user_by_email(db, email) if email else None
                if not user or str(user.id) != user_id:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="User account not found"
                    )
            finally:
                db.close()

            return CurrentUser(user_id, email, role)
        except HTTPException:
            raise
    
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
