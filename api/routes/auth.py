"""
Authentication Endpoints
POST /api/v1/auth/token - Get access token
POST /api/v1/auth/api-key - Create API key
GET /api/v1/auth/me - Get current user
"""
from fastapi import APIRouter, HTTPException, status, Depends
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from database import get_db, db_session
from db.models import APIKey
from api.auth import create_access_token, create_api_key_record, CurrentUser, get_current_user, verify_password
from api.auth import create_access_token, generate_api_key, hash_api_key, CurrentUser, get_current_user
from api.models import TokenResponse, APIKeyCreate, APIKeyResponse
from api.rate_limits import check_api_key_creation_limit
from database import get_db, APIKey
import structlog
from core.log_redaction import mask_email
from fastapi import Request
from api.auth import revoke_jwt_token as api_revoke_jwt

_pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])
logger = structlog.get_logger(__name__)


@router.post(
    "/token",
    response_model=TokenResponse,
    summary="Get access token",
    dependencies=[Depends(RateLimit(use_auth_defaults=True))]
)
async def get_token(
    request: TokenRequest
) -> TokenResponse:
    """
    Authenticate user and get access token
    
    Request body:
    - **username**: User email or username
    - **password**: User password
    
    Returns JWT token valid for 24 hours
    """
    from database import get_user_by_email

    logger.info("token_request_received", username=mask_email(username))

    with db_session() as db:
        user = get_user_by_email(db, username)
        if not user or not user.password_hash:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )
        
        if not verify_password(password, user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )
        
        token = create_access_token({"sub": str(user.id), "email": user.email})
        
        return TokenResponse(
            access_token=token,
            token_type="bearer",
            expires_in=24 * 3600  # 24 hours in seconds
        )


@router.post(
    "/api-keys",
    response_model=APIKeyResponse,
    summary="Create API key"
)
async def create_api_key(
    request: APIKeyCreate,
    current_user: CurrentUser = Depends(get_current_user),
    _: None = Depends(check_api_key_creation_limit),
    db: Session = Depends(get_db),
) -> APIKeyResponse:
    """
    Create new API key for programmatic access
    
    - **name**: Name for the API key
    - **expires_in_days**: Optional expiration (1-365 days)
    
    Returns API key (only shown on creation - save it!)
    """
    
    logger.info(
        "Creating API key",
        user_id=current_user.user_id,
        key_name=request.name
    )
    
    key, api_key_record = create_api_key_record(
        db=db,
        name=request.name,
        expires_in_days=request.expires_in_days,
        user_id=current_user.user_id
    )
    
    record = APIKey(
        user_id=int(current_user.user_id),
        name=request.name,
        key_hash=key_hash,
        expires_at=expires_at,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    
    return APIKeyResponse(
        id=str(record.id),
        name=record.name,
        key=key,
        created_at=record.created_at,
        expires_at=record.expires_at
    )


@router.get(
    "/api-keys",
    summary="List API keys"
)
async def list_api_keys(
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    """List all API keys for current user"""
    
    logger.info("Listing API keys", user_id=current_user.user_id)
    
    records = db.query(APIKey).filter(
        APIKey.user_id == int(current_user.user_id),
        APIKey.is_active == True,
    ).all()
    
    return {
        "user_id": current_user.user_id,
        "keys": [
            {
                "id": str(r.id),
                "name": r.name,
                "created_at": r.created_at.isoformat(),
                "expires_at": r.expires_at.isoformat() if r.expires_at else None,
                "last_used": r.last_used_at.isoformat() if r.last_used_at else None,
            }
            for r in records
        ]
    }


@router.delete(
    "/api-keys/{key_id}",
    summary="Delete API key"
)
async def delete_api_key(
    key_id: str,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    """Delete an API key"""
    
    logger.info(
        "Deleting API key",
        user_id=current_user.user_id,
        key_id=key_id
    )
    
    record = db.query(APIKey).filter(
        APIKey.id == int(key_id),
        APIKey.user_id == int(current_user.user_id),
        APIKey.is_active == True,
    ).first()
    
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found or already deleted"
        )
    
    record.is_active = False
    db.commit()
    
    return {"status": "deleted", "key_id": key_id}


@router.get(
    "/me",
    summary="Get current user info"
)
async def get_current_user_info(
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> dict:
    """Get information about current user"""
    
    user = db.query(User).filter(User.id == int(current_user.user_id)).first()
    subscription_tier = user.subscription_tier if user else "free"
    
    return {
        "user_id": current_user.user_id,
        "email": current_user.email,
        "role": current_user.role,
        "subscription_tier": subscription_tier
    }


@router.post(
    "/logout",
    summary="Logout and revoke current JWT token"
)
async def logout(
    request: Request,
    current_user: CurrentUser = Depends(get_current_user),
) -> dict:
    """Revoke the current JWT token.

    Requires the user to be authenticated. Extracts the token from the
    Authorization header, validates that its subject matches the requesting
    user, and blacklists it so it can no longer be used.
    """
    auth = request.headers.get("Authorization") or request.headers.get("authorization")
    token = None
    if auth and auth.lower().startswith("bearer "):
        token = auth.split(None, 1)[1].strip()

    if not token:
        return {"status": "ok", "revoked": False, "detail": "No token to revoke"}

    revoke_jwt_token(token, int(current_user.user_id))
    return {"status": "ok", "revoked": True}
