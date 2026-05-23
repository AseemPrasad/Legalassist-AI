"""
Authentication Endpoints
POST /api/v1/auth/token - Get access token
POST /api/v1/auth/api-key - Create API key
GET /api/v1/auth/me - Get current user
"""
from fastapi import APIRouter, HTTPException, status, Depends
from datetime import datetime, timedelta, timezone
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from database import get_db
from db.models import APIKey
from db.models import APIKey
from api.auth import create_access_token, create_api_key_record, CurrentUser, get_current_user, verify_password
from database import SessionLocal
from api.models import TokenResponse, APIKeyCreate, APIKeyResponse
from api.limiter import RateLimit
import structlog
from core.log_redaction import mask_email

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
    username: str,
    password: str
) -> TokenResponse:
    """
    Authenticate user and get access token.

    Validates credentials against the database.
    """
    from database import get_user_by_email

    logger.info("token_request_received", username=mask_email(username))

    db = SessionLocal()
    try:
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
            expires_in=24 * 3600
        )
    finally:
        db.close()


@router.post(
    "/api-keys",
    response_model=APIKeyResponse,
    summary="Create API key"
)
async def create_api_key(
    request: APIKeyCreate,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db)
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
    
    return APIKeyResponse(
        id=api_key_record.key_id,
        name=api_key_record.name,
        key=key,  # This is the combined key: key_id.secret
        created_at=api_key_record.created_at,
        expires_at=api_key_record.expires_at
    )


@router.get(
    "/api-keys",
    summary="List API keys"
)
async def list_api_keys(
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> dict:
    """List all API keys for current user"""
    
    logger.info("Listing API keys", user_id=current_user.user_id)
    
    keys = db.query(APIKey).filter(APIKey.user_id == current_user.user_id).all()
    
    return {
        "user_id": current_user.user_id,
        "keys": [
            {
                "id": k.key_id,
                "name": k.name,
                "created_at": k.created_at.isoformat() if k.created_at else None,
                "expires_at": k.expires_at.isoformat() if k.expires_at else None,
                "last_used": None
            }
            for k in keys
        ]
    }


@router.delete(
    "/api-keys/{key_id}",
    summary="Delete API key"
)
async def delete_api_key(
    key_id: str,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> dict:
    """Delete an API key"""
    
    logger.info(
        "Deleting API key",
        user_id=current_user.user_id,
        key_id=key_id
    )
    
    key_record = db.query(APIKey).filter(
        APIKey.key_id == key_id,
        APIKey.user_id == current_user.user_id
    ).first()
    
    if not key_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found"
        )
        
    db.delete(key_record)
    db.commit()
    
    return {"status": "deleted", "key_id": key_id}


@router.get(
    "/me",
    summary="Get current user info"
)
async def get_current_user_info(
    current_user: CurrentUser = Depends(get_current_user)
) -> dict:
    """Get information about current user"""
    
    return {
        "user_id": current_user.user_id,
        "email": current_user.email,
        "role": current_user.role,
        "subscription_tier": "pro"
    }
