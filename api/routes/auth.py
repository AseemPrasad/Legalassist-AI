"""
Authentication Endpoints
POST /api/v1/auth/token - Get access token
POST /api/v1/auth/api-key - Create API key
GET /api/v1/auth/me - Get current user
"""
from fastapi import APIRouter, HTTPException, status, Depends
from datetime import datetime, timedelta
from api.auth import create_access_token, generate_api_key, hash_api_key, CurrentUser, get_current_user
from api.models import TokenRequest, TokenResponse, APIKeyCreate, APIKeyResponse
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
    
    # In production, validate against database
    logger.info("Token request", username=request.username)
    
    token = create_access_token({"sub": "user123", "email": request.username, "role": "user"})
    
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
    db: Session = Depends(get_db_rls)
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
    db: Session = Depends(get_db_rls)
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
    db: Session = Depends(get_db_rls)
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




@router.post(
    "/logout",
    summary="Logout and revoke current JWT token",
    dependencies=[Depends(RateLimit(use_auth_defaults=True))]
)
async def logout(request: Request) -> dict:
    """Revoke the JWT presented in Authorization header (if any)."""
    auth = request.headers.get("Authorization") or request.headers.get("authorization")
    token = None
    if auth and auth.lower().startswith("bearer "):
        token = auth.split(None, 1)[1].strip()

    if not token:
        # Nothing to revoke, but return success to avoid token probing
        return {"status": "ok", "revoked": False}

    try:
        success = api_revoke_jwt(token)
        if success:
            logger.info("api_logout_revoked")
        else:
            logger.info("api_logout_no_revoke_needed")
        return {"status": "ok", "revoked": bool(success)}
    except Exception as e:
        logger.error("api_logout_failed", error=str(e))
        return {"status": "error", "revoked": False}
