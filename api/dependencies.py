"""
Dependency injection and common dependencies
"""
from typing import Optional
from fastapi import Depends, HTTPException, status
from api.auth import get_current_user, get_current_user_optional, CurrentUser


async def get_rate_limit_key(
    current_user: Optional[CurrentUser] = Depends(get_current_user_optional)
) -> str:
    """Get rate limit key for current user/API key.

    Uses get_current_user_optional so that unauthenticated requests are not
    rejected during dependency resolution — they fall back to an anonymous
    identifier instead of bypassing rate-limit evaluation entirely.
    """
    if current_user:
        return f"user:{current_user.user_id}"
    return "anonymous"


async def verify_api_version(
    api_version: Optional[str] = None
) -> str:
    """Verify API version from query parameter"""
    if api_version and api_version not in ["v1"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported API version: {api_version}. Use v1"
        )
    return api_version or "v1"
