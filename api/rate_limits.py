
import time
from collections import defaultdict, deque
from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session
from database import get_db, APIKey
from api.auth import CurrentUser, get_current_user

MAX_KEYS_PER_USER = 5
RATE_LIMIT_WINDOW = 60
RATE_LIMIT_MAX = 3


class _PerUserRateLimiter:
    def __init__(self):
        self._buckets: dict[str, deque] = defaultdict(deque)

    def is_allowed(self, user_id: str) -> bool:
        now = time.time()
        bucket = self._buckets[user_id]
        while bucket and bucket[0] < now - RATE_LIMIT_WINDOW:
            bucket.popleft()
        if len(bucket) >= RATE_LIMIT_MAX:
            return False
        bucket.append(now)
        return True


_api_key_limiter = _PerUserRateLimiter()


async def check_api_key_creation_limit(
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> None:
    active_count = db.query(APIKey).filter(
        APIKey.user_id == int(current_user.user_id),
        APIKey.is_active == True,
    ).count()
    if active_count >= MAX_KEYS_PER_USER:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Maximum of {MAX_KEYS_PER_USER} active API keys reached",
        )
    if not _api_key_limiter.is_allowed(current_user.user_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded for API key creation",
        )
