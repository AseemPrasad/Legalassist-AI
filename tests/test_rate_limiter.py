import pytest
from unittest.mock import patch
from api.limiter import DistributedRateLimiter


@pytest.mark.asyncio
async def test_rate_limiter_in_memory_fallback():
    limiter = DistributedRateLimiter()
    limiter.enabled = True
    
    # Force get_redis to raise connection error
    with patch.object(limiter, "get_redis", side_effect=Exception("Redis connection refused")):
        # First request should be allowed
        allowed1 = await limiter.check_rate_limit("user1", "/test", limit=2, window_seconds=60)
        assert allowed1 is True
        
        # Second request should be allowed
        allowed2 = await limiter.check_rate_limit("user1", "/test", limit=2, window_seconds=60)
        assert allowed2 is True
        
        # Third request should exceed the limit of 2 and return False
        allowed3 = await limiter.check_rate_limit("user1", "/test", limit=2, window_seconds=60)
        assert allowed3 is False
        
        # Remaining TTL should be calculated from in-memory fallback oldest timestamp
        ttl = await limiter.get_remaining_ttl("user1", "/test", window_seconds=60)
        assert 0 < ttl <= 60
