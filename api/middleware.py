"""
API Rate Limiting and Middleware
"""
import time
from typing import Callable
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
import redis
import structlog

logger = structlog.get_logger(__name__)


class RateLimiter:
    """Token bucket rate limiter using Redis"""

    # Lua script: atomically increment the counter and set TTL on first write.
    # Redis executes Lua scripts as a single atomic operation, so there is no
    # window between INCR and EXPIRE where the key can be left without a TTL.
    _INCR_EXPIRE_SCRIPT = """
local current = redis.call('INCR', KEYS[1])
if current == 1 then
    redis.call('EXPIRE', KEYS[1], ARGV[1])
end
return current
"""

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis = redis.from_url(redis_url, decode_responses=True)
        self.requests = 100  # requests
        self.window = 60  # seconds
        self._script = self.redis.register_script(self._INCR_EXPIRE_SCRIPT)

    def is_allowed(self, key: str) -> bool:
        """Check if request is allowed"""
        try:
            current = int(self._script(keys=[key], args=[self.window]))
            return current <= self.requests
        except Exception as e:
            logger.error("Rate limiter error", error=str(e))
            # Fail open - allow request if Redis unavailable
            return True
    
    def get_retry_after(self, key: str) -> int:
        """Get seconds until next request allowed"""
        try:
            ttl = self.redis.ttl(key)
            return ttl if ttl > 0 else self.window
        except:
            return self.window


async def rate_limit_middleware(request: Request, call_next: Callable):
    """Rate limiting middleware"""
    
    # Skip rate limiting for health checks
    if request.url.path in ["/api/v1/health", "/api/v1/health/ready", "/api/v1/health/live"]:
        return await call_next(request)
    
    # Get client identifier
    client_ip = request.client.host if request.client else "unknown"
    user_id = request.headers.get("X-User-Id", client_ip)
    
    rate_limiter = RateLimiter()
    rate_limit_key = f"ratelimit:{user_id}:{int(time.time() // 60)}"
    
    if not rate_limiter.is_allowed(rate_limit_key):
        logger.warning(
            "Rate limit exceeded",
            user_id=user_id,
            ip=client_ip
        )
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "error_code": "RATE_LIMIT_EXCEEDED",
                "message": f"Rate limit exceeded. Max {rate_limiter.requests} requests per {rate_limiter.window} seconds",
                "retry_after": rate_limiter.get_retry_after(rate_limit_key)
            },
            headers={"Retry-After": str(rate_limiter.get_retry_after(rate_limit_key))}
        )
    
    response = await call_next(request)
    response.headers["X-RateLimit-Limit"] = str(rate_limiter.requests)
    response.headers["X-RateLimit-Remaining"] = str(max(0, rate_limiter.requests - int(self.redis.get(rate_limit_key) or 0)))
    
    return response


async def add_correlation_id_middleware(request: Request, call_next: Callable):
    """Add correlation ID to all requests"""
    
    correlation_id = request.headers.get("X-Correlation-Id")
    if not correlation_id:
        import uuid
        correlation_id = str(uuid.uuid4())
    
    request.state.correlation_id = correlation_id
    
    response = await call_next(request)
    response.headers["X-Correlation-Id"] = correlation_id
    
    return response


async def error_handling_middleware(request: Request, call_next: Callable):
    """Global error handling middleware"""
    
    try:
        response = await call_next(request)
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Unhandled error",
            path=request.url.path,
            method=request.method,
            error=str(e)
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error_code": "INTERNAL_SERVER_ERROR",
                "message": "An internal error occurred"
            }
        )


async def logging_middleware(request: Request, call_next: Callable):
    """Log all requests and responses"""
    
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    
    logger.info(
        "HTTP Request",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=process_time * 1000
    )
    
    response.headers["X-Process-Time"] = str(process_time)
    
    return response
