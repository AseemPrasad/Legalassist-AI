"""
API Rate Limiting and Middleware
"""
import hashlib
import time
from typing import Callable, Optional
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
import redis
import structlog

from api.config import get_settings
from observability.instrumentation import (
    bind_request_context,
    clear_request_context,
    capture_exception,
    generate_correlation_id,
    observe_request,
    record_api_error,
    traced_operation,
)

logger = structlog.get_logger(__name__)
settings = get_settings()


class RateLimiter:
    """Sliding-window rate limiter with per-endpoint and global enforcement."""

    _SLIDING_WINDOW_SCRIPT = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
local cutoff = now - (window * 1000)

redis.call('ZREMRANGEBYSCORE', key, 0, cutoff)
local count = redis.call('ZCARD', key)

if count >= limit then
    local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
    local retry_after = math.ceil((tonumber(oldest[2]) + window * 1000 - now) / 1000)
    return {0, retry_after}
end

redis.call('ZADD', key, now, now .. ':' .. ARGV[4])
redis.call('PEXPIRE', key, window * 1000 + 1000)
return {1, 0}
"""

    def __init__(self):
        self._redis: Optional[redis.Redis] = None
        self._script = None

    def _get_client(self) -> redis.Redis:
        if self._redis is None:
            self._redis = redis.from_url(settings.REDIS_URL or "redis://localhost:6379/0", decode_responses=True)
            self._script = self._redis.register_script(self._SLIDING_WINDOW_SCRIPT)
        return self._redis

    def _endpoint_key(self, user_id: str, path: str) -> str:
        ep_hash = hashlib.sha256(path.encode("utf-8")).hexdigest()[:12]
        return f"ratelimit:ep:{ep_hash}:{user_id}"

    def _global_key(self, user_id: str) -> str:
        return f"ratelimit:global:{user_id}"

    def check(self, key: str, limit: int, window: int) -> tuple[bool, int]:
        """Returns (allowed, retry_after_seconds)."""
        try:
            client = self._get_client()
            now_ms = int(time.time() * 1000)
            unique = str(time.monotonic_ns())
            result = self._script(keys=[key], args=[now_ms, window, limit, unique])
            allowed = bool(int(result[0]))
            retry_after = int(result[1])
            return allowed, retry_after
        except Exception as e:
            logger.error("Rate limiter error", error=str(e))
            return True, 0

    def get_retry_after(self, key: str) -> int:
        try:
            client = self._get_client()
            now_ms = int(time.time() * 1000)
            oldest = client.zrange(key, 0, 0, withscores=True)
            if oldest:
                return max(1, int((oldest[0][1] + 60000 - now_ms) / 1000))
        except Exception:
            pass
        return 60

    def current_count(self, key: str) -> int:
        try:
            client = self._get_client()
            now_ms = int(time.time() * 1000)
            cutoff = now_ms - 60000
            client.zremrangebyscore(key, 0, cutoff)
            return int(client.zcard(key) or 0)
        except Exception:
            return 0


_limiter: Optional[RateLimiter] = None


def get_limiter() -> RateLimiter:
    global _limiter
    if _limiter is None:
        _limiter = RateLimiter()
    return _limiter


async def rate_limit_middleware(request: Request, call_next: Callable):
    """Rate limiting middleware — enforces per-endpoint and global limits."""

    if not settings.RATE_LIMIT_ENABLED:
        return await call_next(request)

    if request.url.path in ["/api/v1/health", "/api/v1/health/ready", "/api/v1/health/live"]:
        return await call_next(request)

    client_ip = request.client.host if request.client else "unknown"
    user_id = request.headers.get("X-User-Id", client_ip)

    limiter = get_limiter()
    path = request.url.path

    # Per-endpoint check
    ep_key = limiter._endpoint_key(user_id, path)
    ep_allowed, ep_retry = limiter.check(ep_key, settings.RATE_LIMIT_REQUESTS, settings.RATE_LIMIT_WINDOW)

    if not ep_allowed:
        logger.warning("rate_limit_exceeded_endpoint", user_id=user_id, path=path, retry_after=ep_retry)
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "error_code": "RATE_LIMIT_EXCEEDED",
                "message": f"Endpoint rate limit exceeded. Max {settings.RATE_LIMIT_REQUESTS} requests per {settings.RATE_LIMIT_WINDOW} seconds.",
                "retry_after": ep_retry,
            },
            headers={"Retry-After": str(ep_retry)},
        )

    # Global check
    gbl_key = limiter._global_key(user_id)
    gbl_allowed, gbl_retry = limiter.check(gbl_key, settings.GLOBAL_RATE_LIMIT_REQUESTS, settings.GLOBAL_RATE_LIMIT_WINDOW)

    if not gbl_allowed:
        logger.warning("rate_limit_exceeded_global", user_id=user_id, retry_after=gbl_retry)
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "error_code": "RATE_LIMIT_EXCEEDED",
                "message": f"Global rate limit exceeded. Max {settings.GLOBAL_RATE_LIMIT_REQUESTS} requests per {settings.GLOBAL_RATE_LIMIT_WINDOW} seconds.",
                "retry_after": gbl_retry,
            },
            headers={"Retry-After": str(gbl_retry)},
        )

    response = await call_next(request)
    response.headers["X-RateLimit-Limit"] = str(settings.RATE_LIMIT_REQUESTS)
    response.headers["X-RateLimit-Global-Limit"] = str(settings.GLOBAL_RATE_LIMIT_REQUESTS)
    return response


async def add_correlation_id_middleware(request: Request, call_next: Callable):
    """Add correlation ID to all requests"""
    
    correlation_id = request.headers.get("X-Correlation-Id")
    if not correlation_id:
        correlation_id = generate_correlation_id()
    
    request.state.correlation_id = correlation_id
    request.state.request_id = correlation_id
    request.state.user_id = request.headers.get("X-User-Id") or request.headers.get("Authorization")
    
    try:
        response = await call_next(request)
        response.headers["X-Correlation-Id"] = correlation_id
        response.headers["X-Request-Id"] = correlation_id
        return response
    finally:
        pass


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
        record_api_error(request.url.path, e)
        capture_exception(e, path=request.url.path, method=request.method)
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
    endpoint = request.url.path
    request_id = getattr(request.state, "request_id", request.headers.get("X-Correlation-Id") or generate_correlation_id())
    user_id = getattr(request.state, "user_id", request.headers.get("X-User-Id"))

    bind_request_context(request_id=request_id, user_id=user_id)

    with traced_operation(
        f"http {request.method} {endpoint}",
        {
            "http.method": request.method,
            "http.target": endpoint,
            "request.id": request_id,
            "user.id": user_id or "anonymous",
        },
    ):
        try:
            response = await call_next(request)
        except Exception as exc:
            duration = time.time() - start_time
            observe_request(endpoint, request.method, 500, duration)
            logger.error(
                "http_request_failed",
                method=request.method,
                path=endpoint,
                status_code=500,
                duration_ms=round(duration * 1000, 2),
                request_id=request_id,
                user_id=user_id,
                error=str(exc),
            )
            raise
        finally:
            clear_request_context()

    process_time = time.time() - start_time
    observe_request(endpoint, request.method, response.status_code, process_time)

    logger.info(
        "http_request_completed",
        method=request.method,
        path=endpoint,
        status_code=response.status_code,
        duration_ms=round(process_time * 1000, 2),
        request_id=request_id,
        user_id=user_id,
    )

    response.headers["X-Process-Time"] = str(process_time)
    response.headers["X-Request-Id"] = request_id
    return response
