"""API middleware for request context, error handling, and logging.

The composable security middlewares now live in api.middlewares.* and are
re-exported here for backward compatibility.
"""
API Rate Limiting and Middleware
"""
import hashlib
import time
import threading
from typing import Callable
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
import redis

# ---------------------------------------------------------------------------
# Request size enforcement configuration
# ---------------------------------------------------------------------------

# Maximum allowed request body in bytes (50 MB).
MAX_BODY_SIZE: int = 50 * 1024 * 1024

# URL path prefixes whose endpoints accept uploaded/streamed bodies and must
# therefore have strict size enforcement even when Content-Length is absent.
UPLOAD_PATH_PREFIXES: tuple = (
    "/api/v1/analyze",
    "/api/v1/documents",
    "/api/v1/cases",
    "/api/v1/reports",
)
import structlog
from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse

from api.config import get_settings
from observability.instrumentation import (
    bind_request_context,
    capture_exception,
    clear_request_context,
    generate_correlation_id,
    get_current_trace_headers,
    observe_request,
    record_api_error,
    traced_operation,
    use_extracted_trace_context,
)

logger = structlog.get_logger(__name__)
settings = get_settings()


class RateLimiter:
    """Token bucket rate limiter using Redis with application-level locking.

    Thread safety:
    - Redis-side:  the INCR + EXPIRE Lua script runs atomically in Redis.
    - App-side:    a module-level ``_lock`` serialises local state access so
                   that concurrent ASGI workers see consistent bucket values.
    """

    _instance = None
    _lock = threading.Lock()

    # Lua script: atomically increment the counter and set TTL on first write.
    # Redis executes Lua scripts as a single atomic operation, so there is no
    # window between INCR and EXPIRE where the key can be left without a TTL.
    _INCR_EXPIRE_SCRIPT = """
local current = redis.call('INCR', KEYS[1])
if current == 1 then
    redis.call('EXPIRE', KEYS[1], ARGV[1])
end

redis.call('ZADD', key, now, now .. ':' .. ARGV[4])
redis.call('PEXPIRE', key, window * 1000 + 1000)
return {1, 0}
"""

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        if not hasattr(self, "_initialised"):
            self.redis = redis.from_url(redis_url, decode_responses=True)
            self.requests = 100  # requests
            self.window = 60  # seconds
            self._script = self.redis.register_script(self._INCR_EXPIRE_SCRIPT)
            self._initialised = True

    def is_allowed(self, key: str) -> bool:
        """Check if request is allowed under the rate limit."""
        try:
            with self._lock:
                current = int(self._script(keys=[key], args=[self.window]))
            return current <= self.requests
        except Exception as e:
            logger.error("Rate limiter error", error=str(e))
            return True
    
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


async def add_correlation_id_middleware(request: Request, call_next: Callable):
    correlation_id = (
        request.headers.get("X-Correlation-Id")
        or request.headers.get("X-Request-Id")
        or request.headers.get("x-correlation-id")
        or request.headers.get("x-request-id")
        or generate_correlation_id()
    )
    request.state.correlation_id = correlation_id
    request.state.request_id = correlation_id
    request.state.user_id = getattr(request.state, "rate_limit_identifier", request.headers.get("X-User-Id", "anonymous"))

    incoming_trace_headers = {
        key.lower(): value
        for key, value in request.headers.items()
        if key.lower() in {"traceparent", "tracestate", "baggage"}
    }
    request.state.trace_headers = incoming_trace_headers

    with use_extracted_trace_context(incoming_trace_headers):
        response = await call_next(request)

    trace_headers = get_current_trace_headers()
    for header_name, header_value in trace_headers.items():
        response.headers[header_name] = header_value
    response.headers["X-Correlation-Id"] = correlation_id
    response.headers["X-Request-Id"] = correlation_id
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


async def error_handling_middleware(request: Request, call_next: Callable):
    """Convert uncaught exceptions into a structured JSON 500 response."""

    try:
        return await call_next(request)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "unhandled_error",
            path=request.url.path,
            method=request.method,
            error=sanitize_log_text(str(exc)),
        )
        record_api_error(request.url.path, exc)
        capture_exception(exc, path=request.url.path, method=request.method)
        return structured_error_response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="INTERNAL_SERVER_ERROR",
            message="An internal error occurred",
            request=request,
        )


async def logging_middleware(request: Request, call_next: Callable):
    """Log all requests and responses
    
    Note: Error handling and tracing blocks are strictly enclosed inside this
    async function scope to prevent global scope exception masking.
    """
    
    start_time = time.time()
    endpoint = request.url.path
    request_id = getattr(
        request.state,
        "request_id",
        request.headers.get("X-Correlation-Id")
        or request.headers.get("X-Request-Id")
        or generate_correlation_id(),
    )
    user_id_attr = getattr(request.state, "user_id", request.headers.get("X-User-Id", "anonymous"))

    bind_request_context(request_id=request_id, user_id=user_id_attr)

    if apply_rls_context and _is_postgres and user_id_attr not in (None, "anonymous", ""):
        # Normalize common identifier shapes ("user:123", numeric strings, ints)
        rls_id = None
        try:
            if isinstance(user_id_attr, int):
                rls_id = int(user_id_attr)
            elif isinstance(user_id_attr, str):
                if user_id_attr.isdigit():
                    rls_id = int(user_id_attr)
                elif user_id_attr.startswith("user:"):
                    parts = user_id_attr.split(":", 1)
                    if len(parts) == 2 and parts[1].isdigit():
                        rls_id = int(parts[1])
        except Exception:
            rls_id = None

        if rls_id is not None:
            request.state.db_rls_user_id = rls_id

    response = None
    error_occurred = False

    try:
        with traced_operation(
            f"http {request.method} {endpoint}",
            {
                "http.method": request.method,
                "http.target": endpoint,
                "request.id": request_id,
                "user.id": user_id_attr,
            },
        ):
            try:
                response = await call_next(request)
            except Exception as exc:
                error_occurred = True
                duration = time.time() - start_time
                observe_request(endpoint, request.method, 500, duration)
                logger.error(
                    "http_request_failed",
                    method=request.method,
                    path=endpoint,
                    status_code=500,
                    duration_ms=round(duration * 1000, 2),
                    request_id=request_id,
                    user_id=user_id_attr,
                    error=sanitize_log_text(str(exc)),
                )
                raise

        process_time = time.time() - start_time

        if not error_occurred and response:
            observe_request(endpoint, request.method, response.status_code, process_time)
            logger.info(
                "http_request_completed",
                method=request.method,
                path=endpoint,
                status_code=response.status_code,
                duration_ms=round(process_time * 1000, 2),
                request_id=request_id,
                user_id=user_id_attr,
            )
            response.headers["X-Process-Time"] = str(process_time)
            response.headers["X-Request-Id"] = request_id

    finally:
        clear_request_context()

    return response
 

__all__ = [
    "add_correlation_id_middleware",
    "error_handling_middleware",
    "http_idempotency_manager",
    "idempotency_middleware",
    "is_safe_to_cache",
    "limiter",
    "logging_middleware",
    "rate_limit_middleware",
    "request_size_limit_middleware",
    "settings",
]

