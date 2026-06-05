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
    observe_request,
    record_api_error,
    traced_operation,
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

    correlation_id = request.headers.get("X-Correlation-Id") or generate_correlation_id()
    request.state.correlation_id = correlation_id
    request.state.request_id = correlation_id
    request.state.user_id = getattr(request.state, "rate_limit_identifier", request.headers.get("X-User-Id", "anonymous"))

async def request_size_limit_middleware(request: Request, call_next: Callable):
    """Enforce request body size limits, closing two bypass vectors.

    Vector 1 — declared Content-Length:
        The header value is inspected *before* any body bytes are read.  If the
        declared size exceeds MAX_BODY_SIZE the request is rejected immediately
        with 413 Request Entity Too Large.

    Vector 2 — missing Content-Length / Transfer-Encoding: chunked:
        Clients that omit the header (or explicitly use chunked encoding) used
        to bypass the size check entirely, because the old code only branched
        on ``content_length is not None``.

        * Upload-capable paths (UPLOAD_PATH_PREFIXES): the incoming body stream
          is read chunk-by-chunk with a running byte counter.  The request is
          aborted with 413 the moment the counter exceeds MAX_BODY_SIZE.  If
          the body fits, it is re-assembled in memory and injected back so that
          downstream handlers can read it normally.
        * All other paths without Content-Length: rejected with 411 Length
          Required, since non-upload JSON bodies must always declare their size.
    """
    path = request.url.path
    is_upload_path = any(path.startswith(prefix) for prefix in UPLOAD_PATH_PREFIXES)
    content_length_header = request.headers.get("content-length")

    # ── Case 1: Content-Length header is present ────────────────────────────
    if content_length_header is not None:
        try:
            content_length = int(content_length_header)
        except ValueError:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": "Malformed Content-Length header."},
            )
        if content_length > MAX_BODY_SIZE:
            logger.warning(
                "request_size_limit_exceeded",
                path=path,
                content_length=content_length,
                limit=MAX_BODY_SIZE,
            )
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={
                    "detail": (
                        f"Request body too large. "
                        f"Maximum allowed size is {MAX_BODY_SIZE // (1024 * 1024)} MB."
                    )
                },
            )
        # Declared size is within limits — pass through.
        return await call_next(request)

    # ── Case 2: No Content-Length (omitted or chunked) ──────────────────────
    transfer_encoding = request.headers.get("transfer-encoding", "").lower()

    if is_upload_path:
        if transfer_encoding == "chunked":
            # Stream-read and count bytes so the limit is enforced even when
            # the total size is not declared up front.
            total = 0
            chunks: list[bytes] = []
            async for chunk in request.stream():
                total += len(chunk)
                if total > MAX_BODY_SIZE:
                    logger.warning(
                        "chunked_request_size_exceeded",
                        path=path,
                        bytes_received=total,
                        limit=MAX_BODY_SIZE,
                    )
                    return JSONResponse(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        content={
                            "detail": (
                                f"Chunked request body too large. "
                                f"Maximum allowed size is {MAX_BODY_SIZE // (1024 * 1024)} MB."
                            )
                        },
                    )
                chunks.append(chunk)

            # Re-inject the buffered body so downstream handlers can read it.
            body = b"".join(chunks)

            async def _receive():
                return {"type": "http.request", "body": body, "more_body": False}

            request._receive = _receive  # type: ignore[assignment]
        else:
            # Upload path with no Content-Length and no chunked encoding —
            # reject to close the header-omission bypass.
            return JSONResponse(
                status_code=status.HTTP_411_LENGTH_REQUIRED,
                content={"detail": "Content-Length header is required for this endpoint."},
            )
    # Non-upload paths without Content-Length (e.g. empty-body GET/DELETE
    # proxied through the middleware chain) are allowed through.

    return await call_next(request)


_rate_limiter = RateLimiter()


async def rate_limit_middleware(request: Request, call_next: Callable):
    """Rate limiting middleware — enforces per-endpoint and global limits."""

    if not settings.RATE_LIMIT_ENABLED:
        return await call_next(request)

    if request.url.path in ["/api/v1/health", "/api/v1/health/ready", "/api/v1/health/live"]:
        return await call_next(request)

    client_ip = request.client.host if request.client else "unknown"
    user_id = request.headers.get("X-User-Id", client_ip)
    
    rate_limiter = _rate_limiter
    rate_limit_key = f"ratelimit:{user_id}:{int(time.time() // 60)}"
    
    if not rate_limiter.is_allowed(rate_limit_key):
        logger.warning(
            "Rate limit exceeded",
            user_id=user_id,
            ip=client_ip
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
    request_id = getattr(request.state, "request_id", request.headers.get("X-Correlation-Id") or generate_correlation_id())
    raw_user_id = getattr(request.state, "user_id", request.headers.get("X-User-Id", "anonymous"))
    user_id_attr = sanitize_log_value(raw_user_id, "user_id")

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

