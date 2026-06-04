"""
API Rate Limiting and Middleware
"""
import time
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
    try:
        current_count = int(rate_limiter.redis.get(rate_limit_key) or 0)
    except Exception:
        current_count = 0
    response.headers["X-RateLimit-Remaining"] = str(max(0, rate_limiter.requests - current_count))
    
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
