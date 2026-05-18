"""API middleware for abuse protection, request context, and logging."""

from __future__ import annotations

import time
from typing import Callable

import structlog
from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse

from api.config import get_settings
from api.limiter import (
    build_rate_limit_response,
    get_rate_limit_policy,
    is_whitelisted,
    limiter,
    resolve_rate_limit_identifier,
)
from api.validation import PayloadTooLargeError, ValidationConfig
from observability.instrumentation import (
    bind_request_context,
    capture_exception,
    clear_request_context,
    generate_correlation_id,
    observe_request,
    record_api_error,
    traced_operation,
)

settings = get_settings()
logger = structlog.get_logger(__name__)

SKIP_PATHS = {"/api/v1/health", "/api/v1/health/ready", "/api/v1/health/live", "/metrics", "/"}
UPLOAD_PATH_PREFIXES = (
    "/api/v1/analyze/upload",
    "/api/v1/analyze/document",
    "/api/v1/documents",
)
ANALYTICS_PATH_PREFIXES = (
    "/api/v1/analytics",
)


async def rate_limit_middleware(request: Request, call_next: Callable):
    """Apply global and endpoint-specific Redis sliding-window throttling."""

    if not settings.RATE_LIMIT_ENABLED:
        return await call_next(request)

    path = request.url.path
    if path in SKIP_PATHS:
        return await call_next(request)

    identifier = resolve_rate_limit_identifier(request)
    request.state.rate_limit_identifier = identifier
    request.state.user_id = identifier

    if is_whitelisted(identifier):
        response = await call_next(request)
        response.headers["X-RateLimit-Scope"] = "whitelist"
        return response

    rule, matched_override = get_rate_limit_policy(path, request.method)
    allowed = await limiter.check_rate_limit(
        identifier=identifier,
        endpoint=f"{request.method.upper()} {path}",
        limit=rule.requests,
        window_seconds=rule.window,
        request_id=getattr(request.state, "request_id", None),
    )

    if not allowed:
        retry_after = await limiter.get_remaining_ttl(identifier, f"{request.method.upper()} {path}", rule.window)
        logger.warning(
            "rate_limit_exceeded",
            identifier=identifier,
            path=path,
            method=request.method,
            limit=rule.requests,
            window_seconds=rule.window,
            retry_after=retry_after,
        )
        return build_rate_limit_response(
            retry_after=retry_after,
            message=f"Rate limit exceeded. Limit is {rule.requests} requests per {rule.window} seconds.",
        )

    response = await call_next(request)
    response.headers["X-RateLimit-Limit"] = str(rule.requests)
    response.headers["X-RateLimit-Window"] = str(rule.window)
    response.headers["X-RateLimit-Scope"] = "endpoint" if matched_override else "global"
    return response


async def add_correlation_id_middleware(request: Request, call_next: Callable):
    """Attach correlation and request IDs to the request context."""

    correlation_id = request.headers.get("X-Correlation-Id") or generate_correlation_id()
    request.state.correlation_id = correlation_id
    request.state.request_id = correlation_id
    request.state.user_id = getattr(request.state, "rate_limit_identifier", request.headers.get("X-User-Id", "anonymous"))

    response = await call_next(request)
    response.headers["X-Correlation-Id"] = correlation_id
    response.headers["X-Request-Id"] = correlation_id
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
            error=str(exc),
        )
        record_api_error(request.url.path, exc)
        capture_exception(exc, path=request.url.path, method=request.method)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error_code": "INTERNAL_SERVER_ERROR",
                "message": "An internal error occurred",
            },
        )


async def logging_middleware(request: Request, call_next: Callable):
    """Log request metadata and emit tracing/metrics events."""

    start_time = time.time()
    endpoint = request.url.path
    request_id = getattr(request.state, "request_id", request.headers.get("X-Correlation-Id") or generate_correlation_id())
    user_id = getattr(request.state, "user_id", request.headers.get("X-User-Id", "anonymous"))

    bind_request_context(request_id=request_id, user_id=user_id)

    response = None
    error_occurred = False
    
    with traced_operation(
        f"http {request.method} {endpoint}",
        {
            "http.method": request.method,
            "http.target": endpoint,
            "request.id": request_id,
            "user.id": user_id,
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
                user_id=user_id,
                error=str(exc),
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
            user_id=user_id,
        )
        
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-Request-Id"] = request_id
    
    clear_request_context()
    return response

def _request_size_limit_for_path(path: str) -> int:
    if any(path.startswith(prefix) for prefix in UPLOAD_PATH_PREFIXES):
        return ValidationConfig.MAX_UPLOAD_SIZE
    if any(path.startswith(prefix) for prefix in ANALYTICS_PATH_PREFIXES):
        return ValidationConfig.MAX_ANALYTICS_PAYLOAD
    return ValidationConfig.MAX_JSON_BODY


async def request_size_limit_middleware(request: Request, call_next: Callable):
    """Reject oversized requests before they reach the application layer."""

    if request.url.path in SKIP_PATHS:
        return await call_next(request)

    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            content_length_bytes = int(content_length)
        except (TypeError, ValueError):
            content_length_bytes = None

        if content_length_bytes is not None:
            max_size = _request_size_limit_for_path(request.url.path)
            if content_length_bytes > max_size:
                logger.warning(
                    "request_size_limit_exceeded",
                    path=request.url.path,
                    content_length=content_length_bytes,
                    max_size=max_size,
                    size_mb=round(content_length_bytes / 1024 / 1024, 2),
                )
                return JSONResponse(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    content={
                        "error_code": "PAYLOAD_TOO_LARGE",
                        "message": (
                            f"Request body too large: {round(content_length_bytes / 1024 / 1024, 2)} MB "
                            f"(max {round(max_size / 1024 / 1024, 2)} MB)"
                        ),
                    },
                )

    try:
        return await call_next(request)
    except PayloadTooLargeError as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error_code": "PAYLOAD_TOO_LARGE",
                "message": str(exc.detail),
            },
        )

