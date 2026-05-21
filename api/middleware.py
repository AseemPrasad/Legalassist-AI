"""API middleware for request context, error handling, and logging.

The composable security middlewares now live in api.middlewares.* and are
re-exported here for backward compatibility.
"""

from __future__ import annotations

import time
from typing import Callable

import structlog
from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse

from api.config import get_settings
from api.middlewares.idempotency import http_idempotency_manager, idempotency_middleware, is_safe_to_cache
from api.middlewares.rate_limit import rate_limit_middleware
from api.middlewares.request_size import request_size_limit_middleware
from api.limiter import limiter
from observability.instrumentation import (
    bind_request_context,
    capture_exception,
    clear_request_context,
    generate_correlation_id,
    observe_request,
    record_api_error,
    traced_operation,
)

try:
    from db.session import apply_rls_context, clear_rls_context, _is_postgres
except Exception:
    apply_rls_context = None
    clear_rls_context = None
    _is_postgres = False

try:
    from api.csrf import validate_csrf as _csrf_validate
except Exception:
    _csrf_validate = None

settings = get_settings()
logger = structlog.get_logger(__name__)


async def add_correlation_id_middleware(request: Request, call_next: Callable):
    """Attach correlation and request IDs to the request context."""

    correlation_id = request.headers.get("X-Correlation-Id") or generate_correlation_id()
    request.state.correlation_id = correlation_id
    request.state.request_id = correlation_id
    request.state.user_id = getattr(request.state, "rate_limit_identifier", request.headers.get("X-User-Id", "anonymous"))

    response = await call_next(request)
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
    user_id_attr = getattr(request.state, "user_id", request.headers.get("X-User-Id", "anonymous"))

    bind_request_context(request_id=request_id, user_id=user_id_attr)

    if apply_rls_context and _is_postgres and user_id_attr not in (None, "anonymous", ""):
        request.state.db_rls_user_id = user_id_attr

    if _csrf_validate and request.method not in {"GET", "HEAD", "OPTIONS"}:
        try:
            user_id_int = int(user_id_attr) if str(user_id_attr).isdigit() else None
            if user_id_int:
                _csrf_validate(request, current_user_id=user_id_int, allowed_hosts=None)
        except Exception as exc:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=403,
                content={"detail": getattr(exc, "detail", "CSRF validation failed")},
            )

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
    transfer_encoding = request.headers.get("transfer-encoding", "").lower()

    # For upload endpoints, require Content-Length header (no chunked fallback).
    content_length = request.headers.get("content-length")
    max_size = _request_size_limit_for_path(request.url.path)

    if any(request.url.path.startswith(p) for p in UPLOAD_PATH_PREFIXES):
        # Must provide explicit content-length for uploads
        if content_length is None:
            return JSONResponse(
                status_code=status.HTTP_411_LENGTH_REQUIRED,
                content={
                    "error_code": "LENGTH_REQUIRED",
                    "message": "Content-Length header is required for upload endpoints.",
                },
            )

    # Reject explicit chunked transfer encoding as ambiguous
    if "chunked" in transfer_encoding:
        return JSONResponse(
            status_code=status.HTTP_411_LENGTH_REQUIRED,
            content={
                "error_code": "CHUNKED_ENCODING_NOT_SUPPORTED",
                "message": "Chunked transfer encoding is not supported. Provide Content-Length header.",
            },
        )

    content_length = request.headers.get("content-length")
    if content_length is None:
        return JSONResponse(
            status_code=status.HTTP_411_LENGTH_REQUIRED,
            content={
                "error_code": "LENGTH_REQUIRED",
                "message": "Content-Length header is required for all requests.",
            },
        )

    try:
        content_length_bytes = int(content_length)
    except (TypeError, ValueError):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error_code": "INVALID_CONTENT_LENGTH",
                "message": "Content-Length must be a valid integer.",
            },
        )

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

