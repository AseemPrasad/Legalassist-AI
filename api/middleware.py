"""API middleware for request context, error handling, and logging."""
import hashlib
import time
import threading
from typing import Callable, Optional

from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
import redis
import structlog

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


def sanitize_log_text(text: str) -> str:
    """Strip control characters from log text."""
    return text.replace("\x00", "").replace("\n", " ").replace("\r", " ")[:1000]


def structured_error_response(status_code: int, error_code: str, message: str, request: Request):
    """Build standardized error JSONResponse."""
    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "error_code": error_code,
            "message": message,
            "path": request.url.path,
        },
    )


async def add_correlation_id_middleware(request: Request, call_next: Callable):
    """Inject W3C Trace Context traceparent on every request and propagate through response."""
    try:
        from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
        from opentelemetry import trace as otel_trace
        propagator = TraceContextTextMapPropagator()
        tracer = otel_trace.get_tracer(__name__)
    except Exception:
        propagator = None
        tracer = None

    # Extract incoming trace context from headers
    incoming_carrier = {
        key.lower(): value
        for key, value in request.headers.items()
        if key.lower() in {"traceparent", "tracestate", "baggage"}
    }

    # If no traceparent exists, start a new trace
    if "traceparent" not in incoming_carrier:
        if tracer:
            with tracer.start_as_current_span("http_request") as span:
                trace_id = format(span.get_span_context().trace_id, "032x")
                span_id = format(span.get_span_context().span_id, "016x")
                incoming_carrier["traceparent"] = f"00-{trace_id}-{span_id}-01"
        else:
            correlation_id = generate_correlation_id()
            incoming_carrier["traceparent"] = f"00-{correlation_id}-0000000000000001-01"

    traceparent = incoming_carrier.get("traceparent", "")
    correlation_id = traceparent.split("-")[1] if "-" in traceparent else generate_correlation_id()


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
    request.state.traceparent = traceparent
    request.state.trace_headers = incoming_carrier

    bind_request_context(request_id=correlation_id, user_id=getattr(request.state, "user_id", None))

    with use_extracted_trace_context(incoming_carrier):
        response = await call_next(request)

    # Propagate trace context in response headers
    response.headers["traceparent"] = traceparent
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
    """Log all requests and responses with trace context."""
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


# Re-export middlewares from api.middlewares.* for backward compatibility
from api.middlewares.idempotency import http_idempotency_manager, idempotency_middleware, is_safe_to_cache
from api.middlewares.rate_limit import rate_limit_middleware, settings as rate_limit_settings
from api.middlewares.request_size import request_size_limit_middleware

__all__ = [
    "add_correlation_id_middleware",
    "error_handling_middleware",
    "http_idempotency_manager",
    "idempotency_middleware",
    "is_safe_to_cache",
    "logging_middleware",
    "rate_limit_middleware",
    "request_size_limit_middleware",
    "sanitize_log_text",
    "structured_error_response",
]