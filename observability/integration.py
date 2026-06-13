"""
Integration module for observability instrumentation
Add to app.py and other entry points
"""

import os
import logging
from datetime import datetime, timezone
from observability.instrumentation import (
    initialize_observability,
    log,
    tracer,
    traced_operation,
    correlation_context,
    generate_correlation_id,
)
from observability.slack_notifier import initialize_slack_notifier


def setup_production_observability():
    """Initialize all observability components in production"""
    
    # 1. Setup structured logging
    initialize_observability()
    
    # 2. Setup Slack alert notifications
    initialize_slack_notifier()
    
    # 3. Log startup
    log.info(
        "application_started",
        environment=os.getenv("ENVIRONMENT", "development"),
        version="1.0.0",
        timestamp=str(datetime.now(timezone.utc)),
    )
    
    # 4. Verify all services are connected
    verify_observability_services()


def verify_observability_services():
    """Verify observability services are accessible"""
    import requests
    
    services = {
        "prometheus": os.getenv("PROMETHEUS_URL", "http://prometheus:9090"),
        "jaeger": os.getenv("JAEGER_AGENT_HOST", "localhost"),
        "elasticsearch": os.getenv("ELASTICSEARCH_URL", "http://elasticsearch:9200"),
    }
    
    for service_name, url in services.items():
        try:
            if "prometheus" in service_name:
                requests.get(f"{url}/-/healthy", timeout=5)
            elif "elasticsearch" in service_name:
                requests.get(url, timeout=5)
            log.info(f"observability_service_available", service=service_name)
        except Exception as e:
            log.warning(
                f"observability_service_unavailable",
                service=service_name,
                error=str(e),
            )


# ==================== Middleware for Streamlit ====================

def add_correlation_id_to_requests(app):
    """Streamlit-compatible correlation ID tracking"""
    import streamlit as st
    
    # Initialize correlation context if not present
    if "correlation_id" not in st.session_state:
        st.session_state.correlation_id = generate_correlation_id()
        correlation_context.correlation_id = st.session_state.correlation_id
    
    return st.session_state.correlation_id


# ==================== Flask/FastAPI Middleware ====================

def create_flask_middleware(app):
    """Flask middleware for observability"""
    from flask import request, g
    from observability.instrumentation import (
        http_requests_total,
        http_request_duration_seconds,
        correlation_context,
        generate_correlation_id,
    )
    import time
    
    @app.before_request
    def before_request():
        g.start_time = time.time()
        g.correlation_id = request.headers.get("X-Correlation-ID") or generate_correlation_id()
        correlation_context.correlation_id = g.correlation_id
        correlation_context.user_id = request.headers.get("X-User-ID")
    
    @app.after_request
    def after_request(response):
        duration = time.time() - g.start_time
        
        # Track metrics
        http_requests_total.labels(
            method=request.method,
            endpoint=request.endpoint or "unknown",
            status=response.status_code,
        ).inc()
        
        http_request_duration_seconds.labels(
            method=request.method,
            endpoint=request.endpoint or "unknown",
        ).observe(duration)
        
        # Add correlation ID to response
        response.headers["X-Correlation-ID"] = g.correlation_id
        
        return response
    
    return app


# ==================== FastAPI Middleware ====================

def create_fastapi_middleware(app):
    """FastAPI middleware for observability request correlation"""
    from fastapi import Request
    from starlette.middleware.base import BaseHTTPMiddleware
    from observability.instrumentation import (
        correlation_context,
        generate_correlation_id,
        use_extracted_trace_context,
        get_current_trace_headers,
    )

    class FastAPICorrelationMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            correlation_id = (
                request.headers.get("X-Correlation-ID")
                or request.headers.get("X-Request-ID")
                or request.headers.get("x-correlation-id")
                or request.headers.get("x-request-id")
                or generate_correlation_id()
            )
            # Bind in context thread-local
            correlation_context.correlation_id = correlation_id
            
            # Set request state attributes expected by background workers
            request.state.correlation_id = correlation_id
            request.state.request_id = correlation_id
            
            user_id = request.headers.get("X-User-ID")
            if user_id:
                correlation_context.user_id = user_id
                request.state.user_id = user_id

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
            response.headers["X-Correlation-ID"] = correlation_id
            response.headers["X-Request-ID"] = correlation_id
            return response

    app.add_middleware(FastAPICorrelationMiddleware)
    return app


# ==================== Initialization Helper ====================

def initialize_observability_for_environment():
    """Auto-detect environment and initialize appropriately"""
    env = os.getenv("ENVIRONMENT", "development")
    
    if env == "production":
        setup_production_observability()
    elif env in ["staging", "testing"]:
        initialize_observability()
        log.info("observability_initialized_staging_mode")
    else:
        # Development mode: minimal logging
        logging.basicConfig(level=logging.DEBUG)
        log.info("observability_initialized_dev_mode")
