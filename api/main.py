"""
Main FastAPI Application
"""
from fastapi import FastAPI, Request
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.openapi.utils import get_openapi
from fastapi import status
import logging
import structlog

from api.config import get_settings
from api.middleware import (
    rate_limit_middleware,
    add_correlation_id_middleware,
    error_handling_middleware,
    logging_middleware,
    request_size_limit_middleware,
    security_headers_middleware,
)
from api.idempotency_middleware import idempotency_middleware
from observability.integration import initialize_observability_for_environment
from observability.instrumentation import get_metrics
from api.errors import register_structured_error_handlers
from api.validation import (
    ValidationConfig,
    ValidationError,
    PayloadTooLargeError,
)
from database import init_db

# Import routes
from api.routes import documents, cases, reports, analytics, deadlines, auth, health, case_search, speech, document_verification, argument_strength, deadline_engine, efiling, notifications as notifications_webhooks, anonymized_cases

logger = structlog.get_logger(__name__)


# ============================================================================
# FastAPI Application
# ============================================================================

def create_app() -> FastAPI:
    """Create FastAPI application"""

    settings = get_settings()

    # Force explicit origins when credentials are enabled — never allow *
    _origins = settings.CORS_ORIGINS
    had_wildcard = False
    if isinstance(_origins, str):
        _origins = [o.strip() for o in _origins.split(",") if o.strip()]
    if "*" in _origins:
        had_wildcard = True
        _origins = [o for o in _origins if o != "*"]
    if not _origins:
        _origins = ["http://localhost:8080"]
    if had_wildcard:
        logging.getLogger(__name__).warning(
            "Removed wildcard '*' from CORS_ORIGINS because allow_credentials=True. "
            "Explicit origins required: %s",
            _origins,
        )

    middleware = [
        Middleware(
            CORSMiddleware,
            allow_origins=_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        ),
        Middleware(
            TrustedHostMiddleware,
            allowed_hosts=["localhost", "127.0.0.1", "*.example.com"]
        ),
    ]

    app = FastAPI(
        title=settings.API_TITLE,
        description="Comprehensive legal case analysis and deadline management API",
        version=settings.API_VERSION,
        middleware=middleware
    )
    
    # Initialize validation config from settings
    ValidationConfig.from_settings(settings)
    
    # Add middleware
    app.middleware("http")(request_size_limit_middleware)
    # Idempotency middleware should run early for POST/PUT/PATCH/DELETE
    app.middleware("http")(idempotency_middleware)
    app.middleware("http")(add_correlation_id_middleware)
    app.middleware("http")(logging_middleware)
    app.middleware("http")(error_handling_middleware)
    app.middleware("http")(security_headers_middleware)
    
    if settings.RATE_LIMIT_ENABLED:
        app.middleware("http")(rate_limit_middleware)
    
    # ========================================================================
    # Include Routers
    # ========================================================================
    
    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(cases.router)
    app.include_router(reports.router)
    app.include_router(analytics.router)
    app.include_router(deadlines.router)
    app.include_router(auth.router)
    app.include_router(case_search.router)  # Case search and precedent matching
    app.include_router(speech.router)
    app.include_router(document_verification.router)
    app.include_router(argument_strength.router)
    app.include_router(deadline_engine.router)
    app.include_router(efiling.router)
    app.include_router(notifications_webhooks.router)
    app.include_router(notifications_webhooks.pref_router)
    app.include_router(anonymized_cases.router)
    # Model feedback & optimization
    from api.routes import models as models_router
    app.include_router(models_router.router)
    
    # ========================================================================
    # Global Exception Handlers
    # ========================================================================

    register_structured_error_handlers(app)
    
    @app.exception_handler(ValidationError)
    async def validation_error_handler(request: Request, exc: ValidationError):
        """Handle validation errors"""
        logger.warning(
            "validation_error",
            path=request.url.path,
            detail=exc.detail
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error_code": "VALIDATION_ERROR",
                "message": exc.detail,
                "status_code": exc.status_code,
            }
        )
    
    @app.exception_handler(PayloadTooLargeError)
    async def payload_too_large_handler(request: Request, exc: PayloadTooLargeError):
        """Handle payload too large errors"""
        logger.warning(
            "payload_too_large",
            path=request.url.path,
            detail=exc.detail
        )
        return JSONResponse(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            content={
                "success": False,
                "error_code": "PAYLOAD_TOO_LARGE",
                "message": exc.detail,
                "status_code": 413,
            },
            headers={"Retry-After": "60"}
        )
    
    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        """Handle all uncaught exceptions"""
        logger.error(
            "Unhandled exception",
            path=request.url.path,
            error=str(exc)
        )
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error_code": "INTERNAL_SERVER_ERROR",
                "message": "An internal error occurred",
                "status_code": 500,
            }
        )
    
    # ========================================================================
    # Startup/Shutdown Events
    # ========================================================================
    
    @app.on_event("startup")
    async def startup_event():
        """Initialize on startup"""
        init_db()
        initialize_observability_for_environment()
        # Attempt to instrument FastAPI with OpenTelemetry (if available)
        try:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
            FastAPIInstrumentor().instrument_app(app)
            logger.info("fastapi_instrumented")
        except Exception:
            logger.debug("fastapi_instrumentation_unavailable_or_failed")
        logger.info(
            "API Starting",
            version=settings.API_VERSION,
            environment=settings.LOG_LEVEL
        )
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on shutdown"""
        logger.info("API Shutting down")
    
    # ========================================================================
    # OpenAPI Customization
    # ========================================================================
    
    def custom_openapi():
        """Customize OpenAPI schema"""
        if app.openapi_schema:
            return app.openapi_schema
        
        openapi_schema = get_openapi(
            title=settings.API_TITLE,
            version=settings.API_VERSION,
            description="Comprehensive legal case analysis and deadline management API",
            routes=app.routes,
        )
        
        # Add security scheme
        openapi_schema["components"]["securitySchemes"] = {
            "bearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT",
                "description": "JWT token from /api/v1/auth/token"
            },
            "apiKeyAuth": {
                "type": "apiKey",
                "in": "header",
                "name": "X-API-Key",
                "description": "API key from /api/v1/auth/api-keys"
            }
        }
        
        # Add examples to paths
        for path_key, path_item in openapi_schema["paths"].items():
            for method_key, operation in path_item.items():
                if isinstance(operation, dict):
                    if "tags" not in operation:
                        operation["tags"] = ["API"]
        
        app.openapi_schema = openapi_schema
        return app.openapi_schema
    
    app.openapi = custom_openapi
    
    # ========================================================================
    # Root Endpoint
    # ========================================================================
    
    @app.get("/")
    async def root():
        """API root endpoint"""
        return {
            "name": settings.API_TITLE,
            "version": settings.API_VERSION,
            "docs": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json"
        }

    @app.get("/metrics")
    async def metrics_endpoint():
        """Prometheus metrics endpoint."""
        return Response(content=get_metrics(), media_type="text/plain; version=0.0.4; charset=utf-8")
    
    # Register WebSocket endpoints
    if getattr(settings, "ENABLE_WEBSOCKET", True):
        from api.websockets.case_timeline import register_case_timeline_endpoint
        from api.websockets.job_progress import register_job_progress_endpoint
        register_case_timeline_endpoint(app)
        register_job_progress_endpoint(app)

    return app


# Lazy app instance — created on first access so tests can call
# create_app() directly with fresh configuration.
_app = None


def get_app() -> FastAPI:
    global _app
    if _app is None:
        _app = create_app()
    return _app


def __getattr__(name):
    if name == "app":
        return get_app()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.main:app",
        host=get_settings().API_HOST,
        port=get_settings().API_PORT,
        workers=get_settings().API_WORKERS,
        reload=True
    )
