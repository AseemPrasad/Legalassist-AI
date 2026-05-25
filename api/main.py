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
import structlog

from api.config import get_settings
from api.middleware import (
    rate_limit_middleware,
    add_correlation_id_middleware,
    error_handling_middleware,
    logging_middleware,
    request_size_limit_middleware
)
from api.idempotency_middleware import idempotency_middleware
from observability.integration import initialize_observability_for_environment
from observability.instrumentation import get_metrics
from api.validation import (
    ValidationConfig,
    ValidationError,
    PayloadTooLargeError,
)

# Import routes
from api.routes import documents, cases, reports, analytics, deadlines, auth, health, case_search, speech, document_verification, argument_strength, deadline_engine, efiling

settings = get_settings()
logger = structlog.get_logger(__name__)


# ============================================================================
# Middleware Configuration
# ============================================================================

middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    ),
    Middleware(
        TrustedHostMiddleware,
        allowed_hosts=["localhost", "127.0.0.1", "*.example.com"]
    ),
]


# ============================================================================
# FastAPI Application
# ============================================================================

def create_app() -> FastAPI:
    """Create FastAPI application"""
    
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
    # Model feedback & optimization
    from api.routes import models as models_router
    app.include_router(models_router.router)
    
    # ========================================================================
    # Global Exception Handlers
    # ========================================================================
    
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
                "error_code": "VALIDATION_ERROR",
                "message": exc.detail
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
                "error_code": "PAYLOAD_TOO_LARGE",
                "message": exc.detail,
                "status_code": 413
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
                "error_code": "INTERNAL_SERVER_ERROR",
                "message": "An internal error occurred"
            }
        )
    
    # ========================================================================
    # Startup/Shutdown Events
    # ========================================================================
    
    @app.on_event("startup")
    async def startup_event():
        """Initialize on startup"""
        initialize_observability_for_environment()
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
    
    return app


# Create app instance
app = create_app()


# ============================================================================
# WebSocket Support (Optional)
# ============================================================================

if settings.ENABLE_WEBSOCKET:
    from fastapi import WebSocket
    
    @app.websocket("/ws/progress/{job_id}")
    async def websocket_progress_endpoint(websocket: WebSocket, job_id: str):
        """
        WebSocket endpoint for real-time job progress
        
        Usage:
        ws = new WebSocket('ws://localhost:8000/ws/progress/job_id')
        ws.onmessage = (event) => console.log(event.data)
        """
        await websocket.accept()
        
        try:
            from celery_app import TaskStatus
            
            while True:
                import asyncio
                
                status_info = TaskStatus.get_task_status(job_id)
                
                await websocket.send_json({
                    "job_id": job_id,
                    "status": status_info["status"],
                    "progress": status_info["info"].get("progress", 0),
                    "timestamp": status_info["timestamp"]
                })
                
                # Update every 2 seconds
                await asyncio.sleep(2)
                
                # Stop if completed
                if status_info["status"] in ["completed", "failed", "cancelled"]:
                    await websocket.send_json({
                        "job_id": job_id,
                        "status": status_info["status"],
                        "message": "Job completed"
                    })
                    break
        
        except Exception as e:
            logger.error("WebSocket error", job_id=job_id, error=str(e))
            await websocket.close(code=1011)


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        workers=settings.API_WORKERS,
        reload=True
    )
