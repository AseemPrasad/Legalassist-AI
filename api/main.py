"""
Main FastAPI Application
"""
# asyncio imported at module level for performance - avoids repeated import
# resolution inside async hot paths like WebSocket loops
import asyncio
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
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
from api.csrf import CSRFProtectionMiddleware
from api.limiter import cleanup_limiter
from observability.integration import initialize_observability_for_environment
from observability.instrumentation import get_metrics
from api.validation import (
    ValidationConfig,
    ValidationError,
    PayloadTooLargeError,
)

# Import routes
from api.routes import documents, cases, reports, analytics, deadlines, auth, health, case_search
from api.auth import get_current_user_optional

settings = get_settings()
logger = structlog.get_logger(__name__)


def _sanitize_log_text(value: str) -> str:
    """Make log text single-line and safe for structured log sinks."""
    return value.replace("\r", "\\r").replace("\n", "\\n")


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
        allowed_hosts=settings.ALLOWED_HOSTS
    ),
    Middleware(
        CSRFProtectionMiddleware,
        allowed_hosts=set(settings.ALLOWED_HOSTS),
        exempt_paths={"/health", "/ready", "/metrics", "/docs", "/openapi.json"}
    ),
]


# ============================================================================
# FastAPI Application
# ============================================================================

def create_app() -> FastAPI:
    """Create FastAPI application"""

    settings.validate_runtime_security()
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Application lifespan manager"""
        initialize_observability_for_environment()
        
        if settings.RATE_LIMIT_ENABLED:
            logger.info(
                "Rate limiter enabled",
                redis_url=settings.REDIS_URL,
                requests=settings.RATE_LIMIT_REQUESTS,
                window=settings.RATE_LIMIT_WINDOW
            )
        
        logger.info("API Starting", version=settings.API_VERSION)
        
        yield
        
        await cleanup_limiter()
        logger.info("API Shutting down")
    
    app = FastAPI(
        title=settings.API_TITLE,
        description="Comprehensive legal case analysis and deadline management API",
        version=settings.API_VERSION,
        lifespan=lifespan,
        middleware=middleware
    )
    
    # Initialize validation config from settings
    ValidationConfig.from_settings(settings)
    
    # Add middleware
    app.middleware("http")(request_size_limit_middleware)
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
    # Model feedback & optimization
    from api.routes import models as models_router
    app.include_router(models_router.router)
    
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
        components = openapi_schema.setdefault("components", {})
        components["securitySchemes"] = {
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
            status_code=exc.status_code,
            content={
                "error_code": "PAYLOAD_TOO_LARGE",
                "message": exc.detail
            }
        )
    
    # ========================================================================
    # Root Endpoint
    # ========================================================================
    
    @app.get("/")
    async def root(user=Depends(get_current_user_optional)):
        """API root endpoint"""
        user_info = {"authenticated": True, "user_id": user.user_id} if user else {"authenticated": False}
        return {
            "name": settings.API_TITLE,
            "version": settings.API_VERSION,
            "docs": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json",
            "user": user_info
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
    from fastapi import WebSocket, WebSocketDisconnect, Query
    from celery_app import TaskStatus
    from api.auth import AuthError, TokenExpiredError, InvalidTokenError
    
    @app.websocket("/ws/progress/{job_id}")
    async def websocket_progress_endpoint(
        websocket: WebSocket,
        job_id: str,
        token: str = Query(None)
    ):
        """
        WebSocket endpoint for real-time job progress
        
        Requires authentication via token query parameter.
        """
        if not token:
            await websocket.close(code=4001, reason="Authentication required")
            return
        
        try:
            from api.auth import verify_token
            payload = verify_token(token)
            user_id = payload.get("sub")
            
            if not user_id:
                await websocket.close(code=4003, reason="Invalid token")
                return
        except (TokenExpiredError, InvalidTokenError, AuthError):
            await websocket.close(code=4001, reason="Invalid or expired token")
            return
        
        await websocket.accept()

        async def stream_progress_updates():
            while True:
                status_info = TaskStatus.get_task_status(job_id)

                try:
                    await websocket.send_json({
                        "job_id": job_id,
                        "status": status_info["status"],
                        "progress": status_info["info"].get("progress", 0),
                        "timestamp": status_info["timestamp"]
                    })
                except WebSocketDisconnect:
                    return

                if status_info["status"] in ["completed", "failed", "cancelled"]:
                    try:
                        await websocket.send_json({
                            "job_id": job_id,
                            "status": status_info["status"],
                            "message": "Job completed"
                        })
                    except WebSocketDisconnect:
                        return
                    return

                # Update every 2 seconds
                await asyncio.sleep(2)

        async def watch_for_disconnect():
            try:
                async for _ in websocket.iter_text():
                    pass
            except WebSocketDisconnect:
                return
        
        try:
            progress_task = asyncio.create_task(stream_progress_updates())
            disconnect_task = asyncio.create_task(watch_for_disconnect())

            done, pending = await asyncio.wait(
                {progress_task, disconnect_task},
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in pending:
                task.cancel()

            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

            for task in done:
                task.result()

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
