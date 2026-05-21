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
from api.middlewares import register_middlewares
from api.csrf import CSRFProtectionMiddleware
from api.limiter import cleanup_limiter
from observability.integration import initialize_observability_for_environment
from observability.instrumentation import get_metrics

try:
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.trace import get_tracer
    _FASTAPI_INSTRUMENTOR = FastAPIInstrumentor
except Exception:
    _FASTAPI_INSTRUMENTOR = None

try:
    from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
    _SQLALCHEMY_INSTRUMENTOR = SQLAlchemyInstrumentor
except Exception:
    _SQLALCHEMY_INSTRUMENTOR = None

try:
    from opentelemetry.instrumentation.redis import RedisInstrumentor
    _REDIS_INSTRUMENTOR = RedisInstrumentor
except Exception:
    _REDIS_INSTRUMENTOR = None

try:
    import strawberry
    from strawberry.fastapi import GraphQLRouter
    from api.graphql_schema import schema as graphql_schema
    _GRAPHQL_ROUTER = GraphQLRouter
    _GRAPHQL_SCHEMA = graphql_schema
except Exception:
    _GRAPHQL_ROUTER = None
    _GRAPHQL_SCHEMA = None

from api.validation import (
    ValidationConfig,
    ValidationError,
    PayloadTooLargeError,
)

# Import routes
from api.routes import documents, cases, reports, analytics, deadlines, auth, health, case_search, speech, document_verification

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

        if _FASTAPI_INSTRUMENTOR:
            _FASTAPI_INSTRUMENTOR.instrument_app(app)

        if _SQLALCHEMY_INSTRUMENTOR:
            try:
                _SQLALCHEMY_INSTRUMENTOR().instrument()
            except Exception:
                pass

        if _REDIS_INSTRUMENTOR:
            try:
                _REDIS_INSTRUMENTOR().instrument()
            except Exception:
                pass

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
        try:
            from services.timeline_realtime import timeline_realtime_bus

            await timeline_realtime_bus.close()
        except Exception:
            pass
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

    # Add middleware through a single registration entry to preserve order.
    register_middlewares(app)

    if _GRAPHQL_ROUTER is not None:
        app.include_router(
            _GRAPHQL_ROUTER(_GRAPHQL_SCHEMA, path="/graphql", graphiql=True),
            prefix="",
        )

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
        return structured_error_response(
            status_code=exc.status_code,
            error_code="VALIDATION_ERROR",
            message=exc.detail,
            request=request,
        )
    
    @app.exception_handler(PayloadTooLargeError)
    async def payload_too_large_handler(request: Request, exc: PayloadTooLargeError):
        """Handle payload too large errors"""
        logger.warning(
            "payload_too_large",
            path=request.url.path,
            detail=exc.detail
        )
        return structured_error_response(
            status_code=exc.status_code,
            error_code="PAYLOAD_TOO_LARGE",
            message=exc.detail,
            request=request,
        )

    @app.exception_handler(StructuredAPIError)
    async def structured_api_error_handler(request: Request, exc: StructuredAPIError):
        """Handle structured security errors from auth and CSRF helpers."""
        logger.warning(
            "structured_api_error",
            path=request.url.path,
            error_code=exc.error_code,
            detail=exc.message,
        )
        return structured_error_response(
            status_code=exc.status_code,
            error_code=exc.error_code,
            message=exc.message,
            request=request,
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
        
        Requires authentication via token query parameter or Sec-WebSocket-Protocol header.
        """
        auth_token = token
        requested_protocols = []
        
        if "sec-websocket-protocol" in websocket.headers:
            header_val = websocket.headers["sec-websocket-protocol"]
            requested_protocols = [p.strip() for p in header_val.split(",")]
            if "access_token" in requested_protocols:
                idx = requested_protocols.index("access_token")
                if idx + 1 < len(requested_protocols):
                    auth_token = requested_protocols[idx + 1]

        if not auth_token:
            await websocket.close(code=4001, reason="Authentication required")
            return
        
        try:
            from api.auth import verify_token
            payload = verify_token(auth_token)
            user_id = payload.get("sub")
            
            if not user_id:
                await websocket.close(code=4003, reason="Invalid token")
                return
        except (TokenExpiredError, InvalidTokenError, AuthError):
            await websocket.close(code=4001, reason="Invalid or expired token")
            return
        
        subprotocol = "access_token" if "access_token" in requested_protocols else None
        await websocket.accept(subprotocol=subprotocol)

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

    from api.websockets.case_timeline import register_case_timeline_endpoint

    register_case_timeline_endpoint(app)


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        workers=settings.API_WORKERS,
        reload=True
    )
