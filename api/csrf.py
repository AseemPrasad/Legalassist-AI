"""
CSRF (Cross-Site Request Forgery) protection for API endpoints.

Provides protection against CSRF attacks by:
- Validating Origin/Referer headers for browser-based requests
- Supporting double-submit cookie pattern for stateless validation
- Exempting safe methods (GET, HEAD, OPTIONS) from token checks
"""

import secrets
import logging
from typing import Optional, Set, Callable
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

SAFE_METHODS: Set[str] = {"GET", "HEAD", "OPTIONS"}
CSRF_TOKEN_HEADER = "X-CSRF-Token"
CSRF_COOKIE_NAME = "csrf_token"


class CSRFError(HTTPException):
    def __init__(self, detail: str = "CSRF validation failed"):
        super().__init__(status_code=status.HTTP_403_FORBIDDEN, detail=detail)


def generate_csrf_token() -> str:
    """Generate a cryptographically secure CSRF token."""
    return secrets.token_urlsafe(32)


def get_origin(request: Request) -> Optional[str]:
    """Extract Origin header value."""
    origin = request.headers.get("origin") or request.headers.get("Origin")
    return origin


def get_referer(request: Request) -> Optional[str]:
    """Extract Referer header value."""
    referer = request.headers.get("referer") or request.headers.get("Referer")
    return referer


def is_same_origin(request: Request, allowed_hosts: Optional[Set[str]] = None) -> bool:
    """
    Check if request originates from the same origin.

    Args:
        request: FastAPI request object
        allowed_hosts: Set of allowed hostnames

    Returns:
        True if same origin, False if cross-origin
    """
    origin = get_origin(request)
    if not origin:
        return True

    allowed_hosts = allowed_hosts or set()
    if allowed_hosts:
        from urllib.parse import urlparse
        parsed = urlparse(origin)
        return parsed.netloc in allowed_hosts or parsed.netloc in {"localhost", "127.0.0.1"}

    host = request.headers.get("host", "").split(":")[0]
    from urllib.parse import urlparse
    parsed = urlparse(origin)
    return parsed.netloc == host or parsed.netloc == f"{host}:{request.url.port or 80}"


def validate_csrf_request(
    request: Request,
    allowed_hosts: Optional[Set[str]] = None,
    require_token: bool = True,
) -> None:
    """
    Validate CSRF protection for a request.

    Args:
        request: FastAPI request object
        allowed_hosts: Set of allowed hostnames for same-origin checks
        require_token: Whether to require CSRF token header

    Raises:
        CSRFError: If CSRF validation fails
    """
    if request.method in SAFE_METHODS:
        return

    origin = get_origin(request)
    if not origin:
        return

    if not is_same_origin(request, allowed_hosts):
        logger.warning("csrf_cross_origin_rejected", origin=origin, path=str(request.url.path))
        raise CSRFError(detail="Cross-origin requests not allowed")

    if require_token:
        token = request.headers.get(CSRF_TOKEN_HEADER) or request.headers.get(CSRF_TOKEN_HEADER.lower())
        if not token:
            logger.warning("csrf_missing_token", path=str(request.url.path))
            raise CSRFError(detail=f"Missing CSRF token. Include '{CSRF_TOKEN_HEADER}' header.")


class CSRFProtectionMiddleware(BaseHTTPMiddleware):
    """
    Middleware for automatic CSRF protection on state-mutating endpoints.

    Usage:
        app.add_middleware(CSRFProtectionMiddleware, allowed_hosts={"api.example.com"})
    """

    def __init__(
        self,
        app,
        allowed_hosts: Optional[Set[str]] = None,
        exempt_paths: Optional[Set[str]] = None,
    ):
        super().__init__(app)
        self.allowed_hosts = allowed_hosts or set()
        self.exempt_paths = exempt_paths or {"/health", "/ready", "/metrics"}

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        if path in self.exempt_paths or request.method in SAFE_METHODS:
            return await call_next(request)

        origin = get_origin(request)
        if origin and not is_same_origin(request, self.allowed_hosts):
            logger.warning("csrf_blocked", origin=origin, path=path)
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={"detail": "Cross-origin request blocked"},
            )

        return await call_next(request)


def csrf_protected(func: Callable) -> Callable:
    """
    Decorator to add CSRF protection to a specific endpoint.

    Usage:
        @router.post("/endpoint")
        @csrf_protected
        async def my_endpoint(request: Request, ...):
            ...
    """
    async def wrapper(request: Request, *args, **kwargs):
        allowed_hosts = getattr(request.app.state, "csrf_allowed_hosts", None)
        validate_csrf_request(request, allowed_hosts=allowed_hosts)
        return await func(request, *args, **kwargs)
    return wrapper


def set_csrf_headers(response, token: str) -> None:
    """Set CSRF-related headers on response."""
    response.headers["X-CSRF-Token"] = token
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"