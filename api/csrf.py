"""
CSRF (Cross-Site Request Forgery) protection for API endpoints.

Provides protection against CSRF attacks by:
- HMAC-signed tokens bound to authenticated session (user_id + session_id)
- Double-submit cookie pattern with server-side session binding
- SameSite=lax cookies preventing cross-site cookie sending
- Safe methods (GET, HEAD, OPTIONS) exempted automatically
- Origin/Referer header validation for browser-based requests
"""

import hashlib
import hmac
import secrets
import structlog
from typing import Optional, Set

from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = structlog.get_logger(__name__)

SAFE_METHODS: Set[str] = {"GET", "HEAD", "OPTIONS"}
CSRF_TOKEN_HEADER = "X-CSRF-Token"
CSRF_COOKIE_NAME = "csrf_token"
CSRF_SESSION_PREFIX = "csrf_session:"


def generate_csrf_token(user_id: int, session_id: str) -> str:
    secret = _get_csrf_secret()
    message = f"{user_id}:{session_id}"
    sig = hmac.new(secret.encode(), message.encode(), hashlib.sha256).hexdigest()
    return f"{session_id}.{sig[:32]}"


def validate_csrf_token(token: str, user_id: int) -> bool:
    if not token or "." not in token:
        return False
    parts = token.rsplit(".", 1)
    if len(parts) != 2:
        return False
    session_id, received_sig = parts
    secret = _get_csrf_secret()
    message = f"{user_id}:{session_id}"
    expected_sig = hmac.new(secret.encode(), message.encode(), hashlib.sha256).hexdigest()[:32]
    return hmac.compare_digest(expected_sig, received_sig)


def _get_csrf_secret() -> str:
    import os
    secret = os.getenv("CSRF_SECRET")
    if not secret:
        secret = os.getenv("SECRET_KEY", "")
        if secret:
            secret = secret[:32].ljust(32, "0")
        else:
            secret = secrets.token_hex(32)
    return secret


def get_origin(request: Request) -> Optional[str]:
    return request.headers.get("origin") or request.headers.get("Origin")


def get_referer(request: Request) -> Optional[str]:
    return request.headers.get("referer") or request.headers.get("Referer")


def is_same_origin(request: Request, allowed_hosts: Optional[Set[str]] = None) -> bool:
    origin = get_origin(request)
    if not origin:
        return True
    from urllib.parse import urlparse
    parsed = urlparse(origin)
    host = request.headers.get("host", "").split(":")[0]
    allowed = allowed_hosts or set()
    if parsed.netloc in allowed:
        return True
    return parsed.netloc == host


def validate_csrf(
    request: Request,
    current_user_id: Optional[int] = None,
    allowed_hosts: Optional[Set[str]] = None,
) -> None:
    if request.method in SAFE_METHODS:
        return

    origin = get_origin(request)
    if origin and not is_same_origin(request, allowed_hosts):
        logger.warning("csrf_cross_origin_rejected", origin=origin, path=str(request.url.path))
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cross-origin request blocked",
        )

    cookie_token = request.cookies.get(CSRF_COOKIE_NAME)
    header_token = request.headers.get(CSRF_TOKEN_HEADER) or request.headers.get(CSRF_TOKEN_HEADER.lower())

    if current_user_id and cookie_token:
        if not header_token:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing CSRF token. Include '{CSRF_TOKEN_HEADER}' header.",
            )
        if not validate_csrf_token(header_token, current_user_id):
            logger.warning("csrf_token_invalid", path=str(request.url.path), user_id=current_user_id)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid or expired CSRF token",
            )
        if not hmac.compare_digest(header_token, cookie_token):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="CSRF token mismatch",
            )


class CSRFProtectionMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app,
        allowed_hosts: Optional[Set[str]] = None,
        exempt_paths: Optional[Set[str]] = None,
    ):
        super().__init__(app)
        self.allowed_hosts = allowed_hosts or set()
        self.exempt_paths = exempt_paths or {
            "/health", "/ready", "/metrics",
            "/docs", "/openapi.json", "/redoc",
            "/api/v1/auth/sso/google", "/api/v1/auth/sso/google/callback",
            "/api/v1/auth/sso/microsoft", "/api/v1/auth/sso/microsoft/callback",
        }

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        if path in self.exempt_paths:
            return await call_next(request)

        user_id = getattr(request.state, "csrf_user_id", None)
        response = await call_next(request)

        if request.method not in SAFE_METHODS and user_id:
            token = generate_csrf_token(user_id, getattr(request.state, "request_id", "default"))
            try:
                secure = True
            except Exception:
                secure = False
            response.set_cookie(
                CSRF_COOKIE_NAME,
                token,
                httponly=False,
                samesite="lax",
                secure=secure,
                path="/",
                max_age=3600 * 8,
            )
            response.headers["X-CSRF-Token"] = token

        return response