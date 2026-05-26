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
import os
import secrets
import structlog
from typing import Optional, Set

from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from api.errors import StructuredAPIError, structured_error_response

logger = structlog.get_logger(__name__)

SAFE_METHODS: Set[str] = {"GET", "HEAD", "OPTIONS"}
CSRF_TOKEN_HEADER = "X-CSRF-Token"
CSRF_COOKIE_NAME = "csrf_token"
ACCESS_TOKEN_COOKIE_NAME = "access_token"
CSRF_SESSION_PREFIX = "csrf_session:"


class CSRFError(StructuredAPIError):
    """CSRF validation error with standardized error response."""
    pass


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


_CSRF_SECRET_CACHE: Optional[str] = None


def _get_csrf_secret() -> str:
    global _CSRF_SECRET_CACHE
    if _CSRF_SECRET_CACHE is not None:
        return _CSRF_SECRET_CACHE

    from api.config import get_settings
    settings = get_settings()
    secret = settings.CSRF_SECRET
    if not secret or len(secret) < 16:
        secret = os.environ.get("SECRET_KEY", "")
        if not secret or len(secret) < 16:
            is_prod = settings.ENVIRONMENT in ("production", "prod", "live")
            if is_prod:
                raise RuntimeError(
                    "CSRF_SECRET environment variable must be set in production. "
                    "Auto-generation is not allowed."
                )
            logger.warning("csrf_secret_auto_generated", message="CSRF_SECRET not set. Auto-generated per-process secret — cross-worker CSRF will fail in multi-worker deployments.")
            secret = secrets.token_hex(32)
    _CSRF_SECRET_CACHE = secret
    return secret


def get_origin(request: Request) -> Optional[str]:
    return request.headers.get("origin") or request.headers.get("Origin")


def get_referer(request: Request) -> Optional[str]:
    return request.headers.get("referer") or request.headers.get("Referer")


def is_same_origin(request: Request, allowed_hosts: Optional[Set[str]] = None) -> bool:
    origin = get_origin(request)
    if not origin:
        referer = get_referer(request)
        if referer:
            from urllib.parse import urlparse
            parsed = urlparse(referer)
            allowed = allowed_hosts or set()
            host = request.headers.get("host", "").split(":")[0]
            if parsed.netloc in allowed or parsed.netloc == f"{host}:443":
                return True
            return parsed.netloc == f"{host}:443" or parsed.netloc == host
        return False
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
        raise StructuredAPIError(status_code=status.HTTP_403_FORBIDDEN, error_code="CSRF_ORIGIN_BLOCKED", message="Cross-origin request blocked")

    access_token = request.cookies.get(ACCESS_TOKEN_COOKIE_NAME)
    cookie_token = request.cookies.get(CSRF_COOKIE_NAME)
    header_token = request.headers.get(CSRF_TOKEN_HEADER) or request.headers.get(CSRF_TOKEN_HEADER.lower())

    if not access_token and not cookie_token:
        return

    if not access_token:
        if not header_token:
            raise StructuredAPIError(status_code=status.HTTP_403_FORBIDDEN, error_code="CSRF_MISSING_TOKEN", message=f"Missing CSRF token. Include '{CSRF_TOKEN_HEADER}' header.")
        if not hmac.compare_digest(header_token, cookie_token or ""):
            raise StructuredAPIError(status_code=status.HTTP_403_FORBIDDEN, error_code="CSRF_TOKEN_MISMATCH", message="CSRF token mismatch")
        return

    if current_user_id is None:
        from api.jwt_auth import InvalidTokenError, TokenExpiredError, verify_token

        try:
            payload = verify_token(access_token)
            current_user_id = int(payload.get("sub"))
        except TokenExpiredError as exc:
            raise StructuredAPIError(status_code=status.HTTP_401_UNAUTHORIZED, error_code="TOKEN_EXPIRED", message=str(exc))
        except (InvalidTokenError, TypeError, ValueError):
            raise StructuredAPIError(status_code=status.HTTP_401_UNAUTHORIZED, error_code="INVALID_TOKEN", message="Invalid token")

    if not cookie_token:
        raise StructuredAPIError(status_code=status.HTTP_403_FORBIDDEN, error_code="CSRF_MISSING_COOKIE", message="Missing CSRF cookie")

    if not header_token:
        raise StructuredAPIError(status_code=status.HTTP_403_FORBIDDEN, error_code="CSRF_MISSING_TOKEN", message=f"Missing CSRF token. Include '{CSRF_TOKEN_HEADER}' header.")

    if not validate_csrf_token(header_token, current_user_id):
        logger.warning("csrf_token_invalid", path=str(request.url.path), user_id=current_user_id)
        raise StructuredAPIError(status_code=status.HTTP_403_FORBIDDEN, error_code="CSRF_TOKEN_INVALID", message="Invalid or expired CSRF token")

    if not hmac.compare_digest(header_token, cookie_token):
        raise StructuredAPIError(status_code=status.HTTP_403_FORBIDDEN, error_code="CSRF_TOKEN_MISMATCH", message="CSRF token mismatch")


def validate_csrf_request(
    request: Request,
    current_user_id: Optional[int] = None,
    allowed_hosts: Optional[Set[str]] = None,
) -> None:
    """Validate CSRF for a given request. Convenience wrapper around validate_csrf."""
    return validate_csrf(request, current_user_id=current_user_id, allowed_hosts=allowed_hosts)


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

        if request.method not in SAFE_METHODS:
            try:
                validate_csrf(request, allowed_hosts=self.allowed_hosts)
            except StructuredAPIError as exc:
                return structured_error_response(
                    status_code=exc.status_code,
                    error_code=exc.error_code,
                    message=exc.message,
                    request=request,
                )

        user_id = getattr(request.state, "csrf_user_id", None)
        response = await call_next(request)

        access_token = request.cookies.get(ACCESS_TOKEN_COOKIE_NAME)
        csrf_cookie = request.cookies.get(CSRF_COOKIE_NAME)

        if request.method in SAFE_METHODS and not csrf_cookie:
            session_id = secrets.token_urlsafe(16)
            token = generate_csrf_token(int(user_id) if str(user_id).isdigit() else 0, session_id)
            response.set_cookie(
                CSRF_COOKIE_NAME,
                token,
                httponly=False,
                samesite="lax",
                secure=True,
                path="/",
                max_age=3600 * 8,
            )
            response.headers["X-CSRF-Token"] = token
            return response

        if request.method not in SAFE_METHODS and (user_id or access_token):
            resolved_user_id = int(user_id) if str(user_id).isdigit() else None
            jwt_jti = None
            if resolved_user_id is None and access_token:
                try:
                    payload = verify_token(access_token)
                    resolved_user_id = int(payload.get("sub"))
                    jwt_jti = payload.get("jti")
                except Exception:
                    resolved_user_id = None

            if resolved_user_id is not None:
                session_id = jwt_jti or secrets.token_urlsafe(16)
                token = generate_csrf_token(resolved_user_id, session_id)
                response.set_cookie(
                    CSRF_COOKIE_NAME,
                    token,
                    httponly=False,
                    samesite="lax",
                    secure=True,
                    path="/",
                    max_age=3600 * 8,
                )
                response.headers["X-CSRF-Token"] = token

        return response


CSRFError = StructuredAPIError
validate_csrf_request = validate_csrf