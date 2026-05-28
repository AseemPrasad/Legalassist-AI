"""
OAuth2/OpenID Connect SSO Integration

Supports:
    - Google Workspace (Google OAuth 2.0)
    - Microsoft Entra ID (Azure AD / Microsoft OAuth 2.0)

Environment Variables:
    # Google
    GOOGLE_CLIENT_ID=
    GOOGLE_CLIENT_SECRET=
    GOOGLE_REDIRECT_URI=http://localhost:8000/api/v1/auth/sso/google/callback

    # Microsoft Entra ID
    MICROSOFT_CLIENT_ID=
    MICROSOFT_CLIENT_SECRET=
    MICROSOFT_REDIRECT_URI=http://localhost:8000/api/v1/auth/sso/microsoft/callback
    MICROSOFT_TENANT_ID=common  # or specific tenant ID

    # General
    SSO_ENABLED=true
    AUTO_PROVISION_USERS=true  # auto-create users from SSO if not exists
"""

import hashlib
import hmac
import logging
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

import structlog
from authlib.integrations.starlette_client import OAuth
from fastapi import APIRouter, HTTPException, status, Request, Depends, Query
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from api.auth import create_access_token, CurrentUser
from core.log_redaction import mask_email
from api.csrf import CSRF_COOKIE_NAME, generate_csrf_token
from database import get_db, SessionLocal, User

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/auth/sso", tags=["sso"])


class SSOConfig:
    GOOGLE_CLIENT_ID: Optional[str] = None
    GOOGLE_CLIENT_SECRET: Optional[str] = None
    MICROSOFT_CLIENT_ID: Optional[str] = None
    MICROSOFT_CLIENT_SECRET: Optional[str] = None
    MICROSOFT_TENANT_ID: str = "common"
    SSO_ENABLED: bool = False
    AUTO_PROVISION_USERS: bool = True

    @classmethod
    def from_env(cls) -> "SSOConfig":
        import os
        cls.GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
        cls.GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
        cls.MICROSOFT_CLIENT_ID = os.getenv("MICROSOFT_CLIENT_ID")
        cls.MICROSOFT_CLIENT_SECRET = os.getenv("MICROSOFT_CLIENT_SECRET")
        cls.MICROSOFT_TENANT_ID = os.getenv("MICROSOFT_TENANT_ID", "common")
        cls.SSO_ENABLED = os.getenv("SSO_ENABLED", "false").lower() == "true"
        cls.AUTO_PROVISION_USERS = os.getenv("AUTO_PROVISION_USERS", "true").lower() == "true"
        return cls


SSOConfig.from_env()

_oauth = OAuth()

_state_secret: str = ""


def _get_state_secret() -> str:
    """Return a shared secret for HMAC-signing OAuth state tokens.

    Uses the CSRF secret so that all workers share the same key.
    Falls back to a per-process random value cached at module scope.
    """
    global _state_secret
    if _state_secret:
        return _state_secret
    import os
    _state_secret = os.getenv("CSRF_SECRET") or os.getenv("SECRET_KEY") or ""
    if not _state_secret:
        _state_secret = secrets.token_hex(32)
    return _state_secret


def _generate_state(provider: str, redirect_uri: str) -> str:
    """Create a stateless HMAC-signed OAuth state token.

    Encodes provider, redirect_uri, and a 10-minute expiry inside the
    token so no shared storage is needed across workers.
    """
    exp = int((datetime.now(timezone.utc) + timedelta(minutes=10)).timestamp())
    payload = f"{exp}:{provider}:{redirect_uri}"
    sig = hmac.new(_get_state_secret().encode(), payload.encode(), hashlib.sha256).hexdigest()[:16]
    return f"{payload}:{sig}"


def _consume_state(state: str) -> Optional[dict]:
    """Verify and decode a stateless OAuth state token.

    Returns None if the signature is invalid or the token has expired.
    """
    try:
        parts = state.rsplit(":", 3)
        if len(parts) != 4:
            return None
        exp_str, provider, redirect_uri, sig = parts
        exp = int(exp_str)
        if datetime.now(timezone.utc).timestamp() > exp:
            return None
        payload = f"{exp_str}:{provider}:{redirect_uri}"
        expected_sig = hmac.new(_get_state_secret().encode(), payload.encode(), hashlib.sha256).hexdigest()[:16]
        if not hmac.compare_digest(expected_sig, sig):
            return None
        return {"provider": provider, "redirect_uri": redirect_uri, "exp": datetime.fromtimestamp(exp, tz=timezone.utc)}
    except (ValueError, IndexError):
        return None


def _get_or_create_user(email: str, name: str, provider: str, provider_id: str) -> User:
    db = SessionLocal()
    try:
        user = db.query(User).filter(
            User.sso_provider == provider,
            User.sso_provider_id == provider_id,
        ).first()
        if user:
            if hasattr(user, "last_login"):
                user.last_login = datetime.now(timezone.utc)
            db.commit()
            return user

        existing = db.query(User).filter(User.email == email).first()
        if existing:
            if existing.sso_provider is not None and existing.sso_provider != provider:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="This email is already linked to a different SSO provider. Contact support to merge accounts.",
                )
            existing.sso_provider = provider
            existing.sso_provider_id = provider_id
            existing.last_login = datetime.now(timezone.utc)
            db.commit()
            logger.info("sso_user_linked", email=mask_email(email), provider=provider)
            return existing

        if not SSOConfig.AUTO_PROVISION_USERS:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User not found. Contact your administrator to provision an account.",
            )

        from db.models.auth import UserRole
        user = User(
            email=email,
            role=UserRole.CLIENT,
            sso_provider=provider,
            sso_provider_id=provider_id,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        logger.info("sso_user_provisioned", email=mask_email(email), provider=provider)
        return user
    finally:
        db.close()


def _build_token_response(user: User, provider: str) -> RedirectResponse:
    from api.config import get_settings
    _sso_settings = get_settings()
    role = user.role.value if user.role else "client"
    token = create_access_token(
        data={"sub": str(user.id), "email": user.email, "role": role, "provider": provider}
    )
    token_max_age = _sso_settings.JWT_ACCESS_TOKEN_MINUTES * 60
    response = RedirectResponse(url="/", status_code=302)
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        secure=True,
        samesite="lax",
        path="/",
        max_age=token_max_age,
    )
    csrf_token = generate_csrf_token(user.id, secrets.token_urlsafe(16))
    response.set_cookie(
        key=CSRF_COOKIE_NAME,
        value=csrf_token,
        httponly=False,
        secure=True,
        samesite="lax",
        path="/",
        max_age=token_max_age,
    )
    return response


def _configure_google():
    if not SSOConfig.GOOGLE_CLIENT_ID or not SSOConfig.GOOGLE_CLIENT_SECRET:
        return None
    _oauth.register(
        name="google",
        client_id=SSOConfig.GOOGLE_CLIENT_ID,
        client_secret=SSOConfig.GOOGLE_CLIENT_SECRET,
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )
    return _oauth.google


def _configure_microsoft():
    if not SSOConfig.MICROSOFT_CLIENT_ID or not SSOConfig.MICROSOFT_CLIENT_SECRET:
        return None
    tenant = SSOConfig.MICROSOFT_TENANT_ID
    _oauth.register(
        name="microsoft",
        client_id=SSOConfig.MICROSOFT_CLIENT_ID,
        client_secret=SSOConfig.MICROSOFT_CLIENT_SECRET,
        server_metadata_url=f"https://login.microsoftonline.com/{tenant}/v2.0/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )
    return _oauth.microsoft


@router.get("/google")
async def sso_google(request: Request):
    """Initiate Google Workspace OAuth flow."""
    if not SSOConfig.SSO_ENABLED:
        raise HTTPException(status_code=501, detail="SSO is not enabled")
    client = _configure_google()
    if client is None:
        raise HTTPException(status_code=501, detail="Google SSO is not configured")
    redirect_uri = str(request.url_for("sso_google_callback"))
    state = _generate_state("google", redirect_uri)
    return await client.authorize_redirect(request, redirect_uri, state=state)


@router.get("/google/callback", name="sso_google_callback")
async def sso_google_callback(request: Request, code: str = Query(...), state: str = Query(...)):
    """Handle Google OAuth callback."""
    ctx = _consume_state(state)
    if ctx is None or ctx.get("provider") != "google":
        raise HTTPException(status_code=400, detail="Invalid or expired SSO state")

    client = _configure_google()
    if client is None:
        raise HTTPException(status_code=501, detail="Google SSO not configured")

    try:
        token = await client.authorize_access_token(request)
    except Exception as e:
        logger.error("google_sso_auth_failed", error=str(e))
        raise HTTPException(status_code=401, detail="Google authentication failed")

    userinfo = token.get("userinfo")
    if not userinfo:
        raise HTTPException(status_code=401, detail="Failed to retrieve user info")

    email = userinfo.get("email", "")
    name = userinfo.get("name", "")
    provider_id = userinfo.get("sub", "")

    user = _get_or_create_user(email, name, "google", provider_id)
    return _build_token_response(user, "google")


@router.get("/microsoft")
async def sso_microsoft(request: Request):
    """Initiate Microsoft Entra ID OAuth flow."""
    if not SSOConfig.SSO_ENABLED:
        raise HTTPException(status_code=501, detail="SSO is not enabled")
    client = _configure_microsoft()
    if client is None:
        raise HTTPException(status_code=501, detail="Microsoft SSO is not configured")
    redirect_uri = str(request.url_for("sso_microsoft_callback"))
    state = _generate_state("microsoft", redirect_uri)
    return await client.authorize_redirect(request, redirect_uri, state=state)


@router.get("/microsoft/callback", name="sso_microsoft_callback")
async def sso_microsoft_callback(request: Request, code: str = Query(...), state: str = Query(...)):
    """Handle Microsoft Entra ID OAuth callback."""
    ctx = _consume_state(state)
    if ctx is None or ctx.get("provider") != "microsoft":
        raise HTTPException(status_code=400, detail="Invalid or expired SSO state")

    client = _configure_microsoft()
    if client is None:
        raise HTTPException(status_code=501, detail="Microsoft SSO not configured")

    try:
        token = await client.authorize_access_token(request)
    except Exception as e:
        logger.error("microsoft_sso_auth_failed", error=str(e))
        raise HTTPException(status_code=401, detail="Microsoft authentication failed")

    userinfo = token.get("userinfo")
    if not userinfo:
        raise HTTPException(status_code=401, detail="Failed to retrieve user info")

    email = userinfo.get("email", "")
    name = userinfo.get("name", "")
    provider_id = userinfo.get("sub", "")

    user = _get_or_create_user(email, name, "microsoft", provider_id)
    return _build_token_response(user, "microsoft")


@router.get("/config")
async def sso_config():
    """Return SSO provider configuration status (no secrets)."""
    return {
        "enabled": SSOConfig.SSO_ENABLED,
        "providers": {
            "google": bool(SSOConfig.GOOGLE_CLIENT_ID and SSOConfig.GOOGLE_CLIENT_SECRET),
            "microsoft": bool(SSOConfig.MICROSOFT_CLIENT_ID and SSOConfig.MICROSOFT_CLIENT_SECRET),
        },
        "auto_provision": SSOConfig.AUTO_PROVISION_USERS,
    }