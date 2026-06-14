"""
Dependency injection and common dependencies
"""
from typing import Any, Generator, Optional

import structlog
from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from api.auth import get_current_user, get_current_user_optional, CurrentUser
from core.policy_engine import PolicyDecision, UserContext, evaluate, PolicyEngine

logger = structlog.get_logger(__name__)


async def get_rate_limit_key(
    request: Request,
    current_user: Optional[CurrentUser] = Depends(get_current_user_optional),
) -> str:
    """Return a per-identity rate-limit key.

    Resolution order:
    1. Authenticated user  → ``user:<user_id>``   (unchanged)
    2. Unauthenticated     → ``ip:<client_ip>``   (was: ``"anonymous"``)
    3. No IP available     → unique per-request token so no shared bucket
    """
    if current_user:
        return f"user:{current_user.user_id}"

    from api.limiter import resolve_rate_limit_identifier
    return resolve_rate_limit_identifier(request, current_user=current_user)


async def verify_api_version(
    api_version: Optional[str] = None
) -> str:
    """Verify API version from query parameter"""
    if api_version and api_version not in ["v1"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported API version: {api_version}. Use v1"
        )
    return api_version or "v1"


def get_db_rls(
    request: Request,
    current_user: CurrentUser = Depends(get_current_user),
) -> Generator[Session, None, None]:
    """Request-scoped database session with PostgreSQL RLS applied."""
    from db.session import SessionLocal, apply_rls_context, clear_rls_context, _is_postgres

    db: Session = SessionLocal()
    try:
        if _is_postgres:
            user_id_str = str(current_user.user_id)
            if user_id_str:
                apply_rls_context(db, user_id_str)
            else:
                logger.warning(
                    "rls_skipped_empty_user_id",
                    user_id=user_id_str,
                )
        yield db
    finally:
        if _is_postgres:
            try:
                clear_rls_context(db)
            except Exception as e:
                import logging
                logging.error(f"Dependency error: {e}")
                pass
        db.close()


def get_db_rls_optional(
    request: Request,
    current_user: Optional[CurrentUser] = Depends(get_current_user_optional),
) -> Generator[Session, None, None]:
    """Like ``get_db_rls`` but for endpoints that allow unauthenticated access."""
    from db.session import SessionLocal, apply_rls_context, clear_rls_context, _is_postgres

    db: Session = SessionLocal()
    try:
        if _is_postgres and current_user is not None:
            user_id_str = str(current_user.user_id)
            if user_id_str.isdigit():
                apply_rls_context(db, int(user_id_str))
        yield db
    finally:
        if _is_postgres:
            try:
                clear_rls_context(db)
            except Exception:
                pass
        db.close()


def get_db_no_rls() -> Generator[Session, None, None]:
    """DB session with NO RLS context set — for public endpoints."""
    from db.session import SessionLocal

    db: Session = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# ============================================================================
# POLICY ENGINE DEPENDENCIES
# ============================================================================

def _current_user_to_context(current_user: CurrentUser) -> UserContext:
    """Convert FastAPI CurrentUser to policy engine UserContext."""
    return UserContext(
        user_id=current_user.user_id,
        email=current_user.email,
        role=current_user.role,
    )


async def require_policy(
    resource_type: str,
    action: str,
    resource: Optional[Any] = None,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db_rls),
) -> None:
    """FastAPI dependency that enforces a policy decision.

    Usage in route definitions::

        @router.get("/{case_id}")
        async def get_case(
            case_id: str,
            _authorized: None = Depends(require_policy("case", "view")),
            ...
        ):
            ...

    For resource-level checks (e.g., ownership), fetch the resource first
    and use the lower-level ``evaluate_policy`` function inside the handler.
    """
    user_ctx = _current_user_to_context(current_user)
    decision = evaluate(user_ctx, resource_type, action, resource, db)

    if decision == PolicyDecision.DENY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access denied: {action} on {resource_type}",
        )
    if decision == PolicyDecision.ABSTAIN:
        logger.warning(
            "policy_abstain_fallback_deny",
            resource_type=resource_type,
            action=action,
            user_id=current_user.user_id,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access denied: no policy defined for {action} on {resource_type}",
        )


def evaluate_policy(
    current_user: CurrentUser,
    resource_type: str,
    action: str,
    resource: Any,
    db: Session,
) -> PolicyDecision:
    """Evaluate policy for a specific resource instance.

    Use this inside route handlers after fetching the resource from the DB.
    """
    user_ctx = _current_user_to_context(current_user)
    return evaluate(user_ctx, resource_type, action, resource, db)
def check_permission(permission: str):
    """Dependency that enforces a specific permission on the current user."""
    from api.rbac import has_permission
    async def dependency(current_user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        if not has_permission(current_user.role, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required permission: {permission}"
            )
        return current_user
    return dependency


def check_min_role(min_role: str):
    """Dependency that enforces a minimum hierarchical role on the current user."""
    ROLE_HIERARCHY = ["client", "paralegal", "attorney", "admin"]
    async def dependency(current_user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        user_level = ROLE_HIERARCHY.index(current_user.role) if current_user.role in ROLE_HIERARCHY else 0
        min_level = ROLE_HIERARCHY.index(min_role) if min_role in ROLE_HIERARCHY else 0
        if user_level < min_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{current_user.role}' is below minimum required role '{min_role}'",
            )
        return current_user
    return dependency
