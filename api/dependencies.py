"""
Dependency injection and common dependencies
"""
from typing import Generator, Optional

import structlog
from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from api.auth import get_current_user, get_current_user_optional, CurrentUser

logger = structlog.get_logger(__name__)


async def get_rate_limit_key(
    current_user: Optional[CurrentUser] = Depends(get_current_user_optional)
) -> str:
    """Get rate limit key for current user/API key.

    Uses get_current_user_optional so that unauthenticated requests are not
    rejected during dependency resolution — they fall back to an anonymous
    identifier instead of bypassing rate-limit evaluation entirely.
    """
    if current_user:
        return f"user:{current_user.user_id}"
    return "anonymous"


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
    """Request-scoped database session with PostgreSQL Row-Level Security applied.

    For every authenticated API request this dependency:
    1. Opens a new SQLAlchemy session.
    2. Sets the ``app.current_user_id`` PostgreSQL session variable so that
       all RLS policies defined by ``scripts/setup_rls.py`` are enforced for
       the duration of the request.
    3. Clears the variable and closes the session in the ``finally`` block,
       ensuring no user context leaks across connection-pool reuse.

    On SQLite (local development) the RLS calls are no-ops, so behaviour is
    unchanged for developers who do not run PostgreSQL locally.

    Usage::

        @router.get("/resource")
        async def my_endpoint(db: Session = Depends(get_db_rls)):
            ...

    Note: ``scripts/setup_rls.py`` must be run against the PostgreSQL database
    **after** ``Base.metadata.create_all`` (i.e. after migrations) for the
    policies to exist.  Without that step this dependency still works correctly
    — it simply sets a session variable that no policy reads yet.
    """
    from db.session import SessionLocal, apply_rls_context, clear_rls_context, _is_postgres

    db: Session = SessionLocal()
    try:
        if _is_postgres:
            user_id_str = str(current_user.user_id)
            if user_id_str.isdigit():
                apply_rls_context(db, int(user_id_str))
            else:
                logger.warning(
                    "rls_skipped_non_numeric_user_id",
                    user_id=user_id_str,
                )
        yield db
    finally:
        if _is_postgres:
            try:
                clear_rls_context(db)
            except Exception:
                pass  # session may already be closed / rolled back
        db.close()


def get_db_rls_optional(
    request: Request,
    current_user: Optional[CurrentUser] = Depends(get_current_user_optional),
) -> Generator[Session, None, None]:
    """Like ``get_db_rls`` but for endpoints that allow unauthenticated access.

    When no authenticated user is present the session is opened without setting
    the RLS context variable, which means PostgreSQL RLS policies that require
    ``app.current_user_id`` will block all rows — providing a safe default.
    """
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
