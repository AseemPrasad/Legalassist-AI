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
    request: Request,
    current_user: Optional[CurrentUser] = Depends(get_current_user_optional),
) -> str:
    """Return a per-identity rate-limit key.

    Unauthenticated requests are keyed by source IP rather than the shared
    literal 'anonymous' to prevent a single attacker from exhausting the
    entire unauthenticated quota.
    """
    if current_user:
        return f"user:{current_user.user_id}"

    from api.limiter import resolve_rate_limit_identifier
    return resolve_rate_limit_identifier(request)


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
    """Request-scoped DB session with PostgreSQL RLS applied for the authenticated user."""
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
                pass
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


def get_db_no_rls() -> Generator[Session, None, None]:
    """DB session with NO RLS context set — for public endpoints.

    This dependency must only be used on endpoints that are intentionally
    public and whose queries are already scoped by a non-user-owned token
    (e.g. an anonymized_id capability token).

    Security contract:
    - ``app.current_user_id`` is never set, so PostgreSQL RLS policies that
      filter by the current user will not apply.
    - The caller is responsible for ensuring the query cannot leak data
      across ownership boundaries through other means (e.g. the
      anonymized_id is the only lookup key and it is not guessable).
    - This dependency must NEVER be used on authenticated endpoints.
    """
    from db.session import SessionLocal

    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
