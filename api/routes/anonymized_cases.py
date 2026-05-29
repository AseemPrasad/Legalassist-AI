"""Anonymized case sharing endpoint.

GET /api/v1/anonymized-cases/{anonymized_id}

Returns a redacted case payload for the given anonymized share ID.  No
authentication is required — the anonymized_id itself acts as a capability
token.  Owner identity and PII are never exposed.

All lookups are recorded in the immutable audit log.

RLS note
--------
This endpoint uses ``get_db_no_rls`` (no RLS context set) rather than
``get_db_rls_optional``.  The previous use of ``get_db_rls_optional`` had
two problems:

1. When an authenticated user called this endpoint their ``user_id`` was set
   as the PostgreSQL ``app.current_user_id`` session variable.  The raw-SQL
   lookup in ``lookup_anonymized_case`` filters only on ``anonymized_id``, not
   on ``user_id``, so the RLS context was set but never enforced — giving a
   false sense of security.

2. An authenticated user could look up any case's anonymized view regardless
   of ownership, because the query does not check ``user_id``.

The correct model for a capability-token endpoint is:
- Never set an RLS context (use ``get_db_no_rls``).
- The anonymized_id IS the access control — it is an HMAC-derived token that
  cannot be guessed without the server-side secret.
- The response is always fully redacted by the privacy profile.
- Owner identity is never included in the payload.
"""

from __future__ import annotations

import re

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from api.dependencies import get_db_no_rls
from api.auth import CurrentUser, get_current_user_optional
from db.crud.audit import record_immutable_audit_event
from services.anonymized_case_lookup import lookup_anonymized_case

router = APIRouter(
    prefix="/api/v1/anonymized-cases",
    tags=["anonymized-cases"],
)

logger = structlog.get_logger(__name__)

# Validate that the anonymized_id looks like a hex string (12–64 chars).
_ANON_ID_RE = re.compile(r"^[0-9a-f]{12,64}$", re.IGNORECASE)


@router.get(
    "/{anonymized_id}",
    summary="View shared anonymized case",
    response_description="Redacted case summary, documents, and timeline",
)
async def get_anonymized_case(
    anonymized_id: str,
    request: Request,
    # Use get_db_no_rls: this is a public capability-token endpoint.
    # Setting an RLS context here would be misleading — the lookup is scoped
    # by anonymized_id, not by user_id, so any RLS context set would be
    # silently ignored by the raw-SQL query while creating a false impression
    # of row-level isolation.
    db: Session = Depends(get_db_no_rls),
    current_user: CurrentUser | None = Depends(get_current_user_optional),
) -> dict:
    """Resolve an *anonymized_id* to a redacted case view.

    - Valid IDs return a redacted payload consistent with the privacy profile
      applied when the case was first anonymized.
    - Owner identity and PII are **never** exposed.
    - Invalid or unknown IDs return **404**.
    - Every lookup is recorded in the immutable audit log.
    """
    if not _ANON_ID_RE.match(anonymized_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Anonymized case not found",
        )

    ip_address: str | None = None
    if request.client:
        ip_address = request.client.host

    actor_user_id: int | None = None
    if current_user is not None:
        try:
            actor_user_id = int(current_user.user_id)
        except (TypeError, ValueError):
            pass

    payload = lookup_anonymized_case(db, anonymized_id=anonymized_id)

    if payload is None:
        # Log failed lookup attempts (unknown ID) for security monitoring.
        record_immutable_audit_event(
            event_type="anonymized_case.lookup",
            action="lookup_not_found",
            actor_user_id=actor_user_id,
            actor_type="user" if actor_user_id else "anonymous",
            resource_type="anonymized_case",
            resource_id=anonymized_id,
            outcome="not_found",
            metadata={"anonymized_id": anonymized_id},
            ip_address=ip_address,
            user_agent=request.headers.get("user-agent"),
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Anonymized case not found",
        )

    # Log successful lookup.
    record_immutable_audit_event(
        event_type="anonymized_case.lookup",
        action="lookup_success",
        actor_user_id=actor_user_id,
        actor_type="user" if actor_user_id else "anonymous",
        resource_type="anonymized_case",
        resource_id=anonymized_id,
        outcome="success",
        metadata={
            "anonymized_id": anonymized_id,
            "privacy_profile": payload.get("privacy_profile"),
        },
        ip_address=ip_address,
        user_agent=request.headers.get("user-agent"),
    )

    logger.info(
        "anonymized_case_lookup",
        anonymized_id=anonymized_id,
        actor_user_id=actor_user_id,
        outcome="success",
    )

    return payload
