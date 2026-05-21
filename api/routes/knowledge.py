"""Knowledge freshness and invalidation endpoints."""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from api.auth import CurrentUser, get_current_user
from api.feature_flags import is_feature_enabled_for_user, get_feature_flag_manager
from api.models import KnowledgeInvalidationItem, KnowledgeInvalidationListResponse
from db.crud.knowledge import get_knowledge_freshness_summary, list_knowledge_invalidations
from database import get_db


router = APIRouter(prefix="/api/v1/knowledge", tags=["knowledge"])


@router.get("/invalidations", response_model=KnowledgeInvalidationListResponse, summary="List knowledge invalidations")
async def list_invalidations(
    case_id: int | None = Query(default=None),
    document_id: int | None = Query(default=None),
    status: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
    current_user: CurrentUser = Depends(get_current_user),
) -> KnowledgeInvalidationListResponse:
    """Return the invalidation ledger for the current user.

    Admin users can still see their own invalidations; this endpoint keeps the
    initial scope user-centric and can be expanded later with org-level filters.
    """

    feature_enabled = is_feature_enabled_for_user(
        "knowledge_status_dashboard",
        str(current_user.user_id),
        attributes={"role": current_user.role, "email": current_user.email},
        surface="api",
    )

    if not feature_enabled:
        return KnowledgeInvalidationListResponse(
            items=[],
            total=0,
            stale_count=0,
            fresh_count=0,
            next_recompute_at=None,
            generated_at=datetime.now(timezone.utc),
        )

    get_feature_flag_manager().mark_flag_used(
        "knowledge_status_dashboard",
        user_id=str(current_user.user_id),
        surface="api",
    )

    rows = list_knowledge_invalidations(
        db,
        user_id=current_user.user_id if current_user.role != "admin" else None,
        case_id=case_id,
        document_id=document_id,
        status=status,
        limit=limit,
    )
    summary = get_knowledge_freshness_summary(
        db,
        user_id=current_user.user_id if current_user.role != "admin" else None,
        case_id=case_id,
    )

    items = [
        KnowledgeInvalidationItem(
            id=row.id,
            user_id=row.user_id,
            case_id=row.case_id,
            document_id=row.document_id,
            scope_type=row.scope_type,
            scope_value=row.scope_value,
            reason=row.reason,
            details=row.details,
            status=row.status,
            invalidated_at=row.invalidated_at,
            scheduled_for=row.scheduled_for,
            recompute_started_at=row.recompute_started_at,
            recompute_completed_at=row.recompute_completed_at,
            error_message=row.error_message,
            recompute_attempts=row.recompute_attempts,
        )
        for row in rows
    ]

    return KnowledgeInvalidationListResponse(
        items=items,
        total=summary["total"],
        stale_count=summary["stale"],
        fresh_count=summary["fresh"],
        next_recompute_at=summary["next_recompute_at"],
        generated_at=datetime.now(timezone.utc),
    )
