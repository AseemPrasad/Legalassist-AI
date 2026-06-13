"""Audit trail endpoints."""

from __future__ import annotations

import csv
from io import StringIO

from fastapi import APIRouter, Depends, HTTPException, Response, status, Query
from sqlalchemy.orm import Session

from api.auth import CurrentUser, get_current_user, get_admin_user
from api.models import AuditEventItem, AuditEventListResponse
from database import Case
from api.dependencies import get_db_rls
from db.models import AuditEvent
from db.crud.audit import list_audit_events, audit_events_to_csv

router = APIRouter(prefix="/api/v1/audit", tags=["audit"])


@router.get("/cases/{case_id}", response_model=AuditEventListResponse, summary="View audit events for a case")
async def get_case_audit_events(
    case_id: int,
    limit: int = Query(default=100, ge=1, le=500, description="Maximum number of audit events to return (1–500)"),
    db: Session = Depends(get_db),
    current_user: CurrentUser = Depends(get_current_user),
) -> AuditEventListResponse:
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Case not found")
    if current_user.role != "admin" and case.user_id != current_user.user_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Case not found")

    events = list_audit_events(db, case_id=case_id, limit=limit)
    return AuditEventListResponse(
        case_id=case_id,
        total=len(events),
        events=[
            AuditEventItem(
                id=event.id,
                actor=event.actor,
                actor_user_id=event.actor_user_id,
                action=event.action,
                resource=event.resource,
                case_id=event.case_id,
                occurred_at=event.occurred_at,
                metadata=event.event_metadata or {},
            )
            for event in events
        ],
    )


@router.get("/cases/{case_id}/export", summary="Export case audit trail CSV")
async def export_case_audit_events(
    case_id: int,
    db: Session = Depends(get_db_rls),
    admin_user: CurrentUser = Depends(get_admin_user),
) -> Response:
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Case not found")

    events = list_audit_events(db, case_id=case_id, limit=1000)
    csv_bytes = audit_events_to_csv(events)
    filename = f"case_{case_id}_audit_trail.csv"
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )



