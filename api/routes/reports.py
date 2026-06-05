"""
Report Generation Endpoints

This module provides REST endpoints for generating, tracking, and downloading legal reports.

Endpoints:
- POST /api/v1/reports/generate - Generate report asynchronously
- GET /api/v1/reports/{report_id} - Get report status  
- GET /api/v1/reports/{report_id}/download - Download report
- GET /api/v1/reports - List user's reports

Key refactoring:
- Uses Report DB model instead of glob patterns
- Stores celery_task_id for reliable task tracking
- Validates user ownership on downloads
- No more report_id = job_id confusion
"""
import uuid
from fastapi import APIRouter, HTTPException, status, Depends, Request
from fastapi.responses import FileResponse
from pathlib import Path

from api.models import ReportGenerationRequest, ReportGenerationResponse
from api.auth import get_current_user, CurrentUser
from celery_app import generate_report_task
from report_service import get_report_by_id
import structlog
from datetime import datetime

router = APIRouter(prefix="/api/v1/reports", tags=["reports"])
logger = structlog.get_logger(__name__)


def _record_download_audit(
    *,
    user_id: int,
    report_id: str,
    file_name: str,
    file_size_bytes: int,
) -> None:
    """Persist a structured audit record for a report download.

    Raises any underlying storage exceptions to the caller so it can
    decide whether to propagate or swallow them.  Keeping the logic here
    makes it easy to swap the structlog call for a DB-backed AuditLog
    insert once the schema is available::

        db.add(AuditLog(
            user_id=user_id,
            action="report_download",
            resource_id=report_id,
            ...
        ))
        db.commit()
    """
    logger.info(
        "report_downloaded",
        user_id=user_id,
        report_id=report_id,
        file_name=file_name,
        file_size_bytes=file_size_bytes,
        downloaded_at=datetime.utcnow().isoformat(),
    )
    # TODO: persist to AuditLog table when DB schema migration is ready.


@router.post(
    "/generate",
    response_model=ReportGenerationResponse,
    summary="Generate report asynchronously"
)
async def generate_report(
    request: ReportGenerationRequest,
    http_request: Request,
    db: Session = Depends(get_db_rls),
    current_user: CurrentUser = Depends(get_current_user)
) -> ReportGenerationResponse:
    """
    Generate a legal report asynchronously
    
    - **case_id**: Case ID to generate report for
    - **report_type**: comprehensive, summary, or legal_brief
    - **include_remedies**: Include remedy clauses
    - **include_timeline**: Include case timeline
    - **include_similar_cases**: Include similar cases
    - **format**: pdf or docx
    - **style**: formal or casual
    
    Returns immediately with report_id for polling status.
    Uses DB-backed Report model for reliability.
    """
    
    logger.info(
        "Starting report generation",
        user_id=current_user.user_id,
        case_id=request.case_id,
        report_type=request.report_type
    )

    # Step 1: Create and persist Report record BEFORE enqueueing task
    # This ensures we have report_id and can track the task reliably
    report_id = str(uuid.uuid4())
    
    try:
        # Parse case_id as integer (assumes it's numeric in the DB)
        case_id_int = int(request.case_id)
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid case_id format"
        )
    
    # Create Report record in DB
    db_report = create_report(
        db,
        report_id=report_id,
        user_id=current_user.user_id,
        case_id=case_id_int,
        celery_task_id="pending",  # Will be updated after task enqueue
        report_type=request.report_type,
        format=request.format,
        style=request.style
    )
    
    logger.info("Report record created", report_id=report_id, db_id=db_report.id)
    
    # Step 2: Queue async task with report_id parameter
    task = enqueue_task_from_http_request(
        generate_report_task,
        http_request,
        context_user_id=current_user.user_id,
        user_id=str(current_user.user_id),
        case_id=str(case_id_int),
        report_id=report_id,
        report_type=request.report_type,
        format=request.format,
        privacy_profile=request.privacy_profile,
    )

    # Save job_id to the database record
    db_report.job_id = task.id
    db.commit()
    db.refresh(db_report)
    
    logger.info("Task enqueued", report_id=report_id, task_id=task.id)
    
    return ReportGenerationResponse(
        report_id=db_report.report_id,
        job_id=task.id,
        case_id=request.case_id,
        status="pending",
        report_type=request.report_type,
        format=request.format,
        created_at=db_report.created_at
    )


@router.get(
    "/{report_id}",
    response_model=ReportGenerationResponse,
    summary="Get report status"
)
async def get_report_status(
    report_id: str,
    db: Session = Depends(get_db_rls),
    current_user: CurrentUser = Depends(get_current_user)
) -> ReportGenerationResponse:
    """Get status of report generation job with ownership validation."""

    report = get_report_by_id(report_id, current_user.user_id)
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found",
        )

    return ReportGenerationResponse(
        report_id=report["report_id"],
        job_id=report_id,
        case_id="unknown",
        status=report["status"],
        report_type="comprehensive",
        format="pdf",
        download_url=report["download_url"],
        created_at=datetime.utcnow(),
        completed_at=datetime.utcnow() if report["status"] == "completed" else None,
    )


@router.get(
    "/{report_id}/download",
    summary="Download generated report"
)
async def download_report(
    report_id: str,
    db: Session = Depends(get_db_rls),
    current_user: CurrentUser = Depends(get_current_user)
):
    """Download the generated report file."""

    report = get_report_by_id(report_id, current_user.user_id)
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found",
        )

    if report["status"] != "completed":
        raise HTTPException(
            status_code=status.HTTP_202_ACCEPTED,
            detail=f"Report is still {report['status']}",
        )

    if not report["file_path"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report file not found on disk"
        )
    
    logger.info(
        "Downloading report",
        report_id=report_id,
        user_id=current_user.user_id,
        file_path=str(file_path)
    )

    # Audit logging is a non-critical observability concern.
    # A transient failure (DB unavailable, pending migration, etc.) must
    # never prevent the user from receiving a file that already exists on
    # disk.  We catch all exceptions, record full diagnostic context so
    # on-call engineers can investigate, then continue with delivery.
    try:
        _record_download_audit(
            user_id=current_user.user_id,
            report_id=report_id,
            file_name=file_path.name,
            file_size_bytes=file_path.stat().st_size,
        )
    except Exception as audit_exc:  # noqa: BLE001
        logger.warning(
            "audit_log_failed_for_report_download",
            report_id=report_id,
            user_id=current_user.user_id,
            error=str(audit_exc),
            exc_info=True,
        )

    return FileResponse(
        path=report["file_path"],
        media_type="application/pdf",
        filename=Path(report["file_path"]).name,
    )


@router.get(
    "",
    summary="List user's reports"
)
async def list_reports(
    limit: int = 10,
    offset: int = 0,
    status_filter: str | None = None,
    db: Session = Depends(get_db_rls),
    current_user: CurrentUser = Depends(get_current_user)
) -> dict:
    """
    Get list of generated reports for current user with pagination.
    
    Optional filters:
    - status_filter: Filter by status (pending, processing, completed, failed)
    """
    
    reports, total = list_reports_by_user(
        db,
        user_id=current_user.user_id,
        limit=limit,
        offset=offset,
        status=status_filter
    )

    reports_data = []
    for r in reports:
        status_str = r.status
        if r.status in ["pending", "processing"] and r.job_id:
            try:
                status_info = TaskStatus.get_task_status(r.job_id)
                celery_status = status_info["status"]
                if celery_status != r.status:
                    r.status = celery_status
                    if celery_status == "completed":
                        r.completed_at = datetime.utcnow()
                    db.commit()
                    status_str = r.status
            except Exception:
                pass

        reports_data.append({
            "report_id": r.report_id,
            "job_id": r.job_id or "unknown",
            "case_id": r.case_id,
            "status": status_str,
            "report_type": r.report_type or "comprehensive",
            "format": r.format,
            "download_url": f"/api/v1/reports/{r.report_id}/download" if status_str == "completed" else None,
            "created_at": r.created_at,
            "completed_at": r.completed_at
        })

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "reports": reports_data
    }




