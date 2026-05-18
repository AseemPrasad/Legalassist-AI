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
from datetime import datetime
import structlog

from report_service import _get_reports_base_dir
from api.models import ReportGenerationRequest, ReportGenerationResponse
from api.auth import get_current_user, CurrentUser
from celery_app import generate_report_task, TaskStatus, enqueue_task_from_http_request
from db.session import get_db
from db.crud import create_report, get_report_by_id, list_reports_by_user, update_report_status

router = APIRouter(prefix="/api/v1/reports", tags=["reports"])
logger = structlog.get_logger(__name__)


@router.post(
    "/generate",
    response_model=ReportGenerationResponse,
    summary="Generate report asynchronously"
)
async def generate_report(
    request: ReportGenerationRequest,
    http_request: Request,
    current_user: CurrentUser = Depends(get_current_user),
    db = Depends(get_db)
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
        format=request.format
    )
    
    # Step 3: Update Report record with actual celery_task_id
    update_report_status(db, report_id, status="pending")
    db_report = db.query(db_report.__class__).filter(
        db_report.__class__.report_id == report_id
    ).first()
    db_report.celery_task_id = task.id
    db.commit()
    db.refresh(db_report)
    
    logger.info("Task enqueued", report_id=report_id, task_id=task.id)
    
    return ReportGenerationResponse(
        report_id=report_id,
        job_id=task.id,
        case_id=request.case_id,
        status="pending",
        report_type=request.report_type,
        format=request.format,
        created_at=datetime.utcnow()
    )


@router.get(
    "/{report_id}",
    response_model=ReportGenerationResponse,
    summary="Get report status"
)
async def get_report_status(
    report_id: str,
    current_user: CurrentUser = Depends(get_current_user),
    db = Depends(get_db)
) -> ReportGenerationResponse:
    """
    Get status of report generation job.
    
    Now uses DB record for reliable status, using stored celery_task_id
    instead of the fragile report_id-as-job_id pattern.
    """
    
    # Retrieve Report record from DB
    db_report = get_report_by_id(db, report_id, user_id=current_user.user_id)
    
    if not db_report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found"
        )
    
    # Get Celery task status using stored celery_task_id
    status_info = TaskStatus.get_task_status(db_report.celery_task_id)
    
    return ReportGenerationResponse(
        report_id=report_id,
        job_id=db_report.celery_task_id,
        case_id=str(db_report.case_id),
        status=status_info["status"],
        report_type=db_report.report_type.value,
        format=db_report.format.value,
        download_url=f"/api/v1/reports/{report_id}/download" if db_report.status.value == "completed" else None,
        file_size_bytes=db_report.file_size_bytes,
        created_at=db_report.created_at,
        completed_at=db_report.completed_at
    )


@router.get(
    "/{report_id}/download",
    summary="Download generated report"
)
async def download_report(
    report_id: str,
    current_user: CurrentUser = Depends(get_current_user),
    db = Depends(get_db)
):
    """
    Download the generated report file.
    
    Key improvements:
    - Uses stored file_path from DB (no glob patterns)
    - Validates user ownership
    - Confirms status is completed before download
    """
    
    # Retrieve Report record from DB
    db_report = get_report_by_id(db, report_id, user_id=current_user.user_id)
    
    if not db_report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found"
        )
    
    # Validate ownership
    if db_report.user_id != current_user.user_id:
        logger.warning(
            "Unauthorized download attempt",
            report_id=report_id,
            owner_id=db_report.user_id,
            requester_id=current_user.user_id
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to download this report"
        )
    
    # Check status
    if db_report.status.value != "completed":
        raise HTTPException(
            status_code=status.HTTP_202_ACCEPTED,
            detail=f"Report is still {db_report.status.value}"
        )
    
    # Validate file exists at stored path
    if not db_report.file_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report file path not found in database"
        )
    
    file_path = Path(db_report.file_path)
    if not file_path.exists():
        logger.error(
            "Report file missing",
            report_id=report_id,
            expected_path=str(file_path)
        )
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
    
    return FileResponse(
        path=file_path,
        media_type="application/pdf",
        filename=file_path.name,
    )


@router.get(
    "",
    summary="List user's reports"
)
async def list_reports(
    limit: int = 10,
    offset: int = 0,
    status_filter: str = None,
    current_user: CurrentUser = Depends(get_current_user),
    db = Depends(get_db)
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
    
    report_dicts = [
        {
            "report_id": r.report_id,
            "case_id": r.case_id,
            "status": r.status.value,
            "report_type": r.report_type.value,
            "format": r.format.value,
            "file_size_bytes": r.file_size_bytes,
            "created_at": r.created_at.isoformat(),
            "completed_at": r.completed_at.isoformat() if r.completed_at else None,
        }
        for r in reports
    ]
    
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "reports": report_dicts
    }

