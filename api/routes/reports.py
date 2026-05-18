"""
Report Generation Endpoints
POST /api/v1/reports/generate - Generate report asynchronously
GET /api/v1/reports/{report_id} - Get report status
GET /api/v1/reports/{report_id}/download - Download report
"""
import uuid
from fastapi import APIRouter, HTTPException, status, Depends, Request
from fastapi.responses import FileResponse
from pathlib import Path
from report_service import _get_reports_base_dir
from sqlalchemy.orm import Session

from api.models import ReportGenerationRequest, ReportGenerationResponse
from api.auth import get_current_user, CurrentUser
from celery_app import generate_report_task, TaskStatus, enqueue_task_from_http_request
from database import get_db, Report
import structlog
from datetime import datetime

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
    db: Session = Depends(get_db),
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
    
    Returns immediately with job ID
    """
    
    logger.info(
        "Starting report generation",
        user_id=current_user.user_id,
        case_id=request.case_id,
        report_type=request.report_type
    )

    report_id = str(uuid.uuid4())

    # Create the report record in the database
    db_report = Report(
        report_id=report_id,
        user_id=current_user.user_id,
        case_id=request.case_id,
        report_type=request.report_type,
        format=request.format,
        status="pending",
    )
    db.add(db_report)
    db.commit()
    
    # Queue async task
    task = enqueue_task_from_http_request(
        generate_report_task,
        http_request,
        context_user_id=current_user.user_id,
        user_id=current_user.user_id,
        case_id=request.case_id,
        report_type=request.report_type,
        format=request.format,
        report_id=report_id,
    )

    # Save job_id to the database record
    db_report.job_id = task.id
    db.commit()
    db.refresh(db_report)
    
    return ReportGenerationResponse(
        report_id=db_report.report_id,
        job_id=task.id,
        case_id=request.case_id,
        status="pending",
        report_type=request.report_type,
        format=request.format,
<<<<<<< fix/issue-579
        created_at=db_report.created_at
=======
        created_at=datetime.now(timezone.utc)
>>>>>>> main
    )


@router.get(
    "/{report_id}",
    response_model=ReportGenerationResponse,
    summary="Get report status"
)
async def get_report_status(
    report_id: str,
    db: Session = Depends(get_db),
    current_user: CurrentUser = Depends(get_current_user)
) -> ReportGenerationResponse:
    """Get status of report generation job"""
    
    db_report = db.query(Report).filter(
        Report.report_id == report_id,
        Report.user_id == current_user.user_id
    ).first()
    
    if not db_report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found"
        )
    
    status_str = db_report.status
    if db_report.status in ["pending", "processing"] and db_report.job_id:
        try:
            status_info = TaskStatus.get_task_status(db_report.job_id)
            celery_status = status_info["status"]
            if celery_status != db_report.status:
                db_report.status = celery_status
                if celery_status == "completed":
                    db_report.completed_at = datetime.utcnow()
                db.commit()
                db.refresh(db_report)
                status_str = db_report.status
        except Exception:
            pass
    
    return ReportGenerationResponse(
<<<<<<< fix/issue-579
        report_id=db_report.report_id,
        job_id=db_report.job_id or "unknown",
        case_id=db_report.case_id,
        status=status_str,
        report_type=db_report.report_type or "comprehensive",
        format=db_report.format,
        download_url=f"/api/v1/reports/{db_report.report_id}/download" if status_str == "completed" else None,
        created_at=db_report.created_at,
        completed_at=db_report.completed_at
=======
        report_id=report_id,
        job_id=report_id,
        case_id="unknown",
        status=status_info["status"],
        report_type="comprehensive",
        format="pdf",
        download_url=f"/api/v1/reports/{report_id}/download" if status_info["status"] == "completed" else None,
        created_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc) if status_info["status"] == "completed" else None
>>>>>>> main
    )


@router.get(
    "/{report_id}/download",
    summary="Download generated report"
)
async def download_report(
    report_id: str,
    db: Session = Depends(get_db),
    current_user: CurrentUser = Depends(get_current_user)
):
    """Download the generated report file"""
    
    db_report = db.query(Report).filter(
        Report.report_id == report_id,
        Report.user_id == current_user.user_id
    ).first()
    
    if not db_report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found"
        )
    
    status_str = db_report.status
    if db_report.status in ["pending", "processing"] and db_report.job_id:
        try:
            status_info = TaskStatus.get_task_status(db_report.job_id)
            celery_status = status_info["status"]
            if celery_status != db_report.status:
                db_report.status = celery_status
                if celery_status == "completed":
                    db_report.completed_at = datetime.utcnow()
                db.commit()
                db.refresh(db_report)
                status_str = db_report.status
        except Exception:
            pass

    if status_str != "completed":
        raise HTTPException(
            status_code=status.HTTP_202_ACCEPTED,
            detail=f"Report is still {status_str}"
        )
    
    base_dir = _get_reports_base_dir()
    user_dir = base_dir / str(current_user.user_id)

    # Find by any report file that ends with the report_id.
    # Filenames are: <case_id>_<report_type>_<report_id>.<ext>
    if not user_dir.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report output not found",
        )

    ext = ".pdf" if db_report.format == "pdf" else f".{db_report.format}"
    matches = list(user_dir.glob(f"*_{report_id}{ext}"))
    if not matches:
        matches = list(user_dir.glob(f"*{report_id}{ext}"))

    if not matches:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report file not found",
        )

    file_path = matches[0]

    return FileResponse(
        path=file_path,
        media_type="application/pdf" if db_report.format == "pdf" else "application/octet-stream",
        filename=file_path.name,
    )


@router.get(
    "",
    summary="List user's reports"
)
async def list_reports(
    limit: int = 10,
    offset: int = 0,
    db: Session = Depends(get_db),
    current_user: CurrentUser = Depends(get_current_user)
) -> dict:
    """Get list of generated reports for current user"""
    
    query = db.query(Report).filter(Report.user_id == current_user.user_id)
    total = query.count()
    reports = query.order_by(Report.created_at.desc()).offset(offset).limit(limit).all()

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

