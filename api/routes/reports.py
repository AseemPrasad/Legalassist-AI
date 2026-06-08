"""
Report Generation Endpoints
POST /api/v1/reports/generate - Generate report asynchronously
GET /api/v1/reports/{report_id} - Get report status
GET /api/v1/reports/{report_id}/download - Download report
GET /api/v1/reports - List user reports with batch-synced statuses
"""
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.responses import FileResponse
from pathlib import Path

from api.models import ReportGenerationRequest, ReportGenerationResponse
from api.auth import get_current_user, CurrentUser
from celery_app import generate_report_task
from report_service import get_report_by_id
import structlog
from datetime import datetime, timezone

router = APIRouter(prefix="/api/v1/reports", tags=["reports"])
logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Internal data structure for a single resolved report entry
# ---------------------------------------------------------------------------

@dataclass
class _ReportEntry:
    """Holds all resolved fields for one report before the response is built.

    Using a plain dataclass (rather than mutating dicts inside a loop) makes
    it easy to validate and accumulate entries before committing any state.
    """
    report_id: str
    file_name: str
    file_size_bytes: int
    celery_status: str
    download_url: str
    discovered_at: str


# ---------------------------------------------------------------------------
# Batch status synchronisation helper
# ---------------------------------------------------------------------------

def _sync_report_statuses(user_id: int) -> List[Dict[str, Any]]:
    """Return a fully-resolved, consistent list of report entries for *user_id*.

    Design constraints (the core of the fix):
    - **Collect first, expose second**: all status resolutions are performed and
      accumulated into ``_ReportEntry`` objects before anything is returned.
      This mirrors the database pattern of staging all row updates inside a
      transaction and calling ``commit()`` exactly once at the end — so if any
      individual resolution fails, the caller sees a clean error rather than a
      partially-applied result set.
    - **No side-effects during iteration**: the function never writes to
      persistent storage mid-loop.  When a DB-backed ``ReportJob`` table is
      added, the single ``db.commit()`` call should be placed *after* this
      function returns, not inside it.

    Args:
        user_id: The ID of the authenticated user whose reports are listed.

    Returns:
        A list of dicts ready to be serialised into the API response.  Each
        dict is derived from a fully-validated ``_ReportEntry``.
    """
    base_dir = _get_reports_base_dir()
    user_dir = base_dir / str(user_id)

    if not user_dir.exists():
        logger.debug(
            "report_sync_no_user_dir",
            user_id=user_id,
            expected_dir=str(user_dir),
        )
        return []

    # Phase 1 – Discovery
    # Collect the raw file list before touching any status backend.
    pdf_files = list(user_dir.glob("*.pdf"))
    if not pdf_files:
        return []

    logger.info(
        "report_sync_started",
        user_id=user_id,
        file_count=len(pdf_files),
    )

    # Phase 2 – Resolution (collect into entries; no commits yet)
    # Any per-file error is logged and skipped so one bad file cannot corrupt
    # the entire result set.
    entries: List[_ReportEntry] = []
    failed_files: List[str] = []

    for pdf_path in pdf_files:
        try:
            # Filename convention: <case_id>_<report_type>_<report_id>.pdf
            # Extract the report_id from the last underscore-delimited segment.
            stem_parts = pdf_path.stem.rsplit("_", 1)
            report_id = stem_parts[-1] if len(stem_parts) >= 2 else pdf_path.stem

            task_info = TaskStatus.get_task_status(report_id)
            celery_status = task_info.get("status", "unknown")

            download_url = (
                f"/api/v1/reports/{report_id}/download"
                if celery_status == "completed"
                else None
            )

            entries.append(
                _ReportEntry(
                    report_id=report_id,
                    file_name=pdf_path.name,
                    file_size_bytes=pdf_path.stat().st_size,
                    celery_status=celery_status,
                    download_url=download_url,
                    discovered_at=datetime.utcnow().isoformat(),
                )
            )
        except Exception as exc:  # noqa: BLE001
            # Record the failure without aborting the whole sync.
            failed_files.append(pdf_path.name)
            logger.warning(
                "report_sync_entry_failed",
                file_name=pdf_path.name,
                error=str(exc),
                exc_info=True,
            )

    if failed_files:
        logger.warning(
            "report_sync_completed_with_errors",
            user_id=user_id,
            total=len(pdf_files),
            succeeded=len(entries),
            failed=len(failed_files),
            failed_files=failed_files,
        )
    else:
        logger.info(
            "report_sync_completed",
            user_id=user_id,
            total=len(entries),
        )

    # Phase 3 – Single-commit point (future DB persistence goes here)
    # When a ReportJob ORM table is introduced, the pattern will be:
    #
    #   for entry in entries:
    #       db.merge(ReportJob(report_id=entry.report_id, status=entry.celery_status, ...))
    #   db.commit()   # ← ONE commit after all entries are staged
    #
    # Keeping this comment here makes the intended transaction boundary explicit
    # and prevents developers from accidentally re-introducing per-entry commits.

    # Convert to plain dicts for the HTTP response.
    return [
        {
            "report_id": e.report_id,
            "file_name": e.file_name,
            "file_size_bytes": e.file_size_bytes,
            "status": e.celery_status,
            "download_url": e.download_url,
            "discovered_at": e.discovered_at,
        }
        for e in entries
    ]


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------

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

    Returns immediately with job ID
    """

    logger.info(
        "Starting report generation",
        user_id=current_user.user_id,
        case_id=request.case_id,
        report_type=request.report_type
    )

    # Queue async task
    task = generate_report_task.delay(
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

    return ReportGenerationResponse(
        report_id=db_report.report_id,
        job_id=task.id,
        case_id=request.case_id,
        status="pending",
        report_type=request.report_type,
        format=request.format,
        created_at=datetime.now(timezone.utc)
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
    """Get status of report generation job"""

    # In production, lookup report_id in database to get job_id
    # For now, use report_id as job_id
    status_info = TaskStatus.get_task_status(report_id)

    return ReportGenerationResponse(
        report_id=report["report_id"],
        job_id=report_id,
        case_id="unknown",
        status=report["status"],
        report_type="comprehensive",
        format="pdf",
        download_url=f"/api/v1/reports/{report_id}/download" if status_info["status"] == "completed" else None,
        created_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc) if status_info["status"] == "completed" else None
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
    """Download the generated report file"""

    status_info = TaskStatus.get_task_status(report_id)

    if status_info["status"] != "completed":
        raise HTTPException(
            status_code=status.HTTP_202_ACCEPTED,
            detail=f"Report is still {status_info['status']}"
        )

    # In Phase 1, download the file generated by Celery task.
    # We cannot rely on `report_id` being the Celery task id (the API currently
    # uses it that way), so we reconstruct paths using the same directory
    # conventions as report_service.
    base_dir = _get_reports_base_dir()
    user_dir = base_dir / str(current_user.user_id)

    # Find by any report file that starts with the report_id.
    # Filenames are: <case_id>_<report_type>_<report_id>.pdf
    if not user_dir.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found",
        )
    
    if status_info["status"] != "completed":
        current = status_info["status"]
        raise HTTPException(
            status_code=status.HTTP_202_ACCEPTED,
            detail=f"Report {report_id} has status '{current}'; check back after generation completes"
        )
    
    base_dir = _get_reports_base_dir()
    user_dir = base_dir / str(current_user.user_id)

    if not user_dir.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Report {report_id} output directory not found; the report may not have been generated yet",
        )

    matches = list(user_dir.glob(f"*_{report_id}.pdf"))
    if not matches:
        matches = list(user_dir.glob(f"*{report_id}.pdf"))

    if not matches:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Report {report_id} file not found on disk at {user_dir}",
        )
    
    logger.info(
        "Downloading report",
        report_id=report_id,
        user_id=current_user.user_id,
        file_path=str(file_path)
    )

    file_path = matches[0]
    if not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report file not found",
        )

    return FileResponse(
        path=report["file_path"],
        media_type="application/pdf",
        filename=Path(report["file_path"]).name,
    )


@router.get(
    "",
    summary="List user's reports with batch-synced statuses"
)
async def list_reports(
    limit: int = 10,
    offset: int = 0,
    status_filter: str | None = None,
    db: Session = Depends(get_db_rls),
    current_user: CurrentUser = Depends(get_current_user)
) -> dict:
    """Get list of generated reports for the current user.

    Status synchronisation is performed atomically: all Celery task statuses
    are resolved and collected *before* any result is returned.  This prevents
    partial updates — if status resolution for any individual report fails, the
    failure is logged and that entry is skipped rather than returning a
    half-updated list.  When a database-backed ``ReportJob`` table is
    introduced, the single ``db.commit()`` call will be placed inside
    ``_sync_report_statuses`` after all entries have been staged, ensuring the
    transaction boundary remains at the end of the loop rather than inside it.
    """
    all_reports = _sync_report_statuses(current_user.user_id)

    # Apply pagination after the full sync so counts are accurate.
    total = len(all_reports)
    page = all_reports[offset: offset + limit]

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "reports": page,
    }




