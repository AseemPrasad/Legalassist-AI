"""
Document Analysis Endpoints
POST /api/v1/analyze/document - Analyze document asynchronously
GET /api/v1/analyze/{job_id} - Check analysis job status
"""
import os
import uuid
from pathlib import Path
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status, Depends
from fastapi import Request
from sqlalchemy.orm import Session
from api.models import DocumentAnalysisRequest, DocumentAnalysisSummary, AnalysisJobResponse
from api.auth import get_current_user, CurrentUser
from api.dependencies import get_db_rls
from api.limiter import RateLimit
from db.models.cases import Attachment

try:
    from celery_app import analyze_document_task, TaskStatus, enqueue_task_from_http_request
except Exception:
    analyze_document_task = None
    TaskStatus = None
    enqueue_task_from_http_request = None
from api.validation import (
    validate_file_upload,
    validate_file_url,
    validate_text_input,
    ValidationConfig,
    PayloadTooLargeError,
)
from api.job_registry import register_job_owner, get_job_owner
from config import Config
import structlog

router = APIRouter(prefix="/api/v1/analyze", tags=["document-analysis"])
logger = structlog.get_logger(__name__)


def validate_file_path(file_path: str) -> str:
    """Canonicalize and restrict *file_path* to allowed directories.

    Raises HTTPException(400) if the path is not within one of the
    configured allowed directories.
    """
    raw = Path(file_path)
    try:
        resolved = raw.resolve(strict=False)
    except (OSError, RuntimeError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="file_path could not be resolved",
        )

    if not resolved.exists():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="file_path does not exist",
        )

    allowed_dirs = [
        Path(Config.ATTACHMENTS_DIR).resolve(),
    ]
    # Also allow the upload temp dir if configured
    upload_temp = getattr(Config, "UPLOAD_TEMP_DIR", None)
    if upload_temp:
        allowed_dirs.append(Path(upload_temp).resolve())
    # Allow the project-root attachments dir as a fallback
    allowed_dirs.append(Path.cwd().resolve() / "attachments")

    allowed = any(
        resolved == d or str(resolved).startswith(str(d) + os.sep)
        for d in allowed_dirs
    )
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="file_path is outside the allowed directories",
        )

    if resolved.is_symlink():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="file_path must not be a symbolic link",
        )

    return str(resolved)


@router.post(
    "/document",
    response_model=AnalysisJobResponse,
    summary="Analyze document asynchronously",
    description="Upload or provide document text for analysis. Returns immediately with job ID.",
    dependencies=[Depends(RateLimit(requests=10, window=300))],
)
async def analyze_document(
    request: DocumentAnalysisRequest,
    http_request: Request,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db_rls),
) -> AnalysisJobResponse:
    """
    Analyze a legal document asynchronously
    
    - **file_url**: URL to document (if not uploading)
    - **file_path**: Local file path (if not uploading)
    - **text**: Document text directly (if not uploading)
    - **document_type**: Type of document (contract, lawsuit, etc.)
    - **extract_remedies**: Extract remedy clauses
    - **extract_deadlines**: Extract important deadlines
    - **extract_obligations**: Extract obligations
    
    Returns job ID to track progress
    """
    if not any([request.file_url, request.file_path, request.text]):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Must provide file_url, file_path, or text"
        )
    
    # Validate text input if provided
    if request.text:
        validate_text_input(request.text, max_length=ValidationConfig.MAX_TEXT_LENGTH)

    # SSRF validation: block private/internal IPs for file_url
    if request.file_url:
        validate_file_url(request.file_url)

    # Path traversal prevention: canonicalize and restrict file_path
    safe_path = validate_file_path(request.file_path) if request.file_path else None
    
    # Ownership verification: the user must own an Attachment record for this path
    if safe_path:
        attachment = db.query(Attachment).filter(
            Attachment.stored_path == safe_path,
            Attachment.user_id == int(current_user.user_id),
        ).first()
        if not attachment:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to access this file",
            )
    
    # Generate document ID and job ID
    document_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())
    
    logger.info(
        "Starting document analysis",
        user_id=current_user.user_id,
        document_id=document_id,
        job_id=job_id
    )
    
    # Queue async task
    task = enqueue_task_from_http_request(
        analyze_document_task,
        http_request,
        context_user_id=current_user.user_id,
        user_id=current_user.user_id,
        document_id=document_id,
        text=request.text,
        file_path=safe_path,
        file_url=request.file_url,
        document_type=request.document_type,
    )
    
    register_job_owner(task.id, current_user.user_id)
    
    return AnalysisJobResponse(
        job_id=task.id,
        status="pending",
        created_at=datetime.now(timezone.utc)
    )


@router.get(
    "/{job_id}",
    response_model=AnalysisJobResponse,
    summary="Get analysis job status"
)
async def get_analysis_status(
    job_id: str,
    current_user: CurrentUser = Depends(get_current_user)
) -> AnalysisJobResponse:
    """Get status and result of analysis job"""
    
    status_info = TaskStatus.get_task_status(job_id)
    
    return AnalysisJobResponse(
        job_id=job_id,
        status=status_info["status"],
        created_at=datetime.now(timezone.utc),
        result_url=f"/api/v1/analyze/{job_id}/result" if status_info["status"] == "completed" else None
    )


@router.get(
    "/{job_id}/result",
    response_model=DocumentAnalysisSummary,
    summary="Get analysis result"
)
async def get_analysis_result(
    job_id: str,
    current_user: CurrentUser = Depends(get_current_user)
) -> DocumentAnalysisSummary:
    """Get the complete analysis result"""
    
    status_info = TaskStatus.get_task_status(job_id)
    
    if status_info["status"] != "completed":
        raise HTTPException(
            status_code=status.HTTP_202_ACCEPTED,
            detail=f"Job is still {status_info['status']}"
        )
    
    result = status_info["info"]
    
    return DocumentAnalysisSummary(
        document_id=result.get("document_id", job_id),
        title=result.get("title", "Untitled"),
        document_type=result.get("document_type", "unknown"),
        summary=result.get("summary", ""),
        key_points=result.get("key_points", []),
        remedies=result.get("remedies", []),
        deadlines=result.get("deadlines", []),
        obligations=result.get("obligations", []),
        confidence_score=result.get("confidence_score", 0.0),
        remedies_confidence_score=result.get("remedies_confidence_score"),
        remedies_evidence_spans=result.get("remedies_evidence_spans", []),
        analysis_time_seconds=result.get("analysis_time_seconds", 0.0)
    )


@router.post(
    "/{job_id}/cancel",
    summary="Cancel analysis job"
)
async def cancel_analysis(
    job_id: str,
    current_user: CurrentUser = Depends(get_current_user)
) -> dict:
    """Cancel an analysis job"""
    
    owner_id = get_job_owner(job_id)
    if owner_id is not None and owner_id != int(current_user.user_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to cancel this job",
        )
    
    success = TaskStatus.revoke_task(job_id)
    
    if success:
        return {"status": "cancelled", "job_id": job_id}
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to cancel job"
        )


@router.post(
    "/upload",
    response_model=AnalysisJobResponse,
    summary="Upload document file for analysis",
    description="Upload a PDF, Word, or text file for legal analysis.",
    dependencies=[Depends(RateLimit(requests=5, window=300))],
)
async def upload_document_file(
    http_request: Request,
    file: UploadFile = File(...),
    document_type: str = Form(default="unknown"),
    current_user: CurrentUser = Depends(get_current_user)
) -> AnalysisJobResponse:
    """
    Upload and analyze a document file asynchronously
    
    - **file**: Document file (PDF, DOCX, DOC, TXT, HTML, RTF)
    - **document_type**: Type of document (contract, lawsuit, etc.)
    
    Returns job ID to track progress
    """
    import uuid
    
    try:
        # Validate file metadata upfront
        validate_file_upload(
            file,
            max_size=ValidationConfig.MAX_UPLOAD_SIZE,
            allowed_extensions=ValidationConfig.ALLOWED_EXTENSIONS,
            allowed_mime_types=ValidationConfig.ALLOWED_MIME_TYPES,
        )
        
        # Rewind the stream after validation (magic-bytes check in
        # validate_file_upload reads from the underlying SpooledTemporaryFile
        # directly, which can desync the UploadFile async wrapper).
        await file.seek(0)

        # Read file content into memory, then validate size from the buffer.
        file_content = await file.read()
        file_bytes_read = len(file_content)

        if file_bytes_read > ValidationConfig.MAX_UPLOAD_SIZE:
            raise PayloadTooLargeError(
                detail=f"Upload exceeded maximum size limit of {round(ValidationConfig.MAX_UPLOAD_SIZE / 1024 / 1024, 2)} MB"
            )

        logger.info(
            "File uploaded successfully",
            user_id=current_user.user_id,
            filename=file.filename,
            size_bytes=file_bytes_read,
            document_type=document_type,
        )
        
        file_ext = file.filename.split(".")[-1].lower() if file.filename else ""

        # MIME sniff first bytes to catch renamed binaries (e.g. PDF → .txt)
        try:
            import magic
            mime_type = magic.from_buffer(file_content[:2048], mime=True)
        except (ImportError, Exception):
            mime_type = None

        # Generate IDs
        document_id = str(uuid.uuid4())

        logger.info(
            "Starting document analysis from upload",
            user_id=current_user.user_id,
            document_id=document_id,
            filename=file.filename,
            mime_type=mime_type,
        )

        # Pass file content: decode text files, keep PDFs as bytes for worker extraction
        is_text = file_ext in ("txt", "html", "rtf") and (
            mime_type is None or mime_type.startswith("text/")
        )
        if is_text:
            text = file_content.decode("utf-8", errors="ignore")
            file_bytes = None
        else:
            text = None
            file_bytes = file_content

        task = enqueue_task_from_http_request(
            analyze_document_task,
            http_request,
            context_user_id=current_user.user_id,
            user_id=current_user.user_id,
            document_id=document_id,
            text=text,
            file_bytes=file_bytes,
            document_type=document_type,
        )
        
        register_job_owner(task.id, current_user.user_id)
        
        return AnalysisJobResponse(
            job_id=task.id,
            status="pending",
            created_at=datetime.now(timezone.utc)
        )
    
    except Exception as e:
        logger.error(
            "File upload failed",
            user_id=current_user.user_id,
            filename=file.filename,
            error=str(e),
        )
        raise
