"""
Case Search Endpoints
POST /api/v1/cases/search - Search for similar cases
POST /api/v1/cases/similarity-feedback - Save similarity feedback
GET /api/v1/cases/{id}/timeline - Get case timeline
"""
import csv
import io
import re
from typing import Optional
from datetime import datetime, timezone, timedelta
import secrets
from pathlib import Path

from fastapi import APIRouter, HTTPException, status, Depends, UploadFile, File, Form, Request, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, Response
from sqlalchemy.orm import Session
from api.models import (
    CaseSearchRequest, CaseSearchResponse, CaseResult,
    CaseTimeline, CaseEvent, SimilarityFeedbackRequest,
    SimilarityFeedbackResponse,
    CaseNoteDraftRequest,
    CaseNotePublishRequest,
    CaseNoteHistoryResponse,
    CaseNoteVersionItem,
    AnonymizedShareCreateRequest,
    AnonymizedShareResponse,
)
from api.auth import get_current_user, CurrentUser
from api.validation import validate_file_upload, validate_file_upload_streaming, ValidationConfig
import structlog
from sqlalchemy import func
from functools import wraps
from config import Config
from services.privacy_redaction import normalize_privacy_profile

from database import (
    CaseRecord,
    CaseOutcome,
    submit_similarity_feedback,
    Case,
    DocumentType,
    CaseDocument,
    Attachment,
    AnonymizedShareToken,
    SimilarityFeedback,
    get_db,
)
from api.dependencies import get_db_rls
from db.case_service import save_case_note_draft, publish_case_note, get_case_note_history
from db.repositories.case_queries import fetch_latest_documents_per_case
from db.crud.audit import record_immutable_audit_event
from services.timeline_service import timeline_service as _timeline_service
try:
    from celery_app import enqueue_task_from_http_request, process_case_document_upload_task
except Exception:
    enqueue_task_from_http_request = None
    process_case_document_upload_task = None
from analytics_engine import CaseSimilarityCalculator

router = APIRouter(prefix="/api/v1/cases", tags=["cases"])
logger = structlog.get_logger(__name__)

EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_PATTERN = re.compile(r"(?:(?:\+?\d[\d\s().-]{6,}\d))")


def redact_pii(text: Optional[str]) -> Optional[str]:
    if not isinstance(text, str):
        return text
    text = EMAIL_PATTERN.sub("[redacted-email]", text)
    text = PHONE_PATTERN.sub("[redacted-phone]", text)
    return text

def _build_case_summary_payload(case: Case, latest_doc: CaseDocument | None = None) -> dict:
    return {
        "case_id": str(case.id),
        "case_number": case.case_number,
        "title": case.title or case.case_number,
        "parties": (
            (latest_doc.extracted_metadata or {}).get("parties", [])
            if latest_doc and latest_doc.extracted_metadata
            else []
        ),
        "jurisdiction": case.jurisdiction,
        "status": case.status.value if hasattr(case.status, 'value') else str(case.status),
        "summary": latest_doc.summary if latest_doc else "",
    }


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _coerce_datetime(value):
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _build_share_url(token: str) -> str:
    base_url = str(Config.BASE_URL or "").rstrip("/")
    path = f"/api/v1/cases/share/{token}"
    return f"{base_url}{path}" if base_url else path


def _generate_unique_share_token(db: Session) -> str:
    for _ in range(5):
        token = secrets.token_urlsafe(32)
        if not db.query(AnonymizedShareToken).filter(AnonymizedShareToken.token == token).first():
            return token
    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to generate share token")


def _get_share_token_or_404(db: Session, token: str) -> AnonymizedShareToken:
    share = db.query(AnonymizedShareToken).filter(AnonymizedShareToken.token == token).first()
    if not share:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Share token not found")
    expires_at = _coerce_datetime(share.expires_at)
    if expires_at and expires_at < _utcnow():
        raise HTTPException(status_code=status.HTTP_410_GONE, detail="Share token expired")
    return share


def _build_anonymized_share_payload(case: Case, share: AnonymizedShareToken) -> dict:
    from case_manager import generate_anonymized_case_data

    anonymized_data = generate_anonymized_case_data(case.id, profile_name=share.scope)
    if anonymized_data is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Anonymized case not found")

    payload = dict(anonymized_data)
    payload["share"] = {
        "token": share.token,
        "scope": share.scope,
        "expires_at": share.expires_at.isoformat() if share.expires_at else None,
        "created_at": share.created_at.isoformat() if share.created_at else None,
        "anonymized_id": share.anonymized_id,
    }
    return payload


def _build_anonymized_share_pdf(case: Case, share: AnonymizedShareToken, payload: dict) -> bytes:
    from pdf_exporter import generate_anonymized_pdf

    pdf_bytes = generate_anonymized_pdf(
        case.id,
        share.anonymized_id,
        case.user_id,
        profile_name=share.scope,
        anonymized_data=payload,
    )
    if pdf_bytes is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unable to generate anonymized PDF")
    return pdf_bytes


def _audit_case_view_route(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        result = await func(*args, **kwargs)
        case_id = kwargs.get("case_id")
        current_user = kwargs.get("current_user")
        if result is not None and case_id is not None and current_user is not None:
            resource_id = str(case_id)
            record_immutable_audit_event(
                event_type="case.viewed",
                action="viewed",
                actor_user_id=int(current_user.user_id),
                resource_type="case",
                resource_id=resource_id,
                outcome="success",
                metadata={
                    "route": "/api/v1/cases/{case_id}",
                },
            )
        return result

    return wrapper


def get_owned_case(case_id: str, current_user: CurrentUser, db: Session) -> Case:
    try:
        case_id_int = int(case_id)
        user_id_int = int(current_user.user_id)
    except (ValueError, TypeError):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid case ID format")

    case = db.query(Case).filter(Case.id == case_id_int).first()
    if not case:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Case not found")

    # Ownership mismatches intentionally return 404 to avoid leaking whether a
    # case exists to callers who do not own it.
    if current_user.role != "admin" and case.user_id != user_id_int:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Case not found")

    return case


@router.post(
    "/search",
    response_model=CaseSearchResponse,
    summary="Search for similar cases"
)
async def search_cases(
    request: CaseSearchRequest,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db_rls)
) -> CaseSearchResponse:
    """
    Search for similar cases in database
    
    - **case_number**: Case number to search for
    - **keywords**: Keywords to search
    - **jurisdiction**: Jurisdiction (US, UK, etc.)
    - **case_type**: Type of case (civil, criminal, etc.)
    - **year_from**: Start year filter
    - **year_to**: End year filter
    - **limit**: Max results (1-100)
    - **offset**: Pagination offset
    
    Returns paginated list of matching cases
    """
    
    logger.info(
        "Searching cases",
        user_id=current_user.user_id,
        keywords=request.keywords,
        jurisdiction=request.jurisdiction
    )
    
    from time import perf_counter

    start = perf_counter()

    # Similarity constraints/knobs
    min_similarity = request.relevance_threshold
    candidate_limit = 1000  # keeps the response time low

    reference_case = None

    query_signature = request.query_signature or _build_query_signature(request)

    # Build candidate query from filters (cheap DB-side filtering)
    query = db.query(CaseRecord)
    if request.case_type and request.case_type != "general":
        query = query.filter(CaseRecord.case_type == request.case_type)
    if request.jurisdiction:
        query = query.filter(CaseRecord.jurisdiction == request.jurisdiction)
    if request.court_name:
        query = query.filter(CaseRecord.court_name == request.court_name)
    if request.judge_name:
        query = query.filter(CaseRecord.judge_name == request.judge_name)
    if request.plaintiff_type:
        query = query.filter(CaseRecord.plaintiff_type == request.plaintiff_type)
    if request.defendant_type:
        query = query.filter(CaseRecord.defendant_type == request.defendant_type)

    # Restrict time window if requested
    if request.year_from is not None:
        query = query.filter(CaseRecord.created_at >= datetime(request.year_from, 1, 1))
    if request.year_to is not None:
        query = query.filter(CaseRecord.created_at <= datetime(request.year_to, 12, 31, 23, 59, 59))

    # Keep result set small for <2s performance
    candidates = query.order_by(CaseRecord.created_at.desc()).limit(candidate_limit).all()

    # If we cannot get a real reference_case, we use the first candidate as proxy when possible.
    # This still returns meaningful “similar cases” under the attribute-only scoring.
    if candidates:
        reference_case = candidates[0]

    if not reference_case:
        return CaseSearchResponse(
            total_results=0,
            results=[],
            search_time_seconds=round(perf_counter() - start, 4),
        )

    # Pre-fetch all SimilarityFeedback rows for candidate IDs in a single query to avoid N+1 queries.
    candidate_ids = [c.id for c in candidates if c.id != reference_case.id]
    prefetched_feedbacks = []
    if candidate_ids:
        fb_query = db.query(SimilarityFeedback).filter(
            SimilarityFeedback.candidate_case_id.in_(candidate_ids)
        )
        if current_user.user_id is not None:
            fb_query = fb_query.filter(SimilarityFeedback.user_id == str(current_user.user_id))
        if query_signature is not None:
            fb_query = fb_query.filter(SimilarityFeedback.query_signature == query_signature)
        prefetched_feedbacks = fb_query.all()

    feedback_by_case_id = {}
    for fb in prefetched_feedbacks:
        feedback_by_case_id.setdefault(fb.candidate_case_id, []).append(fb)

    # Score candidates and apply threshold
    scored = []
    for c in candidates:
        if c.id == reference_case.id:
            continue
        raw = CaseSimilarityCalculator.case_similarity_score(reference_case, c)
        # raw is 0..100. normalize to 0..1
        score01 = raw / 100.0
      
        try:
            created_at = c.created_at
            if created_at and created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
            recency_days = (datetime.now(timezone.utc) - created_at).days if created_at else 0
            recency_boost = max(0.0, 0.05 - recency_days * 0.0002)  # up to +0.05
        except Exception:
            recency_boost = 0.0
        feedback_boost = CaseSimilarityCalculator.get_feedback_adjustment(
            db,
            c,
            user_id=current_user.user_id,
            query_signature=query_signature,
            prefetched_feedback=feedback_by_case_id.get(c.id),
        )
        score01 = min(1.0, score01 + recency_boost + feedback_boost)

        if score01 > min_similarity:
            scored.append((c, score01))

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[: request.limit]

    # Fetch appeal analytics for the returned set
    result_ids = [c.id for c, _ in top]
    outcome_rows = []
    if result_ids:
        outcome_rows = (
            db.query(CaseOutcome)
            .filter(CaseOutcome.case_id.in_(result_ids))
            .all()
        )

    outcome_map = {row.case_id: row for row in outcome_rows}
    appealed_cases = sum(1 for row in outcome_rows if row.appeal_filed)
    appeal_successful_cases = sum(1 for row in outcome_rows if row.appeal_filed and row.appeal_success)
    appeal_success_rate = (
        round(appeal_successful_cases / appealed_cases, 4) if appealed_cases > 0 else None
    )

    results = []
    for c, score in top:
        verdict = c.outcome
        # We don't have a stored case_number/title on CaseRecord for analytics in current schema.
        # Use placeholders derived from available fields.
        case_number = c.hashed_case_id
        title = c.judge_name or "Precedent"
        outcome = outcome_map.get(c.id)
        case_appeal_success_rate = None
        if outcome and outcome.appeal_filed:
            case_appeal_success_rate = 1.0 if outcome.appeal_success else 0.0

        results.append(
            CaseResult(
                case_id=str(c.id),
                case_number=case_number,
                title=title,
                year=c.created_at.year if c.created_at else 0,
                jurisdiction=c.jurisdiction,
                case_type=c.case_type,
                summary=c.judgment_summary or "",
                verdict=verdict,
                relevance_score=round(float(score), 4),
                appeal_success_rate=case_appeal_success_rate,
                url=None,
            )
        )

    total_results = len(scored)
    return CaseSearchResponse(
        total_results=total_results,
        results=results,
        search_time_seconds=round(perf_counter() - start, 4),
        appeal_success_rate=appeal_success_rate,
        appealed_cases=appealed_cases,
        appeal_successful_cases=appeal_successful_cases,
    )


@router.post(
    "/similarity-feedback",
    response_model=SimilarityFeedbackResponse,
    summary="Save similarity feedback"
)
async def submit_similarity_result_feedback(
    request: SimilarityFeedbackRequest,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db_rls)
) -> SimilarityFeedbackResponse:
    """Persist user feedback for a similarity search result."""
    query_signature = request.query_signature or ""
    feedback = submit_similarity_feedback(
        db,
        user_id=current_user.user_id,
        candidate_case_id=request.candidate_case_id,
        query_signature=query_signature,
        relevance=request.relevance,
    )
    return SimilarityFeedbackResponse(
        success=True,
        saved_at=feedback.created_at,
        feedback_id=feedback.id,
    )


def _build_query_signature(request: CaseSearchRequest) -> str:
    """Derive a stable signature for the current similarity search filters."""
    parts = [
        f"jurisdiction={request.jurisdiction}",
        f"case_type={request.case_type}",
        f"court_name={request.court_name or ''}",
        f"judge_name={request.judge_name or ''}",
        f"plaintiff_type={request.plaintiff_type or ''}",
        f"defendant_type={request.defendant_type or ''}",
        f"year_from={request.year_from or ''}",
        f"year_to={request.year_to or ''}",
    ]
    return "|".join(parts)


def _timeline_event_to_api_event(event) -> CaseEvent:
    metadata = event.event_metadata if isinstance(event.event_metadata, dict) else {}
    documents = metadata.get("documents") if isinstance(metadata.get("documents"), list) else []
    return CaseEvent(
        date=event.event_date,
        event_type=event.event_type,
        description=event.description,
        court=metadata.get("court"),
        judge=metadata.get("judge"),
        location=metadata.get("location"),
        documents=documents,
    )


def _build_case_timeline_payload(case: Case, timeline_events) -> dict:
    api_events = [_timeline_event_to_api_event(event) for event in timeline_events]
    event_dates = [event.date for event in api_events]
    if len(event_dates) >= 2:
        duration_years = round((max(event_dates) - min(event_dates)).days / 365.25, 1)
    else:
        duration_years = 0.0

    return {
        "case_id": str(case.id),
        "case_number": case.case_number,
        "title": case.title or case.case_number,
        "status": case.status.value if hasattr(case.status, "value") else str(case.status),
        "created_at": case.created_at,
        "updated_at": case.updated_at or case.created_at,
        "events": api_events,
        "total_events": len(api_events),
        "duration_years": duration_years,
    }



@router.get(
    "/{case_id}/timeline",
    response_model=CaseTimeline,
    summary="Get case timeline"
)
async def get_case_timeline(
    case_id: str,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db_rls),
) -> CaseTimeline:
    """Get case history and timeline."""
    logger.info(
        "Retrieving case timeline",
        case_id=case_id,
        user_id=current_user.user_id,
    )

    case = get_owned_case(case_id, current_user, db)

    timeline_events = _timeline_service.get_case_timeline(db, case.id)
    return CaseTimeline.model_validate(_build_case_timeline_payload(case, timeline_events))


@router.get(
    "/{case_id}/timeline/export",
    summary="Export case timeline events in CSV or JSON format"
)
async def export_case_timeline(
    case_id: str,
    format: str = Query(..., pattern="^(csv|json)$", description="Export format: csv or json"),
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db_rls),
):
    """
    Export case timeline events in a machine-readable format (CSV or JSON).
    Ensures authorization is enforced and data is PII-free.
    """
    logger.info(
        "Exporting case timeline",
        case_id=case_id,
        format=format,
        user_id=current_user.user_id,
    )

    case = get_owned_case(case_id, current_user, db)

    timeline_events = _timeline_service.get_case_timeline(db, case.id)
    payload = _build_case_timeline_payload(case, timeline_events)

    redacted_events = []
    for event in payload["events"]:
        redacted_event = CaseEvent(
            date=event.date,
            event_type=event.event_type,
            description=redact_pii(event.description) or "",
            court=redact_pii(event.court),
            judge=redact_pii(event.judge),
            location=redact_pii(event.location),
            documents=[redact_pii(doc) or "" for doc in event.documents] if event.documents else [],
        )
        redacted_events.append(redacted_event)

    if format == "csv":
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["Date", "Event Type", "Description", "Court", "Judge", "Location", "Documents"])
        for event in redacted_events:
            date_str = event.date.isoformat() if hasattr(event.date, "isoformat") else str(event.date)
            docs_str = "; ".join(event.documents) if event.documents else ""
            writer.writerow([
                date_str,
                event.event_type,
                event.description,
                event.court or "",
                event.judge or "",
                event.location or "",
                docs_str
            ])
        csv_bytes = buffer.getvalue().encode("utf-8")
        headers = {
            "Content-Disposition": f'attachment; filename="case_{case.id}_timeline.csv"'
        }
        return Response(content=csv_bytes, media_type="text/csv", headers=headers)

    else:
        # JSON format
        redacted_payload = CaseTimeline(
            case_id=str(case.id),
            case_number=redact_pii(case.case_number) or "",
            title=redact_pii(case.title or case.case_number) or "",
            status=case.status.value if hasattr(case.status, "value") else str(case.status),
            created_at=case.created_at,
            updated_at=case.updated_at or case.created_at,
            events=redacted_events,
            total_events=len(redacted_events),
            duration_years=payload["duration_years"],
        )
        return redacted_payload


@router.get(
    "/{case_id}",
    summary="Get case details"
)
@_audit_case_view_route
async def get_case_details(
    case_id: str,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db_rls)
) -> dict:
    """Get complete case details"""
    
    logger.info(
        "Retrieving case details",
        case_id=case_id,
        user_id=current_user.user_id
    )
    case = get_owned_case(case_id, current_user, db)
    latest_docs = fetch_latest_documents_per_case(db, [case.id])

    return _build_case_summary_payload(case, latest_docs.get(case.id))


@router.post(
    "/{case_id}/share-anonymized",
    response_model=AnonymizedShareResponse,
    summary="Create a shareable anonymized case link",
)
async def create_anonymized_case_share(
    case_id: str,
    request: AnonymizedShareCreateRequest,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> AnonymizedShareResponse:
    case = get_owned_case(case_id, current_user, db)
    scope = normalize_privacy_profile(request.scope)
    from case_manager import generate_anonymized_case_data

    anonymized_data = generate_anonymized_case_data(case.id, profile_name=scope)
    if anonymized_data is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Anonymized case not found")

    anonymized_id = str(anonymized_data.get("anonymized_id") or "")
    if not anonymized_id:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to build anonymized case identifier")

    share = AnonymizedShareToken(
        token=_generate_unique_share_token(db),
        case_id=case.id,
        anonymized_id=anonymized_id,
        created_by=int(current_user.user_id),
        created_at=_utcnow(),
        expires_at=_utcnow() + timedelta(hours=request.expires_in_hours),
        scope=scope,
        used_at=None,
    )
    db.add(share)
    db.commit()
    db.refresh(share)

    record_immutable_audit_event(
        event_type="case.share_token_created",
        action="created",
        actor_user_id=int(current_user.user_id),
        resource_type="case",
        resource_id=str(case.id),
        outcome="success",
        metadata={
            "scope": scope,
            "anonymized_id": anonymized_id,
            "expires_at": share.expires_at.isoformat(),
        },
    )

    return AnonymizedShareResponse(
        token=share.token,
        anonymized_id=share.anonymized_id,
        scope=share.scope,
        share_url=_build_share_url(share.token),
        expires_at=share.expires_at,
    )


@router.get(
    "/share/{token}",
    summary="Resolve an anonymized share token",
)
async def resolve_anonymized_case_share(
    token: str,
    format: str = Query(default="json", pattern="^(json|pdf)$"),
    db: Session = Depends(get_db),
):
    share = _get_share_token_or_404(db, token)
    case = db.query(Case).filter(Case.id == share.case_id).first()
    if not case:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Case not found")

    payload = _build_anonymized_share_payload(case, share)

    if share.used_at is None:
        share.used_at = _utcnow()
        db.add(share)
        db.commit()

    record_immutable_audit_event(
        event_type="case.share_token_viewed",
        action="viewed",
        actor_user_id=share.created_by,
        resource_type="case",
        resource_id=str(case.id),
        outcome="success",
        metadata={
            "scope": share.scope,
            "anonymized_id": share.anonymized_id,
            "format": format,
        },
    )

    if format == "pdf":
        pdf_bytes = _build_anonymized_share_pdf(case, share, payload)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'inline; filename="anonymized_case_{share.anonymized_id}.pdf"'},
        )

    return JSONResponse(content=jsonable_encoder(payload))


@router.post(
    "/{case_id}/documents/upload",
    summary="Upload a PDF or image document to a case",
)
async def upload_case_document_endpoint(
    case_id: str,
    http_request: Request,
    file: UploadFile = File(...),
    document_type: str = Form(default="Other"),
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db_rls),
) -> dict:
    """Store an upload, create linked attachment/document rows, and queue OCR extraction."""
    case = get_owned_case(case_id, current_user, db)
    case_id_int = case.id
    user_id_int = int(current_user.user_id)

    allowed_extensions = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
    allowed_mime_types = {
        "application/pdf",
        "image/png",
        "image/jpeg",
        "image/jpg",
        "image/tiff",
        "image/bmp",
        "image/webp",
    }

    validate_file_upload(
        file,
        max_size=ValidationConfig.MAX_UPLOAD_SIZE,
        allowed_extensions=allowed_extensions,
        allowed_mime_types=allowed_mime_types,
    )
    await validate_file_upload_streaming(file, max_size=ValidationConfig.MAX_UPLOAD_SIZE)
    file_bytes = await file.read()

    from case_manager import upload_case_document_file

    safe_filename = Path(file.filename).name
    stored = upload_case_document_file(
        user_id=user_id_int,
        case_id=case_id_int,
        file_bytes=file_bytes,
        filename=safe_filename,
        document_type=getattr(DocumentType, document_type.upper(), DocumentType.OTHER),
        content_type=file.content_type,
    )
    if not stored:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Failed to store uploaded file")

    task = enqueue_task_from_http_request(
        process_case_document_upload_task,
        http_request,
        context_user_id=current_user.user_id,
        user_id=str(current_user.user_id),
        case_id=str(case_id_int),
        attachment_id=str(stored["attachment"]["id"]),
        document_id=str(stored["document"]["id"]),
        original_filename=safe_filename,
    )

    return {
        "status": "queued",
        "task_id": task.id,
        "case_id": case_id_int,
        "attachment_id": stored["attachment"]["id"],
        "document_id": stored["document"]["id"],
        "document_type": stored["document"]["document_type"],
        "filename": safe_filename,
    }


@router.post(
    "/{case_id}/notes/draft",
    summary="Save case note draft",
)
async def save_case_note_draft_endpoint(
    case_id: str,
    request: CaseNoteDraftRequest,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db_rls),
) -> dict:
    case = get_owned_case(case_id, current_user, db)
    case_id_int = case.id
    user_id_int = int(current_user.user_id)

    note = save_case_note_draft(
        db,
        case_id=case_id_int,
        user_id=user_id_int,
        note_text=request.note_text,
        changed_by_email=current_user.email,
    )
    return {
        "case_id": str(case_id_int),
        "note_id": note.id,
        "draft_text": note.draft_text,
        "draft_updated_at": note.draft_updated_at,
        "published_at": note.published_at,
    }


@router.post(
    "/{case_id}/notes/publish",
    summary="Publish case note",
)
async def publish_case_note_endpoint(
    case_id: str,
    request: CaseNotePublishRequest,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db_rls),
) -> dict:
    case = get_owned_case(case_id, current_user, db)
    case_id_int = case.id
    user_id_int = int(current_user.user_id)

    version = publish_case_note(
        db,
        case_id=case_id_int,
        user_id=user_id_int,
        note_text=request.note_text,
        changed_by_email=current_user.email,
    )
    return {
        "case_id": str(case_id_int),
        "version_number": version.version_number,
        "note_text": version.note_text,
        "changed_by_user_id": str(version.changed_by_user_id),
        "changed_by_email": version.changed_by_email,
        "created_at": version.created_at,
        "version_metadata": version.version_metadata,
    }


@router.get(
    "/{case_id}/notes/history",
    response_model=CaseNoteHistoryResponse,
    summary="Get case note history",
)
async def get_case_note_history_endpoint(
    case_id: str,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db_rls),
) -> CaseNoteHistoryResponse:
    case = get_owned_case(case_id, current_user, db)
    case_id_int = case.id
    user_id_int = int(current_user.user_id)

    versions = get_case_note_history(db, case_id_int, user_id_int)
    return CaseNoteHistoryResponse(
        case_id=str(case_id_int),
        case_number=case.case_number,
        title=case.title or case.case_number,
        total_versions=len(versions),
        versions=[
            CaseNoteVersionItem(
                version_number=version.version_number,
                note_text=version.note_text,
                change_type=version.change_type,
                changed_by_user_id=str(version.changed_by_user_id),
                changed_by_email=version.changed_by_email,
                created_at=version.created_at,
                version_metadata={k: v for k, v in (version.version_metadata or {}).items() if k in {"published_from_draft", "source"}},
            )
            for version in versions
        ],
    )


@router.get(
    "/",
    summary="List user's cases"
)
async def list_cases(
    limit: int = 10,
    offset: int = 0,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db_rls)
) -> dict:
    """Get list of cases for current user"""
    
    limit = min(limit, 100)
    offset = max(offset, 0)
    
    logger.info(
        "Listing user cases",
        user_id=current_user.user_id,
        limit=limit,
        offset=offset
    )
    
    try:
        user_id_int = int(current_user.user_id)
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid user ID format"
        )
    
    total = db.query(Case).filter(Case.user_id == user_id_int).count()
    
    cases = (
        db.query(Case)
        .filter(Case.user_id == user_id_int)
        .order_by(Case.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    
    latest_docs = fetch_latest_documents_per_case(db, [c.id for c in cases])

    cases_list = []
    for c in cases:
        cases_list.append(_build_case_summary_payload(c, latest_docs.get(c.id)))
        
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "cases": cases_list
    }

