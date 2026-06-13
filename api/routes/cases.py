"""
Case Search Endpoints
POST /api/v1/cases/search - Search for similar cases
POST /api/v1/cases/similarity-feedback - Save similarity feedback
GET /api/v1/cases/{id}/timeline - Get case timeline
"""
from datetime import datetime, timedelta, timezone
from typing import Dict

from fastapi import APIRouter, HTTPException, status, Depends, UploadFile, File, Form, Request, Query
from sqlalchemy.orm import Session
from fastapi import APIRouter, HTTPException, status, Depends
from api.models import (
    CaseSearchRequest, CaseSearchResponse, CaseResult,
    CaseTimeline, CaseEvent, SimilarityFeedbackRequest,
    SimilarityFeedbackResponse,
)
from api.auth import get_current_user, CurrentUser
import structlog
from sqlalchemy import func
from sqlalchemy.orm import Session

from database import (
    CaseRecord,
    CaseOutcome,
    Case,
    get_db,
    submit_similarity_feedback,
    SimilarityFeedback,
)
from analytics_engine import CaseSimilarityCalculator

router = APIRouter(prefix="/api/v1/cases", tags=["cases"])
logger = structlog.get_logger(__name__)


def _build_case_summary_payload(case: Case, latest_doc: CaseDocument | None = None) -> dict:
    return {
        "case_id": str(case.id),
        "case_number": case.case_number,
        "title": case.title or case.case_number,
        "parties": ["Smith", "Jones"],  # Placeholder
        "jurisdiction": case.jurisdiction,
        "status": case.status.value if hasattr(case.status, 'value') else str(case.status),
        "summary": latest_doc.summary if latest_doc else "",
    }


def _record_case_view_audit(case_id: str, current_user: CurrentUser) -> None:
    """Record an immutable audit event for a case view.

    Called directly inside the route handler so that ``case_id`` and
    ``current_user`` are always the real, dependency-resolved values rather
    than relying on ``**kwargs`` inspection, which is unreliable under
    FastAPI's dependency-injection call convention.
    """
    try:
        record_immutable_audit_event(
            event_type="case.viewed",
            action="viewed",
            actor_user_id=int(current_user.user_id),
            resource_type="case",
            resource_id=str(case_id),
            outcome="success",
            metadata={"route": "/api/v1/cases/{case_id}"},
        )
    except Exception:
        logger.exception(
            "audit_event_failed",
            event_type="case.viewed",
            case_id=case_id,
            user_id=current_user.user_id,
        )


def get_owned_case(case_id: str, current_user: CurrentUser, db: Session) -> Case:
    try:
        case_id_int = int(case_id)
        user_id_int = int(current_user.user_id)
    except (ValueError, TypeError):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid case ID format")

    case = db.query(Case).filter(Case.id == case_id_int).first()
    if not case:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Case not found")

    if current_user.role != "admin" and case.user_id != user_id_int:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden: You do not own this case")

    return case


@router.post(
    "/search",
    response_model=CaseSearchResponse,
    summary="Search for similar cases"
)
async def search_cases(
    request: CaseSearchRequest,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> CaseSearchResponse:
    """
    Search for similar cases in database.

    FastAPI's Depends(get_db) manages the session lifecycle, guaranteeing
    the connection is always returned to the pool regardless of the exit
    path (normal return, early return, or unhandled exception).

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
        jurisdiction=request.jurisdiction,
    )

    from time import perf_counter

    start = perf_counter()

    min_similarity = request.relevance_threshold
    candidate_limit = 1000

    reference_case = None
    candidates = []
    outcome_rows = []
    feedback_rows = []
    db = None
    try:
        db = get_db()
        query_signature = request.query_signature or _build_query_signature(request)

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

        if request.year_from is not None:
            query = query.filter(CaseRecord.created_at >= datetime(request.year_from, 1, 1))
        if request.year_to is not None:
            query = query.filter(CaseRecord.created_at <= datetime(request.year_to, 12, 31, 23, 59, 59))

        candidates = query.order_by(CaseRecord.created_at.desc()).limit(candidate_limit).all()

        reference_case = candidates[0] if candidates else None
        if not reference_case:
            return CaseSearchResponse(
                total_results=0,
                results=[],
                search_time_seconds=round(perf_counter() - start, 4),
            )

        candidate_ids = [c.id for c in candidates if c.id != reference_case.id]

        feedback_rows = []
        if candidate_ids:
            feedback_rows = (
                db.query(SimilarityFeedback)
                .filter(
                    SimilarityFeedback.candidate_case_id.in_(candidate_ids),
                    SimilarityFeedback.user_id == str(current_user.user_id),
                )
                .all()
            )

        outcome_rows = []
        if candidate_ids:
            outcome_rows = (
                db.query(CaseOutcome)
                .filter(CaseOutcome.case_id.in_(candidate_ids))
                .all()
            )

    finally:
        if db is not None:
            db.close()

    outcome_map = {row.case_id: row for row in outcome_rows}
    appealed_cases = sum(1 for row in outcome_rows if row.appeal_filed)
    appeal_successful_cases = sum(1 for row in outcome_rows if row.appeal_filed and row.appeal_success)
    appeal_success_rate = (
        round(appeal_successful_cases / appealed_cases, 4) if appealed_cases > 0 else None
    )

    feedback_adjustments: Dict[int, float] = {}
    for f in feedback_rows:
        current = feedback_adjustments.get(f.candidate_case_id, 0.0)
        adjustment = 0.03 if f.relevance else -0.03
        feedback_adjustments[f.candidate_case_id] = current + adjustment
    for cid in feedback_adjustments:
        feedback_adjustments[cid] = max(-0.03, min(0.03, feedback_adjustments[cid]))

    scored = []
    for c in candidates:
        if c.id == reference_case.id:
            continue
        raw = CaseSimilarityCalculator.case_similarity_score(reference_case, c)
        score01 = raw / 100.0
        try:
            created_at = c.created_at
            if created_at and created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
            recency_days = (datetime.now(timezone.utc) - created_at).days if created_at else 0
            recency_boost = max(0.0, 0.05 - recency_days * 0.0002)
        except Exception:
            recency_boost = 0.0
        feedback_boost = feedback_adjustments.get(c.id, 0.0)
        score01 = min(1.0, score01 + recency_boost + feedback_boost)

        if score01 > min_similarity:
            scored.append((c, score01))

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[: request.limit]

    results = []
    for c, score in top:
        verdict = c.outcome
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
    db: Session = Depends(get_db),
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



@router.get(
    "/{case_id}/timeline",
    response_model=CaseTimeline,
    summary="Get case timeline"
)
async def get_case_timeline(
    case_id: str,
    current_user: CurrentUser = Depends(get_current_user)
) -> CaseTimeline:
    """Get case history and timeline"""
    
    logger.info(
        "Retrieving case timeline",
        case_id=case_id,
        user_id=current_user.user_id
    )
    
    # Mock timeline data
    base_date = datetime.utcnow() - timedelta(days=365)
    events = [
        CaseEvent(
            date=base_date,
            event_type="filing",
            description="Case filed",
            court="District Court",
            location="New York, NY",
            documents=["complaint.pdf"]
        ),
        CaseEvent(
            date=base_date + timedelta(days=30),
            event_type="hearing",
            description="Initial hearing",
            court="District Court",
            judge="Judge Smith",
            location="New York, NY"
        ),
        CaseEvent(
            date=base_date + timedelta(days=90),
            event_type="discovery",
            description="Discovery period",
            court="District Court",
            location="New York, NY"
        ),
        CaseEvent(
            date=base_date + timedelta(days=180),
            event_type="hearing",
            description="Motion hearing",
            court="District Court",
            judge="Judge Smith",
            location="New York, NY"
        ),
        CaseEvent(
            date=base_date + timedelta(days=365),
            event_type="decision",
            description="Court decision rendered",
            court="District Court",
            judge="Judge Smith",
            location="New York, NY",
            documents=["decision.pdf"]
        ),
    ]
    
    return CaseTimeline(
        case_id=case_id,
        case_number="2023-CV-00001",
        title="Example Case",
        status="closed",
        created_at=base_date,
        updated_at=datetime.utcnow(),
        events=events,
        total_events=len(events),
        duration_years=1.0
    )


@router.get(
    "/{case_id}",
    summary="Get case details"
)
async def get_case_details(
    case_id: str,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> dict:
    """Get complete case details"""
    
    logger.info(
        "Retrieving case details",
        case_id=case_id,
        user_id=current_user.user_id
    )
    case = get_owned_case(case_id, current_user, db)
    latest_docs = fetch_latest_documents_per_case(db, [case.id])
    result = _build_case_summary_payload(case, latest_docs.get(case.id))
    _record_case_view_audit(case_id, current_user)
    return result


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
    db: Session = Depends(get_db),
) -> dict:
    """Get complete case details from database."""
    try:
        case_id_int = int(case_id)
    except (ValueError, TypeError):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid case ID")

    db = get_db()
    try:
        case = db.query(Case).filter(
            Case.id == case_id_int,
            Case.user_id == int(current_user.user_id),
        ).first()
        if not case:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Case not found")

    latest_doc = None
    if case.documents:
        latest_doc = sorted(case.documents, key=lambda d: d.uploaded_at, reverse=True)[0]

    return {
        "case_id": str(case.id),
        "case_number": case.case_number,
        "title": case.title or case.case_number,
        "parties": [],
        "jurisdiction": case.jurisdiction,
        "status": case.status.value if hasattr(case.status, 'value') else str(case.status),
        "summary": latest_doc.summary if latest_doc else "",
    }


@router.get(
    "",
    summary="List user's cases"
)
async def list_cases(
    limit: int = Query(default=10, ge=1, le=100, description="Maximum number of cases to return (1–100)"),
    offset: int = Query(default=0, ge=0, description="Number of cases to skip"),
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    """Get list of cases for the current user."""
    try:
        user_id_int = int(current_user.user_id)
    except (ValueError, TypeError):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid user ID")

    total = db.query(Case).filter(Case.user_id == user_id_int).count()
    cases = (
        db.query(Case)
        .filter(Case.user_id == user_id_int)
        .order_by(Case.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    cases_list = []
    for c in cases:
        latest_doc = None
        if c.documents:
            latest_doc = sorted(c.documents, key=lambda d: d.uploaded_at, reverse=True)[0]
        cases_list.append({
            "case_id": str(c.id),
            "case_number": c.case_number,
            "title": c.title or c.case_number,
            "parties": [],
            "jurisdiction": c.jurisdiction,
            "status": c.status.value if hasattr(c.status, 'value') else str(c.status),
            "summary": latest_doc.summary if latest_doc else "",
        })

    return {"total": total, "limit": limit, "offset": offset, "cases": cases_list}
