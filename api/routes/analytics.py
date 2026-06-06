"""
Analytics Endpoints
GET /api/v1/analytics/costs - User cost breakdown
GET /api/v1/analytics/overview - User analytics overview
"""
from fastapi import APIRouter, Depends
from api.models import CostBreakdown, AnalyticsResponse
from api.auth import get_current_user, CurrentUser
from database import ModelPerformance, CaseDocument, Attachment, Case, SessionLocal
from datetime import datetime, timezone
from sqlalchemy import func
import structlog

router = APIRouter(prefix="/api/v1/analytics", tags=["analytics"])
logger = structlog.get_logger(__name__)

# Known task categories for operation-specific cost calculations
_LLM_TASKS = frozenset({"summary", "remedy_extraction", "appeal_estimation", "report_generation", "analysis"})
_DOC_TASKS = frozenset({"document_classification", "document_extraction", "ocr", "text_extraction"})


def _get_per_task_average_costs(db) -> tuple[float, float]:
    """Query ModelPerformance for per-task average costs and return (llm_avg_cents, doc_avg_cents).

    Averages are weighted by samples to avoid skew from low-volume tasks.
    Falls back to the overall weighted average if a category has no matching tasks.
    """
    rows = db.query(
        ModelPerformance.task,
        func.sum(ModelPerformance.average_cost * ModelPerformance.samples).label("total_weighted_cost"),
        func.sum(ModelPerformance.samples).label("total_samples"),
    ).filter(
        ModelPerformance.samples > 0,
        ModelPerformance.average_cost.isnot(None),
    ).group_by(ModelPerformance.task).all()

    if not rows:
        return 0.0, 0.0

    llm_weighted = doc_weighted = 0.0
    llm_samples = doc_samples = 0
    fallback_weighted = fallback_samples = 0.0

    for row in rows:
        w = row.total_weighted_cost or 0
        s = row.total_samples or 0
        if row.task in _LLM_TASKS:
            llm_weighted += w
            llm_samples += s
        elif row.task in _DOC_TASKS:
            doc_weighted += w
            doc_samples += s
        else:
            fallback_weighted += w
            fallback_samples += s

    # If a category has no matching tasks, use fallback (all tasks) weighted average
    if llm_samples == 0 and fallback_samples > 0:
        llm_weighted = fallback_weighted
        llm_samples = fallback_samples
    if doc_samples == 0 and fallback_samples > 0:
        doc_weighted = fallback_weighted
        doc_samples = fallback_samples

    llm_avg_cents = llm_weighted / llm_samples if llm_samples > 0 else 0.0
    doc_avg_cents = doc_weighted / doc_samples if doc_samples > 0 else 0.0
    return llm_avg_cents, doc_avg_cents


def _get_user_doc_count(db, uid: int) -> int:
    """Count documents belonging to the given user through the Case relationship."""
    return db.query(func.count(CaseDocument.id)).select_from(Case).join(
        CaseDocument, Case.id == CaseDocument.case_id
    ).filter(Case.user_id == uid).scalar() or 0


def _get_user_storage_bytes(db, uid: int) -> int:
    """Sum of attachment sizes in bytes for the given user."""
    return db.query(func.coalesce(func.sum(Attachment.size_bytes), 0)).filter(
        Attachment.user_id == uid
    ).scalar() or 0


@router.get(
    "/costs",
    response_model=AnalyticsResponse,
    summary="Get user cost breakdown"
)
async def get_cost_breakdown(
    period: str = "monthly",
    current_user: CurrentUser = Depends(get_current_user)
) -> AnalyticsResponse:
    """
    Get cost breakdown for user API usage

    - **period**: monthly or all_time

    Returns breakdown of API costs by service, using per-task average costs
    from the ModelPerformance table rather than a single global average.
    """
    uid = int(current_user.user_id)

    logger.info("Fetching cost breakdown", user_id=uid, period=period)

    db = SessionLocal()
    try:
        llm_avg_cents, doc_avg_cents = _get_per_task_average_costs(db)

        doc_count = _get_user_doc_count(db, uid)
        storage_bytes = _get_user_storage_bytes(db, uid)

        # Storage cost: $0.023 per GB per month
        storage_gb = storage_bytes / (1024 ** 3)
        storage_cost = round(storage_gb * 0.023, 4)

        # Per-operation costs using category-specific averages (cents → dollars)
        llm_avg_usd = llm_avg_cents / 100.0
        doc_avg_usd = doc_avg_cents / 100.0

        document_processing_cost = round(doc_count * doc_avg_usd, 4)
        llm_api_cost = round(doc_count * llm_avg_usd, 4)

        total = round(llm_api_cost + document_processing_cost + storage_cost, 4)

        cost_breakdown = CostBreakdown(
            period=period,
            total_cost=total,
            llm_api_cost=llm_api_cost,
            document_processing_cost=document_processing_cost,
            storage_cost=storage_cost,
            api_calls=0,
            documents_analyzed=doc_count,
            reports_generated=0,
        )

        now = datetime.now(timezone.utc)

        return AnalyticsResponse(
            user_id=str(uid),
            cost_breakdown=cost_breakdown,
            active_cases=0,
            pending_deadlines=0,
            successful_analyses=doc_count,
            failed_analyses=0,
            average_analysis_time_seconds=0.0,
            top_case_types=[],
            generated_at=now,
        )
    finally:
        db.close()


@router.get(
    "/overview",
    summary="Get analytics overview"
)
async def get_analytics_overview(
    current_user: CurrentUser = Depends(get_current_user)
) -> dict:
    """Get comprehensive analytics overview using operation-specific cost metrics."""
    uid = int(current_user.user_id)

    logger.info("Fetching analytics overview", user_id=uid)

    db = SessionLocal()
    try:
        llm_avg_cents, doc_avg_cents = _get_per_task_average_costs(db)
        llm_avg_usd = llm_avg_cents / 100.0
        doc_avg_usd = doc_avg_cents / 100.0

        doc_count = _get_user_doc_count(db, uid)
        storage_bytes = _get_user_storage_bytes(db, uid)

        now = datetime.now(timezone.utc)
        llm_cost = round(doc_count * llm_avg_usd, 4)
        doc_cost = round(doc_count * doc_avg_usd, 4)
        storage_gb = storage_bytes / (1024 ** 3)
        storage_cost = round(storage_gb * 0.023, 4)

        return {
            "user_id": str(uid),
            "active_cases": 0,
            "pending_deadlines": 0,
            "this_month": {
                "api_calls": 0,
                "documents_analyzed": doc_count,
                "reports_generated": 0,
                "cost": round(llm_cost + doc_cost + storage_cost, 4),
            },
            "last_30_days": {
                "api_calls": 0,
                "documents_analyzed": doc_count,
                "reports_generated": 0,
                "cost": round(llm_cost + doc_cost + storage_cost, 4),
            },
            "top_features": [
                {"feature": "document_analysis", "usage": doc_count},
            ],
            "generated_at": now.isoformat(),
        }
    finally:
        db.close()


@router.get(
    "/usage",
    summary="Get API usage metrics"
)
async def get_usage_metrics(
    days: int = 30,
    current_user: CurrentUser = Depends(get_current_user)
) -> dict:
    """Get API usage metrics for last N days based on actual document activity."""
    uid = int(current_user.user_id)

    db = SessionLocal()
    try:
        doc_count = _get_user_doc_count(db, uid)

        return {
            "user_id": str(uid),
            "period_days": days,
            "total_requests": doc_count,
            "daily_average": round(doc_count / max(days, 1), 1),
            "peak_day": doc_count,
            "peak_hour": 0,
            "endpoints": {
                "POST /analyze/document": doc_count,
            },
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
    finally:
        db.close()
