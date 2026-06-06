"""
Analytics Endpoints
GET /api/v1/analytics/costs - User cost breakdown
GET /api/v1/analytics/overview - User analytics overview
GET /api/v1/analytics/usage - User API usage metrics
GET /api/v1/analytics/dashboard - Dashboard summary for the Streamlit frontend
"""
from collections import Counter

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func

from api.models import CostBreakdown, AnalyticsResponse, DashboardSummaryResponse
from api.auth import get_current_user, CurrentUser
import structlog
from datetime import datetime, timezone

router = APIRouter(prefix="/api/v1/analytics", tags=["analytics"])
logger = structlog.get_logger(__name__)


@router.get(
    "/costs",
    response_model=AnalyticsResponse,
    summary="Get user cost breakdown"
)
async def get_cost_breakdown(
    period: str = "monthly",
    db: Session = Depends(get_db_rls),
    current_user: CurrentUser = Depends(get_current_user)
) -> AnalyticsResponse:
    """
    Get cost breakdown for user API usage

    - **period**: monthly or all_time

    Returns breakdown of API costs by service
    """

    logger.info(
        "Fetching cost breakdown",
        user_id=current_user.user_id,
        period=period
    )
    
    # Ensure each cost category is mutually exclusive to prevent double counting.
    # general_llm_api_cost represents general (non-document-related) LLM calls.
    # doc_proc_cost represents document-related operations (which also use the LLM under the hood).
    # store_cost represents storage cost.
    # total_cost is computed as the sum of these non-overlapping components.
    general_llm_api_cost = 39.50
    doc_proc_cost = 35.50
    store_cost = 15.00
    total_calculated_cost = general_llm_api_cost + doc_proc_cost + store_cost

    cost_breakdown = CostBreakdown(
        period=period,
        total_cost=total_calculated_cost,
        llm_api_cost=general_llm_api_cost,
        document_processing_cost=doc_proc_cost,
        storage_cost=store_cost,
        api_calls=5432,
        documents_analyzed=87,
        reports_generated=12,
    )

    failed = db.query(func.count(CaseDocument.id)).join(
        Case, CaseDocument.case_id == Case.id
    ).filter(
        Case.user_id == uid,
        CaseDocument.summary.is_(None)
    ).scalar() or 0

    return AnalyticsResponse(
        user_id=current_user.user_id,
        cost_breakdown=cost_breakdown,
        active_cases=5,
        pending_deadlines=3,
        successful_analyses=87,
        failed_analyses=2,
        average_analysis_time_seconds=12.5,
        top_case_types=[("civil", 34), ("contract", 28), ("labor", 15)],
        generated_at=datetime.now(timezone.utc)
    )


@router.get(
    "/overview",
    summary="Get analytics overview"
)
async def get_analytics_overview(
    db: Session = Depends(get_db_rls),
    current_user: CurrentUser = Depends(get_current_user)
) -> dict:
    """Get comprehensive analytics overview"""

    logger.info(
        "Fetching analytics overview",
        user_id=current_user.user_id
    )

    uid = int(current_user.user_id)
    now = datetime.now(timezone.utc)
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    thirty_days_ago = now - timedelta(days=30)

    active_cases = db.query(func.count(Case.id)).filter(
        Case.user_id == uid, Case.status == CaseStatus.ACTIVE
    ).scalar() or 0

    pending_deadlines = db.query(func.count(CaseDeadline.id)).filter(
        CaseDeadline.user_id == uid, CaseDeadline.is_completed == False
    ).scalar() or 0

    this_month_docs = db.query(func.count(CaseDocument.id)).join(
        Case, CaseDocument.case_id == Case.id
    ).filter(
        Case.user_id == uid,
        CaseDocument.uploaded_at >= month_start
    ).scalar() or 0

    last_30_docs = db.query(func.count(CaseDocument.id)).join(
        Case, CaseDocument.case_id == Case.id
    ).filter(
        Case.user_id == uid,
        CaseDocument.uploaded_at >= thirty_days_ago
    ).scalar() or 0

    this_month_reports = db.query(func.count(Report.id)).filter(
        Report.user_id == uid,
        Report.created_at >= month_start
    ).scalar() or 0

    last_30_reports = db.query(func.count(Report.id)).filter(
        Report.user_id == uid,
        Report.created_at >= thirty_days_ago
    ).scalar() or 0

    this_month_calls = db.query(func.count(AuditEvent.id)).filter(
        AuditEvent.actor_user_id == uid,
        AuditEvent.occurred_at >= month_start
    ).scalar() or 0

    last_30_calls = db.query(func.count(AuditEvent.id)).filter(
        AuditEvent.actor_user_id == uid,
        AuditEvent.occurred_at >= thirty_days_ago
    ).scalar() or 0

    avg_model_cost = (db.query(func.avg(ModelPerformance.average_cost)).scalar() or 0) / 100.0

    return {
        "user_id": current_user.user_id,
        "active_cases": 5,
        "pending_deadlines": 3,
        "last_30_days": {
            "api_calls": 4567,
            "documents_analyzed": 89,
            "reports_generated": 12,
            "cost": 123.45,
        },
        "top_features": [
            {"feature": "document_analysis", "usage": 45},
            {"feature": "case_search", "usage": 32},
            {"feature": "report_generation", "usage": 12},
        ],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


@router.get(
    "/dashboard",
    response_model=DashboardSummaryResponse,
    summary="Get dashboard summary"
)
def get_dashboard_summary(
    db: Session = Depends(get_db_rls),
    current_user: CurrentUser = Depends(get_current_user),
) -> DashboardSummaryResponse:
    """Get the dashboard summary used by the Streamlit home analytics view."""

    summary = AnalyticsAggregator.get_dashboard_summary(db)
    return DashboardSummaryResponse(**summary)


@router.get(
    "/usage",
    summary="Get API usage metrics"
)
async def get_usage_metrics(
    days: int = 30,
    db: Session = Depends(get_db_rls),
    current_user: CurrentUser = Depends(get_current_user)
) -> dict:
    """Get API usage metrics for last N days"""

    uid = int(current_user.user_id)
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    docs = db.query(func.count(CaseDocument.id)).join(
        Case, CaseDocument.case_id == Case.id
    ).filter(
        Case.user_id == uid,
        CaseDocument.uploaded_at >= cutoff
    ).scalar() or 0

    reports = db.query(func.count(Report.id)).filter(
        Report.user_id == uid,
        Report.created_at >= cutoff
    ).scalar() or 0

    events = db.query(AuditEvent.occurred_at).filter(
        AuditEvent.actor_user_id == uid,
        AuditEvent.occurred_at >= cutoff
    ).all()

    day_counts = Counter(e.occurred_at.date() for e in events)
    hour_counts = Counter(e.occurred_at.hour for e in events)

    peak_day_entry = day_counts.most_common(1)
    peak_hour_entry = hour_counts.most_common(1)

    total_api_requests = len(events)

    return {
        "user_id": current_user.user_id,
        "period_days": days,
        "total_requests": total_api_requests,
        "daily_average": round(total_api_requests / max(days, 1), 1),
        "peak_day": str(peak_day_entry[0][0]) if peak_day_entry else 0,
        "peak_hour": peak_hour_entry[0][0] if peak_hour_entry else 0,
        "endpoints": {
            "POST /analyze/document": docs,
            "POST /reports/generate": reports,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }



