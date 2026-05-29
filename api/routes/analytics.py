"""
Analytics Endpoints
GET /api/v1/analytics/costs - User cost breakdown
GET /api/v1/analytics/overview - User analytics overview
GET /api/v1/analytics/dashboard - Dashboard summary for the Streamlit frontend
"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func

from api.models import CostBreakdown, AnalyticsResponse, DashboardSummaryResponse
from api.auth import get_current_user, CurrentUser
import structlog
from datetime import datetime, timezone, timedelta

from analytics_engine import AnalyticsAggregator
from api.dependencies import get_db_rls
from db.models.cases import Case, CaseDocument, CaseDeadline, CaseStatus
from db.models.reports import Report


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
    
    uid = int(current_user.user_id)
    active_cases = db.query(func.count(Case.id)).filter(
        Case.user_id == uid, Case.status == CaseStatus.ACTIVE
    ).scalar() or 0
    
    pending_deadlines = db.query(func.count(CaseDeadline.id)).filter(
        CaseDeadline.user_id == uid, CaseDeadline.is_completed == False
    ).scalar() or 0
    
    doc_count = db.query(func.count(CaseDocument.id)).join(
        Case, CaseDocument.case_id == Case.id
    ).filter(Case.user_id == uid).scalar() or 0
    
    reports = db.query(func.count(Report.id)).filter(
        Report.user_id == uid
    ).scalar() or 0
    
    case_types = db.query(
        Case.case_type, func.count(Case.id).label("cnt")
    ).filter(Case.user_id == uid).group_by(Case.case_type).order_by(
        func.count(Case.id).desc()
    ).limit(5).all()
    
    cost_breakdown = CostBreakdown(
        period=period,
        total_cost=0.0,
        llm_api_cost=0.0,
        document_processing_cost=0.0,
        storage_cost=0.0,
        api_calls=0,
        documents_analyzed=doc_count,
        reports_generated=reports,
    )
    
    return AnalyticsResponse(
        user_id=current_user.user_id,
        cost_breakdown=cost_breakdown,
        active_cases=active_cases,
        pending_deadlines=pending_deadlines,
        successful_analyses=doc_count,
        failed_analyses=0,
        average_analysis_time_seconds=0.0,
        top_case_types=[(t, c) for t, c in case_types],
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
        CaseDocument.uploaded_at >= now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    ).scalar() or 0
    
    last_30_docs = db.query(func.count(CaseDocument.id)).join(
        Case, CaseDocument.case_id == Case.id
    ).filter(
        Case.user_id == uid,
        CaseDocument.uploaded_at >= now - timedelta(days=30)
    ).scalar() or 0
    
    this_month_reports = db.query(func.count(Report.id)).filter(
        Report.user_id == uid,
        Report.created_at >= now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    ).scalar() or 0
    
    last_30_reports = db.query(func.count(Report.id)).filter(
        Report.user_id == uid,
        Report.created_at >= now - timedelta(days=30)
    ).scalar() or 0
    
    return {
        "user_id": current_user.user_id,
        "active_cases": active_cases,
        "pending_deadlines": pending_deadlines,
        "this_month": {
            "api_calls": 0,
            "documents_analyzed": this_month_docs,
            "reports_generated": this_month_reports,
            "cost": 0.0
        },
        "last_30_days": {
            "api_calls": 0,
            "documents_analyzed": last_30_docs,
            "reports_generated": last_30_reports,
            "cost": 0.0
        },
        "top_features": [
            {"feature": "document_analysis", "usage": last_30_docs},
            {"feature": "report_generation", "usage": last_30_reports},
        ],
        "generated_at": now.isoformat()
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
    
    return {
        "user_id": current_user.user_id,
        "period_days": days,
        "total_requests": docs + reports,
        "daily_average": round((docs + reports) / max(days, 1), 1),
        "peak_day": 0,
        "peak_hour": 0,
        "endpoints": {
            "POST /analyze/document": docs,
            "POST /reports/generate": reports,
        },
        "generated_at": datetime.now(timezone.utc).isoformat()
    }



