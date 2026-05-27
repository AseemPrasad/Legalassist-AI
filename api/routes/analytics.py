"""
Analytics Endpoints
GET /api/v1/analytics/costs - User cost breakdown
GET /api/v1/analytics/overview - User analytics overview
GET /api/v1/analytics/dashboard - Dashboard summary for the Streamlit frontend
"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from api.models import CostBreakdown, AnalyticsResponse, DashboardSummaryResponse
from api.auth import get_current_user, CurrentUser
import structlog
from datetime import datetime

from analytics_engine import AnalyticsAggregator
from api.dependencies import get_db_rls


router = APIRouter(prefix="/api/v1/analytics", tags=["analytics"])
logger = structlog.get_logger(__name__)


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
    
    Returns breakdown of API costs by service
    """
    
    logger.info(
        "Fetching cost breakdown",
        user_id=current_user.user_id,
        period=period
    )
    
    cost_breakdown = CostBreakdown(
        period=period,
        total_cost=125.50,
        llm_api_cost=75.00,
        document_processing_cost=35.50,
        storage_cost=15.00,
        api_calls=5432,
        documents_analyzed=87,
        reports_generated=12
    )
    
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
    current_user: CurrentUser = Depends(get_current_user)
) -> dict:
    """Get comprehensive analytics overview"""
    
    logger.info(
        "Fetching analytics overview",
        user_id=current_user.user_id
    )
    
    return {
        "user_id": current_user.user_id,
        "active_cases": 5,
        "pending_deadlines": 3,
        "this_month": {
            "api_calls": 1234,
            "documents_analyzed": 23,
            "reports_generated": 3,
            "cost": 45.67
        },
        "last_30_days": {
            "api_calls": 4567,
            "documents_analyzed": 89,
            "reports_generated": 12,
            "cost": 123.45
        },
        "top_features": [
            {"feature": "document_analysis", "usage": 45},
            {"feature": "case_search", "usage": 32},
            {"feature": "report_generation", "usage": 12}
        ],
        "generated_at": datetime.now(timezone.utc).isoformat()
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
    current_user: CurrentUser = Depends(get_current_user)
) -> dict:
    """Get API usage metrics for last N days"""
    
    return {
        "user_id": current_user.user_id,
        "period_days": days,
        "total_requests": 4567,
        "daily_average": 152,
        "peak_day": 234,
        "peak_hour": 18,
        "endpoints": {
            "POST /analyze/document": 1234,
            "POST /cases/search": 2345,
            "POST /reports/generate": 456,
            "GET /analytics/costs": 234,
            "GET /deadlines/upcoming": 298
        },
        "generated_at": datetime.now(timezone.utc).isoformat()
    }



