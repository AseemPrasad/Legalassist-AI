"""Model feedback and optimization endpoints"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from api.auth import get_current_user, CurrentUser
from api.dependencies import get_db_rls
from api.models import (
    ModelFeedbackRequest,
    ModelFeedbackResponse,
    ModelPerformanceResponse,
    ModelPerformanceItem,
)
from database import submit_model_feedback, aggregate_model_performance
from typing import List

router = APIRouter(prefix="/api/v1/models", tags=["models"])


@router.post("/feedback", response_model=ModelFeedbackResponse)
async def submit_feedback(
    request: ModelFeedbackRequest,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db_rls),
):
    fb = submit_model_feedback(
        db,
        user_id=current_user.user_id,
        model_name=request.model_name,
        task=request.task,
        case_id=request.case_id,
        is_accurate=request.is_accurate,
        corrected_text=request.corrected_text,
        feedback_notes=request.feedback_notes,
    )
    return ModelFeedbackResponse(success=True, feedback_id=fb.id, saved_at=fb.created_at)


@router.get("/performance", response_model=ModelPerformanceResponse)
async def get_performance(
    task: str = None,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db_rls),
):
    rows = aggregate_model_performance(db, task=task)
    items = [
        ModelPerformanceItem(
            model_name=r.model_name,
            task=r.task,
            case_type=r.case_type,
            jurisdiction=r.jurisdiction,
            samples=r.samples,
            accurate_count=r.accurate_count,
            accuracy=r.accuracy,
        )
        for r in rows
    ]
    return ModelPerformanceResponse(items=items)
