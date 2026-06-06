import datetime as dt
from typing import Optional, List
from sqlalchemy.orm import Session
from db.models import (
    UserFeedback,
)


def submit_user_feedback(
    db: Session,
    user_id: int,
    did_appeal: Optional[bool] = None,
    appeal_outcome: Optional[str] = None,
    appeal_cost: Optional[int] = None,
    time_to_verdict: Optional[int] = None,
    case_type: Optional[str] = None,
    jurisdiction: Optional[str] = None,
    satisfaction_rating: Optional[int] = None,
    feedback_text: Optional[str] = None,
) -> UserFeedback:
    feedback = UserFeedback(
        user_id=user_id,
        did_appeal=did_appeal,
        appeal_outcome=appeal_outcome,
        appeal_cost=appeal_cost,
        time_to_verdict=time_to_verdict,
        case_type=case_type,
        jurisdiction=jurisdiction,
        satisfaction_rating=satisfaction_rating,
        feedback_text=feedback_text,
    )
    db.add(feedback)
    db.commit()
    db.refresh(feedback)
    return feedback


def get_user_feedback(db: Session, user_id: int, limit: int = 50) -> List[UserFeedback]:
    return db.query(UserFeedback).filter(
        UserFeedback.user_id == user_id
    ).order_by(UserFeedback.created_at.desc()).limit(limit).all()
