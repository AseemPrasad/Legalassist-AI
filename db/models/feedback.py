import datetime as dt
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.orm import relationship
from db.base import Base


class UserFeedback(Base):
    __tablename__ = "user_feedback"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    case_id = Column(Integer, ForeignKey("cases.id", ondelete="CASCADE"), nullable=True)

    did_appeal = Column(Boolean, nullable=True)
    appeal_outcome = Column(String(255), nullable=True)
    appeal_cost = Column(Integer, nullable=True)
    time_to_verdict = Column(Integer, nullable=True)
    case_type = Column(String(255), nullable=True)
    jurisdiction = Column(String(255), nullable=True)
    satisfaction_rating = Column(Integer, nullable=True)
    feedback_text = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), nullable=False)
