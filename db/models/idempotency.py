import datetime as dt
import enum

from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, Enum as SQLEnum
from db.base import Base


class IdempotencyKeyStatus(str, enum.Enum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


class IdempotencyKey(Base):
    __tablename__ = "idempotency_keys"

    id = Column(Integer, primary_key=True)
    key = Column(String(255), unique=True, nullable=False, index=True)
    method = Column(String(10), nullable=False)
    path = Column(String(1024), nullable=False)
    status = Column(SQLEnum(IdempotencyKeyStatus), default=IdempotencyKeyStatus.IN_PROGRESS)
    response_status = Column(Integer, nullable=True)
    response_headers = Column(JSON, nullable=True)
    response_body = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    def __repr__(self):
        return f"<IdempotencyKey(key={self.key}, status={self.status})>"
