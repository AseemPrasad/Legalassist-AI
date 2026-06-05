from sqlalchemy import Column, Integer, String, DateTime, Enum as SQLEnum
from sqlalchemy.orm import declarative_base
import enum
import datetime as dt

from db.base import Base

class SchedulerJobStatus(str, enum.Enum):
    SUCCESS = "success"
    FAILED = "failed"

class SchedulerRun(Base):
    __tablename__ = "scheduler_runs"
    __table_args__ = {"extend_existing": True}

    id = Column(Integer, primary_key=True)
    job_name = Column(String(255), nullable=False, index=True)
    started_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), nullable=False)
    finished_at = Column(DateTime(timezone=True), nullable=True)
    sent_count = Column(Integer, default=0, nullable=False)
    status = Column(SQLEnum(SchedulerJobStatus), nullable=False)
    error_code = Column(String(255), nullable=True)
