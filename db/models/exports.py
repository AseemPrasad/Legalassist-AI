import datetime as dt
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Boolean, Text
from sqlalchemy.orm import relationship
from db.base import Base


class ExportJob(Base):
    __tablename__ = "export_jobs"

    id = Column(Integer, primary_key=True)
    created_by = Column(Integer, nullable=True, index=True)
    output_path = Column(String(1024), nullable=False)
    export_format = Column(String(32), default="json")
    total_chunks = Column(Integer, default=0)
    last_completed_chunk = Column(Integer, default=-1)
    status = Column(String(32), default="pending")  # pending, processing, completed, failed
    error = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), onupdate=lambda: dt.datetime.now(dt.timezone.utc))

    chunks = relationship("ExportChunk", back_populates="job", cascade="all, delete-orphan")


class ExportChunk(Base):
    __tablename__ = "export_chunks"

    id = Column(Integer, primary_key=True)
    job_id = Column(Integer, ForeignKey("export_jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    index = Column(Integer, nullable=False, index=True)
    path = Column(String(2048), nullable=False)
    processed = Column(Boolean, default=False, nullable=False)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc))

    job = relationship("ExportJob", back_populates="chunks")
