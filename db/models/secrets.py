import datetime as dt
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.orm import relationship
from db.base import Base


class SecretEntry(Base):
    __tablename__ = "secret_entries"

    id = Column(Integer, primary_key=True)
    name = Column(String(255), unique=True, nullable=False, index=True)
    value_enc = Column(Text, nullable=False)  # encrypted value
    version = Column(Integer, default=1, nullable=False)
    rotated_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), onupdate=lambda: dt.datetime.now(dt.timezone.utc))


class SecretRotationLog(Base):
    __tablename__ = "secret_rotation_log"

    id = Column(Integer, primary_key=True)
    secret_id = Column(Integer, ForeignKey("secret_entries.id", ondelete="CASCADE"), nullable=False, index=True)
    previous_version = Column(Integer, nullable=False)
    new_version = Column(Integer, nullable=False)
    rotated_by = Column(String(255), nullable=True)
    reason = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc))

    secret = relationship("SecretEntry")
