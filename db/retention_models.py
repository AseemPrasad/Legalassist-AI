"""
Data Retention Policy Models

Defines configurable retention rules per data category for GDPR/regulatory compliance.
"""

import datetime as dt
from enum import Enum

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, JSON
from sqlalchemy.orm import Session

from db.base import Base


class RetentionTier(str, Enum):
    """Data retention tier classification"""
    ACTIVE = "active"          # Live data, full access
    ARCHIVED = "archived"       # Soft-deleted, accessible to admins only
    ANONYMIZED = "anonymized"  # PII stripped, aggregate data only
    DELETED = "deleted"        # Hard-deleted, gone forever


class RetentionRule(Base):
    """Configurable retention rules per data category"""
    __tablename__ = "retention_rules"

    id = Column(Integer, primary_key=True)
    data_category = Column(String(100), nullable=False, unique=True, index=True)
    retention_days = Column(Integer, nullable=False)
    archive_after_days = Column(Integer, nullable=True)
    anonymize_after_days = Column(Integer, nullable=True)
    delete_after_days = Column(Integer, nullable=True)
    require_audit_log = Column(Boolean, default=True)
    pii_sensitive = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), onupdate=lambda: dt.datetime.now(dt.timezone.utc))


class RetentionAuditLog(Base):
    """Audit trail for all retention actions"""
    __tablename__ = "retention_audit_log"

    id = Column(Integer, primary_key=True)
    action = Column(String(50), nullable=False, index=True)
    data_category = Column(String(100), nullable=False, index=True)
    record_ids = Column(JSON, nullable=False)
    records_affected = Column(Integer, default=0)
    executed_by = Column(String(100), nullable=True)
    reason = Column(Text, nullable=True)
    policy_version = Column(String(50), nullable=True)
    timestamp = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), index=True)


DEFAULT_RETENTION_RULES = {
    "cases": {"retention_days": 2555, "archive_after_days": 730, "anonymize_after_days": 1825, "delete_after_days": 2555, "pii_sensitive": True},
    "case_documents": {"retention_days": 2555, "archive_after_days": 730, "anonymize_after_days": 1825, "delete_after_days": 2555, "pii_sensitive": True},
    "case_timeline": {"retention_days": 2555, "archive_after_days": 730, "anonymize_after_days": 1825, "delete_after_days": 2555, "pii_sensitive": True},
    "case_deadlines": {"retention_days": 1825, "archive_after_days": 365, "anonymize_after_days": 730, "delete_after_days": 1825, "pii_sensitive": False},
    "attachments": {"retention_days": 1095, "archive_after_days": 180, "anonymize_after_days": 365, "delete_after_days": 1095, "pii_sensitive": True},
    "notifications": {"retention_days": 365, "archive_after_days": 90, "anonymize_after_days": 180, "delete_after_days": 365, "pii_sensitive": True},
    "report_feedback": {"retention_days": 730, "archive_after_days": 180, "anonymize_after_days": 365, "delete_after_days": 730, "pii_sensitive": True},
    "model_feedback": {"retention_days": 1095, "archive_after_days": 365, "anonymize_after_days": 730, "delete_after_days": 1095, "pii_sensitive": False},
    "user_feedback": {"retention_days": 730, "archive_after_days": 180, "anonymize_after_days": 365, "delete_after_days": 730, "pii_sensitive": True},
}


def seed_retention_rules(db: Session) -> int:
    """Seed default retention rules if none exist."""
    existing = db.query(RetentionRule).count()
    if existing > 0:
        return 0

    created = 0
    for category, config in DEFAULT_RETENTION_RULES.items():
        rule = RetentionRule(data_category=category, **config)
        db.add(rule)
        created += 1

    db.commit()
    return created


def get_retention_rule(db: Session, data_category: str) -> RetentionRule | None:
    return db.query(RetentionRule).filter(RetentionRule.data_category == data_category).first()


def log_retention_action(
    db: Session,
    action: str,
    data_category: str,
    record_ids: list,
    records_affected: int,
    executed_by: str | None = None,
    reason: str | None = None,
) -> RetentionAuditLog:
    log = RetentionAuditLog(
        action=action,
        data_category=data_category,
        record_ids=record_ids,
        records_affected=records_affected,
        executed_by=executed_by,
        reason=reason,
    )
    db.add(log)
    db.commit()
    return log