"""
Retention Enforcement Service

Implements automated data retention, archiving, anonymization, and deletion
of expired records per configured retention policies.
"""

import datetime as dt
import hashlib
import logging
import random
import string
from typing import Optional

from sqlalchemy import update, func
from sqlalchemy.orm import Session

from db.session import db_session

logger = logging.getLogger(__name__)

_ANONYMIZATION_SALT = "legalease-anon-v1"


def _anonymize_value(value: str, field_name: str = "") -> str:
    if not value:
        return ""
    h = hashlib.sha256(f"{_ANONYMIZATION_SALT}{field_name}{value}".encode()).hexdigest()[:12]
    suffix = "".join(random.choices(string.ascii_uppercase + string.digits, k=4))
    return f"REDACTED_{h[:8]}_{suffix}"


def _anonymize_record(record, field_map: dict[str, str]) -> dict:
    """Replace PII fields with anonymized values."""
    updated = {}
    for field, ftype in field_map.items():
        val = getattr(record, field, None)
        if val is None:
            continue
        if ftype == "name":
            updated[field] = _anonymize_value(str(val), field)
        elif ftype == "email":
            updated[field] = f"redacted_{hashlib.sha256(val.encode()).hexdigest()[:8]}@anonymized.local"
        elif ftype == "phone":
            updated[field] = "REDACTED"
        elif ftype == "address":
            updated[field] = "REDACTED"
        elif ftype == "text":
            updated[field] = f"[Content redacted on {dt.datetime.now(dt.timezone.utc).date()}]"
        else:
            updated[field] = str(val)
    return updated


def archive_expired_cases(db: Session, cutoff_days: int, dry_run: bool = False) -> tuple[list, int]:
    """Soft-delete (archive) cases older than cutoff_days with no active deadline."""
    from db.models.cases import Case, CaseStatus

    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=cutoff_days)
    active_statuses = {CaseStatus.ACTIVE, CaseStatus.PENDING, CaseStatus.APPEALED}

    q = (
        db.query(Case)
        .filter(Case.status.notin_(active_statuses))
        .filter(Case.updated_at < cutoff)
    )
    if dry_run:
        records = q.all()
        ids = [r.id for r in records]
        return ids, len(ids)

    ids = [r.id for r in q.all()]
    if ids:
        db.query(Case).filter(Case.id.in_(ids)).update(
            {Case.status: CaseStatus.CLOSED}, synchronize_session="fetch"
        )
        db.commit()
    logger.info(f"Archived {len(ids)} cases (cutoff={cutoff_days} days)")
    return ids, len(ids)


def anonymize_expired_records(
    db: Session,
    model_class,
    cutoff_days: int,
    id_field: str = "id",
    pii_fields: Optional[dict] = None,
    dry_run: bool = False,
) -> tuple[list, int]:
    """Anonymize PII fields on records older than cutoff_days."""
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=cutoff_days)
    q = db.query(model_class).filter(getattr(model_class, "updated_at", None) < cutoff)
    records = q.all()

    if not records:
        return [], 0

    anonymized_ids = []
    for record in records:
        if pii_fields:
            changes = _anonymize_record(record, pii_fields)
            for field, value in changes.items():
                setattr(record, field, value)
            anonymized_ids.append(getattr(record, id_field))

    if not dry_run:
        db.commit()
    logger.info(f"Anonymized {len(anonymized_ids)} {model_class.__name__} records")
    return anonymized_ids, len(anonymized_ids)


def hard_delete_expired_records(
    db: Session,
    model_class,
    cutoff_days: int,
    date_field: str = "updated_at",
    dry_run: bool = False,
) -> tuple[list, int]:
    """Hard-delete records older than cutoff_days."""
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=cutoff_days)
    q = db.query(model_class).filter(getattr(model_class, date_field, None) < cutoff)
    ids = [getattr(r, "id") for r in q.all()]

    if dry_run:
        return ids, len(ids)

    if ids:
        db.query(model_class).filter(model_class.id.in_(ids)).delete(synchronize_session="fetch")
        db.commit()
    logger.info(f"Hard-deleted {len(ids)} {model_class.__name__} records (cutoff={cutoff_days} days)")
    return ids, len(ids)


def purge_expired_attachments(db: Session, cutoff_days: int, dry_run: bool = False) -> tuple[list, int]:
    """Delete orphaned or expired file attachments."""
    from db.models.cases import Attachment

    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=cutoff_days)
    q = (
        db.query(Attachment)
        .filter(Attachment.uploaded_at < cutoff)
        .filter(Attachment.case_id == None)  # orphaned
    )
    attachments = q.all()
    ids = [a.id for a in attachments]

    if dry_run:
        return ids, len(ids)

    if ids:
        db.query(Attachment).filter(Attachment.id.in_(ids)).delete(synchronize_session="fetch")
        db.commit()
    logger.info(f"Deleted {len(ids)} orphaned attachments (cutoff={cutoff_days} days)")
    return ids, len(ids)


def purge_expired_notifications(db: Session, cutoff_days: int, dry_run: bool = False) -> tuple[list, int]:
    """Delete old notification logs beyond retention window."""
    from db.models.notifications import NotificationLog

    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=cutoff_days)
    q = db.query(NotificationLog).filter(NotificationLog.sent_at < cutoff)
    records = q.all()
    ids = [r.id for r in records]

    if dry_run:
        return ids, len(ids)

    if ids:
        db.query(NotificationLog).filter(NotificationLog.id.in_(ids)).delete(synchronize_session="fetch")
        db.commit()
    logger.info(f"Deleted {len(ids)} expired notifications (cutoff={cutoff_days} days)")
    return ids, len(ids)


def purge_expired_otl_tokens(db: Session, cutoff_days: int, dry_run: bool = False) -> tuple[list, int]:
    """Delete OTP tokens past their expiry time."""
    from db.models.auth import OTPVerification

    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=cutoff_days)
    q = db.query(OTPVerification).filter(
        OTPVerification.expires_at < dt.datetime.now(dt.timezone.utc)
    )
    records = q.all()
    ids = [r.id for r in records]

    if dry_run:
        return ids, len(ids)

    if ids:
        db.query(OTPVerification).filter(OTPVerification.id.in_(ids)).delete(synchronize_session="fetch")
        db.commit()
    logger.info(f"Deleted {len(ids)} expired OTP tokens")
    return ids, len(ids)