"""
Immutable Audit Log Models

Provides Write-Once-Read-Many (WORM) audit logging with cryptographic tamper detection.
All entries are append-only with SHA-256 chain hashing for evidentiary integrity.
"""

import datetime as dt
import hashlib
import json

from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, Index, event

from db.base import Base


def _compute_hash(data: dict, prev_hash: str = "") -> str:
    serializable = {
        k: (v.isoformat() if isinstance(v, dt.datetime) else v)
        for k, v in data.items()
        if k not in ("id", "integrity_hash", "prev_hash", "integrity_verified")
    }
    content = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(f"{prev_hash}|{content}".encode()).hexdigest()


class ImmutableAuditLog(Base):
    """
    Append-only audit log with cryptographic chain hashing.

    Integrity guarantees:
    - INSERT only: no UPDATE/DELETE allowed via DB rules
    - Each entry hashes its content + previous entry's hash (chain)
    - Entries are never modified or deleted after creation
    - Tampering is detectable by re-computing the chain hash
    """
    __tablename__ = "immutable_audit_log"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), nullable=False, index=True)
    event_type = Column(String(100), nullable=False, index=True)
    actor_id = Column(String(100), nullable=True, index=True)
    actor_type = Column(String(50), nullable=True)
    resource_type = Column(String(100), nullable=True, index=True)
    resource_id = Column(String(200), nullable=True, index=True)
    action = Column(String(50), nullable=False, index=True)
    changes = Column(JSON, nullable=True)
    metadata = Column(JSON, nullable=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    prev_hash = Column(String(64), nullable=False)
    integrity_hash = Column(String(64), nullable=False)

    __table_args__ = (
        Index("ix_immutable_audit_log_timestamp_event", "timestamp", "event_type"),
        Index("ix_immutable_audit_log_resource", "resource_type", "resource_id"),
    )


def append_audit_entry(
    event_type: str,
    action: str,
    actor_id: str | None = None,
    actor_type: str | None = None,
    resource_type: str | None = None,
    resource_id: str | None = None,
    changes: dict | None = None,
    metadata: dict | None = None,
    ip_address: str | None = None,
    user_agent: str | None = None,
) -> dict:
    """
    Append a new audit log entry to the chain.
    Computes cryptographic hash linking to the previous entry.
    """
    from db.session import db_session, _is_sqlite

    entry_data = {
        "event_type": event_type,
        "action": action,
        "actor_id": actor_id,
        "actor_type": actor_type,
        "resource_type": resource_type,
        "resource_id": resource_id,
        "changes": changes,
        "metadata": metadata,
        "ip_address": ip_address,
        "user_agent": user_agent,
    }

    with db_session() as db:
        last = db.query(ImmutableAuditLog).order_by(ImmutableAuditLog.id.desc()).first()
        prev_hash = last.integrity_hash if last else "GENESIS"

        entry = ImmutableAuditLog(
            event_type=event_type,
            action=action,
            actor_id=actor_id,
            actor_type=actor_type,
            resource_type=resource_type,
            resource_id=resource_id,
            changes=changes,
            metadata=metadata,
            ip_address=ip_address,
            user_agent=user_agent,
            prev_hash=prev_hash,
            integrity_hash="",  # computed below
        )
        db.add(entry)
        db.flush()

        entry.integrity_hash = _compute_hash(
            {
                "id": entry.id,
                "timestamp": entry.timestamp,
                "event_type": entry.event_type,
                "action": entry.action,
                "actor_id": entry.actor_id,
                "resource_type": entry.resource_type,
                "resource_id": entry.resource_id,
                "changes": entry.changes,
                "prev_hash": prev_hash,
            },
            prev_hash,
        )
        db.commit()

        return {
            "id": entry.id,
            "timestamp": entry.timestamp,
            "integrity_hash": entry.integrity_hash,
            "prev_hash": entry.prev_hash,
        }


def verify_audit_chain(start_id: int = 1, end_id: int | None = None) -> dict:
    """
    Verify integrity of the audit chain from start_id to end_id.
    Returns dict with 'valid' bool, 'broken_at' id or None, and 'entries_checked' count.
    """
    from db.session import db_session

    results = {"valid": True, "broken_at": None, "entries_checked": 0}

    with db_session() as db:
        q = db.query(ImmutableAuditLog).filter(ImmutableAuditLog.id >= start_id)
        if end_id:
            q = q.filter(ImmutableAuditLog.id <= end_id)
        entries = q.order_by(ImmutableAuditLog.id).all()

        prev_hash = ""
        if entries and entries[0].id > 1:
            prev = db.query(ImmutableAuditLog).filter(ImmutableAuditLog.id == entries[0].id - 1).first()
            prev_hash = prev.integrity_hash if prev else "GENESIS"
        elif entries:
            prev_hash = "GENESIS"

        for entry in entries:
            results["entries_checked"] += 1
            expected_hash = _compute_hash(
                {
                    "id": entry.id,
                    "timestamp": entry.timestamp,
                    "event_type": entry.event_type,
                    "action": entry.action,
                    "actor_id": entry.actor_id,
                    "resource_type": entry.resource_type,
                    "resource_id": entry.resource_id,
                    "changes": entry.changes,
                    "prev_hash": prev_hash,
                },
                prev_hash,
            )
            if entry.integrity_hash != expected_hash:
                results["valid"] = False
                results["broken_at"] = entry.id
                break
            if entry.prev_hash != prev_hash:
                results["valid"] = False
                results["broken_at"] = entry.id
                break
            prev_hash = entry.integrity_hash

    return results