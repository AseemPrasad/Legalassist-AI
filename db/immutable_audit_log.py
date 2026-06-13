"""
Immutable Audit Log Models

Provides Write-Once-Read-Many (WORM) audit logging with cryptographic tamper detection.
All entries are append-only with SHA-256 chain hashing for evidentiary integrity.

Security fixes applied (#1238)
-------------------------------
1. **Dedicated isolated session** — ``append_audit_entry`` now uses its own
   ``SessionLocal()`` instance that is never shared with the application
   request session.  This prevents application code from accidentally (or
   maliciously) rolling back, modifying, or deleting audit rows through the
   same session object.

2. **Hash computed before INSERT** — the ``integrity_hash`` is calculated
   before the row is written to the database.  The previous implementation
   inserted a row with ``integrity_hash=""`` and then issued a second UPDATE
   after ``flush()``.  That UPDATE violated the immutability trigger and left
   a window where the row existed with an empty hash.

3. **Serialised chain reads** — on PostgreSQL the previous-entry query uses
   ``SELECT … FOR UPDATE`` to acquire a row-level lock, preventing two
   concurrent writers from both reading the same "last" entry and producing
   a broken chain.  On SQLite the session is opened with
   ``IMMEDIATE`` transaction isolation for the same effect.
"""

import datetime as dt
import hashlib
import json
import logging
import threading

from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, Index, text

from db.base import Base

logger = logging.getLogger(__name__)


def _compute_hash(data: dict, prev_hash: str = "") -> str:
    serializable = {
        k: (v.isoformat() if isinstance(v, dt.datetime) else v)
        for k, v in data.items()
        if k not in ("id", "integrity_hash", "prev_hash", "integrity_verified")
    }
    content = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(f"{prev_hash}|{content}".encode()).hexdigest()


# Canonical set of fields included in the integrity hash.
# Must be identical in append_audit_entry and verify_audit_chain.
_HASH_FIELDS = (
    "event_type",
    "action",
    "actor_id",
    "actor_user_id",
    "resource_type",
    "resource_id",
    "outcome",
    "changes",
    "audit_metadata",
    "timestamp",
)


class ImmutableAuditLog(Base):
    """
    Append-only audit log with cryptographic chain hashing.

    Integrity guarantees:
    - INSERT only: no UPDATE/DELETE allowed via DB triggers
    - Each entry hashes its content + previous entry's hash (chain)
    - Entries are never modified or deleted after creation
    - Tampering is detectable by re-computing the chain hash
    """
    __tablename__ = "immutable_audit_log"

    id = Column(Integer, primary_key=True)
    timestamp = Column(
        DateTime(timezone=True),
        default=lambda: dt.datetime.now(dt.timezone.utc),
        nullable=False,
        index=True,
    )
    event_type = Column(String(100), nullable=False, index=True)
    actor_id = Column(String(100), nullable=True, index=True)
    actor_user_id = Column(Integer, nullable=True, index=True)
    actor_type = Column(String(50), nullable=True)
    resource_type = Column(String(100), nullable=True, index=True)
    resource_id = Column(String(200), nullable=True, index=True)
    action = Column(String(50), nullable=False, index=True)
    outcome = Column(String(50), nullable=True, index=True)
    changes = Column(JSON, nullable=True)
    audit_metadata = Column("metadata", JSON, nullable=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    prev_hash = Column(String(64), nullable=False)
    integrity_hash = Column(String(64), nullable=False)

    __table_args__ = (
        Index("ix_immutable_audit_log_timestamp_event", "timestamp", "event_type"),
        Index("ix_immutable_audit_log_resource", "resource_type", "resource_id"),
    )


def _get_audit_session():
    """Return a **dedicated** SQLAlchemy session for the audit log.

    This session is intentionally separate from the application request
    session so that:
    - Application-level rollbacks cannot undo audit writes.
    - Application code cannot access audit rows through the same session
      object and issue UPDATE/DELETE statements.
    - The audit session's transaction lifecycle is fully controlled here.
    """
    from db.session import SessionLocal
    return SessionLocal()


def append_audit_entry(
    event_type: str,
    action: str,
    actor_id: str | None = None,
    actor_user_id: int | None = None,
    actor_type: str | None = None,
    resource_type: str | None = None,
    resource_id: str | None = None,
    outcome: str | None = None,
    changes: dict | None = None,
    metadata: dict | None = None,
    ip_address: str | None = None,
    user_agent: str | None = None,
) -> dict:
    """Append a new audit log entry to the chain.

    Computes the cryptographic hash **before** inserting the row so that:
    - The row is written with its final ``integrity_hash`` in a single INSERT.
    - No post-insert UPDATE is needed (which would violate the immutability
      trigger and leave a window with an empty hash).

    The chain read is serialised with a row-level lock (PostgreSQL) or an
    IMMEDIATE transaction (SQLite) to prevent concurrent writers from
    producing a broken chain.
    """
    # Harmonize actor attributes across background tasks and API operations
    if actor_user_id is not None:
        if actor_user_id < 0:
            actor_type = "api"
            actor_id = f"api:{abs(actor_user_id)}"
        elif actor_user_id == 0:
            actor_type = "system"
            actor_id = "system:api_user"
        else:
            actor_type = "user"
            actor_id = f"user:{actor_user_id}"
    else:
        actor_type = actor_type or "system"
        if not actor_id:
            actor_id = "system:worker"

    from db.session import _is_postgres, _is_sqlite

    # Use a dedicated session — never share with the application session.
    db = _get_audit_session()
    try:
        # ------------------------------------------------------------------ #
        # Serialise the chain read to prevent race conditions.               #
        # ------------------------------------------------------------------ #
        if _is_sqlite:
            # SQLite does not support SELECT FOR UPDATE.  BEGIN IMMEDIATE
            # acquires a write lock before any reads, serialising concurrent
            # writers at the transaction level.
            db.execute(text("BEGIN IMMEDIATE"))
        # For PostgreSQL the SELECT FOR UPDATE below provides the lock.

        if _is_postgres:
            last = (
                db.query(ImmutableAuditLog)
                .order_by(ImmutableAuditLog.id.desc())
                .with_for_update()
                .first()
            )
        else:
            last = (
                db.query(ImmutableAuditLog)
                .order_by(ImmutableAuditLog.id.desc())
                .first()
            )

        prev_hash = last.integrity_hash if last else "GENESIS"

        # ------------------------------------------------------------------ #
        # Compute the timestamp now so it is included in the hash.           #
        # Normalize to a naive UTC ISO string (no +00:00 suffix) so the     #
        # hash is stable across databases that strip timezone info on        #
        # read-back (e.g. SQLite stores datetimes as naive strings).         #
        # ------------------------------------------------------------------ #
        now = dt.datetime.now(dt.timezone.utc)
        # Strip tzinfo for hashing — SQLite returns naive datetimes on read,
        # so we must hash the naive form to keep verify_audit_chain consistent.
        now_naive = now.replace(tzinfo=None)
        now_iso = now_naive.isoformat()

        # ------------------------------------------------------------------ #
        # Compute the integrity hash BEFORE inserting the row.               #
        # ------------------------------------------------------------------ #
        hash_data = {
            "event_type": event_type,
            "action": action,
            "actor_id": actor_id,
            "actor_user_id": actor_user_id,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "outcome": outcome,
            "changes": changes,
            "audit_metadata": metadata,
            "timestamp": now_iso,
        }
        integrity_hash = _compute_hash(hash_data, prev_hash)

        entry = ImmutableAuditLog(
            event_type=event_type,
            action=action,
            actor_id=actor_id,
            actor_user_id=actor_user_id,
            actor_type=actor_type,
            resource_type=resource_type,
            resource_id=resource_id,
            outcome=outcome,
            changes=changes,
            audit_metadata=metadata,
            ip_address=ip_address,
            user_agent=user_agent,
            timestamp=now,
            prev_hash=prev_hash,
            integrity_hash=integrity_hash,  # set on first INSERT, no UPDATE needed
        )
        db.add(entry)
        db.commit()
        db.refresh(entry)

        return {
            "id": entry.id,
            "timestamp": entry.timestamp,
            "integrity_hash": entry.integrity_hash,
            "prev_hash": entry.prev_hash,
        }

    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def verify_audit_chain(start_id: int = 1, end_id: int | None = None) -> dict:
    """Verify integrity of the audit chain from start_id to end_id.

    Returns dict with 'valid' bool, 'broken_at' id or None, and
    'entries_checked' count.
    """
    results = {"valid": True, "broken_at": None, "entries_checked": 0}

    db = _get_audit_session()
    try:
        q = db.query(ImmutableAuditLog).filter(ImmutableAuditLog.id >= start_id)
        if end_id:
            q = q.filter(ImmutableAuditLog.id <= end_id)
        entries = q.order_by(ImmutableAuditLog.id).all()

        prev_hash = ""
        if entries and entries[0].id > 1:
            prev = (
                db.query(ImmutableAuditLog)
                .filter(ImmutableAuditLog.id == entries[0].id - 1)
                .first()
            )
            prev_hash = prev.integrity_hash if prev else "GENESIS"
        elif entries:
            prev_hash = "GENESIS"

        for entry in entries:
            results["entries_checked"] += 1
            # Normalize timestamp to naive UTC ISO string — same as
            # append_audit_entry does before hashing, so the hash is stable
            # across databases that strip timezone info on read-back.
            ts = entry.timestamp
            if isinstance(ts, dt.datetime):
                ts_naive = ts.replace(tzinfo=None)
                ts_iso = ts_naive.isoformat()
            else:
                # SQLite may return a string; strip any timezone suffix
                ts_str = str(ts)
                try:
                    parsed = dt.datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    ts_iso = parsed.replace(tzinfo=None).isoformat()
                except Exception:
                    ts_iso = ts_str

            expected_hash = _compute_hash(
                {
                    "event_type": entry.event_type,
                    "action": entry.action,
                    "actor_id": entry.actor_id,
                    "actor_user_id": entry.actor_user_id,
                    "resource_type": entry.resource_type,
                    "resource_id": entry.resource_id,
                    "outcome": entry.outcome,
                    "changes": entry.changes,
                    "audit_metadata": entry.audit_metadata,
                    "timestamp": ts_iso,
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

    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

    return results
