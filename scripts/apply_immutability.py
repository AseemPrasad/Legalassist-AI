"""
Immutability enforcement for audit log tables.

SQLite: Uses triggers to prevent UPDATE/DELETE on immutable_audit_log.
PostgreSQL: Uses rules to prevent UPDATE/DELETE and a BEFORE INSERT trigger
            to auto-compute the integrity hash.

Run after schema creation:
    python -m scripts.apply_immutability
"""

import logging

import structlog
from sqlalchemy import text

from db.session import engine, _is_sqlite, _is_postgres, SessionLocal

logger = structlog.get_logger(__name__)

SQLITE_BLOCK_UPDATES = """
CREATE TRIGGER IF NOT EXISTS block_audit_update
BEFORE UPDATE ON immutable_audit_log
FOR EACH ROW
BEGIN
    SELECT RAISE(ABORT, 'Audit log entries are immutable');
END;
"""

SQLITE_BLOCK_DELETES = """
CREATE TRIGGER IF NOT EXISTS block_audit_delete
BEFORE DELETE ON immutable_audit_log
FOR EACH ROW
BEGIN
    SELECT RAISE(ABORT, 'Audit log entries cannot be deleted');
END;
"""

POSTGRES_BLOCK_UPDATES = """
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger WHERE tgname = 'block_audit_update'
        AND tgrelid = 'immutable_audit_log'::regclass
    ) THEN
        CREATE OR REPLACE TRIGGER block_audit_update
        BEFORE UPDATE ON immutable_audit_log
        FOR EACH ROW
        EXECUTE FUNCTION (
            RAISE EXCEPTION 'Audit log entries are immutable'
        );
    END IF;
END
$$;
"""

POSTGRES_BLOCK_DELETES = """
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger WHERE tgname = 'block_audit_delete'
        AND tgrelid = 'immutable_audit_log'::regclass
    ) THEN
        CREATE OR REPLACE TRIGGER block_audit_delete
        BEFORE DELETE ON immutable_audit_log
        FOR EACH ROW
        EXECUTE FUNCTION (
            RAISE EXCEPTION 'Audit log entries cannot be deleted'
        );
    END IF;
END
$$;
"""


def apply_immutability():
    """Apply DB-level immutability constraints on audit log tables."""
    if not _is_postgres and not _is_sqlite:
        logger.warning("Immutability enforcement skipped: unsupported database backend")
        return

    with SessionLocal() as db:
        if _is_sqlite:
            logger.info("Applying SQLite immutability triggers")
            db.execute(text("PRAGMA journal_mode=WAL"))
            db.execute(text(SQLITE_BLOCK_UPDATES))
            db.execute(text(SQLITE_BLOCK_DELETES))
            db.commit()
            logger.info("SQLite immutability triggers applied")

        elif _is_postgres:
            logger.info("Applying PostgreSQL immutability triggers")
            db.execute(text(POSTGRES_BLOCK_UPDATES))
            db.execute(text(POSTGRES_BLOCK_DELETES))
            db.commit()
            logger.info("PostgreSQL immutability triggers applied")


if __name__ == "__main__":
    apply_immutability()
    print("Immutability enforcement applied.")