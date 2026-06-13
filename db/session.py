import datetime as dt
import logging
import os
from contextlib import contextmanager
from typing import Optional

from sqlalchemy import create_engine, text, event
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import orm

try:
    from fastapi import Request
except ImportError:
    Request = type("Request", (), {})  # type: ignore
from config import Config

logger = logging.getLogger(__name__)

DATABASE_URL = Config.DATABASE_URL
_db_url = make_url(DATABASE_URL)
_is_sqlite = _db_url.get_backend_name() == "sqlite"
_is_postgres = _db_url.get_backend_name() == "postgresql"
engine_kwargs: dict = {}
if _is_sqlite:
    engine_kwargs["connect_args"] = {"check_same_thread": False}
    from sqlalchemy import event
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
else:
    engine_kwargs["pool_size"] = int(os.getenv("DB_POOL_SIZE", "20"))
    engine_kwargs["max_overflow"] = int(os.getenv("DB_MAX_OVERFLOW", "10"))
    engine_kwargs["pool_timeout"] = int(os.getenv("DB_POOL_TIMEOUT", "30"))
    engine_kwargs["pool_recycle"] = int(os.getenv("DB_POOL_RECYCLE", "1800"))
engine = create_engine(DATABASE_URL, **engine_kwargs)

if _is_sqlite:
    @event.listens_for(engine, "connect")
    def _set_sqlite_fk_pragma(db_connection, connection_record):
        cursor = db_connection.cursor()
        cursor.execute("PRAGMA foreign_keys = ON")
        cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, expire_on_commit=False, bind=engine)


def _to_utc_datetime(value: dt.datetime) -> dt.datetime:
    from core.clock import _utc_datetime
    return _utc_datetime(value)


def _datetime_for_db(value: dt.datetime) -> dt.datetime:
    from core.clock import _utc_datetime
    utc_value = _utc_datetime(value)
    if _is_sqlite:
        return utc_value.replace(tzinfo=None)
    return utc_value


def init_db():
    from .base import Base
    from sqlalchemy import text

    Base.metadata.create_all(bind=engine)
    try:
        with engine.begin() as connection:
            connection.execute(text("CREATE INDEX IF NOT EXISTS ix_notification_logs_status ON notification_logs (status)"))
    except Exception as exc:
        logger.warning("Failed to create notification_logs index", error=str(exc))

    try:
        with engine.begin() as connection:
            if _is_sqlite:
                cursor = connection.execute(text("PRAGMA table_info(user_preferences)"))
                cols = [row[1] for row in cursor.fetchall()]
            else:
                cursor = connection.execute(text(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name='user_preferences' AND column_name='reminder_thresholds'"
                ))
                cols = [row[0] for row in cursor.fetchall()]
            
            if "reminder_thresholds" not in cols:
                logger.info("Adding column reminder_thresholds to user_preferences table")
                connection.execute(text("ALTER TABLE user_preferences ADD COLUMN reminder_thresholds TEXT"))
    except Exception as exc:
        logger.warning("Failed to migrate user_preferences schema", error=str(exc))

    try:
        with engine.begin() as connection:
            if _is_sqlite:
                cursor = connection.execute(text("PRAGMA table_info(case_deadlines)"))
                cols = [row[1] for row in cursor.fetchall()]
            else:
                cursor = connection.execute(text(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name='case_deadlines' AND column_name='status'"
                ))
                cols = [row[0] for row in cursor.fetchall()]
            
            if "status" not in cols:
                logger.info("Adding column status to case_deadlines table")
                connection.execute(text("ALTER TABLE case_deadlines ADD COLUMN status VARCHAR(50) DEFAULT 'active'"))
                # Migrate existing completed deadlines
                connection.execute(text("UPDATE case_deadlines SET status = 'completed' WHERE is_completed = 1"))
    except Exception as exc:
        logger.warning("Failed to migrate case_deadlines schema", error=str(exc))

    if _is_sqlite or _is_postgres:
        try:
            import scripts.apply_immutability as imm
            imm.apply_immutability()
        except Exception as exc:
            logger.error(
                "Immutable audit log initialization failed — audit trail may not be tamper-proof",
                error=str(exc),
            )


@contextmanager
def db_session():
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_db():
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def apply_rls_context(db: Session, user_id: int) -> None:
    if not _is_postgres:
        return
    db.execute(text("SET app.current_user_id = :uid"), {"uid": str(user_id)})


def clear_rls_context(db: Session) -> None:
    if not _is_postgres:
        return
    db.execute(text("RESET app.current_user_id"))


def get_db_with_rls(request: "Request") -> Session:
    db = SessionLocal()
    user_id = getattr(request.state, "db_rls_user_id", None) or getattr(request.state, "user_id", None)
    # Normalize common identifier shapes ("user:123", numeric strings, ints)
    normalized = None
    try:
        if isinstance(user_id, int):
            normalized = int(user_id)
        elif isinstance(user_id, str):
            if user_id.isdigit():
                normalized = int(user_id)
            elif user_id.startswith("user:"):
                parts = user_id.split(":", 1)
                if len(parts) == 2 and parts[1].isdigit():
                    normalized = int(parts[1])
    except Exception:
        normalized = None

    if normalized is not None:
        # Apply DB-level RLS if Postgres is used
        if _is_postgres:
            apply_rls_context(db, int(normalized))
        # Also expose tenant_id on the session.info for application-level filtering
        db.info["tenant_id"] = int(normalized)
    else:
        # Ensure tenant_id not set if unknown
        db.info.pop("tenant_id", None)
    return db


class RLSSession:
    def __init__(self, db: Session):
        self.db = db

    def __enter__(self):
        return self.db

    def __exit__(self, *args):
        if _is_postgres:
            clear_rls_context(self.db)
        self.db.close()


def rls_db(request: "Request") -> RLSSession:
    db = get_db_with_rls(request)
    return RLSSession(db)


# Apply application-level tenant scoping for ORM SELECT statements.
# This uses SQLAlchemy's with_loader_criteria to automatically add
# a `user_id = :tenant` predicate for mapped classes that include
# a `user_id` attribute. It only applies when `session.info['tenant_id']`
# is present and the statement is a SELECT.
@event.listens_for(Session, "do_orm_execute")
def _apply_tenant_criteria(execute_state):
    try:
        tenant = execute_state.session.info.get("tenant_id")
    except Exception:
        tenant = None
    if tenant is None:
        return

    # Only apply to SELECT operations
    if not execute_state.is_select:
        return

    # Import Base lazily to avoid circular imports
    try:
        from db.base import Base
    except Exception:
        return

    # For each mapped class, if it has a `user_id` attribute, add a loader criteria
    for mapper in Base.registry.mappers:
        cls = mapper.class_
        if hasattr(cls, "user_id"):
            execute_state.statement = execute_state.statement.options(
                orm.with_loader_criteria(cls, lambda cls_, tid=tenant: cls_.user_id == tid, include_aliases=True)
            )
