import datetime as dt
import logging
from contextlib import contextmanager
from typing import Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker

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
else:
    engine_kwargs["pool_size"] = 20
    engine_kwargs["max_overflow"] = 10
engine = create_engine(DATABASE_URL, **engine_kwargs)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, expire_on_commit=False, bind=engine)


def _to_utc_datetime(value: dt.datetime) -> dt.datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc)


def _datetime_for_db(value: dt.datetime) -> dt.datetime:
    utc_value = _to_utc_datetime(value)
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
    if _is_postgres:
        user_id = getattr(request.state, "db_rls_user_id", None) or getattr(request.state, "user_id", None)
        if user_id and str(user_id).isdigit():
            apply_rls_context(db, int(user_id))
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
