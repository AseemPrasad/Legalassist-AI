import datetime as dt
import logging
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker
from config import Config

logger = logging.getLogger(__name__)

DATABASE_URL = Config.DATABASE_URL
_db_url = make_url(DATABASE_URL)
_is_sqlite = _db_url.get_backend_name() == "sqlite"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if _is_sqlite else {})
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
    # Import here to avoid circular imports at package import time
    from .base import Base
    from sqlalchemy import text

    Base.metadata.create_all(bind=engine)
    try:
        with engine.begin() as connection:
            connection.execute(text("CREATE INDEX IF NOT EXISTS ix_notification_logs_status ON notification_logs (status)"))
    except Exception:
        pass


from contextlib import contextmanager


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
