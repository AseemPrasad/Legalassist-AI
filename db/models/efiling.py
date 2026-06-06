"""SQLAlchemy model for persisting e-filing tracking-ID ownership.

Replaces the previous in-process dict ``_user_filings`` in
``api/routes/efiling.py``.  Storing filing ownership in the database
ensures that status lookups survive server restarts and work correctly
across multiple worker processes.
"""

import datetime as dt
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Index
from db.base import Base


class UserFiling(Base):
    """Maps a user to a court e-filing tracking ID they submitted."""

    __tablename__ = "user_filings"
    __table_args__ = (
        Index("ix_user_filings_user_id", "user_id"),
        Index("ix_user_filings_tracking_id", "tracking_id"),
    )

    id = Column(Integer, primary_key=True)
    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    tracking_id = Column(String(255), nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: dt.datetime.now(dt.timezone.utc),
        nullable=False,
    )
