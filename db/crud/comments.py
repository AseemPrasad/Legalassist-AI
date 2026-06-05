"""
CRUD operations for case collaboration: comments and presence tracking.

Access-control is enforced at the data layer — every read and write path
validates case ownership before returning or persisting records.  Callers are
therefore **not** required to perform a separate ownership check; the
guarantee is unconditional and independent of the access path used.
"""
from __future__ import annotations

import datetime as dt
from typing import List, Optional

from sqlalchemy.orm import Session

from db.models.cases import Case, CaseComment, CasePresence


def create_case_comment(
    db: Session,
    case_id: int,
    user_id: int,
    comment_text: str,
    parent_comment_id: Optional[int] = None,
) -> CaseComment:
    """Create a threaded collaboration comment for a case.

    Ownership validation is enforced before writing: the case must exist and
    be owned by ``user_id``.

    Raises:
        PermissionError: If the case does not exist or is not owned by
            ``user_id``.
        ValueError: If ``parent_comment_id`` is not a valid comment on this
            case, or if ``comment_text`` is blank.
    """
    case = db.query(Case).filter(Case.id == case_id, Case.user_id == user_id).first()
    if not case:
        raise PermissionError("case_id not found or not owned by the provided user_id")

    if parent_comment_id is not None:
        parent = db.query(CaseComment).filter(
            CaseComment.id == parent_comment_id,
            CaseComment.case_id == case_id,
        ).first()
        if not parent:
            raise ValueError("parent_comment_id is invalid for this case")

    text = (comment_text or "").strip()
    if not text:
        raise ValueError("comment_text cannot be empty")

    comment = CaseComment(
        case_id=case_id,
        user_id=user_id,
        parent_comment_id=parent_comment_id,
        comment_text=text,
    )
    db.add(comment)
    db.commit()
    db.refresh(comment)
    return comment


def get_case_comments(db: Session, case_id: int, user_id: int) -> List[CaseComment]:
    """Return threaded comments for a case, enforcing ownership at the retrieval layer.

    Authorization is validated **inside** this function — callers do not need
    to perform a separate ownership check before calling.  This centralizes
    access-control enforcement so that sensitive comment data is protected
    regardless of the code path used to reach this function.

    Args:
        db: Active SQLAlchemy session.
        case_id: Primary key of the case whose comments are requested.
        user_id: ID of the user requesting access.  Must be the case owner.

    Returns:
        Comments for the case ordered by ``created_at`` ascending (oldest
        first, preserving thread reading order).

    Raises:
        PermissionError: If the case does not exist or is not owned by
            ``user_id``.
    """
    case = db.query(Case).filter(Case.id == case_id, Case.user_id == user_id).first()
    if not case:
        raise PermissionError("case_id not found or not owned by the provided user_id")

    return (
        db.query(CaseComment)
        .filter(CaseComment.case_id == case_id)
        .order_by(CaseComment.created_at.asc())
        .all()
    )


def upsert_case_presence(
    db: Session,
    case_id: int,
    user_id: int,
    active_view: Optional[str] = None,
    cursor_anchor: Optional[str] = None,
) -> CasePresence:
    """Mark a collaborator as recently active on a case.

    Creates a new presence record or updates the existing one for this
    (case, user) pair.  Ownership of the case is validated before writing.

    Raises:
        PermissionError: If the case does not exist or is not owned by
            ``user_id``.
    """
    case = db.query(Case).filter(Case.id == case_id, Case.user_id == user_id).first()
    if not case:
        raise PermissionError("case_id not found or not owned by the provided user_id")

    presence = db.query(CasePresence).filter(
        CasePresence.case_id == case_id,
        CasePresence.user_id == user_id,
    ).first()

    now = dt.datetime.now(dt.timezone.utc)
    if presence:
        presence.active_view = active_view
        presence.cursor_anchor = cursor_anchor
        presence.last_seen = now
    else:
        presence = CasePresence(
            case_id=case_id,
            user_id=user_id,
            active_view=active_view,
            cursor_anchor=cursor_anchor,
            last_seen=now,
        )
        db.add(presence)

    db.commit()
    db.refresh(presence)
    return presence


def get_case_presence(
    db: Session,
    case_id: int,
    active_window_minutes: int = 5,
) -> List[CasePresence]:
    """Return collaborators active within a recent time window.

    Args:
        db: Active SQLAlchemy session.
        case_id: Primary key of the case.
        active_window_minutes: How many minutes back to consider a collaborator
            active (default 5).

    Returns:
        Presence records ordered by ``last_seen`` descending (most recently
        active first).
    """
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=active_window_minutes)
    return (
        db.query(CasePresence)
        .filter(
            CasePresence.case_id == case_id,
            CasePresence.last_seen >= cutoff,
        )
        .order_by(CasePresence.last_seen.desc())
        .all()
    )
