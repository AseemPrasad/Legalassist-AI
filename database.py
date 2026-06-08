"""Compatibility shim for the original monolithic `database.py`.

The project has moved models and CRUD helpers into the `db/` package, but many
existing imports still point at `database`. This module re-exports the pieces
needed by the current codebase and keeps the authentication/OTP security path
working while the refactor continues.

CRITICAL: This file should be a PURE RE-EXPORT MODULE. All implementations must
come from db/ subpackages. Do not define duplicate functions here.
"""

from __future__ import annotations

import datetime as dt
import threading
from typing import Optional, List
from sqlalchemy import func
from sqlalchemy.orm import Session

from db.base import Base
from db.session import engine, SessionLocal, init_db, db_session, get_db, _to_utc_datetime, _datetime_for_db

_OTP_RATE_LIMIT_LOCK = threading.RLock()
_OTP_RATE_LIMIT_EVENTS: dict[str, list[dt.datetime]] = {}


def _otp_rate_limit_key(identifier: str) -> str:
    normalized = str(identifier).strip().lower().replace("@", "")
    if not normalized:
        raise ValueError("OTP request identifier is required")
    return f"otp:rate:{normalized}"


from db.models import (
    User,
    OTPVerification,
    NotificationStatus,
    NotificationChannel,
    NotificationLog,
    NotificationTemplate,
    UserPreference,
    CaseDeadline,
    Case,
    CaseDocument,
    Attachment,
    CaseTimeline,
    CaseNote,
    CaseNoteVersion,
    AnonymizedShareToken,
    CaseComment,
    CasePresence,
    CaseStatus,
    DocumentType,
    UserFeedback,
    CaseRecord,
    CaseOutcome,
    CaseAnalytics,
    ModelFeedback,
    ModelPerformance,
    ModelRoutingRule,
    SimilarityFeedback,
    RevokedToken,
    CaseEmbedding,
    CaseIssue,
    CaseArgument,
    KnowledgeGraphEdge,
    PrecedentMatch,
)

from db.crud.notifications import (
    create_case_deadline,
    get_upcoming_deadlines,
    has_notification_been_sent,
    log_notification,
    get_notification_history,
    reserve_notification,
    update_notification_result,
)

from db.crud.users import (
    get_user_by_email,
    create_user,
    update_user_last_login,
    create_otp_verification,
    get_pending_otp,
    mark_otp_as_used,
    is_email_locked_out,
    record_otp_failed_attempt,
    reset_otp_failed_attempts,
    cleanup_expired_otps,
    create_or_update_user_preference,
)

from db.crud.cases import (
    create_case,
    get_user_cases,
    get_case_by_id,
    get_case_by_number,
    update_case_status,
    delete_case,
    create_case_document,
    get_case_documents,
    get_case_document_by_id,
    create_case_record,
    get_case_record,
    get_cases_by_criteria,
    update_case_outcome,
    submit_user_feedback,
    get_user_feedback,
    submit_model_feedback,
    get_case_timeline,
    create_timeline_event,
    create_attachment,
    get_attachments_for_case,
    get_user_stats,
    get_similarity_feedback,
)

from db.crud.tokens import (
    revoke_token,
    cleanup_expired_revoked_tokens,
    is_token_revoked,
)

from db.case_service import (
    save_case_note_draft,
    publish_case_note,
    get_case_note_history,
)

from db.crud.comments import (
    create_case_comment,
    get_case_comments,
    upsert_case_presence,
    get_case_presence,
)

__all__ = [
    "Base",
    "engine",
    "SessionLocal",
    "init_db",
    "db_session",
    "get_db",
    "_to_utc_datetime",
    "_datetime_for_db",
    "NotificationStatus",
    "NotificationChannel",
    "UserPreference",
    "NotificationLog",
    "NotificationTemplate",
    "CaseDeadline",
    "Case",
    "CaseDocument",
    "Attachment",
    "CaseTimeline",
    "CaseNote",
    "CaseComment",
    "CasePresence",
    "CaseStatus",
    "DocumentType",
    "User",
    "OTPVerification",
    "UserFeedback",
    "CaseRecord",
    "CaseOutcome",
    "CaseAnalytics",
    "ModelFeedback",
    "ModelPerformance",
    "ModelRoutingRule",
    "SimilarityFeedback",
    "RevokedToken",
    "CaseEmbedding",
    "CaseIssue",
    "CaseArgument",
    "KnowledgeGraphEdge",
    "PrecedentMatch",
    "create_case_deadline",
    "get_upcoming_deadlines",
    "has_notification_been_sent",
    "log_notification",
    "get_notification_history",
    "reserve_notification",
    "update_notification_result",
    "create_or_update_user_preference",
    "create_user",
    "get_user_by_email",
    "update_user_last_login",
    "create_otp_verification",
    "get_pending_otp",
    "mark_otp_as_used",
    "is_email_locked_out",
    "record_otp_failed_attempt",
    "reset_otp_failed_attempts",
    "cleanup_expired_otps",
    "create_case",
    "get_user_cases",
    "get_case_by_id",
    "get_case_by_number",
    "update_case_status",
    "delete_case",
    "create_case_document",
    "get_case_documents",
    "get_case_document_by_id",
    "create_case_record",
    "get_case_record",
    "get_cases_by_criteria",
    "update_case_outcome",
    "submit_user_feedback",
    "get_user_feedback",
    "submit_model_feedback",
    "get_case_timeline",
    "create_timeline_event",
    "create_attachment",
    "get_attachments_for_case",
    "revoke_token",
    "cleanup_expired_revoked_tokens",
    "is_token_revoked",
    "CaseNote",
    "CaseNoteVersion",
    "save_case_note_draft",
    "publish_case_note",
    "get_case_note_history",
    "create_case_comment",
    "get_case_comments",
    "upsert_case_presence",
    "get_case_presence",
    "get_user_stats",
    "get_similarity_feedback",
]