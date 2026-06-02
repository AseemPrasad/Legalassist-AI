"""Service layer for looking up cases by their anonymized share ID.

The anonymized_id is a 12-character HMAC-SHA256 hex digest stored on the
Case row after the first call to generate_anonymized_case_data().  This
module resolves anonymized_id → redacted case payload without ever exposing
owner identity or PII.

Uses raw SQL via SQLAlchemy text() to avoid the ORM mapper conflict that
arises when database.py (which redefines models with extend_existing=True)
is imported alongside db.models.cases in the same process.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

from services.privacy_redaction import (
    apply_privacy_profile,
    get_privacy_profile_definition,
    normalize_privacy_profile,
)

logger = logging.getLogger(__name__)

# Maximum length accepted for an anonymized_id to prevent abuse.
_ANON_ID_MAX_LEN = 64


def _parse_created_date(raw: Any) -> Optional[str]:
    """Return a human-readable month/year string from a stored datetime value."""
    if not raw:
        return None
    if isinstance(raw, datetime):
        return raw.strftime("%B %Y")
    try:
        # SQLite stores datetimes as ISO strings
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        return dt.strftime("%B %Y")
    except Exception:
        return str(raw)[:7]  # fallback: "YYYY-MM"


def _parse_json_field(raw: Any) -> Any:
    """Safely parse a JSON column that may be a string or already decoded."""
    if raw is None:
        return None
    if isinstance(raw, (list, dict)):
        return raw
    try:
        return json.loads(raw)
    except Exception:
        return raw


def lookup_anonymized_case(
    db: Session,
    anonymized_id: str,
    profile_name: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Return a redacted case payload for the given *anonymized_id*.

    Returns ``None`` when no case with that ID exists.  The caller is
    responsible for translating ``None`` into an appropriate HTTP 404.

    Uses raw SQL to avoid SQLAlchemy ORM mapper conflicts that can occur
    when multiple model registrations exist in the same process.

    Owner identity (``user_id``) is **never** included in the payload.
    """
    if not anonymized_id or len(anonymized_id) > _ANON_ID_MAX_LEN:
        return None

    # ------------------------------------------------------------------ #
    # 1. Fetch the case row by anonymized_id                              #
    # ------------------------------------------------------------------ #
    case_row = db.execute(
        text(
            "SELECT id, case_type, jurisdiction, status, created_at "
            "FROM cases WHERE anonymized_id = :anon_id LIMIT 1"
        ),
        {"anon_id": anonymized_id},
    ).fetchone()

    if case_row is None:
        return None

    case_id = case_row[0]
    case_type = case_row[1]
    jurisdiction = case_row[2]
    status = case_row[3]
    created_at = case_row[4]

    # ------------------------------------------------------------------ #
    # 2. Fetch documents                                                  #
    # ------------------------------------------------------------------ #
    doc_rows = db.execute(
        text(
            "SELECT document_type, summary, remedies "
            "FROM case_documents WHERE case_id = :cid"
        ),
        {"cid": case_id},
    ).fetchall()

    documents: List[Dict[str, Any]] = [
        {
            "type": row[0],
            "summary": row[1],
            "remedies": _parse_json_field(row[2]),
        }
        for row in doc_rows
    ]

    # ------------------------------------------------------------------ #
    # 3. Fetch timeline                                                   #
    # ------------------------------------------------------------------ #
    timeline_rows = db.execute(
        text(
            "SELECT event_type, description "
            "FROM case_timeline WHERE case_id = :cid"
        ),
        {"cid": case_id},
    ).fetchall()

    timeline: List[Dict[str, Any]] = [
        {"event_type": row[0], "description": row[1]}
        for row in timeline_rows
    ]

    # ------------------------------------------------------------------ #
    # 4. Build and redact payload                                         #
    # ------------------------------------------------------------------ #
    selected_profile = normalize_privacy_profile(profile_name)
    profile = get_privacy_profile_definition(selected_profile)

    payload: Dict[str, Any] = {
        "export": {
            "privacy_profile": selected_profile,
            "privacy_profile_label": profile.get("label", selected_profile),
        },
        "anonymized_id": anonymized_id,
        "privacy_profile": selected_profile,
        "privacy_profile_label": profile.get("label", selected_profile),
        "case_type": case_type,
        "jurisdiction": jurisdiction,
        "status": status,
        "document_count": len(documents),
        "documents": documents,
        "timeline": timeline,
        "created_date": _parse_created_date(created_at),
    }

    return apply_privacy_profile(payload, selected_profile, anonymized_id=anonymized_id)
