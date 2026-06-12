"""Anonymized case export helpers."""

from __future__ import annotations

import hashlib
import logging
import hmac
import os
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from db.session import SessionLocal
from db.models import Case, CaseDocument, CaseTimeline
from config import Config
from services.privacy_redaction import (
    apply_privacy_profile,
    get_privacy_profile_definition,
    normalize_privacy_profile,
)
from db.crud.audit import record_audit_event

# Minimum secret length required for anonymization secret
_MIN_SECRET_LENGTH = 32
logger = logging.getLogger(__name__)


def _check_secret_entropy(secret: str, source: str) -> None:
    """Raise if the secret has dangerously low entropy.

    A secret composed entirely of one repeated character (e.g. ``"a" * 40``)
    passes a length check but provides near-zero security.  This guard catches
    the most common weak-secret patterns used in development that could
    accidentally reach a non-production environment connected to real data.
    """
    if len(set(secret)) < 8:
        raise ValueError(
            f"Anonymization secret from {source} has insufficient entropy "
            f"(fewer than 8 distinct characters). Use a randomly generated secret."
        )


def _get_case_anonymization_secret(override: Optional[str] = None) -> str:
    """Resolve the anonymization secret.

    Resolution order
    ----------------
    1. Test-time ``override`` — only accepted when ``Config.TESTING`` is True.
    2. ``CASE_ANONYMIZATION_SECRET`` environment variable / Streamlit secret.

    The previous fallback to the ``.jwt_secret`` file has been removed.
    Reading a secret from a file that may be committed to version control
    creates a systematic risk: any developer or CI runner with repo read
    access can forge valid anonymized IDs and, if the same file is used as
    the JWT secret, forge valid authentication tokens.

    Set ``CASE_ANONYMIZATION_SECRET`` in your environment or secrets manager.
    """
    # Test-time override support
    if override is not None:
        if not Config.TESTING:
            raise RuntimeError("Secret override allowed only in testing mode")
        secret = str(override or "").strip()
        if len(secret) < _MIN_SECRET_LENGTH:
            raise ValueError(f"Anonymization secret must be at least {_MIN_SECRET_LENGTH} characters")
        return secret

    # Primary (and only) source: environment / streamlit secrets via get_settings()
    try:
        from api.config import get_settings
        secret = get_settings().CASE_ANONYMIZATION_SECRET
    except Exception:
        secret = os.getenv("CASE_ANONYMIZATION_SECRET", "").strip()
        if not secret:
            # Also check via Config._get_val so Streamlit secrets are honoured
            try:
                from config import _get_val as _cfg_get
                secret = str(_cfg_get("CASE_ANONYMIZATION_SECRET", "") or "").strip()
            except Exception:
                pass

    if secret:
        if len(secret) < _MIN_SECRET_LENGTH:
            raise ValueError(
                f"Anonymization secret from environment must be at least "
                f"{_MIN_SECRET_LENGTH} characters"
            )
        _check_secret_entropy(secret, "environment")
        return secret

    raise RuntimeError(
        "CASE_ANONYMIZATION_SECRET is not configured. "
        "Set this environment variable to a randomly generated secret of at least "
        f"{_MIN_SECRET_LENGTH} characters."
    )


def _generate_anonymized_case_id(case_id: int, created_at: Any, secret_override: Optional[str] = None) -> str:
    """Generate a deterministic anonymized id for a case.

    Accepts an optional `secret_override` for test determinism (only used when
    `Config.TESTING` is True). Otherwise resolves secret via `_get_case_anonymization_secret()`.
    """
    created_at_str = getattr(created_at, "isoformat", None)
    created_at_str = created_at.isoformat() if callable(created_at_str) else str(created_at)
    if secret_override is not None:
        if not Config.TESTING:
            raise RuntimeError("Secret override allowed only in testing mode")
        if len(str(secret_override or "")) < _MIN_SECRET_LENGTH:
            raise ValueError(f"Anonymization secret must be at least {_MIN_SECRET_LENGTH} characters")
        secret_bytes = str(secret_override).encode("utf-8")
    else:
        secret_bytes = _get_case_anonymization_secret().encode("utf-8")

    msg = f"{case_id}-{created_at_str}".encode("utf-8")
    digest = hmac.new(secret_bytes, msg, hashlib.sha256).hexdigest()
    return digest[:12]


def generate_anonymized_case_data(case_id: int, profile_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    db: Session = SessionLocal()
    try:
        case = db.query(Case).filter(Case.id == case_id).first()
        if not case:
            return None

        documents = db.query(CaseDocument).filter(CaseDocument.case_id == case_id).all()
        timeline = db.query(CaseTimeline).filter(CaseTimeline.case_id == case_id).all()
        selected_profile = normalize_privacy_profile(profile_name)
        profile = get_privacy_profile_definition(selected_profile)
        anonymized_id = _generate_anonymized_case_id(case_id=case_id, created_at=case.created_at)

        # Persist the anonymized_id on the Case row so it can be looked up later.
        if case.anonymized_id != anonymized_id:
            case.anonymized_id = anonymized_id
            db.add(case)
            db.commit()
            db.refresh(case)

        payload = {
            "export": {
                "case_id": case_id,
                "generated_at": None,
                "privacy_profile": selected_profile,
                "privacy_profile_label": profile.get("label", selected_profile),
            },
            "anonymized_id": anonymized_id,
            "privacy_profile": selected_profile,
            "privacy_profile_label": profile.get("label", selected_profile),
            "case_type": case.case_type,
            "jurisdiction": case.jurisdiction,
            "status": case.status.value,
            "document_count": len(documents),
            "documents": [
                {
                    "type": doc.document_type.value,
                    "summary": doc.summary,
                    "remedies": doc.remedies,
                }
                for doc in documents
            ],
            "timeline": [
                {
                    "event_type": e.event_type,
                    "description": e.description,
                }
                for e in timeline
            ],
            "created_date": case.created_at.strftime("%B %Y"),
        }

        record_audit_event(
            db,
            actor=f"user:{case.user_id}",
            actor_user_id=case.user_id,
            action="anonymization_run",
            resource=f"case:{case_id}",
            case_id=case_id,
            metadata={
                "privacy_profile": selected_profile,
                "document_count": len(documents),
                "timeline_events": len(timeline),
            },
        )

        return apply_privacy_profile(payload, selected_profile, anonymized_id=anonymized_id)
    finally:
        db.close()
