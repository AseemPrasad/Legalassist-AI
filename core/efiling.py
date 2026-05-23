import base64
import uuid
from typing import Dict, Optional
from datetime import datetime, timezone

# Simple in-memory store for submissions (POC)
_SUBMISSIONS: Dict[str, Dict] = {}


class EfilingClient:
    """POC client that simulates submission to various court e-filing APIs.

    For production, replace with adapters per court implementing auth, retries,
    file conversion, and polling.
    """

    SUPPORTED_COURTS = {"SUPREME", "HIGH", "DISTRICT"}

    @staticmethod
    def validate_format(file_bytes: bytes, filename: Optional[str] = None) -> bool:
        # Very simple validation: check PDF header or common doc types via filename
        if filename and filename.lower().endswith(".pdf"):
            return True
        # PDF files start with '%PDF'
        try:
            return file_bytes.startswith(b"%PDF")
        except Exception:
            return False

    @classmethod
    def submit(cls, court: str, file_b64: str, metadata: Optional[Dict] = None) -> Dict:
        court_key = (court or "").upper()
        if court_key not in cls.SUPPORTED_COURTS:
            raise ValueError(f"Unsupported court: {court}")

        try:
            file_bytes = base64.b64decode(file_b64)
        except Exception:
            raise ValueError("file must be base64 encoded")

        if not cls.validate_format(file_bytes, metadata.get("filename") if metadata else None):
            raise ValueError("Invalid or unsupported document format")

        tracking_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        entry = {
            "tracking_id": tracking_id,
            "court": court_key,
            "submitted_at": now,
            "status": "submitted",
            "metadata": metadata or {},
        }

        # store submission
        _SUBMISSIONS[tracking_id] = entry

        # Simulate asynchronous processing by setting next status
        # For POC we'll set to 'accepted' immediately for SUPREME, 'pending' for others
        if court_key == "SUPREME":
            entry["status"] = "accepted"
        else:
            entry["status"] = "pending"

        return {"tracking_id": tracking_id, "status": entry["status"], "submitted_at": now}

    @classmethod
    def get_status(cls, tracking_id: str) -> Dict:
        entry = _SUBMISSIONS.get(tracking_id)
        if not entry:
            raise KeyError("tracking id not found")

        # Simulate transition: pending -> accepted after a check
        if entry["status"] == "pending":
            entry["status"] = "accepted"

        return {"tracking_id": tracking_id, "status": entry["status"], "court": entry["court"], "submitted_at": entry["submitted_at"], "metadata": entry.get("metadata", {})}


def clear_submissions():
    _SUBMISSIONS.clear()


__all__ = ["EfilingClient", "clear_submissions"]
