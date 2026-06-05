"""Report generation service.

Phase 1 scope (approved):
- Real PDF generation wired to the existing branded generator in `pdf_exporter.py`.

Storage:
- Supports both local filesystem and S3-compatible storage for distributed deployments.
- Set REPORT_STORAGE_TYPE=s3 for cloud deployments, or local (default) for single-node.
- S3 configuration via AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET env vars.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from core.storage import safe_filename
from pdf_exporter import generate_case_pdf
from celery_app import TaskStatus


@dataclass(frozen=True)
class GeneratedReport:
    report_id: str
    format: str
    file_path: Path
    file_name: str
    mime_type: str
    file_size_bytes: int
    storage_type: str = "local"  # "local" or "s3"


class S3Storage:
    """S3-compatible storage for distributed report access."""
    
    def __init__(self):
        self.bucket = os.getenv("S3_BUCKET", "legalassist-reports")
        self.region = os.getenv("AWS_REGION", "us-east-1")
        self._client = None
    
    @property
    def client(self):
        if self._client is None:
            try:
                import boto3
                self._client = boto3.client(
                    "s3",
                    region_name=self.region,
                    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                )
            except ImportError:
                raise RuntimeError("boto3 required for S3 storage. Install: pip install boto3")
        return self._client
    
    def upload(self, key: str, data: bytes) -> str:
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data)
        return f"s3://{self.bucket}/{key}"
    
    def download(self, key: str) -> bytes:
        response = self.client.get_object(Bucket=self.bucket, Key=key)
        return response["Body"].read()
    
    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:
            return False


_storage_type = os.getenv("REPORT_STORAGE_TYPE", "local").lower()
_s3_storage = S3Storage() if _storage_type == "s3" else None


def _get_reports_base_dir() -> Path:
    if _storage_type == "s3":
        return Path(f"s3://{_s3_storage.bucket}")
    # Keep it in project workspace so it works in local dev
    base = Path(os.getenv("REPORTS_OUTPUT_DIR", "./.report_outputs")).resolve()
    base.mkdir(parents=True, exist_ok=True)
    return base


def _store_report(file_path: Path, data: bytes, storage_type: str) -> str:
    """Store report data using appropriate storage backend."""
    if storage_type == "s3":
        key = str(file_path).lstrip("/")
        return _s3_storage.upload(key, data)
    file_path.write_bytes(data)
    return str(file_path)


def _get_format_meta(format: str) -> tuple[str, str]:
    fmt = (format or "pdf").lower()
    if fmt == "pdf":
        return "application/pdf", ".pdf"
    if fmt == "docx":
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document", ".docx"
    raise ValueError(f"Unsupported format: {format}")


def get_report_by_id(report_id: str, user_id: str) -> Optional[dict]:
    """Look up a report by its ID and validate ownership.

    Returns a dict with report metadata, or None if not found or not owned
    by the given user.  Callers should use the returned dict throughout the
    request lifecycle instead of issuing follow-up queries.
    """
    status_info = TaskStatus.get_task_status(report_id)
    if status_info.get("status") == "unknown":
        return None

    base_dir = _get_reports_base_dir()
    user_dir = base_dir / str(user_id)

    if not user_dir.exists():
        return None

    matches = list(user_dir.glob(f"*_{report_id}.pdf"))
    if not matches:
        matches = list(user_dir.glob(f"*{report_id}.pdf"))

    file_path = str(matches[0]) if matches else None

    return {
        "report_id": report_id,
        "user_id": user_id,
        "status": status_info["status"],
        "file_path": file_path,
        "download_url": f"/api/v1/reports/{report_id}/download" if status_info["status"] == "completed" else None,
    }


def generate_report(
    *,
    user_id: int,
    case_id: int,
    report_type: str = "comprehensive",
    include_remedies: bool = True,
    include_timeline: bool = True,
    format: str = "pdf",
    style: str = "formal",
    report_id: Optional[str] = None,
    watermark: Optional[str] = None,
    privacy_profile: Optional[str] = None,
) -> GeneratedReport:
    """Generate a single report and persist it to disk."""

    report_id = report_id or os.getenv("REPORT_ID", None) or datetime.now(timezone.utc).strftime(
        "%Y%m%d%H%M%S%f"
    )

    mime_type, ext = _get_format_meta(format)

    base_dir = _get_reports_base_dir()
    out_dir = base_dir / str(user_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    file_name = safe_filename(f"{case_id}_{report_type}_{report_id}{ext}")
    file_path = out_dir / file_name

    # Supported formats: pdf (Phase 1), csv/html/docx (planned)
    supported_formats = {"pdf"}
    if format and format.lower() not in supported_formats:
        raise ValueError(
            f"Unsupported report format '{format}'. "
            f"Currently supported: {', '.join(supported_formats)}. "
            "Additional formats (csv, html, docx) coming in Phase 2."
        )

    selected_profile = normalize_privacy_profile(privacy_profile)
    anon_data = generate_anonymized_case_data(case_id=int(case_id), profile_name=selected_profile)
    if anon_data:
        pdf_bytes = generate_anonymized_pdf(
            case_id=int(case_id),
            anon_id=str(anon_data.get("anonymized_id", "anon")),
            user_id=int(user_id),
            profile_name=selected_profile,
            anonymized_data=anon_data,
        )
    else:
        pdf_bytes = generate_case_pdf(user_id=int(user_id), case_id=int(case_id))
    if not pdf_bytes:
        raise RuntimeError("PDF generation returned empty content")

    file_path = out_dir / file_name
    _store_report(file_path, pdf_bytes, _storage_type)

    with SessionLocal() as db:
        record_audit_event(
            db,
            actor=f"user:{user_id}",
            actor_user_id=int(user_id),
            action="report_generated",
            resource=f"case:{case_id}",
            case_id=int(case_id),
            metadata={
                "report_type": report_type,
                "format": format,
                "privacy_profile": selected_profile,
            },
        )

    return GeneratedReport(
        report_id=str(report_id),
        format=report_format,
        file_path=file_path,
        file_name=file_name,
        mime_type=mime_type,
        file_size_bytes=len(pdf_bytes),
        storage_type=_storage_type,
    )

