"""Report generation service.

Phase 1 scope (approved):
- Real PDF generation wired to the existing branded generator in `pdf_exporter.py`.

Notes:
- For now, the service writes generated files to local disk and returns the
  absolute path. Celery returns `report_id` and metadata so API can stream
  the file.
- DOCX/XLSX/batch/signatures will be added in later phases.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pdf_exporter import generate_case_pdf
from docx_exporter import generate_case_docx


@dataclass(frozen=True)
class GeneratedReport:
    report_id: str
    format: str
    file_path: Path
    file_name: str
    mime_type: str
    file_size_bytes: int


def _safe_filename(name: str) -> str:
    name = name or "report"
    # Replace path separators and other unsafe chars
    name = re.sub(r"[\\/:*?\"<>|]", "_", name)
    name = name.strip(" .")
    return name[:180] if len(name) > 180 else name


def _get_reports_base_dir() -> Path:
    # Keep it in project workspace so it works in local dev without object storage.
    base = Path(os.getenv("REPORTS_OUTPUT_DIR", "./.report_outputs")).resolve()
    base.mkdir(parents=True, exist_ok=True)
    return base


def _get_format_meta(format: str) -> tuple[str, str]:
    fmt = (format or "pdf").lower()
    if fmt == "pdf":
        return "application/pdf", ".pdf"
    if fmt == "docx":
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document", ".docx"
    raise ValueError(f"Unsupported format: {format}")


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
) -> GeneratedReport:
    """Generate a single report and persist it to disk."""

    report_id = report_id or os.getenv("REPORT_ID", None) or datetime.now(timezone.utc).strftime(
        "%Y%m%d%H%M%S%f"
    )

    mime_type, ext = _get_format_meta(format)

    base_dir = _get_reports_base_dir()
    out_dir = base_dir / str(user_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    file_name = _safe_filename(f"{case_id}_{report_type}_{report_id}{ext}")
    file_path = out_dir / file_name

    fmt = (format or "pdf").lower()
    if fmt == "pdf":
        report_bytes = generate_case_pdf(user_id=int(user_id), case_id=int(case_id))
        report_format = "pdf"
    elif fmt == "docx":
        report_bytes = generate_case_docx(case_id=int(case_id), user_id=int(user_id))
        report_format = "docx"
    else:
        raise ValueError(f"Unsupported format: {format}")

    if not report_bytes:
        raise RuntimeError(f"{report_format.upper()} generation returned empty content")

    file_path.write_bytes(report_bytes)

    return GeneratedReport(
        report_id=str(report_id),
        format=report_format,
        file_path=file_path,
        file_name=file_name,
        mime_type=mime_type,
        file_size_bytes=len(pdf_bytes),
    )

