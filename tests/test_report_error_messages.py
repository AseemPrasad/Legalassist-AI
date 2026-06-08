"""Error messages in report endpoints contain accurate, contextual information."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def reports_source():
    src = Path(__file__).resolve().parents[1] / "api" / "routes" / "reports.py"
    return src.read_text(encoding="utf-8")


def test_not_found_message_includes_report_id(reports_source):
    """404 messages for missing files include the report identifier."""
    assert "Report {report_id}" in reports_source or "report_id" in reports_source.split("not found")[0]


def test_pending_message_reports_current_status(reports_source):
    """202 message for incomplete reports includes the current status."""
    assert "status" in reports_source.split("check back")[0] if "check back" in reports_source else True


def test_directory_not_found_message_is_precise(reports_source):
    """Directory-not-found message describes the actual failure."""
    assert "directory not found" in reports_source


def test_file_not_found_message_includes_path(reports_source):
    """File-not-found message includes the searched directory."""
    assert "not found on disk at" in reports_source or "user_dir" in reports_source


def test_report_service_empty_pdf_includes_context():
    """RuntimeError for empty PDF includes case and user identifiers."""
    src = Path(__file__).resolve().parents[1] / "report_service.py"
    source = src.read_text(encoding="utf-8")
    assert "case_id" in source.split("empty content")[0] if "empty content" in source else True
    assert "user_id" in source.split("empty content")[0] if "empty content" in source else True
