"""Tests for the report format / report_type allowlist fix (#1571).

Verifies that:
- ReportGenerationRequest rejects invalid format values.
- ReportGenerationRequest rejects invalid report_type values.
- ReportGenerationRequest rejects invalid style values.
- Valid values are accepted.
- generate_report_task raises ValueError for unknown format/report_type.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-value-12345")
os.environ.setdefault("APP_ALLOWED_HOSTS", "localhost")

for _mod in ("streamlit", "pytesseract", "pdf2image"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

import pytest
from pydantic import ValidationError


# ---------------------------------------------------------------------------
# ReportGenerationRequest — Pydantic Literal validation
# ---------------------------------------------------------------------------

def test_valid_report_request_accepted():
    from api.models import ReportGenerationRequest
    req = ReportGenerationRequest(
        case_id="123",
        report_type="comprehensive",
        format="pdf",
        style="formal",
    )
    assert req.format == "pdf"
    assert req.report_type == "comprehensive"
    assert req.style == "formal"


def test_all_valid_formats_accepted():
    from api.models import ReportGenerationRequest
    for fmt in ("pdf", "docx", "html"):
        req = ReportGenerationRequest(case_id="1", format=fmt)
        assert req.format == fmt


def test_all_valid_report_types_accepted():
    from api.models import ReportGenerationRequest
    for rt in ("comprehensive", "summary", "legal_brief"):
        req = ReportGenerationRequest(case_id="1", report_type=rt)
        assert req.report_type == rt


def test_all_valid_styles_accepted():
    from api.models import ReportGenerationRequest
    for style in ("formal", "casual"):
        req = ReportGenerationRequest(case_id="1", style=style)
        assert req.style == style


def test_invalid_format_rejected():
    """A free-form format string must be rejected at the Pydantic layer."""
    from api.models import ReportGenerationRequest
    with pytest.raises(ValidationError) as exc_info:
        ReportGenerationRequest(case_id="1", format="../../evil")
    errors = exc_info.value.errors()
    assert any(e["loc"] == ("format",) for e in errors)


def test_path_traversal_format_rejected():
    from api.models import ReportGenerationRequest
    with pytest.raises(ValidationError):
        ReportGenerationRequest(case_id="1", format="../../../tmp/evil")


def test_invalid_report_type_rejected():
    from api.models import ReportGenerationRequest
    with pytest.raises(ValidationError) as exc_info:
        ReportGenerationRequest(case_id="1", report_type="malicious_type")
    errors = exc_info.value.errors()
    assert any(e["loc"] == ("report_type",) for e in errors)


def test_invalid_style_rejected():
    from api.models import ReportGenerationRequest
    with pytest.raises(ValidationError) as exc_info:
        ReportGenerationRequest(case_id="1", style="hacker")
    errors = exc_info.value.errors()
    assert any(e["loc"] == ("style",) for e in errors)


def test_empty_format_rejected():
    from api.models import ReportGenerationRequest
    with pytest.raises(ValidationError):
        ReportGenerationRequest(case_id="1", format="")


def test_html_injection_in_report_type_rejected():
    from api.models import ReportGenerationRequest
    with pytest.raises(ValidationError):
        ReportGenerationRequest(case_id="1", report_type="<script>alert(1)</script>")


# ---------------------------------------------------------------------------
# generate_report_task — worker-level defence-in-depth
# ---------------------------------------------------------------------------

def test_generate_report_task_rejects_invalid_format():
    """generate_report_task must raise ValueError for unknown format strings."""
    import inspect
    import celery_app as ca

    source = inspect.getsource(ca.generate_report_task)
    # The allowlist check must be present in the task source
    assert "_ALLOWED_FORMATS" in source, (
        "generate_report_task must validate format against an allowlist."
    )
    assert "_ALLOWED_REPORT_TYPES" in source, (
        "generate_report_task must validate report_type against an allowlist."
    )


def test_generate_report_task_allowlists_cover_all_valid_values():
    """The worker allowlists must cover all values accepted by the Pydantic model."""
    import ast
    import celery_app as ca
    import inspect

    source = inspect.getsource(ca.generate_report_task)
    tree = ast.parse(source)

    # Extract the set literals assigned to _ALLOWED_FORMATS and _ALLOWED_REPORT_TYPES
    allowed_formats = set()
    allowed_report_types = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if target.id == "_ALLOWED_FORMATS" and isinstance(node.value, ast.Set):
                        allowed_formats = {
                            elt.s for elt in node.value.elts
                            if isinstance(elt, ast.Constant)
                        }
                    elif target.id == "_ALLOWED_REPORT_TYPES" and isinstance(node.value, ast.Set):
                        allowed_report_types = {
                            elt.s for elt in node.value.elts
                            if isinstance(elt, ast.Constant)
                        }

    assert "pdf" in allowed_formats
    assert "docx" in allowed_formats
    assert "html" in allowed_formats
    assert "comprehensive" in allowed_report_types
    assert "summary" in allowed_report_types
    assert "legal_brief" in allowed_report_types
