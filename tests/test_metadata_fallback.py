"""Unit tests for partial metadata fallback recovery in extract_case_document_metadata."""

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Isolated import: load core.document_metadata without triggering the heavy
# core/__init__.py that pulls openai, fastapi, etc.
# ---------------------------------------------------------------------------
_REPO = Path(__file__).parent.parent
_MODULE_PATH = _REPO / "core" / "document_metadata.py"

# Provide a minimal stub for `core` so the module-level `import core as core_text_utils` resolves.
_core_stub = types.ModuleType("core")
_core_stub.extract_text_with_diagnostics = None
sys.modules["core"] = _core_stub

spec = importlib.util.spec_from_file_location("core.document_metadata", _MODULE_PATH)
_dm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_dm)

extract_case_document_metadata = _dm.extract_case_document_metadata
_extract_party_candidates = _dm._extract_party_candidates
_extract_date_candidates = _dm._extract_date_candidates
_extract_claim_candidates = _dm._extract_claim_candidates
_extract_statute_candidates = _dm._extract_statute_candidates




SAMPLE_TEXT = """
IN THE HIGH COURT OF DELHI AT NEW DELHI
Civil Writ Petition No. 1234 of 2024

Ramesh Kumar Verma                      ...Petitioner
    v.
State of Delhi                          ...Respondent

Date of Hearing: 15 March 2024
Date of Judgment: 22 March 2024

The petitioner seeks a writ of mandamus under Section 226 of the Constitution of India.
The petitioner claims compensation under Section 300 IPC.
The respondent denies the allegations and seeks dismissal of the petition.
The Court finds sufficient cause to grant interim relief by way of injunction.
"""


def test_full_extraction_normal():
    """Normal text: all fields populated and confidence > 0."""
    result = extract_case_document_metadata(SAMPLE_TEXT, filename="case_001.pdf")

    assert result["parties"], "Expected at least one party to be extracted"
    assert result["dates"], "Expected at least one date to be extracted"
    assert result["confidence"]["parties"] > 0.0
    assert result["confidence"]["dates"] > 0.0
    assert result["title_hint"] is not None


def test_empty_text_returns_safe_defaults():
    """Empty text must never raise and must return a well-formed dict."""
    result = extract_case_document_metadata("")

    assert result["parties"] == []
    assert result["dates"] == []
    assert result["claims"] == []
    assert result["statutes"] == []
    assert result["title_hint"] is None
    assert result["confidence"]["parties"] == 0.0


def test_filename_fallback_when_no_parties():
    """When no parties are found, title_hint falls back to the filename stem."""
    result = extract_case_document_metadata(
        "No party information found here.", filename="judgment_2024.pdf"
    )
    assert result["title_hint"] == "judgment_2024"


def test_partial_recovery_when_party_extraction_raises():
    """If party extraction raises, remaining fields are still populated."""
    original = _dm._extract_party_candidates
    try:
        _dm._extract_party_candidates = lambda text: (_ for _ in ()).throw(RuntimeError("simulated OCR corruption"))
        result = extract_case_document_metadata(SAMPLE_TEXT, filename="damaged.pdf")
    finally:
        _dm._extract_party_candidates = original

    # Parties failed but other fields should still be populated
    assert result["parties"] == []
    assert result["confidence"]["parties"] == 0.0
    # Dates and statutes should still be extracted from the sample text
    assert result["dates"] or result["statutes"]


def test_partial_recovery_when_dates_extraction_raises():
    """If date extraction raises, parties and other fields remain accessible."""
    original = _dm._extract_date_candidates
    try:
        _dm._extract_date_candidates = lambda text: (_ for _ in ()).throw(ValueError("regex timeout simulation"))
        result = extract_case_document_metadata(SAMPLE_TEXT)
    finally:
        _dm._extract_date_candidates = original

    assert result["dates"] == []
    assert result["confidence"]["dates"] == 0.0
    # Parties should still be extracted
    assert result["parties"] is not None  # list may be empty or populated


def test_statute_extraction_populated():
    """Statutes matching IPC / Section patterns are captured."""
    result = extract_case_document_metadata(SAMPLE_TEXT)
    assert any("IPC" in s or "Section" in s for s in result["statutes"]), (
        f"Expected IPC or Section in statutes, got: {result['statutes']}"
    )


def test_confidence_shape():
    """Confidence dict always contains all four keys."""
    result = extract_case_document_metadata("")
    assert set(result["confidence"].keys()) == {"parties", "dates", "claims", "statutes"}
