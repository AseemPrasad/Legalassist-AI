"""Tests for meaningful search query validation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from api.query_validation import meaningful_search_query


def _call_validator(query: str) -> str:
    """Helper to call meaningful_search_query as FastAPI would."""
    return meaningful_search_query(query=query)


class TestMeaningfulSearchQuery:
    def test_valid_query_passes(self):
        result = _call_validator("landlord breach of contract")
        assert result == "landlord breach of contract"

    def test_valid_query_with_stop_words_passes(self):
        result = _call_validator("the tenant breached the lease agreement")
        assert result == "the tenant breached the lease agreement"

    def test_too_short_rejected(self):
        with pytest.raises(HTTPException) as exc:
            _call_validator("ab")
        assert exc.value.status_code == 422

    def test_single_word_rejected(self):
        with pytest.raises(HTTPException) as exc:
            _call_validator("lawsuits")
        assert exc.value.status_code == 422
        assert "3 words" in str(exc.value.detail).lower()

    def test_two_words_rejected(self):
        with pytest.raises(HTTPException) as exc:
            _call_validator("breach contract")
        assert exc.value.status_code == 422

    def test_repeated_words_rejected(self):
        with pytest.raises(HTTPException) as exc:
            _call_validator("the the the the the the")
        assert exc.value.status_code == 422
        assert "unique" in str(exc.value.detail).lower()

    def test_all_stop_words_rejected(self):
        with pytest.raises(HTTPException) as exc:
            _call_validator("the and of to for in")
        assert exc.value.status_code == 422
        assert "meaningful" in str(exc.value.detail).lower() or "common" in str(exc.value.detail).lower()

    def test_mostly_stop_words_rejected(self):
        with pytest.raises(HTTPException) as exc:
            _call_validator("the of and to for a an is law")
        assert exc.value.status_code == 422
        assert "too many" in str(exc.value.detail).lower()

    def test_query_with_punctuation(self):
        result = _call_validator("motion to dismiss for lack of jurisdiction")
        assert result == "motion to dismiss for lack of jurisdiction"

    def test_query_with_extra_spaces(self):
        result = _call_validator("  summary   judgment   denied   ")
        assert result == "  summary   judgment   denied   "
