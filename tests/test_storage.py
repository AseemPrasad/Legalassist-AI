"""Regression tests for secure attachment path resolution."""

from pathlib import Path

import pytest

import core.storage as storage


def test_get_attachment_path_keeps_file_inside_attachments_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "ATTACHMENTS_DIR", tmp_path)

    result = storage.get_attachment_path("case-note.pdf")

    assert result == tmp_path / "case-note.pdf"
    assert result.is_relative_to(tmp_path)


def test_get_attachment_path_strips_traversal_segments(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "ATTACHMENTS_DIR", tmp_path)

    result = storage.get_attachment_path("../../../../etc/passwd")

    assert result == tmp_path / "passwd"
    assert result.is_relative_to(tmp_path)


def test_get_attachment_path_rejects_empty_filename(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "ATTACHMENTS_DIR", tmp_path)

    with pytest.raises(ValueError, match="must include a filename"):
        storage.get_attachment_path("   ")
