"""
Comprehensive tests for checkpoint loading and deduplication functionality.

Tests cover:
- Checkpoint corruption detection and tolerance
- Deduplication of duplicate records
- Edge cases (empty files, single records, etc.)
"""

import json
import tempfile
from pathlib import Path
from typing import Dict, List

import pytest

from cli_checkpoint import (
    load_checkpoint,
    dedupe_latest_by_file,
    collect_completed_files,
)
from cli_client import CLIError


class TestLoadCheckpoint:
    """Tests for load_checkpoint function."""

    def test_load_checkpoint_nonexistent_file(self):
        """Nonexistent checkpoint files should return empty list."""
        nonexistent = Path("/tmp/does_not_exist_12345.jsonl")
        result = load_checkpoint(nonexistent)
        assert result == []

    def test_load_checkpoint_empty_file(self):
        """Empty checkpoint file should return empty list."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            result = load_checkpoint(temp_path)
            assert result == []
        finally:
            temp_path.unlink()

    def test_load_checkpoint_single_valid_record(self):
        """Single valid JSON line should be loaded successfully."""
        record = {"file_path": "/tmp/test.pdf", "status": "success", "summary": "Test"}
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps(record) + "\n")
            temp_path = Path(f.name)
        
        try:
            result = load_checkpoint(temp_path)
            assert len(result) == 1
            assert result[0]["file_path"] == "/tmp/test.pdf"
            assert result[0]["status"] == "success"
        finally:
            temp_path.unlink()

    def test_load_checkpoint_multiple_valid_records(self):
        """Multiple valid JSON lines should all be loaded."""
        records = [
            {"file_path": "/tmp/test1.pdf", "status": "success"},
            {"file_path": "/tmp/test2.pdf", "status": "error", "error": "failed"},
            {"file_path": "/tmp/test3.pdf", "status": "success"},
        ]
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
            temp_path = Path(f.name)
        
        try:
            result = load_checkpoint(temp_path)
            assert len(result) == 3
            assert result[0]["file_path"] == "/tmp/test1.pdf"
            assert result[1]["file_path"] == "/tmp/test2.pdf"
            assert result[2]["file_path"] == "/tmp/test3.pdf"
        finally:
            temp_path.unlink()

    def test_load_checkpoint_with_empty_lines(self):
        """Empty lines in checkpoint should be skipped gracefully."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"file_path": "/tmp/test1.pdf", "status": "success"}) + "\n")
            f.write("\n")  # Empty line
            f.write("   \n")  # Whitespace only
            f.write(json.dumps({"file_path": "/tmp/test2.pdf", "status": "success"}) + "\n")
            temp_path = Path(f.name)
        
        try:
            result = load_checkpoint(temp_path)
            assert len(result) == 2
            assert result[0]["file_path"] == "/tmp/test1.pdf"
            assert result[1]["file_path"] == "/tmp/test2.pdf"
        finally:
            temp_path.unlink()

    def test_load_checkpoint_single_corrupted_line_below_threshold(self):
        """Single corrupted line should fail when corruption > threshold (1/2 = 50% > 10%)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"file_path": "/tmp/test1.pdf", "status": "success"}) + "\n")
            f.write("this is not valid json\n")  # Invalid
            temp_path = Path(f.name)
        
        try:
            # 1 out of 2 lines = 50% corruption, exceeds 10% threshold
            with pytest.raises(CLIError) as exc_info:
                load_checkpoint(temp_path, corruption_threshold=0.1)
            
            assert "Checkpoint file corruption rate" in str(exc_info.value)
            assert "50.0%" in str(exc_info.value)
        finally:
            temp_path.unlink()

    def test_load_checkpoint_corruption_rate_5_percent(self):
        """5% corruption (1/25 lines) should be tolerated with 10% threshold."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            # Write 20 valid records
            for i in range(20):
                f.write(json.dumps({"file_path": f"/tmp/test{i}.pdf", "status": "success"}) + "\n")
            # Inject one corrupted line
            f.write("corrupted\n")
            # Continue with more valid records
            for i in range(20, 24):
                f.write(json.dumps({"file_path": f"/tmp/test{i}.pdf", "status": "success"}) + "\n")
            temp_path = Path(f.name)
        
        try:
            # 1 out of 25 lines = 4% corruption, below 10% threshold
            result = load_checkpoint(temp_path, corruption_threshold=0.1)
            assert len(result) == 24  # 20 + 4 valid records (1 skipped)
            assert result[0]["file_path"] == "/tmp/test0.pdf"
        finally:
            temp_path.unlink()

    def test_load_checkpoint_corruption_rate_15_percent(self):
        """23.1% corruption (3/13 lines) should fail with 10% threshold."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            # Write 10 valid records
            for i in range(10):
                f.write(json.dumps({"file_path": f"/tmp/test{i}.pdf", "status": "success"}) + "\n")
            # Inject 3 corrupted lines (3/13 = 23.1%)
            f.write("corrupted1\n")
            f.write("corrupted2\n")
            f.write("corrupted3\n")
            temp_path = Path(f.name)
        
        try:
            with pytest.raises(CLIError) as exc_info:
                load_checkpoint(temp_path, corruption_threshold=0.1)
            
            assert "Checkpoint file corruption rate" in str(exc_info.value)
        finally:
            temp_path.unlink()

    def test_load_checkpoint_high_corruption_threshold(self):
        """High corruption threshold should tolerate more corrupted lines."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            # Write 5 valid records
            for i in range(5):
                f.write(json.dumps({"file_path": f"/tmp/test{i}.pdf", "status": "success"}) + "\n")
            # Inject 4 corrupted lines (4/9 = 44.4%)
            f.write("corrupted1\n")
            f.write("corrupted2\n")
            f.write("corrupted3\n")
            f.write("corrupted4\n")
            temp_path = Path(f.name)
        
        try:
            # With 50% threshold, 44.4% corruption should pass
            result = load_checkpoint(temp_path, corruption_threshold=0.5)
            assert len(result) == 5  # Only valid records
        finally:
            temp_path.unlink()

    def test_load_checkpoint_zero_corruption_threshold(self):
        """Zero corruption threshold should reject any corrupted lines."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"file_path": "/tmp/test1.pdf", "status": "success"}) + "\n")
            f.write("corrupted\n")
            temp_path = Path(f.name)
        
        try:
            with pytest.raises(CLIError) as exc_info:
                load_checkpoint(temp_path, corruption_threshold=0.0)
            
            assert "Checkpoint file corruption rate" in str(exc_info.value)
        finally:
            temp_path.unlink()

    def test_load_checkpoint_100_percent_corruption_threshold(self):
        """100% corruption threshold should accept even fully corrupted files."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write("corrupted1\n")
            f.write("corrupted2\n")
            f.write("corrupted3\n")
            temp_path = Path(f.name)
        
        try:
            # All lines corrupted, but threshold is 100%
            result = load_checkpoint(temp_path, corruption_threshold=1.0)
            assert len(result) == 0  # No valid records recovered
        finally:
            temp_path.unlink()

    def test_load_checkpoint_various_json_errors(self):
        """Different JSON parsing errors should all be caught."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"status": "success"}) + "\n")
            f.write("{incomplete json\n")
            f.write('{"key": undefined}\n')
            f.write("just text\n")
            f.write(json.dumps({"status": "success"}) + "\n")
            temp_path = Path(f.name)
        
        try:
            # 2 valid + 3 corrupted = 60% corruption, exceeds 10% threshold
            with pytest.raises(CLIError):
                load_checkpoint(temp_path, corruption_threshold=0.1)
        finally:
            temp_path.unlink()


class TestDedupeLatestByFile:
    """Tests for dedupe_latest_by_file function."""

    def test_dedupe_empty_list(self):
        """Empty list should return empty list."""
        result = dedupe_latest_by_file([])
        assert result == []

    def test_dedupe_single_record(self):
        """Single record should return unchanged."""
        record = {"file_path": "/tmp/test.pdf", "status": "success"}
        result = dedupe_latest_by_file([record])
        assert len(result) == 1
        assert result[0] == record

    def test_dedupe_no_duplicates(self):
        """Records with different file_paths should all be kept."""
        records = [
            {"file_path": "/tmp/test1.pdf", "status": "success"},
            {"file_path": "/tmp/test2.pdf", "status": "error"},
            {"file_path": "/tmp/test3.pdf", "status": "success"},
        ]
        result = dedupe_latest_by_file(records)
        assert len(result) == 3
        paths = {r["file_path"] for r in result}
        assert paths == {"/tmp/test1.pdf", "/tmp/test2.pdf", "/tmp/test3.pdf"}

    def test_dedupe_exact_duplicates(self):
        """Exact duplicate records should keep only the latest."""
        record = {"file_path": "/tmp/test.pdf", "status": "success", "summary": "test"}
        records = [record, record, record]
        result = dedupe_latest_by_file(records)
        assert len(result) == 1
        assert result[0] == record

    def test_dedupe_same_file_different_statuses(self):
        """Multiple entries for same file should keep the latest (last in list)."""
        records = [
            {"file_path": "/tmp/test.pdf", "status": "pending", "iteration": 1},
            {"file_path": "/tmp/test.pdf", "status": "processing", "iteration": 2},
            {"file_path": "/tmp/test.pdf", "status": "success", "iteration": 3, "summary": "final"},
        ]
        result = dedupe_latest_by_file(records)
        assert len(result) == 1
        assert result[0]["status"] == "success"
        assert result[0]["iteration"] == 3
        assert result[0]["summary"] == "final"

    def test_dedupe_interleaved_duplicates(self):
        """Duplicates interspersed with other records should keep latest per file."""
        records = [
            {"file_path": "/tmp/a.pdf", "version": 1},
            {"file_path": "/tmp/b.pdf", "version": 1},
            {"file_path": "/tmp/a.pdf", "version": 2},
            {"file_path": "/tmp/c.pdf", "version": 1},
            {"file_path": "/tmp/b.pdf", "version": 2},
            {"file_path": "/tmp/a.pdf", "version": 3},
        ]
        result = dedupe_latest_by_file(records)
        assert len(result) == 3
        
        # Create dict for easier verification
        by_path = {r["file_path"]: r for r in result}
        assert by_path["/tmp/a.pdf"]["version"] == 3
        assert by_path["/tmp/b.pdf"]["version"] == 2
        assert by_path["/tmp/c.pdf"]["version"] == 1

    def test_dedupe_preserves_record_structure(self):
        """Deduplicated records should preserve all fields."""
        records = [
            {
                "file_path": "/tmp/test.pdf",
                "status": "success",
                "summary": "complex document",
                "tokens": 1000,
                "cost": 0.05,
                "nested": {"key": "value"},
            },
        ]
        result = dedupe_latest_by_file(records)
        assert len(result) == 1
        assert result[0]["nested"] == {"key": "value"}
        assert result[0]["cost"] == 0.05

    def test_dedupe_missing_file_path_key(self):
        """Records without file_path key should be excluded."""
        records = [
            {"file_path": "/tmp/test1.pdf", "status": "success"},
            {"status": "error"},  # No file_path
            {"file_path": "/tmp/test2.pdf", "status": "success"},
            {},  # No file_path
        ]
        result = dedupe_latest_by_file(records)
        assert len(result) == 2
        paths = {r["file_path"] for r in result}
        assert paths == {"/tmp/test1.pdf", "/tmp/test2.pdf"}

    def test_dedupe_none_file_path_value(self):
        """Records with None as file_path should be excluded."""
        records = [
            {"file_path": "/tmp/test1.pdf", "status": "success"},
            {"file_path": None, "status": "error"},
            {"file_path": "/tmp/test2.pdf", "status": "success"},
        ]
        result = dedupe_latest_by_file(records)
        assert len(result) == 2
        paths = {r["file_path"] for r in result}
        assert paths == {"/tmp/test1.pdf", "/tmp/test2.pdf"}

    def test_dedupe_empty_string_file_path(self):
        """Records with empty string file_path should be excluded."""
        records = [
            {"file_path": "/tmp/test1.pdf", "status": "success"},
            {"file_path": "", "status": "error"},
            {"file_path": "/tmp/test2.pdf", "status": "success"},
        ]
        result = dedupe_latest_by_file(records)
        assert len(result) == 2


class TestCollectCompletedFiles:
    """Tests for collect_completed_files function."""

    def test_collect_completed_empty_list(self):
        """Empty records list should return empty set."""
        result = collect_completed_files([])
        assert result == set()

    def test_collect_completed_success_status(self):
        """Records with status='success' should be collected."""
        records = [
            {"file_path": "/tmp/test1.pdf", "status": "success"},
            {"file_path": "/tmp/test2.pdf", "status": "error"},
            {"file_path": "/tmp/test3.pdf", "status": "success"},
        ]
        result = collect_completed_files(records, status_filter="success")
        assert result == {"/tmp/test1.pdf", "/tmp/test3.pdf"}

    def test_collect_completed_error_status(self):
        """Records with status='error' should be collected when filtering for errors."""
        records = [
            {"file_path": "/tmp/test1.pdf", "status": "success"},
            {"file_path": "/tmp/test2.pdf", "status": "error"},
            {"file_path": "/tmp/test3.pdf", "status": "error"},
        ]
        result = collect_completed_files(records, status_filter="error")
        assert result == {"/tmp/test2.pdf", "/tmp/test3.pdf"}

    def test_collect_completed_missing_status_key(self):
        """Records without status key should be excluded."""
        records = [
            {"file_path": "/tmp/test1.pdf", "status": "success"},
            {"file_path": "/tmp/test2.pdf"},  # No status
            {"file_path": "/tmp/test3.pdf", "status": "success"},
        ]
        result = collect_completed_files(records, status_filter="success")
        assert result == {"/tmp/test1.pdf", "/tmp/test3.pdf"}

    def test_collect_completed_missing_file_path_key(self):
        """Records without file_path should be excluded."""
        records = [
            {"file_path": "/tmp/test1.pdf", "status": "success"},
            {"status": "success"},  # No file_path
            {"file_path": "/tmp/test2.pdf", "status": "success"},
        ]
        result = collect_completed_files(records, status_filter="success")
        assert result == {"/tmp/test1.pdf", "/tmp/test2.pdf"}

    def test_collect_completed_case_sensitive(self):
        """Status filtering should be case-sensitive."""
        records = [
            {"file_path": "/tmp/test1.pdf", "status": "success"},
            {"file_path": "/tmp/test2.pdf", "status": "Success"},  # Different case
            {"file_path": "/tmp/test3.pdf", "status": "SUCCESS"},  # Different case
        ]
        result = collect_completed_files(records, status_filter="success")
        assert result == {"/tmp/test1.pdf"}  # Only exact match

    def test_collect_completed_custom_status_filter(self):
        """Custom status filter values should work."""
        records = [
            {"file_path": "/tmp/test1.pdf", "status": "pending"},
            {"file_path": "/tmp/test2.pdf", "status": "processing"},
            {"file_path": "/tmp/test3.pdf", "status": "pending"},
        ]
        result = collect_completed_files(records, status_filter="pending")
        assert result == {"/tmp/test1.pdf", "/tmp/test3.pdf"}

    def test_collect_completed_returns_set(self):
        """Result should be a set (unordered, unique)."""
        records = [
            {"file_path": "/tmp/test1.pdf", "status": "success"},
            {"file_path": "/tmp/test1.pdf", "status": "success"},  # Duplicate
            {"file_path": "/tmp/test2.pdf", "status": "success"},
        ]
        result = collect_completed_files(records, status_filter="success")
        assert isinstance(result, set)
        assert result == {"/tmp/test1.pdf", "/tmp/test2.pdf"}


class TestIntegrationCheckpointAndDedupe:
    """Integration tests combining checkpoint loading and deduplication."""

    def test_integration_checkpoint_with_dedupe(self):
        """Load checkpoint and deduplicate in sequence."""
        records = [
            {"file_path": "/tmp/test1.pdf", "status": "success", "version": 1},
            {"file_path": "/tmp/test2.pdf", "status": "success", "version": 1},
            {"file_path": "/tmp/test1.pdf", "status": "success", "version": 2},
        ]
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
            temp_path = Path(f.name)
        
        try:
            # Load checkpoint
            loaded = load_checkpoint(temp_path, corruption_threshold=0.1)
            assert len(loaded) == 3
            
            # Deduplicate
            deduped = dedupe_latest_by_file(loaded)
            assert len(deduped) == 2
            
            # Verify latest versions kept
            by_path = {r["file_path"]: r for r in deduped}
            assert by_path["/tmp/test1.pdf"]["version"] == 2
            assert by_path["/tmp/test2.pdf"]["version"] == 1
        finally:
            temp_path.unlink()

    def test_integration_checkpoint_dedupe_collect(self):
        """Full workflow: load, dedupe, collect completed."""
        records = [
            {"file_path": "/tmp/test1.pdf", "status": "success"},
            {"file_path": "/tmp/test2.pdf", "status": "error"},
            {"file_path": "/tmp/test1.pdf", "status": "success"},  # Duplicate of test1
            {"file_path": "/tmp/test3.pdf", "status": "success"},
        ]
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
            temp_path = Path(f.name)
        
        try:
            loaded = load_checkpoint(temp_path)
            deduped = dedupe_latest_by_file(loaded)
            completed = collect_completed_files(deduped, status_filter="success")
            
            assert len(deduped) == 3
            assert completed == {"/tmp/test1.pdf", "/tmp/test3.pdf"}
        finally:
            temp_path.unlink()
