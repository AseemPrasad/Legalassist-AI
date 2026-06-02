"""
Result export and formatting for CLI.

This module handles:
- Exporting results to CSV and/or JSON formats
- Collecting PDF files from directories
- Formatting cost summaries
- Consistent field ordering
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import structlog

from cli_checkpoint import dedupe_latest_by_file

LOGGER = structlog.get_logger(__name__)

# Standard fieldnames for CSV export (ensures consistent column order)
DEFAULT_FIELDNAMES = [
    "file_name",
    "file_path",
    "status",
    "error",
    "language",
    "summary",
    "what_happened",
    "can_appeal",
    "appeal_days",
    "appeal_court",
    "cost_estimate",
    "first_action",
    "deadline",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "api_cost_usd",
    "duration_seconds",
    "processed_at",
    "extraction_method",
    "ocr_enabled",
    "ocr_used",
    "extraction_confidence",
]


def export_results(
    records: List[Dict[str, object]],
    output_path: Path,
    export_format: str = "both"
) -> Tuple[Path, Path]:
    """
    Export processing results to CSV and/or JSON formats.
    
    Features:
    - Deduplicates records by file (latest only)
    - Sorts results by filename for deterministic output
    - Supports CSV, JSON, or both formats
    - Creates parent directories if needed
    - Handles field ordering in CSV
    
    Args:
        records: List of result records to export
        output_path: Base path for output (used to determine .csv/.json paths)
        export_format: Format(s) to export ('csv', 'json', or 'both')
        
    Returns:
        Tuple of (csv_path, json_path) - only the requested formats are written
        
    Raises:
        IOError: If file writing fails
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stem = output_path.with_suffix("")

    csv_path = stem.with_suffix(".csv")
    json_path = stem.with_suffix(".json")

    # Deduplicate keeping only latest per file, then sort by filename
    ordered = dedupe_latest_by_file(records)
    ordered.sort(key=lambda x: str(x.get("file_name", "")))

    if export_format in {"csv", "both"}:
        # Use keys from first record, fallback to default fieldnames if empty
        if ordered:
            fieldnames = list(ordered[0].keys())
        else:
            fieldnames = DEFAULT_FIELDNAMES
            
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in ordered:
                writer.writerow(row)
        
        LOGGER.debug("csv_exported", path=str(csv_path), records=len(ordered))

    if export_format in {"json", "both"}:
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(ordered, f, ensure_ascii=False, indent=2)
        
        LOGGER.debug("json_exported", path=str(json_path), records=len(ordered))

    return csv_path, json_path


def collect_pdf_files(
    folder: Path,
    recursive: bool = False
) -> List[Path]:
    """
    Collect all PDF files from a directory.
    
    Args:
        folder: Directory to search
        recursive: If True, search subdirectories recursively
        
    Returns:
        Sorted list of PDF file paths
    """
    pattern = "**/*.pdf" if recursive else "*.pdf"
    pdf_files = sorted([p for p in folder.glob(pattern) if p.is_file()])
    LOGGER.debug("pdf_files_collected", folder=str(folder), count=len(pdf_files), recursive=recursive)
    return pdf_files


def print_cost_summary(snapshot: Dict[str, float]) -> None:
    """
    Print a formatted cost summary to stdout.
    
    Args:
        snapshot: Dictionary with cost metrics
    """
    parts = [f"batch_cost_summary"]
    parts.extend(f"{k}={v}" for k, v in snapshot.items())
    print(" ".join(parts))
