"""
Checkpoint and deduplication logic for batch processing.

This module handles:
- Loading checkpoint files with corruption tolerance
- Deduplicating records by file path (keeping latest)
- Tracking progress across batch runs
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import structlog

from cli_client import CLIError

LOGGER = structlog.get_logger(__name__)


def load_checkpoint(
    checkpoint_file: Path,
    corruption_threshold: float = 0.1
) -> List[Dict[str, object]]:
    """
    Load checkpoint records from a JSONL file with corruption tolerance.
    
    Checkpoint files are JSONL format (one JSON object per line) and may
    have partial corruption if the process crashed mid-write.
    
    Strategy:
    - Skip individual corrupted lines
    - Fail if corruption rate exceeds threshold
    - Log details of skipped lines
    - Return all successfully parsed records
    
    Args:
        checkpoint_file: Path to the checkpoint file
        corruption_threshold: Maximum fraction of lines that can be corrupted (0.0-1.0)
                             Defaults to 0.1 (10%)
    
    Returns:
        List of valid checkpoint records (dicts)
    
    Raises:
        CLIError: If corruption rate exceeds threshold
    """
    if not checkpoint_file.exists():
        LOGGER.debug("checkpoint_not_found", path=str(checkpoint_file))
        return []

    records: List[Dict[str, object]] = []
    skipped_lines: List[Tuple[int, str]] = []
    line_num = 0
    
    with checkpoint_file.open("r", encoding="utf-8") as f:
        for line in f:
            line_num += 1
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                skipped_lines.append((line_num, str(e)))
                LOGGER.warning(
                    "checkpoint_line_corrupted",
                    line_number=line_num,
                    error=str(e),
                    line_preview=line[:100],
                )
    
    # Check if corruption exceeds threshold
    total_lines = line_num
    if total_lines > 0 and skipped_lines:
        corruption_rate = len(skipped_lines) / total_lines
        if corruption_rate > corruption_threshold:
            error_msg = (
                f"Checkpoint file corruption rate {corruption_rate:.1%} exceeds threshold {corruption_threshold:.1%}. "
                f"Skipped {len(skipped_lines)} out of {total_lines} lines. "
                f"First corrupted line: {skipped_lines[0][0]} ({skipped_lines[0][1]})"
            )
            LOGGER.error(
                "checkpoint_corruption_threshold_exceeded",
                corruption_rate=corruption_rate,
                skipped_count=len(skipped_lines)
            )
            raise CLIError(error_msg)
        elif skipped_lines:
            LOGGER.info(
                "checkpoint_partially_corrupted",
                skipped_count=len(skipped_lines),
                total_lines=total_lines,
                corruption_rate=f"{corruption_rate:.1%}",
                recovered_records=len(records),
            )
    
    LOGGER.debug("checkpoint_loaded", records_loaded=len(records), file=str(checkpoint_file))
    return records


def dedupe_latest_by_file(records: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """
    Deduplicate records keeping only the latest entry for each file.
    
    When a batch is resumed, the same file might be processed multiple times
    (if the run was interrupted). This function keeps only the newest result
    for each file (determined by iteration order, which is insertion order).
    
    Algorithm:
    - Iterate through records in order
    - Track the latest record for each file_path
    - Return the latest records as a list
    
    Args:
        records: List of result records
        
    Returns:
        List of deduplicated records (latest for each file_path)
        
    Example:
        >>> records = [
        ...     {"file_path": "/a.pdf", "status": "success"},
        ...     {"file_path": "/b.pdf", "status": "error"},
        ...     {"file_path": "/a.pdf", "status": "success"},  # New result for /a.pdf
        ... ]
        >>> dedupe_latest_by_file(records)
        [{"file_path": "/b.pdf", "status": "error"}, {"file_path": "/a.pdf", "status": "success"}]
    """
    latest: Dict[str, Dict[str, object]] = {}
    for rec in records:
        file_path = rec.get("file_path")
        # Exclude records where file_path is missing, None, or empty string
        if file_path and isinstance(file_path, str) and file_path.strip():
            latest[file_path] = rec
    return list(latest.values())


def collect_completed_files(
    records: List[Dict[str, object]],
    status_filter: str = "success"
) -> set:
    """
    Extract the set of completed file paths from checkpoint records.
    
    Args:
        records: List of checkpoint records
        status_filter: Only include records with this status (default: 'success')
        
    Returns:
        Set of completed file_path strings (as resolved absolute paths)
    """
    done_files = set()
    for rec in records:
        if rec.get("status") == status_filter and rec.get("file_path"):
            done_files.add(str(rec.get("file_path")))
    return done_files
