"""
Simple file storage manager for user data exports.
Saves exported files to local directory with metadata.
"""

import re
import uuid
from pathlib import Path
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import Optional
import structlog

from config import Config

_ALLOWED_EXPORT_FORMATS = frozenset({"csv", "json", "pdf", "xlsx"})

logger = structlog.get_logger(__name__)


@dataclass
class ExportFile:
    """Metadata for an exported file"""
    export_id: str
    file_path: str
    file_size_bytes: int
    created_at: datetime
    expires_at: datetime


def save_export_file(
    user_id: str,
    file_bytes: bytes,
    format: str,
    export_id: Optional[str] = None
) -> ExportFile:
    """
    Save export file to local storage.
    
    Args:
        user_id: User ID (used for organizing files)
        file_bytes: File content as bytes
        format: File format (csv, json, etc.)
        export_id: Optional custom export ID (auto-generated if not provided)
        
    Returns:
        ExportFile: Metadata including file path and expiry time
        
    Raises:
        RuntimeError: If file cannot be saved
    """
    try:
        if not re.match(r"^\d+$", str(user_id)):
            raise ValueError(f"Invalid user_id: {user_id!r}")

        clean_format = str(format).strip().lower()
        if clean_format not in _ALLOWED_EXPORT_FORMATS:
            raise ValueError(f"Unsupported export format: {format!r}")

        export_id = export_id or str(uuid.uuid4())
        created_at = datetime.now(timezone.utc)
        expires_at = created_at + timedelta(hours=getattr(Config, "EXPORT_FILE_EXPIRY_HOURS", 24))
        
        max_bytes = getattr(Config, "EXPORT_MAX_SIZE_BYTES", 100 * 1024 * 1024)
        if len(file_bytes) > max_bytes:
            raise ValueError(f"Export file size {len(file_bytes)} exceeds maximum {max_bytes}")

        base_dir = Path(getattr(Config, "EXPORTS_DIR", "./exports")).resolve()
        user_dir = base_dir / str(user_id)
        user_dir.mkdir(parents=True, exist_ok=True)
        
        file_name = f"export_{export_id}.{clean_format}"
        file_path = (user_dir / file_name).resolve()

        if not str(file_path).startswith(str(base_dir)):
            raise ValueError(f"File path escapes export directory: {file_path}")

        if file_path.exists():
            raise ValueError(f"Export file already exists: {file_path}")

        file_path.write_bytes(file_bytes)

        if file_path.is_symlink():
            file_path.unlink()
            raise ValueError("Symlink detected after write — rejecting export")
        
        logger.info(
            "Export file saved",
            export_id=export_id,
            user_id=user_id,
            file_size=len(file_bytes)
        )
        
        return ExportFile(
            export_id=export_id,
            file_path=str(file_path),
            file_size_bytes=len(file_bytes),
            created_at=created_at,
            expires_at=expires_at
        )
        
    except Exception as e:
        logger.error(
            "Failed to save export file",
            export_id=export_id,
            user_id=user_id,
            error=str(e)
        )
        raise RuntimeError(f"Export storage failed: {str(e)}")


def cleanup_expired_exports(max_age_hours: Optional[int] = None) -> int:
    """
    Remove export files whose expiry time has passed.

    Args:
        max_age_hours: Override expiry threshold (defaults to Config or 24).

    Returns:
        int: Number of files removed.
    """
    if max_age_hours is None:
        max_age_hours = getattr(Config, "EXPORT_FILE_EXPIRY_HOURS", 24)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    base_dir = Path(getattr(Config, "EXPORTS_DIR", "./exports")).resolve()
    removed = 0
    for user_dir in base_dir.iterdir():
        if not user_dir.is_dir():
            continue
        for file_path in user_dir.iterdir():
            if not file_path.is_file():
                continue
            mtime = datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc)
            if mtime < cutoff:
                file_path.unlink()
                removed += 1
        if not any(user_dir.iterdir()):
            user_dir.rmdir()
    if removed:
        logger.info("Expired export files cleaned up", count=removed, max_age_hours=max_age_hours)
    return removed
