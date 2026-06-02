import logging
import os
import uuid
from pathlib import Path
from typing import Tuple
from config import Config

logger = logging.getLogger(__name__)

ATTACHMENTS_DIR = Path(Config.ATTACHMENTS_DIR)

# Ensure directory exists
os.makedirs(ATTACHMENTS_DIR, exist_ok=True)


def _is_safe_attachment_path(path: str) -> bool:
    """Return True if the resolved canonical path falls within ATTACHMENTS_DIR."""
    try:
        resolved = Path(path).resolve(strict=False)
    except (OSError, RuntimeError):
        return False
    return resolved.is_relative_to(ATTACHMENTS_DIR.resolve())


def save_attachment(file_bytes: bytes, original_filename: str) -> Tuple[str, int]:
    """
    Save attachment bytes to the attachments directory.
    Returns (stored_path, size_bytes).
    """
    ext = Path(original_filename).suffix or ""
    if not _is_safe_attachment_path(str(ATTACHMENTS_DIR / original_filename)):
        logger.warning("Traversal detected in original_filename, randomizing", filename=original_filename)
        stored_name = f"{uuid.uuid4().hex}{ext}"
    elif Config.ATTACHMENTS_RANDOMIZE_FILENAMES:
        stored_name = f"{uuid.uuid4().hex}{ext}"
    else:
        safe_name = Path(original_filename).name
        stored_name = safe_name

    stored_path = ATTACHMENTS_DIR / stored_name

    # Verify the resolved path stays within the attachments directory
    resolved = stored_path.resolve()
    if not resolved.is_relative_to(ATTACHMENTS_DIR.resolve()):
        logger.warning("Blocked resolved path outside attachments directory", path=str(resolved))
        raise ValueError("Invalid storage path")

    # Write file
    with open(resolved, "wb") as f:
        f.write(file_bytes)

    size = resolved.stat().st_size
    return str(resolved), size


def get_attachment_path(stored_path: str) -> str:
    """Return safe path for stored attachment, rejecting traversal attempts."""
    if not stored_path:
        return ""

    if not _is_safe_attachment_path(stored_path):
        logger.warning("Blocked path traversal attempt", path=stored_path)
        return ""

    resolved = Path(stored_path).resolve()

    if not resolved.exists():
        return ""

    return str(resolved)
