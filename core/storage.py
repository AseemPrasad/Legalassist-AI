import logging
import os
import uuid
from pathlib import Path
from typing import Tuple
from urllib.parse import unquote
from config import Config

logger = logging.getLogger(__name__)

# Path traversal patterns (literal and URL-encoded variants)
_TRAVERSAL_PATTERNS = ["..", "%2e%2e", "%2E%2E", "%252e%252e", "%252E%252E"]

ATTACHMENTS_DIR = Path(Config.ATTACHMENTS_DIR)

# Ensure directory exists
os.makedirs(ATTACHMENTS_DIR, exist_ok=True)


def _has_traversal(path: str) -> bool:
    """Check if path contains directory traversal sequences (plain or URL-encoded)."""
    decoded = unquote(path)
    for pattern in _TRAVERSAL_PATTERNS:
        if pattern in path or pattern in decoded:
            return True
    return False


def save_attachment(file_bytes: bytes, original_filename: str) -> Tuple[str, int]:
    """
    Save attachment bytes to the attachments directory.
    Returns (stored_path, size_bytes).
    """
    # Randomize filename to avoid collisions and sensitive names
    ext = Path(original_filename).suffix or ""
    if _has_traversal(original_filename):
        logger.warning("Traversal detected in original_filename, randomizing", filename=original_filename)
        stored_name = f"{uuid.uuid4().hex}{ext}"
    elif Config.ATTACHMENTS_RANDOMIZE_FILENAMES:
        stored_name = f"{uuid.uuid4().hex}{ext}"
    else:
        safe_name = Path(original_filename).name
        stored_name = safe_name

    stored_path = ATTACHMENTS_DIR / stored_name

    # Write file
    with open(stored_path, "wb") as f:
        f.write(file_bytes)

    size = stored_path.stat().st_size
    return str(stored_path), size


def get_attachment_path(stored_path: str) -> str:
    """Return safe path for stored attachment, rejecting traversal attempts."""
    if not stored_path:
        return ""

    if _has_traversal(stored_path):
        logger.warning("Blocked path traversal attempt", path=stored_path)
        return ""

    resolved = Path(stored_path).resolve()
    attachments_dir = Path(ATTACHMENTS_DIR).resolve()

    if resolved.is_relative_to(attachments_dir) is False:
        logger.warning("Blocked path outside attachments directory", path=str(resolved))
        return ""

    if not resolved.exists():
        return ""

    return str(resolved)
