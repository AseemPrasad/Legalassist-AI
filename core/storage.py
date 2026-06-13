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

def delete_attachment_file(stored_path: str) -> bool:
    """Delete an attachment file from storage."""
    if not stored_path:
        logger.warning("Empty stored_path provided for deletion")
        return False

    p = Path(stored_path)

    if p.is_symlink():
        logger.warning("Rejected symlink in attachment deletion path: %s", stored_path)
        return False

    if not _is_safe_attachment_path(p):
        logger.warning("Blocked path traversal in deletion attempt: %s", stored_path)
        return False

    try:
        resolved = p.resolve(strict=True)
        if resolved.exists():
            resolved.unlink()
            logger.info("Deleted attachment file: %s", stored_path)
            return True
        else:
            logger.warning("Attachment file not found for deletion: %s", stored_path)
            return False
    except (OSError, RuntimeError, FileNotFoundError) as e:
        logger.error("Failed to delete attachment file %s: %s", stored_path, e)
        return False


def bulk_delete_attachments(stored_paths: list) -> dict:
    """Delete multiple attachment files from storage."""
    results = {"deleted": 0, "failed": 0, "errors": []}

    for stored_path in stored_paths:
        try:
            if delete_attachment_file(stored_path):
                results["deleted"] += 1
            else:
                results["failed"] += 1
        except Exception as e:
            results["failed"] += 1
            results["errors"].append({"path": stored_path, "error": str(e)})
            logger.error("Error deleting attachment %s: %s", stored_path, e)

    logger.info("Bulk deletion completed: %d deleted, %d failed", results["deleted"], results["failed"])
    return results


