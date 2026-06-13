import logging
import os
import re
import uuid
import logging
from pathlib import Path
from typing import Tuple
from config import Config

logger = logging.getLogger(__name__)

ATTACHMENTS_DIR = Path(Config.ATTACHMENTS_DIR)

# Ensure directory exists
os.makedirs(ATTACHMENTS_DIR, exist_ok=True)


def _is_safe_attachment_path(path: Path) -> bool:
    """Return True if the resolved canonical path falls within ATTACHMENTS_DIR.

    Uses ``strict=True`` resolution so non-existent paths are rejected,
    preventing information disclosure via path canonicalization.
    """
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError, FileNotFoundError):
        return False

    allowed = ATTACHMENTS_DIR.resolve(strict=True)
    try:
        return resolved.is_relative_to(allowed)
    except (OSError, RuntimeError, ValueError):
        return False


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
        safe_name = Path(original_filename).name.replace("..", "")
        stored_name = safe_name

    stored_path = ATTACHMENTS_DIR / stored_name

    if stored_path.is_symlink():
        raise ValueError(f"Attachment path is a symlink: {stored_path}")

    with open(stored_path, "wb") as f:
        f.write(file_bytes)

    size = resolved.stat().st_size
    return str(resolved), size


def get_attachment_path(stored_path: str) -> str:
    """Return safe path for stored attachment, rejecting traversal attempts.

    Resolves the path within ATTACHMENTS_DIR and verifies existence.
    Returns an empty string if the path is invalid or unsafe.
    """
    if not stored_path:
        return ""

    p = Path(stored_path)
    if p.is_symlink():
        logger.warning("Rejected symlink in attachment path: %s", stored_path)
        return ""

    if not _is_safe_attachment_path(p):
        logger.warning("Blocked path traversal attempt: %s", stored_path)
        return ""

    resolved = p.resolve(strict=True)
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

def delete_attachment_file(stored_path: str) -> bool:
    """Delete an attachment file from storage.

    This function safely deletes a file from the attachments directory,
    ensuring the path is within the allowed directory and not a symlink.

    Args:
        stored_path: The stored path of the attachment to delete

    Returns:
        True if the file was deleted successfully, False otherwise
    """
    if not stored_path:
        logger.warning("Empty stored_path provided for deletion")
        return False

    p = Path(stored_path)

    # Reject symlinks to prevent directory traversal
    if p.is_symlink():
        logger.warning("Rejected symlink in attachment deletion path: %s", stored_path)
        return False

    # Verify the path is within ATTACHMENTS_DIR
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
    """Delete multiple attachment files from storage.

    Args:
        stored_paths: List of stored paths for attachments to delete

    Returns:
        Dictionary with counts of deleted and failed deletions
    """
    results = {
        "deleted": 0,
        "failed": 0,
        "errors": []
    }

    for stored_path in stored_paths:
        try:
            if delete_attachment_file(stored_path):
                results["deleted"] += 1
            else:
                results["failed"] += 1
        except Exception as e:
            results["failed"] += 1
            results["errors"].append({
                "path": stored_path,
                "error": str(e)
            })
            logger.error("Error deleting attachment %s: %s", stored_path, e)

    logger.info("Bulk deletion completed: %d deleted, %d failed",
                results["deleted"], results["failed"])

    return results
