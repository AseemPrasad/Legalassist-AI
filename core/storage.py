import os
import uuid
from pathlib import Path
from typing import Tuple
from config import Config

ATTACHMENTS_DIR = Path(Config.ATTACHMENTS_DIR)

# Ensure directory exists
os.makedirs(ATTACHMENTS_DIR, exist_ok=True)


def save_attachment(file_bytes: bytes, original_filename: str) -> Tuple[str, int]:
    """
    Save attachment bytes to the attachments directory.
    Returns (stored_path, size_bytes).
    """
    # Randomize filename to avoid collisions and sensitive names
    ext = Path(original_filename).suffix or ""
    if Config.ATTACHMENTS_RANDOMIZE_FILENAMES:
        stored_name = f"{uuid.uuid4().hex}{ext}"
    else:
        # sanitize filename minimally
        safe_name = Path(original_filename).name.replace("..", "")
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

    resolved = Path(stored_path).resolve()
    attachments_dir = Path(ATTACHMENTS_DIR).resolve()

    if ".." in stored_path or resolved.is_relative_to(attachments_dir) is False:
        return ""

    if not resolved.exists():
        return ""

    return str(resolved)
