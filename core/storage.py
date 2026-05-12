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
    """Return full path for a stored attachment (no security checks)."""
    if not stored_path:
        return ""
    p = Path(stored_path)
    if not p.exists():
        return ""
    return str(p)
