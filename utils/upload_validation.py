"""Content-based validation for uploaded files.

Streamlit's ``type=[...]`` filter only looks at the file extension, so a renamed
executable passes it. These helpers check the real content type from the first
bytes of the file and compare it with what the extension claims.
"""
from typing import Optional

# extension -> MIME types accepted for it
ALLOWED_MIME_BY_EXTENSION = {
    "pdf": {"application/pdf"},
    "png": {"image/png"},
    "jpg": {"image/jpeg"},
    "jpeg": {"image/jpeg"},
}

_SIGNATURES = (
    (b"%PDF-", "application/pdf"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
)


def detect_mime(head: bytes) -> Optional[str]:
    """Return the MIME type of ``head`` (the first ~2 KB), or None if unknown."""
    try:
        import magic

        return magic.from_buffer(head[:2048], mime=True)
    except Exception:
        # libmagic missing or unusable: fall back to the signatures we accept.
        for sig, mime in _SIGNATURES:
            if head.startswith(sig):
                return mime
        return None


def validate_upload(filename: str, head: bytes) -> Optional[str]:
    """Check that the content of an upload matches its extension.

    Returns None when the file is fine, otherwise a message for the user.
    """
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    allowed = ALLOWED_MIME_BY_EXTENSION.get(ext)
    if allowed is None:
        return f"Unsupported file type: .{ext}" if ext else "Unsupported file type."
    detected = detect_mime(head)
    if detected not in allowed:
        return (
            f"'{filename}' does not look like a real .{ext} file "
            f"(detected type: {detected or 'unknown'}). Please upload the original file."
        )
    return None
