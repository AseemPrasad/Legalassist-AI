"""
Input validation and request size enforcement utilities

This module provides comprehensive validation for:
- Request body sizes
- File uploads (size, type, extension, magic bytes)
- JSON payload validation
- Form data validation
"""

import ipaddress
import os
import socket
from typing import Optional, Set, Tuple, NamedTuple
from pathlib import Path
from urllib.parse import urlparse
from fastapi import HTTPException, status, UploadFile

import requests
import structlog

logger = structlog.get_logger(__name__)


class PinnedUrl(NamedTuple):
    scheme: str
    hostname: str
    port: int
    path: str
    ip: str


# SSRF protection: private / reserved / metadata IP ranges
_FORBIDDEN_IP_NETWORKS = [
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("198.18.0.0/15"),
    ipaddress.ip_network("224.0.0.0/4"),
    ipaddress.ip_network("240.0.0.0/4"),
    ipaddress.ip_network("255.255.255.255/32"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]

# Cloud metadata endpoints
_CLOUD_METADATA_HOSTS = {
    "169.254.169.254",
    "metadata.google.internal",
    "metadata.heptio.com",
    "100.100.100.200",
    "fd00:ec2::254",
}

_ALLOWED_URL_SCHEMES = frozenset({"http", "https"})
_ALLOWED_URL_PORTS = frozenset({80, 443, 8080, 8443})


# Magic bytes for trusted file type identification
MAGIC_BYTES: dict = {
    ".pdf": [
        (b"%PDF-", "PDF document"),
    ],
    ".doc": [
        (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1", "Microsoft Word (Legacy)"),
    ],
    ".docx": [
        (b"PK\x03\x04", "Office Open XML (ZIP-based)"),
    ],
    ".txt": [
        (None, "Plain text (no magic bytes)"),
    ],
    ".html": [
        (b"<!DOCTYPE", "HTML document"),
        (b"<html", "HTML document"),
    ],
    ".rtf": [
        (b"{\\rtf", "Rich Text Format"),
    ],
}


class ValidationConfig:
    """Configuration for input validation limits"""

    MAX_UPLOAD_SIZE: int = 500 * 1024 * 1024
    MAX_UPLOAD_SIZE_JSON: int = 50 * 1024 * 1024
    MAX_TEXT_LENGTH: int = 10 * 1024 * 1024

    ALLOWED_EXTENSIONS: Set[str] = {".pdf", ".doc", ".docx", ".txt", ".html", ".rtf"}
    ALLOWED_MIME_TYPES: Set[str] = {
        "application/pdf",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/plain",
        "text/html",
        "application/rtf",
    }

    MAX_BATCH_SIZE: int = 100
    MAX_ANALYTICS_PAYLOAD: int = 100 * 1024 * 1024
    MAX_JSON_BODY: int = 10 * 1024 * 1024
    MAX_BASE64_DECODED_BYTES: int = 25 * 1024 * 1024  # 25 MB

    @classmethod
    def from_settings(cls, settings):
        cls.MAX_UPLOAD_SIZE = getattr(settings, "UPLOAD_MAX_SIZE", 25 * 1024 * 1024)
        cls.ALLOWED_EXTENSIONS = set(getattr(settings, "UPLOAD_EXTENSIONS", [".pdf", ".doc", ".docx", ".txt", ".html"]))
        return cls


class ValidationError(HTTPException):
    def __init__(self, detail: str, status_code: int = status.HTTP_400_BAD_REQUEST):
        super().__init__(status_code=status_code, detail=detail)


class PayloadTooLargeError(HTTPException):
    def __init__(self, detail: str):
        super().__init__(status_code=status.HTTP_413_CONTENT_TOO_LARGE, detail=detail)


def decode_base64_safe(data: str, max_decoded_bytes: int = ValidationConfig.MAX_BASE64_DECODED_BYTES) -> bytes:
    """Decode base64 string with decoded-size validation to prevent memory exhaustion."""
    # Base64 encodes 3 bytes into 4 characters; estimate max_input_len safely
    max_input_len = ((max_decoded_bytes + 2) // 3) * 4
    if len(data) > max_input_len:
        raise PayloadTooLargeError(
            f"Base64 payload exceeds maximum decoded size of {max_decoded_bytes // (1024 * 1024)} MB"
        )
    try:
        return base64.b64decode(data)
    except Exception as exc:
        raise ValidationError(f"Invalid base64 encoding: {exc}")


def validate_magic_bytes(file_content: bytes, expected_extension: str) -> Tuple[bool, str]:
    """Validate file content matches expected file type via magic bytes."""
    expected_extension = expected_extension.lower()
    magic_signatures = MAGIC_BYTES.get(expected_extension, [])

    if not magic_signatures:
        return False, f"No magic bytes defined for extension '{expected_extension}'"

    for magic_bytes, file_type in magic_signatures:
        if magic_bytes is None:
            return True, f"{file_type} - no magic bytes required"
        if file_content.startswith(magic_bytes):
            return True, file_type

    return False, f"File content does not match expected signature for {expected_extension}"


def validate_file_upload(
    file: UploadFile,
    max_size: Optional[int] = None,
    allowed_extensions: Optional[Set[str]] = None,
    allowed_mime_types: Optional[Set[str]] = None,
    check_magic_bytes: bool = True,
) -> None:
    """
    Validate uploaded file meets size, extension, MIME type, and magic bytes requirements.
    """
    max_size = max_size or ValidationConfig.MAX_UPLOAD_SIZE
    allowed_extensions = allowed_extensions or ValidationConfig.ALLOWED_EXTENSIONS
    allowed_mime_types = allowed_mime_types or ValidationConfig.ALLOWED_MIME_TYPES

    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        logger.warning("invalid_upload_extension", filename=file.filename, extension=file_ext)
        raise ValidationError(
            detail=f"File extension '{file_ext}' not allowed. Allowed: {', '.join(sorted(allowed_extensions))}"
        )

    if file.content_type and file.content_type not in allowed_mime_types:
        logger.warning("invalid_upload_mime_type", filename=file.filename, mime_type=file.content_type)
        raise ValidationError(
            detail=f"File type '{file.content_type}' not allowed. Allowed: {', '.join(allowed_mime_types)}"
        )
    # Reject files that declare no MIME type — client-provided content_type
    # cannot be trusted as the sole gate; absent type is also unacceptable.
    if not file.content_type:
        logger.warning("missing_upload_mime_type", filename=file.filename)
        raise ValidationError(
            detail="File MIME type is missing. Ensure the file is a supported type."
        )

    # Content-based validation: inspect magic bytes independently of the
    # client-supplied content_type.  A malicious actor can set any MIME type
    # they like; only the actual file header is authoritative.
    if check_magic_bytes and file_ext in MAGIC_BYTES:
        try:
            if hasattr(file.file, 'read'):
                original_pos = file.file.tell() if hasattr(file.file, 'tell') else None
                header_bytes = file.file.read(16)
                if original_pos is not None:
                    file.file.seek(original_pos)
            else:
                header_bytes = b""

            is_valid, message = validate_magic_bytes(header_bytes, file_ext)
            if not is_valid:
                logger.warning("invalid_magic_bytes", filename=file.filename, extension=file_ext, reason=message)
                raise ValidationError(detail="File content does not match the declared type.")
        except ValidationError:
            raise
        except Exception as exc:
            # Log and re-raise so the caller is never silently passed a file
            # whose content could not be verified — fail closed.
            logger.error("magic_bytes_check_failed", filename=file.filename, error=str(exc))
            raise ValidationError(detail="File content could not be verified. Upload rejected.") from exc

    if file.size and file.size > max_size:
        logger.warning("upload_exceeds_max_size", filename=file.filename, size_bytes=file.size, max_size_bytes=max_size)
        raise PayloadTooLargeError(
            detail=f"File size ({round(file.size / 1024 / 1024, 2)} MB) exceeds maximum ({round(max_size / 1024 / 1024, 2)} MB)"
        )


async def validate_file_upload_streaming(
    file: UploadFile,
    max_size: Optional[int] = None,
    chunk_size: int = 1024 * 1024,
) -> int:
    """Validate file size during streaming read."""
    max_size = max_size or ValidationConfig.MAX_UPLOAD_SIZE
    bytes_read = 0

    try:
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            bytes_read += len(chunk)
            if bytes_read > max_size:
                logger.error("upload_exceeded_max_size_during_stream", filename=file.filename, bytes_read=bytes_read)
                raise PayloadTooLargeError(
                    detail=f"Upload exceeded maximum size limit of {round(max_size / 1024 / 1024, 2)} MB"
                )
    finally:
        await file.seek(0)

    return bytes_read


def validate_json_payload(payload_size: int, max_size: Optional[int] = None) -> None:
    max_size = max_size or ValidationConfig.MAX_JSON_BODY
    if payload_size > max_size:
        logger.warning("json_payload_exceeds_limit", size_bytes=payload_size, max_size_bytes=max_size)
        raise PayloadTooLargeError(
            detail=f"Request body size ({round(payload_size / 1024 / 1024, 2)} MB) exceeds maximum ({round(max_size / 1024 / 1024, 2)} MB)"
        )


def validate_file_url(url: str) -> PinnedUrl:
    """
    Validate a user-supplied URL against SSRF attacks.
    - Only http/https schemes allowed
    - No private, loopback, link-local, or metadata IPs
    - Only standard HTTP(S) ports
    Returns a PinnedUrl with the single validated IP to use for the connection,
    preventing DNS rebinding between validation and fetch.
    """
    try:
        parsed = urlparse(url)
    except Exception as exc:
        logger.warning("url_parse_failed", url=url, error=str(exc))
        raise ValidationError(detail="Invalid URL format")

    if parsed.scheme not in _ALLOWED_URL_SCHEMES:
        logger.warning("url_scheme_denied", scheme=parsed.scheme, url=url)
        raise ValidationError(detail=f"URL scheme '{parsed.scheme}' is not allowed. Only http/https are permitted.")

    if parsed.port is not None and parsed.port not in _ALLOWED_URL_PORTS:
        logger.warning("url_port_denied", port=parsed.port, url=url)
        raise ValidationError(detail=f"URL port {parsed.port} is not allowed.")

    hostname = parsed.hostname or ""
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"

    if hostname.lower() in _CLOUD_METADATA_HOSTS:
        logger.warning("url_metadata_host_denied", host=hostname, url=url)
        raise ValidationError(detail="URL points to a cloud metadata endpoint and is not allowed.")

    try:
        ips = socket.getaddrinfo(hostname, port)
    except socket.gaierror as exc:
        logger.warning("url_dns_resolution_failed", host=hostname, error=str(exc))
        raise ValidationError(detail="Could not resolve the URL hostname.")
    except OSError:
        logger.warning("url_dns_resolution_os_error", host=hostname)
        raise ValidationError(detail="Could not resolve the URL hostname.")

    resolved_ip = None
    for addr_info in ips:
        ip_str = addr_info[4][0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved:
            logger.warning("url_ip_blocked", ip=ip_str, host=hostname, url=url)
            raise ValidationError(detail=f"URL resolves to a blocked IP range ({ip_str}).")

        for net in _FORBIDDEN_IP_NETWORKS:
            if ip in net:
                logger.warning("url_ip_in_forbidden_network", ip=ip_str, network=str(net), host=hostname, url=url)
                raise ValidationError(detail=f"URL resolves to a blocked IP range ({ip_str}).")

        if resolved_ip is None:
            resolved_ip = ip_str

    if resolved_ip is None:
        raise ValidationError(detail="Could not resolve the URL to a valid IP address.")

    return PinnedUrl(
        scheme=parsed.scheme,
        hostname=hostname,
        port=port,
        path=path,
        ip=resolved_ip,
    )


class _PinnedResponse:
    """Minimal response wrapper returned by fetch_url_safe."""
    def __init__(self, status_code: int, headers: dict, content: bytes):
        self.status_code = status_code
        self.headers = headers
        self.content = content

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests as _req
            exc = _req.HTTPError(f"HTTP {self.status_code}")
            exc.response = self
            raise exc


def fetch_url_safe(pinned: PinnedUrl, timeout: int = 30) -> _PinnedResponse:
    """Fetch a URL using the pinned IP from validate_file_url, preventing DNS rebinding."""
    import urllib3
    headers = {"Host": pinned.hostname, "User-Agent": "LegalAssist-AI/1.0"}
    if pinned.scheme == "https":
        conn = urllib3.HTTPSConnection(
            host=pinned.ip,
            port=pinned.port,
            server_hostname=pinned.hostname,
            assert_hostname=pinned.hostname,
            timeout=timeout,
        )
    else:
        conn = urllib3.HTTPConnection(
            host=pinned.ip,
            port=pinned.port,
            timeout=timeout,
        )
    conn.request("GET", pinned.path, headers=headers)
    http_response = conn.getresponse()
    content = http_response.data
    resp_headers = dict(http_response.getheaders())
    status = http_response.status
    conn.close()
    return _PinnedResponse(status_code=status, headers=resp_headers, content=content)


def validate_text_input(text: str, max_length: Optional[int] = None) -> None:
    max_length = max_length or ValidationConfig.MAX_TEXT_LENGTH
    text_bytes = len(text.encode("utf-8"))
    if text_bytes > max_length:
        logger.warning("text_input_exceeds_limit", size_bytes=text_bytes, max_size_bytes=max_length)
        raise PayloadTooLargeError(
            detail=f"Text input size ({round(text_bytes / 1024 / 1024, 2)} MB) exceeds maximum ({round(max_length / 1024 / 1024, 2)} MB)"
        )


def validate_batch_size(items: list, max_items: Optional[int] = None) -> None:
    max_items = max_items or ValidationConfig.MAX_BATCH_SIZE
    if len(items) > max_items:
        logger.warning("batch_request_exceeds_limit", item_count=len(items), max_items=max_items)
        raise ValidationError(
            detail=f"Batch size ({len(items)} items) exceeds maximum allowed ({max_items} items)"
        )


def validate_query_string(query_string: str, max_length: int = 2048) -> None:
    if len(query_string) > max_length:
        logger.warning("query_string_exceeds_limit", length=len(query_string), max_length=max_length)
        raise ValidationError(detail=f"Query string too long ({len(query_string)} chars, max {max_length})")


def validate_upload_file_path(file_path: str, allowed_root: Optional[str] = None) -> str:
    """Validate and canonicalize a file path so it stays within the upload jail.

    This prevents path traversal attacks where a crafted ``file_path`` value
    (e.g. ``../../etc/passwd``) passed through the Celery task queue could
    cause the worker to read arbitrary files outside the upload directory.

    Parameters
    ----------
    file_path:
        The raw path string received from the task arguments.
    allowed_root:
        The directory that the resolved path must be a descendant of.
        Defaults to the configured ``UPLOAD_TEMP_DIR``.

    Returns
    -------
    str
        The canonicalized absolute path, guaranteed to be inside *allowed_root*.

    Raises
    ------
    ValidationError
        If the resolved path escapes *allowed_root* or the path is empty.
    """
    if not file_path or not file_path.strip():
        raise ValidationError(detail="file_path must not be empty.")

    if allowed_root is None:
        # Import lazily to avoid circular imports at module load time.
        try:
            from api.config import get_settings as _get_settings
            _settings = _get_settings()
            allowed_root = _settings.UPLOAD_TEMP_DIR
        except Exception:
            import tempfile
            allowed_root = tempfile.gettempdir()

    try:
        resolved = Path(file_path).resolve()
        jail = Path(allowed_root).resolve()
    except Exception as exc:
        raise ValidationError(detail=f"Invalid file_path: {exc}") from exc

    # Path.is_relative_to() is Python 3.9+; use str prefix check for 3.8 compat.
    try:
        resolved.relative_to(jail)
    except ValueError:
        logger.warning(
            "path_traversal_attempt_blocked",
            file_path=file_path,
            resolved=str(resolved),
            jail=str(jail),
        )
        raise ValidationError(
            detail="file_path must be within the upload directory.",
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    return str(resolved)