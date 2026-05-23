from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Optional

EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_PATTERN = re.compile(r"(?:(?:\+?\d[\d\s().-]{6,}\d))")
JWT_PATTERN = re.compile(r"\b[a-zA-Z0-9_-]{8,}\.[a-zA-Z0-9_-]{8,}\.[a-zA-Z0-9_-]{8,}\b")
BEARER_PATTERN = re.compile(r"(?i)\b(?:bearer|jwt|token|access[_ -]?token|refresh[_ -]?token)[:=\s]+([A-Za-z0-9._-]{16,})")
OTP_PATTERN = re.compile(r"(?i)\b(?:otp|one[-\s]?time password)\b[^0-9]{0,24}([0-9]{4,8})")

SENSITIVE_KEYS = {
    "authorization",
    "body",
    "content",
    "email",
    "html",
    "html_content",
    "message",
    "message_body",
    "otp",
    "password",
    "payload",
    "phone",
    "phone_number",
    "recipient",
    "secret",
    "summary",
    "text",
    "token",
}


def mask_email(email: Optional[str]) -> str:
    if not email:
        return "[redacted-email]"

    value = str(email).strip()
    if "@" not in value:
        return "[redacted-email]"

    local, domain = value.split("@", 1)
    if not local:
        masked_local = "***"
    elif len(local) == 1:
        masked_local = f"{local[0]}***"
    elif len(local) == 2:
        masked_local = f"{local[0]}***{local[-1]}"
    else:
        masked_local = f"{local[0]}***{local[-1]}"
    return f"{masked_local}@{domain}"


def mask_phone(phone: Optional[str]) -> str:
    if not phone:
        return "[redacted-phone]"

    value = str(phone).strip()
    digits = re.sub(r"\D", "", value)
    if len(digits) < 4:
        return "[redacted-phone]"

    visible_prefix = value[:3] if value.startswith("+") else ""
    masked_length = max(len(value) - len(visible_prefix) - 4, 4)
    return f"{visible_prefix}{'*' * masked_length}{digits[-4:]}"


def mask_recipient(recipient: Optional[str]) -> str:
    if not recipient:
        return "[redacted-recipient]"

    value = str(recipient).strip()
    if "@" in value:
        return mask_email(value)
    if re.search(r"\d", value):
        return mask_phone(value)
    return "[redacted-recipient]"


def _replace_sensitive_text(value: str) -> str:
    value = EMAIL_PATTERN.sub("[redacted-email]", value)
    value = PHONE_PATTERN.sub("[redacted-phone]", value)
    value = JWT_PATTERN.sub("[redacted-token]", value)
    value = BEARER_PATTERN.sub(lambda match: f"{match.group(0).split()[0]} [redacted-token]", value)
    value = OTP_PATTERN.sub(lambda match: match.group(0).replace(match.group(1), "[redacted-otp]"), value)
    return value


def sanitize_log_text(value: Optional[str]) -> str:
    if value is None:
        return ""
    return _replace_sensitive_text(str(value)).replace("\r", "\\r").replace("\n", "\\n")


def sanitize_log_value(value: Any, key: Optional[str] = None) -> Any:
    key_name = (key or "").lower()
    if value is None:
        return None
    if key_name in SENSITIVE_KEYS:
        if key_name in {"email", "recipient"}:
            return mask_recipient(str(value))
        if key_name in {"phone", "phone_number"}:
            return mask_phone(str(value))
        return "[redacted]"
    if isinstance(value, Mapping):
        return {str(item_key): sanitize_log_value(item_value, str(item_key)) for item_key, item_value in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [sanitize_log_value(item, key_name) for item in value]
    if isinstance(value, str):
        if "@" in value:
            value = mask_email(value)
        return sanitize_log_text(value)[:240]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return sanitize_log_text(str(value))[:240]


def sanitize_log_fields(**fields: Any) -> dict[str, Any]:
    return {key: sanitize_log_value(value, key) for key, value in fields.items()}