"""Config-driven privacy redaction profiles.

Profiles are used by anonymized exports and privacy-aware report payloads.
"""

from __future__ import annotations

import copy
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from config import Config

DEFAULT_PRIVACY_PROFILE = "personal_identifiers"
PRIVACY_PROFILES_ENV = "PRIVACY_REDACTION_PROFILES_JSON"

EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_PATTERN = re.compile(r"(?:(?:\+?\d[\d\s().-]{6,}\d))")

DEFAULT_PRIVACY_PROFILES: Dict[str, Dict[str, Any]] = {
    "personal_identifiers": {
        "label": "Personal identifiers only",
        "description": "Hide direct identifiers while keeping case structure and narrative context.",
        "case_number_style": "anonymized",
        "title_style": "redacted",
        "mask_free_text": True,
        "keep_document_summaries": True,
        "keep_deadline_descriptions": True,
        "keep_timeline_descriptions": True,
        "keep_attachment_names": False,
        "keep_remedies": True,
        "keep_document_metadata": True,
    },
    "full_party_removal": {
        "label": "Full party removal",
        "description": "Remove free-form party-sensitive content and suppress narrative fields.",
        "case_number_style": "redacted",
        "title_style": "redacted",
        "mask_free_text": True,
        "keep_document_summaries": False,
        "keep_deadline_descriptions": False,
        "keep_timeline_descriptions": False,
        "keep_attachment_names": False,
        "keep_remedies": False,
        "keep_document_metadata": False,
    },
}


@dataclass(frozen=True)
class PrivacyProfileOption:
    name: str
    label: str
    description: str


def _load_profile_overrides() -> Dict[str, Dict[str, Any]]:
    raw = str(getattr(Config, "PRIVACY_REDACTION_PROFILES_JSON", "") or os.getenv(PRIVACY_PROFILES_ENV, "") or "").strip()
    if not raw:
        return {}
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(loaded, dict):
        return {}
    overrides: Dict[str, Dict[str, Any]] = {}
    for profile_name, profile_value in loaded.items():
        if isinstance(profile_name, str) and isinstance(profile_value, dict):
            overrides[profile_name] = profile_value
    return overrides


def get_privacy_profiles() -> Dict[str, Dict[str, Any]]:
    profiles = copy.deepcopy(DEFAULT_PRIVACY_PROFILES)
    for profile_name, override in _load_profile_overrides().items():
        base = profiles.get(profile_name, {})
        base.update(override)
        profiles[profile_name] = base
    return profiles


def get_privacy_profile_options() -> List[Dict[str, str]]:
    profiles = get_privacy_profiles()
    return [
        {
            "name": name,
            "label": profile.get("label", name.replace("_", " ").title()),
            "description": profile.get("description", ""),
        }
        for name, profile in profiles.items()
    ]


def get_default_privacy_profile() -> str:
    default_name = str(getattr(Config, "DEFAULT_PRIVACY_PROFILE", DEFAULT_PRIVACY_PROFILE) or DEFAULT_PRIVACY_PROFILE).strip()
    if default_name in get_privacy_profiles():
        return default_name
    return DEFAULT_PRIVACY_PROFILE


def normalize_privacy_profile(profile_name: Optional[str]) -> str:
    profiles = get_privacy_profiles()
    candidate = str(profile_name or "").strip()
    if candidate in profiles:
        return candidate
    return get_default_privacy_profile()


def get_privacy_profile_definition(profile_name: Optional[str]) -> Dict[str, Any]:
    profiles = get_privacy_profiles()
    normalized = normalize_privacy_profile(profile_name)
    return profiles.get(normalized, profiles[get_default_privacy_profile()])


def _mask_free_text(value: Any) -> Any:
    if not isinstance(value, str) or not value:
        return value
    masked = EMAIL_PATTERN.sub("[redacted-email]", value)
    masked = PHONE_PATTERN.sub("[redacted-phone]", masked)
    return masked


def _redact_case_identifier(case_number: Optional[str], anonymized_id: Optional[str], profile: Dict[str, Any]) -> str:
    style = profile.get("case_number_style", "anonymized")
    if style == "redacted":
        return "REDACTED"
    if anonymized_id:
        return f"ANON-{anonymized_id}"
    if case_number:
        return f"ANON-{case_number[-6:]}" if len(case_number) >= 6 else "ANON-CASE"
    return "ANON-CASE"


def _redact_title(title: Optional[str], profile: Dict[str, Any]) -> str:
    style = profile.get("title_style", "redacted")
    if style == "redacted":
        return "Redacted Matter"
    return title or "Redacted Matter"


def _redact_mapping(mapping: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(mapping)
    if not result:
        return result

    if not profile.get("keep_document_metadata", True):
        result.pop("extracted_metadata", None)
        result.pop("source_attachment_id", None)
    if not profile.get("keep_document_summaries", True):
        result["summary"] = None
    elif profile.get("mask_free_text", True):
        result["summary"] = _mask_free_text(result.get("summary"))

    if not profile.get("keep_deadline_descriptions", True) and "description" in result:
        result["description"] = None
    elif profile.get("mask_free_text", True) and "description" in result:
        result["description"] = _mask_free_text(result.get("description"))

    if not profile.get("keep_timeline_descriptions", True) and "description" in result:
        result["description"] = None
    elif profile.get("mask_free_text", True) and "description" in result:
        result["description"] = _mask_free_text(result.get("description"))

    if not profile.get("keep_attachment_names", True) and "original_filename" in result:
        result["original_filename"] = None

    if not profile.get("keep_remedies", True) and "remedies" in result:
        result["remedies"] = None
    elif profile.get("mask_free_text", True) and "remedies" in result:
        result["remedies"] = _redact_any(result.get("remedies"), profile)

    for key, value in list(result.items()):
        if isinstance(value, str) and profile.get("mask_free_text", True):
            result[key] = _mask_free_text(value)
        elif isinstance(value, dict):
            result[key] = _redact_mapping(value, profile)
        elif isinstance(value, list):
            result[key] = [_redact_any(item, profile) for item in value]

    return result


def _redact_any(value: Any, profile: Dict[str, Any]) -> Any:
    if isinstance(value, dict):
        return _redact_mapping(value, profile)
    if isinstance(value, list):
        return [_redact_any(item, profile) for item in value]
    if isinstance(value, str) and profile.get("mask_free_text", True):
        return _mask_free_text(value)
    return value


def apply_privacy_profile(
    payload: Dict[str, Any],
    profile_name: Optional[str],
    anonymized_id: Optional[str] = None,
) -> Dict[str, Any]:
    profile = get_privacy_profile_definition(profile_name)
    result = copy.deepcopy(payload)

    export_meta = result.setdefault("export", {})
    export_meta["privacy_profile"] = normalize_privacy_profile(profile_name)
    export_meta["privacy_profile_label"] = profile.get("label", export_meta["privacy_profile"])

    case_section = result.get("case")
    if isinstance(case_section, dict):
        case_section["case_number"] = _redact_case_identifier(
            case_section.get("case_number"),
            anonymized_id,
            profile,
        )
        case_section["title"] = _redact_title(case_section.get("title"), profile)

    if "latest_document" in result and isinstance(result["latest_document"], dict):
        result["latest_document"] = _redact_mapping(result["latest_document"], profile)
    if "next_deadline" in result and isinstance(result["next_deadline"], dict):
        result["next_deadline"] = _redact_mapping(result["next_deadline"], profile)

    for section_name in ("documents", "deadlines", "timeline", "attachments"):
        section = result.get(section_name)
        if isinstance(section, list):
            result[section_name] = [_redact_any(item, profile) for item in section]

    if "remedies" in result and not profile.get("keep_remedies", True):
        result["remedies"] = None
    elif "remedies" in result:
        result["remedies"] = _redact_any(result["remedies"], profile)

    preserve_top_level_strings = {
        "anonymized_id",
        "privacy_profile",
        "privacy_profile_label",
        "case_type",
        "jurisdiction",
        "status",
        "created_date",
        "document_count",
    }
    if profile.get("mask_free_text", True):
        for key, value in list(result.items()):
            if isinstance(value, str) and key not in preserve_top_level_strings and key != "export":
                result[key] = _mask_free_text(value)

    return result
