"""Jurisdiction normalization and similarity helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional


_ALIASES: Dict[str, Dict[str, str]] = {
    "in": {
        "delhi": "IN:DL",
        "delhi high court": "IN:DL:HC",
        "high court of delhi": "IN:DL:HC",
        "supreme court of india": "IN:SC",
        "india": "IN",
    },
    "us": {
        "united states": "US",
        "u.s.": "US",
        "us": "US",
        "california": "US:CA",
        "new york": "US:NY",
        "texas": "US:TX",
        "federal": "US:FED",
    },
    "uk": {
        "united kingdom": "UK",
        "england": "UK:ENG",
        "wales": "UK:WLS",
    },
}


@dataclass(frozen=True)
class JurisdictionProfile:
    raw: str
    canonical: str
    country: str
    family: str


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s\.]+", " ", value.lower())).strip()


def normalize_jurisdiction(value: Optional[str]) -> JurisdictionProfile:
    raw = (value or "").strip()
    cleaned = _clean(raw)
    if not cleaned:
        return JurisdictionProfile(raw=raw, canonical="", country="unknown", family="unknown")

    for country, aliases in _ALIASES.items():
        if cleaned in aliases:
            canonical = aliases[cleaned]
            parts = canonical.split(":")
            return JurisdictionProfile(raw=raw, canonical=canonical, country=parts[0], family=parts[0])

    if ":" in cleaned:
        canonical = cleaned.upper().replace(" ", "")
        country = canonical.split(":", 1)[0]
        return JurisdictionProfile(raw=raw, canonical=canonical, country=country, family=country)

    country = cleaned.split(" ", 1)[0].upper()
    return JurisdictionProfile(raw=raw, canonical=cleaned.upper().replace(" ", "_"), country=country, family=country)


def jurisdiction_similarity(left: Optional[str], right: Optional[str]) -> float:
    left_profile = normalize_jurisdiction(left)
    right_profile = normalize_jurisdiction(right)

    if not left_profile.canonical or not right_profile.canonical:
        return 0.0
    if left_profile.canonical == right_profile.canonical:
        return 1.0
    if left_profile.country == right_profile.country and left_profile.country != "unknown":
        return 0.8
    if left_profile.family == right_profile.family and left_profile.family != "unknown":
        return 0.6
    return 0.25
