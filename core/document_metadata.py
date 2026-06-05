"""Helpers for OCR-based case document ingestion and structured metadata extraction."""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import core as core_text_utils

_DATE_PATTERNS = [
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
    r"\b\d{1,2}\s+[A-Z][a-z]+\s+\d{4}\b",
    r"\b[A-Z][a-z]+\s+\d{1,2},\s+\d{4}\b",
]

_STATUTE_PATTERN = re.compile(
    r"\b(?:Section|Sec\.?|\u00a7)\s*\d+[A-Za-z0-9()\-]*\s*(?:of\s+[A-Z][A-Za-z0-9&\s]{2,60})?"
    r"|\b(?:IPC|CrPC|CPC|Evidence Act|Contract Act|Companies Act|Negotiable Instruments Act)\b",
    re.IGNORECASE,
)

_PARTY_LINE_PATTERNS = [
    re.compile(r"\b(.{2,120}?)\s+v(?:s\.?|\.)\s+(.{2,120}?)\b", re.IGNORECASE),
    re.compile(r"\b(.{2,120}?)\s+versus\s+(.{2,120}?)\b", re.IGNORECASE),
]

_CLAIM_KEYWORDS = (
    "claim",
    "prayer",
    "relief",
    "alleg",
    "petition",
    "seek",
    "seeks",
    "damages",
    "compensation",
    "injunction",
    "writ",
)


def _read_image_text(image_bytes: bytes, ocr_languages: str, ocr_dpi: int) -> Dict[str, Any]:
    try:
        from PIL import Image
        import pytesseract
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError(
            "OCR dependencies are missing. Install Pillow and pytesseract to process image uploads."
        ) from exc

    image = Image.open(io.BytesIO(image_bytes))
    text = pytesseract.image_to_string(image, lang=ocr_languages)
    return {
        "text": text.strip(),
        "method": "ocr_tesseract",
        "ocr_used": True,
        "confidence": None,
    }


def extract_text_from_uploaded_file(
    file_path: str,
    *,
    original_filename: Optional[str] = None,
    enable_ocr: bool = True,
    ocr_languages: str = "eng+hin",
    ocr_dpi: int = 300,
) -> Dict[str, Any]:
    """Extract text from a stored upload and report whether OCR was used."""
    path = Path(file_path)
    suffix = (original_filename or path.name or "").lower()
    data = path.read_bytes()

    if suffix.endswith(".pdf"):
        diagnostics = core_text_utils.extract_text_with_diagnostics(
            io.BytesIO(data),
            enable_ocr=enable_ocr,
            ocr_languages=ocr_languages,
            ocr_dpi=ocr_dpi,
        )
        diagnostics["source_format"] = "pdf"
        return diagnostics

    if suffix.endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")):
        diagnostics = _read_image_text(data, ocr_languages=ocr_languages, ocr_dpi=ocr_dpi)
        diagnostics["source_format"] = "image"
        return diagnostics

    # Fall back to plain text decode for unexpected types.
    text = data.decode("utf-8", errors="ignore").strip()
    return {
        "text": text,
        "method": "text_decode",
        "ocr_used": False,
        "confidence": None,
        "source_format": "text",
    }


def _clean_candidate(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip(" ,;:\n\t")
    value = re.sub(r"^[\d\-\(\)\[\]\.:\s]+", "", value)
    return value[:120]


def _extract_party_candidates(text: str) -> List[str]:
    candidates: List[str] = []
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    for line in lines[:40]:
        for pattern in _PARTY_LINE_PATTERNS:
            match = pattern.search(line)
            if match:
                left = _clean_candidate(match.group(1))
                right = _clean_candidate(match.group(2))
                if left:
                    candidates.append(left)
                if right:
                    candidates.append(right)
        if len(candidates) >= 4:
            break

    if not candidates:
        for keyword in ("petitioner", "respondent", "appellant", "defendant", "plaintiff"):
            for line in lines[:50]:
                if keyword in line.lower():
                    candidates.append(_clean_candidate(line))
                    if len(candidates) >= 4:
                        return _dedupe(candidates)

    return _dedupe(candidates)[:6]


def _extract_date_candidates(text: str) -> List[str]:
    candidates: List[str] = []
    for pattern in _DATE_PATTERNS:
        candidates.extend(re.findall(pattern, text, flags=re.IGNORECASE))
    return _dedupe(candidates)[:10]


def _extract_claim_candidates(text: str) -> List[str]:
    candidates: List[str] = []
    for line in text.splitlines():
        normalized = line.strip()
        if len(normalized) < 25:
            continue
        lower = normalized.lower()
        if any(keyword in lower for keyword in _CLAIM_KEYWORDS):
            candidates.append(_clean_candidate(normalized))
    return _dedupe(candidates)[:8]


def _extract_statute_candidates(text: str) -> List[str]:
    return _dedupe(match.group(0).strip() for match in _STATUTE_PATTERN.finditer(text))[:10]


def _dedupe(values) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        if not value:
            continue
        cleaned = _clean_candidate(str(value))
        key = cleaned.casefold()
        if key and key not in seen:
            seen.add(key)
            ordered.append(cleaned)
    return ordered


def extract_case_document_metadata(text: str, *, filename: Optional[str] = None) -> Dict[str, Any]:
    """Extract lightweight structured metadata from a case document.

    On any unexpected extraction failure the function returns a *partial* result
    dictionary populated with whichever fields were successfully computed before
    the error, rather than re-raising and returning nothing to the caller.
    Fields that could not be computed are substituted with empty-list defaults.

    Args:
        text: Raw plain-text content of the legal document.
        filename: Optional original filename used as a title hint fallback.

    Returns:
        Dict with keys: title_hint, parties, dates, claims, statutes, confidence,
        and (on error) partial_result=True plus an error_hint string.
    """
    partial: Dict[str, Any] = {
        "parties": [],
        "dates": [],
        "claims": [],
        "statutes": [],
    }

    try:
        partial["parties"] = _extract_party_candidates(text)
    except Exception:  # noqa: BLE001
        pass

    try:
        partial["dates"] = _extract_date_candidates(text)
    except Exception:  # noqa: BLE001
        pass

    try:
        partial["claims"] = _extract_claim_candidates(text)
    except Exception:  # noqa: BLE001
        pass

    try:
        partial["statutes"] = _extract_statute_candidates(text)
    except Exception:  # noqa: BLE001
        pass

    parties = partial["parties"]
    dates = partial["dates"]
    claims = partial["claims"]
    statutes = partial["statutes"]

    title_hint = None
    try:
        if parties:
            title_hint = " v. ".join(parties[:2]) if len(parties) >= 2 else parties[0]
        elif filename:
            title_hint = Path(filename).stem
    except Exception:  # noqa: BLE001
        pass

    return {
        "title_hint": title_hint,
        "parties": parties,
        "dates": dates,
        "claims": claims,
        "statutes": statutes,
        "confidence": {
            "parties": 0.6 if parties else 0.0,
            "dates": 0.7 if dates else 0.0,
            "claims": 0.5 if claims else 0.0,
            "statutes": 0.5 if statutes else 0.0,
        },
    }

