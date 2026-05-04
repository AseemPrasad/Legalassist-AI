"""
Shared utilities for LegalEase AI
- PDF extraction and text processing
- LLM prompts and remedies advisor
- Styling and UI constants

PATCHES APPLIED (2 bugs fixed):
  FIX-1: Language consistency — remedies LLM system prompt now enforces target language;
          _normalize_yes_no now returns localized values; _validate_court_name no longer
          strips non-English court names; render_shareable_result_box passes ui_text down
          correctly to _build_result_body_html.
  FIX-2: "What you can do" layout — build_judgment_result_text now emits a structured
          dict alongside the plain-text string; render_shareable_result_box uses the
          structured dict to build guaranteed question/answer pairs in the HTML renderer,
          so questions and answers are always correctly paired and answers are never cut short.
          max_tokens for remedies raised from 500 → 900.
"""

import re
import logging
import os
import json
from pathlib import Path
from openai import OpenAI
from pypdf import PdfReader
from langdetect import detect, DetectorFactory, detect_langs
import pdfplumber
from typing import Any, Dict, List

try:
    from dotenv import dotenv_values, load_dotenv
except ModuleNotFoundError:
    dotenv_values = None
    load_dotenv = None

PROJECT_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"

if load_dotenv:
    load_dotenv(dotenv_path=PROJECT_ENV_PATH)

# For consistent language detection results
DetectorFactory.seed = 0
_LOCALIZED_UI_TEXT_CACHE = {}

# ==================== MODEL CONFIGURATION ====================

DEFAULT_MODEL = "meta-llama/llama-3.1-8b-instruct"


def get_default_model():
    try:
        import streamlit as st
        return st.secrets.get("DEFAULT_MODEL", DEFAULT_MODEL)
    except (KeyError, FileNotFoundError, RuntimeError, AttributeError):
        return DEFAULT_MODEL


# ==================== API CLIENT ====================

PLACEHOLDER_CONFIG_VALUES = {
    "",
    "change_me",
    "changeme",
    "dummy",
    "none",
    "null",
    "test",
    "test_key",
    "test_url",
    "your_key_here",
}


def _clean_config_value(value):
    """Normalize config values from env vars or Streamlit secrets."""
    if value is None:
        return ""
    return str(value).strip().strip('"').strip("'")


def _is_placeholder_config(value):
    return _clean_config_value(value).lower() in PLACEHOLDER_CONFIG_VALUES


def _is_usable_api_key(value):
    value = _clean_config_value(value)
    return bool(value) and not _is_placeholder_config(value) and len(value) >= 8


def _is_usable_base_url(value):
    value = _clean_config_value(value)
    return (
        bool(value)
        and not _is_placeholder_config(value)
        and value.startswith(("http://", "https://"))
    )


def _select_openrouter_config(candidates):
    """Return the first complete, non-placeholder OpenRouter config pair."""
    for api_key, base_url in candidates:
        api_key = _clean_config_value(api_key)
        base_url = _clean_config_value(base_url)
        if _is_usable_api_key(api_key) and _is_usable_base_url(base_url):
            return api_key, base_url

    raise ValueError(
        "OPENROUTER_API_KEY and OPENROUTER_BASE_URL must be set to real values. "
        "Placeholder values like dummy/test_url are ignored."
    )


def _read_streamlit_openrouter_secrets():
    try:
        import streamlit as st

        return (
            st.secrets.get("OPENROUTER_API_KEY"),
            st.secrets.get("OPENROUTER_BASE_URL"),
        )
    except (AttributeError, FileNotFoundError, KeyError, RuntimeError):
        return "", ""


def _read_dotenv_openrouter_config():
    if not dotenv_values or not PROJECT_ENV_PATH.exists():
        return "", ""

    values = dotenv_values(PROJECT_ENV_PATH)
    return (
        values.get("OPENROUTER_API_KEY"),
        values.get("OPENROUTER_BASE_URL"),
    )


def _initialize_openai_client():
    """
    Internal function to initialize the OpenAI client using Streamlit secrets or environment variables.
    Uses Streamlit caching to avoid recreating the client.
    """
    secrets_api_key, secrets_base_url = _read_streamlit_openrouter_secrets()
    dotenv_api_key, dotenv_base_url = _read_dotenv_openrouter_config()
    api_key, base_url = _select_openrouter_config(
        [
            (secrets_api_key, secrets_base_url),
            (os.getenv("OPENROUTER_API_KEY"), os.getenv("OPENROUTER_BASE_URL")),
            (dotenv_api_key, dotenv_base_url),
        ]
    )
    return OpenAI(api_key=api_key, base_url=base_url)

    return OpenAI(api_key=api_key, base_url=base_url)


def get_client():
    import streamlit as st

    @st.cache_resource
    def _get_cached_client():
        try:
            return _initialize_openai_client()
        except (ValueError, KeyError, FileNotFoundError, RuntimeError, AttributeError) as e:
            # Graceful fallback for environments where secrets are not available (e.g., tests)
            logging.error(f"Failed to initialize OpenAI client: {e}")
            return None

    return _get_cached_client()


# ==================== TEXT PROCESSING ====================

def _extract_pages_pypdf(reader: PdfReader) -> str:
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip()


def _extract_layout_text_from_tesseract_data(data: Dict[str, List[Any]]) -> str:
    lines: Dict[tuple, Dict[str, Any]] = {}
    for i, token in enumerate(data.get("text", [])):
        token = (token or "").strip()
        if not token:
            continue
        try:
            conf = float(data["conf"][i])
        except Exception:
            conf = -1.0
        if conf < 0:
            continue
        key = (data["page_num"][i], data["block_num"][i], data["par_num"][i], data["line_num"][i])
        if key not in lines:
            lines[key] = {"tokens": [], "left": data["left"][i], "top": data["top"][i], "right": data["left"][i] + data["width"][i]}
        lines[key]["tokens"].append(token)
        lines[key]["left"] = min(lines[key]["left"], data["left"][i])
        lines[key]["right"] = max(lines[key]["right"], data["left"][i] + data["width"][i])
        lines[key]["top"] = min(lines[key]["top"], data["top"][i])

    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for key, value in lines.items():
        page_num = int(key[0])
        value["text"] = " ".join(value["tokens"]).strip()
        grouped.setdefault(page_num, []).append(value)

    pages_out: List[str] = []
    for page_num in sorted(grouped.keys()):
        page_lines = [ln for ln in grouped[page_num] if ln["text"]]
        if not page_lines:
            continue
        min_left = min(ln["left"] for ln in page_lines)
        max_right = max(ln["right"] for ln in page_lines)
        width = max(1, max_right - min_left)
        threshold = min_left + (width // 2)
        left_col = [ln for ln in page_lines if ln["left"] <= threshold]
        right_col = [ln for ln in page_lines if ln["left"] > threshold]
        use_two_cols = len(left_col) > 4 and len(right_col) > 4 and (min(ln["left"] for ln in right_col) - max(ln["right"] for ln in left_col)) > 10
        if use_two_cols:
            ordered = sorted(left_col, key=lambda x: (x["top"], x["left"])) + sorted(right_col, key=lambda x: (x["top"], x["left"]))
        else:
            ordered = sorted(page_lines, key=lambda x: (x["top"], x["left"]))
        pages_out.append("\n".join(ln["text"] for ln in ordered))
    return "\n\n".join(pages_out).strip()


def extract_text_from_pdf(uploaded_pdf, enable_ocr: bool = False, ocr_languages: str = "eng+hin", ocr_dpi: int = 300):
    """Extract text from PDF. Uses parser extraction first, then optional OCR fallback."""
    text = ""
    try:
        with pdfplumber.open(uploaded_pdf) as pdf:
            pages = []
            for page in pdf.pages:
                page_text = page.extract_text(x_tolerance=3, y_tolerance=3)
                if page_text:
                    pages.append(page_text)
            text = "\n".join(pages).strip()
            if text:
                return text
    except Exception:
        pass

    try:
        if hasattr(uploaded_pdf, "seek"):
            uploaded_pdf.seek(0)
        reader = PdfReader(uploaded_pdf)
        text = _extract_pages_pypdf(reader)
        if text:
            return text
    except Exception:
        pass

    if not enable_ocr:
        raise ValueError("No extractable text found. The PDF may be image-only or empty. Re-run with OCR enabled.")

    try:
        from pdf2image import convert_from_bytes
        import pytesseract
        from pytesseract import Output
    except Exception as e:
        raise RuntimeError(
            f"OCR dependencies are missing ({e}). Install pytesseract, pdf2image, Pillow and Tesseract binaries."
        ) from e

    if hasattr(uploaded_pdf, "seek"):
        uploaded_pdf.seek(0)
    data = uploaded_pdf.read() if hasattr(uploaded_pdf, "read") else None
    if hasattr(uploaded_pdf, "seek"):
        uploaded_pdf.seek(0)
    if not data:
        raise ValueError("Unable to read PDF bytes for OCR.")

    images = convert_from_bytes(data, dpi=ocr_dpi)
    pages_out: List[str] = []
    conf_scores: List[float] = []
    for image in images:
        ocr_data = pytesseract.image_to_data(image, lang=ocr_languages, output_type=Output.DICT)
        page_text = _extract_layout_text_from_tesseract_data(ocr_data)
        if page_text:
            pages_out.append(page_text)
        vals = []
        for c in ocr_data.get("conf", []):
            try:
                cv = float(c)
                if cv >= 0:
                    vals.append(cv)
            except Exception:
                continue
        if vals:
            conf_scores.append(sum(vals) / len(vals))
    text = "\n\n".join(pages_out).strip()
    if not text:
        raise ValueError("OCR completed but no readable text was extracted.")
    if conf_scores:
        logging.info("ocr_confidence_avg=%.2f", sum(conf_scores) / len(conf_scores))
    return text


def validate_pdf_metadata(uploaded_file):
    """
    Check PDF size and page count to warn user if it's too large.
    Returns (is_valid, message, level) where level is 'warning' or 'error'.
    """
    if not uploaded_file:
        return True, None, None
    
    # Size check (25MB)
    if uploaded_file.size > 25 * 1024 * 1024:
        return True, "⚠️ This file is quite large. Processing may take longer than usual.", "warning"
        
    try:
        reader = PdfReader(uploaded_file)
        num_pages = len(reader.pages)
        if num_pages > 1000:
            return False, "🛑 Extremely large PDF (1000+ pages) detected. Character limits will be exceeded, leading to a very poor summary. Please upload a shorter excerpt.", "error"
        if num_pages > 100:
            return True, f"⚠️ This document has {num_pages} pages. Summaries of long judgments may be less precise.", "warning"
    except Exception as e:
        logging.error(f"Validation PDF reader failed: {str(e)}")
        return False, "Could not read PDF metadata. The file might be corrupted.", "error"
        
    return True, None, None


def compress_text(text, limit=6000):
    if len(text) <= limit:
        return text
    head = text[:3000]
    tail = text[-3000:]
    return head + "\n\n... [TRUNCATED] ...\n\n" + tail


def english_leakage_detected(output_text, threshold=8):
    if not output_text or len(output_text.strip()) < 10:
        return False
    try:
        langs = detect_langs(output_text)
        for l in langs:
            if l.lang == 'en' and l.prob > 0.8:
                return True
            if l.lang == 'en' and langs[0].lang == 'en' and l.prob > 0.5:
                return True
    except Exception as e:
        logging.debug(f"langdetect failed: {e}")

    common_english = [
        " the ", " and ", " of ", " to ", " in ", " is ", " that ", " it ", " for ", " on ",
        " with ", " as ", " this ", " was ", " are ", " at ", " by ", " be ", " or ", " has "
    ]
    text_lower = " " + re.sub(r'[^\w\s]', ' ', output_text.lower()) + " "
    count = sum(1 for word in common_english if word in text_lower)
    return count >= threshold


LANGUAGE_OUTPUT_RULES = {
    "Assamese": "Use Assamese language in the Assamese form of the Bengali-Assamese script. Do not write Bengali, Tamil, Hindi, English, or the PDF's original language.",
    "Bengali": "Use Bengali language in Bengali script. Do not write Assamese, Tamil, Hindi, English, or the PDF's original language.",
    "Bodo": "Use Bodo language in Devanagari script.",
    "Dogri": "Use Dogri language in Devanagari script.",
    "Gujarati": "Use Gujarati language in Gujarati script.",
    "Hindi": "Use Hindi language in Devanagari script.",
    "Kannada": "Use Kannada language in Kannada script.",
    "Kashmiri": "Use Kashmiri language in the script most natural for Indian Kashmiri readers.",
    "Konkani": "Use Konkani language in Devanagari script.",
    "Maithili": "Use Maithili language in Devanagari script.",
    "Malayalam": "Use Malayalam language in Malayalam script.",
    "Manipuri": "Use Manipuri language in Meetei Mayek or Bengali script, but keep the vocabulary and grammar Manipuri.",
    "Marathi": "Use Marathi language in Devanagari script.",
    "Nepali": "Use Nepali language in Devanagari script.",
    "Odia": "Use Odia language in Odia script.",
    "Punjabi": "Use Punjabi language in Gurmukhi script.",
    "Sanskrit": "Use Sanskrit language in Devanagari script.",
    "Santhali": "Use Santhali language in Ol Chiki script.",
    "Sindhi": "Use Sindhi language in the script most natural for Indian Sindhi readers.",
    "Tamil": "Use Tamil language in Tamil script.",
    "Telugu": "Use Telugu language in Telugu script.",
    "Urdu": "Use Urdu language in Perso-Arabic script.",
}

INDIC_SCRIPT_RANGES = {
    "Devanagari": (0x0900, 0x097F),
    "Bengali-Assamese": (0x0980, 0x09FF),
    "Gurmukhi": (0x0A00, 0x0A7F),
    "Gujarati": (0x0A80, 0x0AFF),
    "Odia": (0x0B00, 0x0B7F),
    "Tamil": (0x0B80, 0x0BFF),
    "Telugu": (0x0C00, 0x0C7F),
    "Kannada": (0x0C80, 0x0CFF),
    "Malayalam": (0x0D00, 0x0D7F),
    "Sinhala": (0x0D80, 0x0DFF),
    "Ol Chiki": (0x1C50, 0x1C7F),
    "Arabic": (0x0600, 0x06FF),
    "Meetei Mayek": (0xABC0, 0xABFF),
}

LANGUAGE_ALLOWED_SCRIPTS = {
    "Assamese": {"Bengali-Assamese"},
    "Bengali": {"Bengali-Assamese"},
    "Bodo": {"Devanagari"},
    "Dogri": {"Devanagari"},
    "Gujarati": {"Gujarati"},
    "Hindi": {"Devanagari"},
    "Kannada": {"Kannada"},
    "Kashmiri": {"Arabic", "Devanagari"},
    "Konkani": {"Devanagari"},
    "Maithili": {"Devanagari"},
    "Malayalam": {"Malayalam"},
    "Manipuri": {"Meetei Mayek", "Bengali-Assamese"},
    "Marathi": {"Devanagari"},
    "Nepali": {"Devanagari"},
    "Odia": {"Odia"},
    "Punjabi": {"Gurmukhi"},
    "Sanskrit": {"Devanagari"},
    "Santhali": {"Ol Chiki", "Devanagari"},
    "Sindhi": {"Arabic", "Devanagari"},
    "Tamil": {"Tamil"},
    "Telugu": {"Telugu"},
    "Urdu": {"Arabic"},
}


def _language_output_rule(language):
    if not language or language == "English":
        return "Use clear English."
    return LANGUAGE_OUTPUT_RULES.get(
        language,
        f"Use natural {language}. Do not write English or the PDF's original language.",
    )


def _count_script_chars(text, script_names):
    total = 0
    for char in text or "":
        codepoint = ord(char)
        for script_name in script_names:
            start, end = INDIC_SCRIPT_RANGES[script_name]
            if start <= codepoint <= end:
                total += 1
                break
    return total


def output_language_mismatch_detected(output_text, language, min_wrong_chars=6):
    """
    Detect obvious wrong-language output for retry decisions.
    This is intentionally conservative because names, courts, and citations may
    contain Latin text even in a localized summary.
    """
    if not output_text or not language or language == "English":
        return False

    if english_leakage_detected(output_text, threshold=4):
        return True

    allowed_scripts = LANGUAGE_ALLOWED_SCRIPTS.get(language)
    if not allowed_scripts:
        return False

    wrong_scripts = set(INDIC_SCRIPT_RANGES) - allowed_scripts
    allowed_count = _count_script_chars(output_text, allowed_scripts)
    wrong_count = _count_script_chars(output_text, wrong_scripts)

    return wrong_count >= min_wrong_chars and (
        allowed_count == 0 or wrong_count / max(allowed_count, 1) > 0.15
    )


# ==================== LLM PROMPTS ====================

def build_prompt(safe_text, language):
    """Build prompt for judgment simplification"""
    language_rule = _language_output_rule(language)
    return f"""
You are LegalEase AI — an expert judicial-simplification and translation engine.

MISSION:
Convert the judgment text into a simple, citizen-friendly summary in the user's selected language.

TARGET LANGUAGE:
- Language: {language}
- Rule: {language_rule}
- The source PDF may be in another language. Ignore the source language for output.
- Never mix languages. If a sentence is not in {language}, rewrite it before answering.

INSTRUCTIONS:
1. Extract ONLY the final judgment outcome.
2. Remove all legal jargon and case history.
3. Produce AT LEAST 5 bullet points. More than 5 is allowed if needed.
4. Write ONLY in {language}. ZERO English allowed if language ≠ English.
5. Each bullet must be 1–2 very short sentences.
6. Put every bullet point on its own new line.
7. No extra headings. No disclaimers.

TEXT TO ANALYZE:
{safe_text}

OUTPUT REQUIRED:
- Minimum 5 bullet points in {language} only
"""


def build_retry_prompt(safe_text, language):
    """Build retry prompt when output language mismatch is detected"""
    language_rule = _language_output_rule(language)
    return f"""
Your previous answer used the wrong language or mixed languages.
Now STRICTLY produce the answer ONLY in {language}.

TARGET LANGUAGE:
- Language: {language}
- Rule: {language_rule}
- Translate the result into {language}; do not copy the PDF's original language.

REQUIREMENTS:
- Minimum 5 bullet points
- VERY simple {language}
- No non-target language text at all
- No introductions, headings, or explanations

TEXT:
{safe_text}

OUTPUT NOW:
Minimum 5 bullet points in {language} only.
"""


# ==================== FIX-1: REMEDIES PROMPT (language-enforced) ====================
# OLD: prompt asked for answers in {language} but system message was generic English.
# FIX: system message now explicitly enforces the target language and its script rule.

def build_remedies_prompt(judgment_text, language):
    """Build prompt for remedies advisor to analyze legal options"""
    language_rule = _language_output_rule(language)
    return f"""
You are a Legal Rights Advisor. Read this judgment and answer the questions below.

JUDGMENT:
{judgment_text}

Answer ONLY these questions in {language}. Be practical and direct.

TARGET LANGUAGE:
- Language: {language}
- Rule: {language_rule}
- The source PDF may be in another language. Ignore the source language for output.
- Never mix languages. If a sentence is not in {language}, rewrite it before answering.
- Do not repeat the question labels; write only the answer after each number.

1. What happened? (Who won and who lost; 1 sentence)
2. Can the loser appeal? (yes/no equivalent in {language} + reason; 1-2 sentences)
3. Appeal timeline: How many days? (Just number)
4. Appeal court: Which court should they go to? (Court name only)
5. Cost estimate: Rough cost in rupees? (e.g., 5000-15000)
6. First action: What should they do first? (1 sentence)
7. Important deadline: What key deadline should they remember? (1 sentence)

Output in numbered form like:
1. ...\n2. ...\n3. ... etc.
"""


# ==================== REMEDIES PARSER ====================

KNOWN_COURTS = {
    "supreme court", "high court", "district court", "sessions court",
    "session court", "civil court", "family court", "consumer court", "tribunal",
}


def _clean_answer(value: str):
    cleaned = re.sub(r"\s+", " ", (value or "")).strip(" -:\t\n")
    return cleaned or ""


def _strip_question_label(key: str, value: str) -> str:
    if not value:
        return ""
    patterns = {
        "what_happened": r"^(?:\*\*)?(what happened\??)(?:\*\*)?\s*",
        "can_appeal": r"^(?:\*\*)?(can the loser appeal\??)(?:\*\*)?\s*",
        "appeal_days": r"^(?:\*\*)?(appeal timeline\??|how many days\??)(?:\*\*)?\s*",
        "appeal_court": r"^(?:\*\*)?(appeal court\??|which court(?: should they go to)?\??)(?:\*\*)?\s*",
        "cost_estimate": r"^(?:\*\*)?(cost estimate\??|rough cost(?: in rupees)?\??)(?:\*\*)?\s*",
        "first_action": r"^(?:\*\*)?(first action\??|what should they do first\??)(?:\*\*)?\s*",
        "deadline": r"^(?:\*\*)?(important deadline\??|important dates?\??)(?:\*\*)?\s*",
    }
    pattern = patterns.get(key)
    if pattern:
        value = re.sub(pattern, "", value, flags=re.IGNORECASE).strip()
    return value or ""


YES_ALIASES = {
    "haan",
    "\u0907\u092f\u0938",
    "\u0939\u093e\u0901",
    "\u0939\u093e\u0902",
    "\u09b9\u09af\u09bc",
    "\u09b9\u09cd\u09af\u09be\u0981",
    "\u0a39\u0a3e\u0a02",
    "\u0ab9\u0abe",
    "\u0b39\u0b01",
    "\u0b86\u0bae\u0bcd",
    "\u0c05\u0c35\u0c41\u0c28\u0c41",
    "\u0cb9\u0ccc\u0ca6\u0cc1",
    "\u0d05\u0d24\u0d46",
    "\u06c1\u0627\u06ba",
}

NO_ALIASES = {
    "nahin",
    "nahi",
    "\u0928\u0939\u0940\u0902",
    "\u0928\u0939\u0940",
    "\u09a8\u09b9\u09af\u09bc",
    "\u09a8\u09be\u0987",
    "\u09a8\u09be",
    "\u0a28\u0abe",
    "\u0aa8\u0abe",
    "\u0b28\u0b3e",
    "\u0b87\u0bb2\u0bcd\u0bb2\u0bc8",
    "\u0c15\u0c3e\u0c26\u0c41",
    "\u0c87\u0cb2\u0ccd\u0cb2",
    "\u0d07\u0d32\u0d4d\u0d32",
    "\u0646\u06c1\u06cc\u06ba",
    "\u0646\u0627",
}


def _contains_alias(value, aliases):
    return any(alias in value for alias in aliases if alias)


def _normalize_yes_no(value: str) -> str:
    """
    Normalize yes/no answer. Returns 'yes', 'no', or the original value unchanged.
    The caller (localize_yes_no) handles display translation.
    """
    if not value:
        return ""
    lower = value.lower()
    if re.search(r"\bno\b", lower) or any(x in lower for x in ["cannot appeal", "not available", "no right", "no appeal"]) or _contains_alias(lower, NO_ALIASES):
        return "no"
    if re.search(r"\byes\b", lower) or any(x in lower for x in ["can appeal", "available", "allowed", "has right"]) or _contains_alias(lower, YES_ALIASES):
        return "yes"
    return ""

def _extract_number(value: str) -> str:
    if not value:
        return ""
    match = re.search(r"\b(\d{1,4})\b", value)
    return match.group(1) if match else ""


# ==================== FIX-1: _validate_court_name — don't strip non-English names ====================
# OLD: only returned a value if it matched English KNOWN_COURTS list, otherwise returned "".
#      This silently dropped court names written in Indic scripts.
# FIX: if the value doesn't match English known courts, return it as-is (trust the LLM).

def _validate_court_name(value: str) -> str:
    if not value:
        return ""
    cleaned = _clean_answer(value)
    if not cleaned:
        return ""
    normalized = cleaned.lower()
    # If it matches an English known court name, return the cleaned version.
    if normalized in KNOWN_COURTS or any(court in normalized for court in KNOWN_COURTS):
        return cleaned
    # FIX: For Indic / non-English court names, trust the LLM output rather than discarding it.
    return cleaned


def parse_remedies_response(response_text):
    """
    Extract structured info from LLM response using flexible numbered-line parsing.
    """
    mapping = {
        1: "what_happened",
        2: "can_appeal",
        3: "appeal_days",
        4: "appeal_court",
        5: "cost_estimate",
        6: "first_action",
        7: "deadline",
    }
    remedies = {
        "what_happened": "",
        "can_appeal": "",
        "appeal_days": "",
        "appeal_court": "",
        "cost_estimate": "",
        "cost": "",
        "first_action": "",
        "deadline": "",
        "appeal_details": "",
        "_is_partial": False,
        "_warning": "",
    }

    text = response_text.strip()
    if not text:
        return remedies

    # Use robust marker-based parsing
    marker_pattern = re.compile(r"(?m)^\s*(?:\*\*)?(\d{1,2})(?:\*\*)?\s*[\.|\)|:|-]\s*(.*)$")
    matches = list(marker_pattern.finditer(text))

    if not matches:
        logging.warning("parse_remedies_response: no numbered sections found")
        return remedies

    for idx, match in enumerate(matches):
        section_num = int(match.group(1))
        key = mapping.get(section_num)
        if not key:
            continue
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        inline_text = _clean_answer(match.group(2))
        block_text = _clean_answer(text[start:end])
        section_text = _clean_answer(
            " ".join(part for part in [inline_text, block_text] if part)
        )
        cleaned = _strip_question_label(key, section_text)
        if cleaned:
            remedies[key] = cleaned

    # Normalize
    if remedies["can_appeal"]:
        remedies["can_appeal"] = _normalize_yes_no(remedies["can_appeal"])

    if remedies["appeal_days"]:
        remedies["appeal_days"] = _extract_number(remedies["appeal_days"])

    if remedies["appeal_court"]:
        remedies["appeal_court"] = _validate_court_name(remedies["appeal_court"])

    if remedies["cost_estimate"]:
        remedies["cost"] = remedies["cost_estimate"]

    required = ["what_happened", "can_appeal", "appeal_days", "appeal_court", "cost", "first_action", "deadline"]
    missing = [f for f in required if not remedies[f]]
    if missing:
        remedies["_is_partial"] = True
        remedies["_warning"] = "Note: Some information may be incomplete."

    return remedies


def extract_appeal_info(appeal_details_text):
    info = {"days": "", "court": "", "cost": ""}
    text = appeal_details_text or ""
    days_match = re.search(r"(\d+)\s*(?:days?|day)", text.lower())
    if days_match:
        info["days"] = days_match.group(1)
    court_keywords = ["High Court", "District Court", "Supreme Court", "Lower Court"]
    for keyword in court_keywords:
        if keyword.lower() in text.lower():
            info["court"] = keyword
            break
    cost_match = re.search(r"₹?([\d,]+(?:[-–][\d,]+)?)", text.replace(" ", ""))
    if cost_match:
        info["cost"] = cost_match.group(1)
    return info


# ==================== FIX-1+2: get_remedies_advice — language-enforced system prompt + more tokens ====================
# OLD: system message was generic English "helpful legal advisor"; max_tokens=500 (too short).
# FIX: system message now names the target language explicitly; max_tokens raised to 900.

def get_remedies_advice(judgment_text, language, client=None):
    """Call LLM to get remedies for this judgment — answers in the selected language."""
    if client is None:
        client = get_client()
    if not client:
        return None

    prompt = build_remedies_prompt(compress_text(judgment_text), language)
    language_rule = _language_output_rule(language)

    try:
        response = client.chat.completions.create(
            model=get_default_model(),
            messages=[
                {
                    "role": "system",
                    # FIX-1: was generic English; now strictly enforces the target language.
                    "content": (
                        f"You are a legal rights advisor for Indian citizens. "
                        f"You MUST answer ONLY in {language}. {language_rule} "
                        f"Never use English words or sentences unless {language} is English. "
                        f"Never mix languages. Be thorough and write 2-3 sentences per answer."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            # FIX-2: was 500 — too low for detailed multi-language answers; raised to 900.
            max_tokens=900,
            temperature=0.1,
        )

        response_text = response.choices[0].message.content.strip()
        remedies = parse_remedies_response(response_text)

        if remedies is None:
            return {k: None for k in
                    ["what_happened", "can_appeal", "appeal_days", "appeal_court",
                     "cost_estimate", "cost", "first_action", "deadline"]}
        return remedies

    except Exception as e:
        logging.error(f"Failed to get remedies advice: {str(e)}")
        return None


# ==================== UI STYLING & CONSTANTS ====================

RETRO_STYLING = """
<style>
    body { background-color: #0d0d0f; color: #e0e0e0; font-family: 'Inter', sans-serif; }
    .main { background-color: #0d0d0f; }
    .stButton>button {
        background: linear-gradient(90deg, #2d2dff, #8a2be2);
        border-radius: 8px; color: white; font-weight: 600;
        border: none; padding: 0.6rem 1.2rem;
    }
    .stSelectbox>div>div { background-color: #1a1a1d; color: #e0e0e0; border-radius: 6px; }
    .stTextArea>div>textarea { background-color: #121214; color: #e0e0e0; }
</style>
"""

LEGAL_HELP_RESOURCES = """
**You don't have to handle this alone. Here are free resources:**

🔗 **National Legal Services (Free Lawyer)**
- Phone: 1800-180-8111
- Website: nalsa.gov.in
- For: Everyone (especially poor citizens)

🔗 **Bar Council of India (Find Verified Lawyers)**
- Website: bci.org.in
- For: Finding qualified lawyers in your area

🔗 **Legal Clinics (Law Colleges)**
- Most law colleges offer free consultation
- Search: "[Your City] law college legal clinic"

🔗 **NGOs for Specific Cases**
- Family cases: National Commission for Women (1800-123-4344)
- Criminal cases: Criminal Law Clinic (project39a.com)
- Tenant rights: Housing rights organizations

**Tip:** Start with National Legal Services. They are free and available.
"""

UI_TEXT = {
    "app_subtitle": "Legal Judgment Simplifier",
    "app_intro": (
        "LegalEase AI breaks the Information Barrier in the Judiciary by converting "
        "complex court judgments into clear, 3-point summaries in your chosen language."
    ),
    "language_label": "🌐 Select your language",
    "upload_label": "📄 Upload Judgment PDF",
    "generate_summary": "🚀 Generate Summary",
    "processing": "Processing judgment...",
    "api_client_failed": "Failed to initialize API client",
    "openrouter_not_configured": "OpenRouter client not configured. Check your API keys.",
    "empty_summary": "The model returned an empty summary. Try a shorter file or switch to English.",
    "simplified_judgment": "✅ Simplified Judgment",
    "summary_success": "The judgment has been simplified successfully.",
    "remedies_title": "⚖️ What Can You Do Now?",
    "remedies_spinner": "Analyzing your legal options...",
    "what_happened": "What Happened?",
    "can_appeal": "Can You Appeal?",
    "appeal_details": "Appeal Details",
    "days_to_file_appeal": "Days to File Appeal",
    "appeal_to": "Appeal to",
    "estimated_cost": "Estimated Cost",
    "first_action": "What Should You Do First?",
    "important_deadline": "⏰ Important Deadline",
    "remedies_error": "Could not get remedies advice",
    "partial_warning": "Note: Some information may be incomplete.",
    "track_title": "📊 Track Your Case & See Statistics",
    "track_info": (
        "**Help us build better predictions!**\n\n"
        "By tracking your case, you help us understand appeal success rates in your jurisdiction. "
        "Later, when you know the outcome of your appeal, you can report it back."
    ),
    "view_analytics": "📈 View Analytics",
    "estimate_chances": "🎯 Estimate Appeal Chances",
    "report_outcome": "📝 Report Outcome",
    "quick_analytics_preview": "📊 Quick Analytics Preview",
    "total_cases_tracked": "Total Cases Tracked",
    "appeals_success_rate": "Appeals Success Rate",
    "appeals_filed": "Appeals Filed",
    "analytics_link_text": "Visit Analytics Dashboard for detailed insights",
    "analytics_empty": "Analytics will be available as more cases are tracked.",
    "analytics_not_ready": "Analytics module not ready yet.",
    "free_legal_help": "📞 Free Legal Help",
    "legal_help_resources": LEGAL_HELP_RESOURCES,
    "not_enough_credits": "❌ Not enough OpenRouter credits. Please top up.",
    "connection_error": (
        "⚠️ Connection Error: {error}\n\n"
        "Troubleshooting:\n- Check your internet connection\n- Verify API key in .env\n"
        "- Check OpenRouter status at openrouter.ai"
    ),
    "generic_error": "❌ Error: {error}\n\n📋 Details for debugging: Check terminal logs",
    "yes": "Yes",
    "no": "No",
}

UI_TEXT_TRANSLATIONS = {}

try:
    _translations_path = Path(__file__).parent / "all_translations.json"
    if _translations_path.exists():
        with open(_translations_path, "r", encoding="utf-8") as _f:
            UI_TEXT_TRANSLATIONS = json.load(_f)
except Exception as e:
    logging.error("Failed to load all_translations.json: %s", e)


STATIC_UI_TEXT_TRANSLATIONS = {
    "Assamese": {
        "app_subtitle": "আইনী ৰায়ৰ সৰলীকৰণ",
        "app_intro": (
            "LegalEase AI-এ জটিল আদালতৰ ৰায়সমূহ আপোনাৰ বাছনি কৰা ভাষাত স্পষ্ট, "
            "৩টা মূল কথাৰ সাৰাংশলৈ ৰূপান্তৰ কৰি ন্যায়ব্যৱস্থাৰ তথ্য-অৱৰোধ কমায়।"
        ),
        "language_label": "🌐 আপোনাৰ ভাষা বাছনি কৰক",
        "upload_label": "📄 ৰায়ৰ পি ডি এফ আপলোড কৰক",
        "generate_summary": "🚀 সাৰাংশ সৃষ্টি কৰক",
        "processing": "ৰায় প্ৰক্ৰিয়াকৰণ হৈ আছে...",
        "api_client_failed": "এপিআই ক্লায়েণ্ট আৰম্ভ কৰাত ব্যৰ্থ",
        "openrouter_not_configured": "OpenRouter ক্লায়েণ্ট কনফিগাৰ কৰা হোৱা নাই। আপোনাৰ এপিআই কী পৰীক্ষা কৰক।",
        "empty_summary": "মডেলে খালী সাৰাংশ ঘূৰাইছে। সৰু ফাইল চেষ্টা কৰক বা ইংৰাজীলৈ সলনি কৰক।",
        "simplified_judgment": "✅ সৰলীকৃত ৰায়",
        "summary_success": "ৰায়টো সফলতাৰে সৰল কৰা হৈছে।",
        "remedies_title": "⚖️ এতিয়া আপুনি কি কৰিব পাৰে?",
        "remedies_spinner": "আপোনাৰ আইনী বিকল্পসমূহ বিশ্লেষণ কৰি থকা হৈছে...",
        "what_happened": "কি ঘটিল?",
        "can_appeal": "আপুনি আপীল কৰিব পাৰিবনে?",
        "appeal_details": "আপীলৰ বিৱৰণ",
        "days_to_file_appeal": "আপীল দাখিল কৰিবলৈ দিন",
        "appeal_to": "আপীল কৰিবলগীয়া আদালত",
        "estimated_cost": "আনুমানিক খৰচ",
        "first_action": "প্ৰথমে আপুনি কি কৰিব লাগে?",
        "important_deadline": "⏰ গুৰুত্বপূর্ণ সময়সীমা",
        "remedies_error": "আইনী পৰামৰ্শ আনিব পৰা নগ'ল",
        "partial_warning": "টোকা: কিছু তথ্য অসম্পূৰ্ণ হ'ব পাৰে।",
        "track_title": "📊 আপোনাৰ গোচৰ অনুসৰণ কৰক আৰু পৰিসংখ্যা চাওক",
        "track_info": (
            "**উন্নত অনুমান তৈয়াৰ কৰাত আমাক সহায় কৰক!**\n\n"
            "আপোনাৰ গোচৰ অনুসৰণ কৰিলে, আপোনাৰ অঞ্চলত আপীল সফলতাৰ হাৰ বুজিবলৈ আমাক সহায় হয়। "
            "পিছত আপীলৰ ফলাফল জানিলে আপুনি আমাক জনাব পাৰে।"
        ),
        "view_analytics": "📈 বিশ্লেষণ চাওক",
        "estimate_chances": "🎯 আপীলৰ সম্ভাৱনা অনুমান কৰক",
        "report_outcome": "📝 ফলাফল জনাওক",
        "quick_analytics_preview": "📊 দ্ৰুত বিশ্লেষণ পূৰ্বদৰ্শন",
        "total_cases_tracked": "মুঠ অনুসৰণ কৰা গোচৰ",
        "appeals_success_rate": "আপীলৰ সফলতাৰ হাৰ",
        "appeals_filed": "দাখিল কৰা আপীল",
        "analytics_link_text": "বিশদ তথ্যৰ বাবে বিশ্লেষণ ডেশ্বব'ৰ্ড চাওক",
        "analytics_empty": "অধিক গোচৰ অনুসৰণ হ'লে বিশ্লেষণ উপলব্ধ হ'ব।",
        "analytics_not_ready": "বিশ্লেষণ মডিউল এতিয়াও সাজু নহয়।",
        "free_legal_help": "📞 বিনামূলীয়া আইনী সহায়",
        "legal_help_resources": (
            "**এইটো আপুনি অকলে সামলাব লাগিব নালাগে। ইয়াত বিনামূলীয়া সহায়ৰ পথ আছে:**\n\n"
            "🔗 **ৰাষ্ট্ৰীয় আইনী সেৱা (বিনামূলীয়া অধিবক্তা)**\n"
            "- ফোন: 1800-180-8111\n"
            "- ৱেবছাইট: nalsa.gov.in\n"
            "- সকলোৰে বাবে, বিশেষকৈ আৰ্থিকভাৱে দুৰ্বল নাগৰিকৰ বাবে\n\n"
            "🔗 **বাৰ কাউন্সিল অৱ ইণ্ডিয়া (যাচাইকৃত অধিবক্তা বিচাৰক)**\n"
            "- ৱেবছাইট: bci.org.in\n\n"
            "🔗 **আইনী ক্লিনিক (আইন মহাবিদ্যালয়)**\n"
            "- বহু আইন মহাবিদ্যালয়ে বিনামূলীয়া পৰামৰ্শ দিয়ে\n\n"
            "**পৰামৰ্শ:** প্ৰথমে ৰাষ্ট্ৰীয় আইনী সেৱাৰ সৈতে যোগাযোগ কৰক।"
        ),
        "not_enough_credits": "❌ পৰ্যাপ্ত OpenRouter ক্ৰেডিট নাই। অনুগ্ৰহ কৰি টপ আপ কৰক।",
        "connection_error": (
            "⚠️ সংযোগ ত্ৰুটি: {error}\n\n"
            "সমাধান:\n- ইণ্টাৰনেট সংযোগ পৰীক্ষা কৰক\n- .env-ত এপিআই কী পৰীক্ষা কৰক\n"
            "- openrouter.ai-ত OpenRouter অৱস্থা চাওক"
        ),
        "generic_error": "❌ ত্ৰুটি: {error}\n\n📋 ডিবাগৰ বিৱৰণ: টাৰ্মিনেল লগ চাওক",
        "yes": "হয়",
        "no": "নহয়",
    }
}


def _parse_json_object(text):
    if not text:
        return {}

    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return {}
    return {}


def _translate_ui_text(language, text_map, client=None):
    if not client:
        return {}

    language_rule = _language_output_rule(language)
    prompt = f"""
Translate this JSON object's values into {language}.
Rules:
- Return JSON only.
- Keep the exact same keys.
- Keep emoji, Markdown, placeholders like {{error}}, URLs, phone numbers, and product names unchanged.
- IMPORTANT: The brand name "LegalEase AI" and "⚡ LegalEase AI" MUST NOT be translated. Keep it exactly as it is.
- {language_rule}
- Do not use English or any non-target language except unchanged brand names, URLs, phone numbers, and placeholders.
- Do not add commentary.

JSON:
{json.dumps(text_map, ensure_ascii=False, indent=2)}
"""
    try:
        response = client.chat.completions.create(
            model=get_default_model(),
            messages=[
                {"role": "system", "content": "You translate UI copy accurately and return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=2200,
            temperature=0.0,
            timeout=45.0,
        )
        translated = _parse_json_object(response.choices[0].message.content)
    except Exception as exc:
        logging.warning("Failed to translate UI text for %s: %s", language, exc)
        return {}

    return {
        key: str(value)
        for key, value in translated.items()
        if key in text_map
        and isinstance(value, (str, int, float))
        and not output_language_mismatch_detected(str(value), language)
    }


def _is_untranslated_ui_value(key, value):
    """Treat copied English strings as missing translations."""
    if key not in UI_TEXT:
        return False
    return str(value).strip() == str(UI_TEXT[key]).strip()


def get_localized_ui_text(language, client=None):
    """Return user-facing UI text in the selected language when available."""
    if not language or language == "English":
        return dict(UI_TEXT)

    text = dict(UI_TEXT)
    stored_translation = {
        **UI_TEXT_TRANSLATIONS.get(language, {}),
        **STATIC_UI_TEXT_TRANSLATIONS.get(language, {}),
    }
    usable_translation = {
        key: value
        for key, value in stored_translation.items()
        if not _is_untranslated_ui_value(key, value)
    }
    text.update(usable_translation)

    missing_keys = [key for key in UI_TEXT if key not in usable_translation]
    if missing_keys and client:
        cached_translation = _LOCALIZED_UI_TEXT_CACHE.setdefault(language, {})
        uncached_keys = [key for key in missing_keys if key not in cached_translation]
        if uncached_keys:
            missing_text = {key: UI_TEXT[key] for key in uncached_keys}
            cached_translation.update(_translate_ui_text(language, missing_text, client))
        text.update(cached_translation)

    return text


def localize_yes_no(value, ui_text):
    normalized = (value or "").strip().lower()
    if normalized == "yes":
        return ui_text.get("yes", "Yes")
    if normalized == "no":
        return ui_text.get("no", "No")
    return value

SCHEDULED_INDIAN_LANGUAGES = [
    "Assamese",
    "Bengali",
    "Bodo",
    "Dogri",
    "Gujarati",
    "Hindi",
    "Kannada",
    "Kashmiri",
    "Konkani",
    "Maithili",
    "Malayalam",
    "Manipuri",
    "Marathi",
    "Nepali",
    "Odia",
    "Punjabi",
    "Sanskrit",
    "Santhali",
    "Sindhi",
    "Tamil",
    "Telugu",
    "Urdu",
]

LANGUAGES = ["English", *SCHEDULED_INDIAN_LANGUAGES]

LANGUAGE_ALIASES = {
    **{language.lower(): language for language in LANGUAGES},
    "oriya": "Odia",
    "santali": "Santhali",
    "meitei": "Manipuri",
    "meiteilon": "Manipuri",
}

LANGUAGE_CODE_TO_NAME = {
    "as": "Assamese",
    "bn": "Bengali",
    "brx": "Bodo",
    "doi": "Dogri",
    "en": "English",
    "gu": "Gujarati",
    "hi": "Hindi",
    "kn": "Kannada",
    "ks": "Kashmiri",
    "kok": "Konkani",
    "mai": "Maithili",
    "ml": "Malayalam",
    "mni": "Manipuri",
    "mr": "Marathi",
    "ne": "Nepali",
    "or": "Odia",
    "pa": "Punjabi",
    "sa": "Sanskrit",
    "sat": "Santhali",
    "sd": "Sindhi",
    "ta": "Tamil",
    "te": "Telugu",
    "ur": "Urdu",
}

# Alias for backward compatibility
build_summary_prompt = build_prompt

def parse_summary_bullets(raw_text):
    """
    Structured parser to ensure exactly 3 bullet points are extracted from LLM output.
    This eliminates introductory text (e.g., 'Here is your summary:') and 
    excessive output beyond the requested 3 bullets.
    """
    if not raw_text:
        return ""
    
    # Regex to identify lines starting with common bullet markers
    # Covers: - *, •, and numbered bullets like 1. or 1)
    # Added optional markdown bolding support
    bullet_marker_regex = re.compile(r"^\s*([\-\*\u2022\u25cf]|(?:\*\*)?(\d+)(?:\*\*)?[\.\)])\s*(.*)$")
    
    lines = raw_text.split('\n')
    bullets = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        match = bullet_marker_regex.match(line)
        if match:
            # We found a bullet! Extract the content after the marker
            content = match.group(3).strip()
            if content:
                bullets.append(content)
        elif len(bullets) < 3:
            # If no marker is found, but we still need bullets, check if this line
            # is a substantive sentence (longer than 20 chars) and doesn't look like a heading.
            if len(line) > 20 and not line.endswith(':'):
                # Filter out obvious intro/outro phrases that LLMs sometimes add
                lower_line = line.lower()
                intro_keywords = ["here is", "summary", "analysis", "judgment", "result"]
                if not any(keyword in lower_line for keyword in intro_keywords):
                    bullets.append(line)
        
        # Stop once we have 3 points
        if len(bullets) >= 3:
            break
                    
    # Re-format as a clean markdown list with consistent bullet markers
    final_bullets = bullets[:3]
    
    if not final_bullets:
        # Fallback for very unstructured output: take first 3 substantial non-intro lines
        final_bullets = [l.strip() for l in lines if len(l.strip()) > 15 and not l.endswith(':')][:3]
        
    if not final_bullets:
        # Final fallback: return the raw text if all parsing heuristics failed
        return raw_text
        
    return "\n".join([f"- {b}" for b in final_bullets])
