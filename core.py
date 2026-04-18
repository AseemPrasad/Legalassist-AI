from pypdf import PdfReader
import re
import logging
from pathlib import Path
from typing import Dict, Optional, List

LOGGER = logging.getLogger(__name__)

# -----------------------------
# Configuration
# -----------------------------
DEFAULT_MODEL = "meta-llama/llama-3.1-8b-instruct"

# -----------------------------
# PDF to text
# -----------------------------
def extract_text_from_pdf(pdf_input) -> str:
    """
    Extracts text from a PDF file or file-like object.
    """
    # Handle both Path objects and file-like objects (from Streamlit)
    if isinstance(pdf_input, (str, Path)):
        with open(pdf_input, "rb") as f:
            reader = PdfReader(f)
            text = _extract_pages(reader)
    else:
        # Streamlit's UploadedFile is a file-like object
        reader = PdfReader(pdf_input)
        text = _extract_pages(reader)
    
    if not text.strip():
        raise ValueError("No extractable text found. The PDF may be image-only or empty.")
    return text

def _extract_pages(reader: PdfReader) -> str:
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip()

# -----------------------------
# Compress text for token safety
# -----------------------------
def compress_text(text: str, limit: int = 6000) -> str:
    """
    Truncates text by taking the first and last portions, 
    ensuring we break at sentence or paragraph boundaries.
    """
    if len(text) <= limit:
        return text

    half = limit // 2
    
    # 1. Process Head: Try to find the last sentence/para boundary in the first half
    head_raw = text[:half]
    # Look for . ! ? followed by space/newline, or multiple newlines
    match_head = list(re.finditer(r'([.!?]\s+|\n+)', head_raw))
    if match_head:
        head_end = match_head[-1].end()
        # Only truncate at the boundary if it doesn't discard too much (at least 70% of half)
        if head_end > half * 0.7:
            head = head_raw[:head_end]
        else:
            head = head_raw
    else:
        # Fallback: search for last space to avoid cutting a word
        last_space = head_raw.rfind(' ')
        head = head_raw[:last_space] if last_space != -1 else head_raw

    # 2. Process Tail: Try to find the first sentence/para boundary in the last half
    tail_raw = text[-half:]
    match_tail = list(re.finditer(r'([.!?]\s+|\n+)', tail_raw))
    if match_tail:
        # Start after the first boundary found in the first 30% of the tail segment
        tail_start = match_tail[0].end()
        if tail_start < half * 0.3:
            tail = tail_raw[tail_start:]
        else:
            tail = tail_raw
    else:
        # Fallback: search for first space
        first_space = tail_raw.find(' ')
        tail = tail_raw[first_space+1:] if first_space != -1 else tail_raw

    return head.strip() + "\n\n... [TRUNCATED] ...\n\n" + tail.strip()

# -----------------------------
# Detect English leakage
# -----------------------------
def english_leakage_detected(output_text: str, threshold: int = 5) -> bool:
    common = [" the ", " and ", " of ", " to ", " in ", " is ", " that ", " it ", " for ", " on "]
    text_lower = " " + output_text.lower() + " "
    count = sum(1 for w in common if w in text_lower)
    return count >= threshold

# -----------------------------
# Build prompts
# -----------------------------
def build_summary_prompt(safe_text: str, language: str) -> str:
    return f"""
You are LegalEase AI — an expert judicial-simplification and translation engine.

MISSION:
Convert the judgment text into a simple, citizen-friendly summary.

INSTRUCTIONS:
1. Extract ONLY the final judgment outcome.
2. Remove all legal jargon and case history.
3. Produce EXACTLY 3 bullet points.
4. Write ONLY in {language}. ZERO English allowed if language ≠ English.
5. Each bullet must be 1–2 very short sentences.
6. No extra headings. No disclaimers.

TEXT TO ANALYZE:
{safe_text}

OUTPUT REQUIRED:
- 3 bullet points in {language} only
"""

def build_retry_prompt(safe_text: str, language: str) -> str:
    return f"""
Your previous answer included English. Now STRICTLY produce the answer ONLY in {language}.

REQUIREMENTS:
- Exactly 3 bullet points
- VERY simple {language}
- No English at all
- No introductions, headings, or explanations

TEXT:
{safe_text}

OUTPUT NOW:
3 bullet points in {language} only.
"""

def build_remedies_prompt(judgment_text: str, language: str) -> str:
    """
    Ask LLM to analyze what remedies are available
    based on the actual judgment content
    """
    return f"""
You are a Legal Rights Advisor. Read this judgment and answer in SIMPLE format.

JUDGMENT:
{judgment_text}

Answer ONLY these questions in {language}. Be practical and direct.

1. What happened? (Who won and who lost; 1 sentence)
2. Can the loser appeal? (Yes/No + reason; 1-2 sentences)
3. Appeal timeline: How many days? (Just number)
4. Appeal court: Which court should they go to? (Court name only)
5. Cost estimate: Rough cost in rupees? (e.g., 5000-15000)
6. First action: What should they do first? (1 sentence)
7. Important deadline: What key deadline should they remember? (1 sentence)

Output in numbered form like:
1. ...\n2. ...\n3. ... etc.
"""

# -----------------------------
# Remedies Parsing Logic
# -----------------------------

KNOWN_COURTS = {
    "supreme court",
    "high court",
    "district court",
    "sessions court",
    "session court",
    "civil court",
    "family court",
    "consumer court",
    "tribunal",
}

def _clean_answer(value: str) -> Optional[str]:
    cleaned = re.sub(r"\s+", " ", (value or "")).strip(" -:\t\n")
    return cleaned or None

def _strip_question_label(key: str, value: Optional[str]) -> Optional[str]:
    if not value:
        return None

    patterns = {
        "what_happened": r"^(what happened\??)\s*",
        "can_appeal": r"^(can the loser appeal\??)\s*",
        "appeal_days": r"^(appeal timeline\??|how many days\??)\s*",
        "appeal_court": r"^(appeal court\??|which court(?: should they go to)?\??)\s*",
        "cost_estimate": r"^(cost estimate\??|rough cost(?: in rupees)?\??)\s*",
        "first_action": r"^(first action\??|what should they do first\??)\s*",
        "deadline": r"^(important deadline\??|important dates?\??)\s*",
    }
    pattern = patterns.get(key)
    if pattern:
        value = re.sub(pattern, "", value, flags=re.IGNORECASE).strip()
    return value or None

def _normalize_yes_no(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    lower = value.lower()
    if re.search(r"\byes\b", lower):
        return "yes"
    if re.search(r"\bno\b", lower):
        return "no"
    return None

def _extract_number(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    match = re.search(r"\b(\d{1,4})\b", value)
    return match.group(1) if match else None

def _validate_court_name(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    cleaned = _clean_answer(value)
    if not cleaned:
        return None

    normalized = cleaned.lower()
    if normalized in KNOWN_COURTS or any(court in normalized for court in KNOWN_COURTS):
        return cleaned
    
    # NEW: Log as warning and return None if not a known court
    return None

def parse_remedies_response(response_text: str) -> Dict[str, Optional[str]]:
    """
    Extract structured info from LLM response using flexible numbered-line parsing.
    Supports multiple formats and performs normalization.
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
    remedies: Dict[str, Optional[str]] = {
        "what_happened": None,
        "can_appeal": None,
        "appeal_days": None,
        "appeal_court": None,
        "cost_estimate": None,
        "cost": None, # For backward compatibility in app.py
        "first_action": None,
        "deadline": None,
        "appeal_details": None # For backward compatibility in app.py
    }

    text = response_text.strip()
    if not text:
        LOGGER.warning("parse_remedies_response: empty response text")
        return None

    # Use the more robust parsing from cli.py
    marker_pattern = re.compile(r"(?m)^\s*(\d{1,2})\s*[\.|\)|:|-]\s*(.*)$")
    matches = list(marker_pattern.finditer(text))

    if not matches:
        LOGGER.warning("parse_remedies_response: no numbered sections found")
        return None

    parsed_sections = 0
    for idx, match in enumerate(matches):
        section_num = int(match.group(1))
        key = mapping.get(section_num)
        if not key:
            continue

        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        inline_text = _clean_answer(match.group(2))
        block_text = _clean_answer(text[start:end])
        section_text = _clean_answer(" ".join(part for part in [inline_text, block_text] if part))
        cleaned = _strip_question_label(key, section_text)
        
        if cleaned is not None:
            remedies[key] = cleaned
            parsed_sections += 1

    # Normalization & Compatibility
    if remedies["can_appeal"]:
        orig = remedies["can_appeal"]
        normalized = _normalize_yes_no(orig)
        if normalized is None:
            LOGGER.warning("parse_remedies_response: invalid can_appeal value: %s", orig)
        remedies["can_appeal"] = normalized
    
    if remedies["appeal_days"]:
        orig = remedies["appeal_days"]
        normalized = _extract_number(orig)
        if normalized is None:
            LOGGER.warning("parse_remedies_response: invalid appeal_days value: %s", orig)
        remedies["appeal_days"] = normalized
    
    if remedies["appeal_court"]:
        orig = remedies["appeal_court"]
        normalized = _validate_court_name(orig)
        if normalized is None:
            LOGGER.warning("parse_remedies_response: unknown appeal_court: %s", orig)
        remedies["appeal_court"] = normalized
    
    # Map 'cost_estimate' to 'cost' for app.py
    if remedies["cost_estimate"]:
        remedies["cost"] = remedies["cost_estimate"]
    
    # Provide appeal_details for 5-section fallback if needed
    if not remedies["appeal_days"] and not remedies["appeal_court"]:
         # This part is tricky as the new logic is more structured.
         # For now, we'll favor the 7-section structure.
         pass

    return remedies
