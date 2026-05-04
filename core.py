from pypdf import PdfReader
import pdfplumber
import re
import logging
from pathlib import Path
from typing import Dict, Optional, List, Union, Any

# Allow this module to coexist with the core/ package so imports such as
# `from core.app_utils import ...` continue to resolve.
__path__ = [str(Path(__file__).with_name("core"))]

# Allow this module to coexist with the core/ package so imports such as
# `from core.app_utils import ...` continue to resolve.
__path__ = [str(Path(__file__).with_name("core"))]

# Allow this module to coexist with the core/ package so imports such as
# `from core.app_utils import ...` continue to resolve.
__path__ = [str(Path(__file__).with_name("core"))]

# Allow this module to coexist with the core/ package so imports such as
# `from core.app_utils import ...` continue to resolve.
__path__ = [str(Path(__file__).with_name("core"))]

LOGGER = logging.getLogger(__name__)

# -----------------------------
# Configuration
# -----------------------------
DEFAULT_MODEL = "meta-llama/llama-3.1-8b-instruct"

# -----------------------------
# PDF to text
# -----------------------------
def _read_pdf_bytes(pdf_input: Union[str, Path, object]) -> Optional[bytes]:
    """Read PDF bytes when possible for OCR conversion."""
    if isinstance(pdf_input, (str, Path)):
        try:
            with open(pdf_input, "rb") as f:
                return f.read()
        except Exception:
            return None
    try:
        if hasattr(pdf_input, "seek"):
            pdf_input.seek(0)
        data = pdf_input.read()
        if hasattr(pdf_input, "seek"):
            pdf_input.seek(0)
        return data
    except Exception:
        return None


def _extract_layout_text_from_tesseract_data(data: Dict[str, List[Any]]) -> str:
    """Reconstruct text using OCR bounding boxes with simple column-aware ordering."""
    lines: Dict[tuple, Dict[str, Any]] = {}
    texts = data.get("text", [])
    confs = data.get("conf", [])
    lefts = data.get("left", [])
    tops = data.get("top", [])
    widths = data.get("width", [])

    for i, token in enumerate(texts):
        token = (token or "").strip()
        if not token:
            continue
        try:
            conf = float(confs[i])
        except Exception:
            conf = -1.0
        if conf < 0:
            continue

        key = (data["page_num"][i], data["block_num"][i], data["par_num"][i], data["line_num"][i])
        if key not in lines:
            lines[key] = {
                "tokens": [],
                "left": lefts[i],
                "top": tops[i],
                "right": lefts[i] + widths[i],
            }
        lines[key]["tokens"].append(token)
        lines[key]["left"] = min(lines[key]["left"], lefts[i])
        lines[key]["right"] = max(lines[key]["right"], lefts[i] + widths[i])
        lines[key]["top"] = min(lines[key]["top"], tops[i])

    if not lines:
        return ""

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


def extract_text_with_diagnostics(
    pdf_input: Union[str, Path, object],
    enable_ocr: bool = False,
    ocr_languages: str = "eng+hin",
    ocr_dpi: int = 300,
) -> Dict[str, Any]:
    """Extract text using PDF parsers, optionally falling back to OCR with confidence."""
    text = ""
    diagnostics: Dict[str, Any] = {
        "text": "",
        "method": "",
        "ocr_used": False,
        "confidence": None,
    }

    # 1. Try pdfplumber (more robust for complex layouts)
    try:
        with pdfplumber.open(pdf_input) as pdf:
            pages_text = []
            for page in pdf.pages:
                page_text = page.extract_text(x_tolerance=3, y_tolerance=3)
                if page_text:
                    pages_text.append(page_text)
            text = "\n".join(pages_text).strip()
            if text:
                diagnostics.update({"text": text, "method": "pdfplumber", "ocr_used": False})
                LOGGER.info("Extracted text using pdfplumber.")
                return diagnostics
    except Exception as e:
        LOGGER.warning(f"pdfplumber extraction failed or not available: {e}. Falling back to pypdf.")

    # 2. Fallback to pypdf
    try:
        if isinstance(pdf_input, (str, Path)):
            with open(pdf_input, "rb") as f:
                reader = PdfReader(f)
                text = _extract_pages_pypdf(reader)
        else:
            reader = PdfReader(pdf_input)
            text = _extract_pages_pypdf(reader)
        if text:
            diagnostics.update({"text": text, "method": "pypdf", "ocr_used": False})
            LOGGER.info("Extracted text using pypdf fallback.")
            return diagnostics
    except Exception as e:
        LOGGER.warning(f"pypdf extraction failed: {e}")

    if not enable_ocr:
        raise ValueError("No extractable text found. The PDF may be image-only or empty. Re-run with OCR enabled.")

    # 3. OCR path for scanned/image PDFs
    try:
        from pdf2image import convert_from_bytes, convert_from_path
        import pytesseract
        from pytesseract import Output
    except Exception as e:
        raise RuntimeError(
            f"OCR dependencies are missing ({e}). Install pytesseract, pdf2image, Pillow and Tesseract binaries."
        ) from e

    try:
        images = []
        if isinstance(pdf_input, (str, Path)):
            images = convert_from_path(str(pdf_input), dpi=ocr_dpi)
        else:
            data = _read_pdf_bytes(pdf_input)
            if not data:
                raise ValueError("Unable to read PDF bytes for OCR.")
            images = convert_from_bytes(data, dpi=ocr_dpi)
    except Exception as e:
        raise RuntimeError(f"Failed to convert PDF pages to images for OCR: {e}") from e

    if not images:
        raise ValueError("OCR could not read any pages from PDF.")

    ocr_pages: List[str] = []
    confidences: List[float] = []
    for image in images:
        data = pytesseract.image_to_data(image, lang=ocr_languages, output_type=Output.DICT)
        ocr_text = _extract_layout_text_from_tesseract_data(data)
        if ocr_text:
            ocr_pages.append(ocr_text)
        raw_conf = data.get("conf", [])
        valid_conf = []
        for c in raw_conf:
            try:
                val = float(c)
                if val >= 0:
                    valid_conf.append(val)
            except Exception:
                continue
        if valid_conf:
            confidences.append(sum(valid_conf) / len(valid_conf))

    final_text = "\n\n".join(ocr_pages).strip()
    if not final_text:
        raise ValueError("OCR completed but no readable text was extracted.")

    avg_conf = round(sum(confidences) / len(confidences), 2) if confidences else None
    diagnostics.update(
        {
            "text": final_text,
            "method": "ocr_tesseract",
            "ocr_used": True,
            "confidence": avg_conf,
        }
    )
    return diagnostics


def extract_text_from_pdf(
    pdf_input: Union[str, Path, object],
    enable_ocr: bool = False,
    ocr_languages: str = "eng+hin",
    ocr_dpi: int = 300,
) -> str:
    """
    Extracts text from a PDF file or file-like object using pdfplumber 
    for robustness, falling back to pypdf if necessary.
    """
    diagnostics = extract_text_with_diagnostics(
        pdf_input=pdf_input,
        enable_ocr=enable_ocr,
        ocr_languages=ocr_languages,
        ocr_dpi=ocr_dpi,
    )
    return diagnostics["text"]

def _extract_pages_pypdf(reader: PdfReader) -> str:
    """Helper for pypdf extraction fallback."""
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

def build_retry_prompt(safe_text: str, language: str) -> str:
    return f"""
Your previous answer included English. Now STRICTLY produce the answer ONLY in {language}.

REQUIREMENTS:
- Minimum 5 bullet points
- VERY simple {language}
- No English at all
- Put every bullet point on its own new line
- No introductions, headings, or explanations

TEXT:
{safe_text}

OUTPUT NOW:
Minimum 5 bullet points in {language} only.
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
    return value or None

def _normalize_yes_no(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    lower = value.lower()
    if re.search(r"\byes\b", lower) or any(x in lower for x in ["can appeal", "available", "allowed", "has right"]):
        return "yes"
    if re.search(r"\bno\b", lower) or any(x in lower for x in ["cannot appeal", "not available", "no right", "no appeal"]):
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

def parse_remedies_response(response_text: str) -> Optional[Dict[str, Optional[str]]]:
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
    remedies: Dict[str, str] = {
        "what_happened": "",
        "can_appeal": "",
        "appeal_days": "",
        "appeal_court": "",
        "cost_estimate": "",
        "cost": "", # For backward compatibility in app.py
        "first_action": "",
        "deadline": "",
        "appeal_details": "", # For backward compatibility in app.py
        "_is_partial": False, # New field for status
    }
    text = response_text.strip()
    if not text:
        LOGGER.warning("parse_remedies_response: empty response text")
        return remedies

    # Use the more robust parsing from cli.py
    marker_pattern = re.compile(r"(?m)^\s*(?:\*\*)?(\d{1,2})(?:\*\*)?\s*[\.|\)|:|-]\s*(.*)$")
    matches = list(marker_pattern.finditer(text))

    if not matches:
        LOGGER.warning("parse_remedies_response: no numbered sections found")
        return remedies

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

    if parsed_sections == 0:
        LOGGER.warning("parse_remedies_response: no valid sections parsed")
        return None

    # Normalization & Compatibility
    if remedies["can_appeal"]:
        orig = remedies["can_appeal"]
        normalized = _normalize_yes_no(orig)
        if normalized is None:
            LOGGER.warning("parse_remedies_response: invalid can_appeal value: %s", orig)
        remedies["can_appeal"] = normalized or ""
    
    if remedies["appeal_days"]:
        orig = remedies["appeal_days"]
        normalized = _extract_number(orig)
        if normalized is None:
            LOGGER.warning("parse_remedies_response: invalid appeal_days value: %s", orig)
        remedies["appeal_days"] = normalized or ""
    
    if remedies["appeal_court"]:
        orig = remedies["appeal_court"]
        normalized = _validate_court_name(orig)
        if normalized is None:
            LOGGER.warning("parse_remedies_response: unknown appeal_court: %s", orig)
        remedies["appeal_court"] = normalized or ""
    
    # Map 'cost_estimate' to 'cost' for app.py
    if remedies["cost_estimate"]:
        remedies["cost"] = remedies["cost_estimate"]
    
    # Track if all main sections are present
    required = ["what_happened", "can_appeal", "appeal_days", "appeal_court", "cost_estimate", "first_action", "deadline"]
    missing = [f for f in required if not remedies[f]]
    if missing:
        remedies["_is_partial"] = True

    return remedies
