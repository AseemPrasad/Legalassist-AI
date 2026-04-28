"""
Shared utilities for LegalEase AI
- PDF extraction and text processing
- LLM prompts and remedies advisor
- Styling and UI constants
"""

import re
import logging
from openai import OpenAI
from pypdf import PdfReader
from langdetect import detect, DetectorFactory, detect_langs

# For consistent language detection results
DetectorFactory.seed = 0

# ==================== MODEL CONFIGURATION ====================

DEFAULT_MODEL = "meta-llama/llama-3.1-8b-instruct"

def get_default_model():
    """
    Returns the default model to use, safely accessing st.secrets.
    Falls back gracefully if secrets are unavailable (e.g., during testing).
    """
    try:
        import streamlit as st
        return st.secrets.get("DEFAULT_MODEL", DEFAULT_MODEL)
    except (KeyError, FileNotFoundError, RuntimeError, AttributeError):
        # Fallback to module DEFAULT_MODEL if secrets are unavailable
        return DEFAULT_MODEL


# ==================== API CLIENT ====================

def _initialize_openai_client():
    """
    Internal function to initialize the OpenAI client using Streamlit secrets.
    Uses Streamlit caching to avoid recreating the client.
    """
    import streamlit as st
    return OpenAI(
        api_key=st.secrets["OPENROUTER_API_KEY"],
        base_url=st.secrets["OPENROUTER_BASE_URL"]
    )


def get_client():
    """
    Returns the OpenAI client, initializing it only when needed.
    This prevents the application from crashing on import if st.secrets are missing.
    Uses Streamlit's cache_resource for efficiency.
    """
    import streamlit as st
    
    @st.cache_resource
    def _get_cached_client():
        """Internal cached client initialization"""
        try:
            return _initialize_openai_client()
        except (KeyError, FileNotFoundError, RuntimeError, AttributeError) as e:
            # Graceful fallback for environments where secrets are not available (e.g., tests)
            logging.error(f"Failed to initialize OpenAI client: {e}")
            return None
    
    return _get_cached_client()


# ==================== TEXT PROCESSING ====================

def extract_text_from_pdf(uploaded_pdf):
    """Extract text from uploaded PDF file with validation"""
    reader = PdfReader(uploaded_pdf)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    
    if not text.strip():
        raise ValueError("No extractable text found. The PDF may be image-only or empty.")
    return text


def compress_text(text, limit=6000):
    """Compress text for token safety by keeping head and tail"""
    if len(text) <= limit:
        return text
    head = text[:3000]
    tail = text[-3000:]
    return head + "\n\n... [TRUNCATED] ...\n\n" + tail


def english_leakage_detected(output_text, threshold=8):
    """
    Detect if English words have leaked into non-English output.
    Uses langdetect for primary detection and a refined word-list heuristic for fallback.
    """
    if not output_text or len(output_text.strip()) < 10:
        return False

    try:
        # Get probabilities for all detected languages
        langs = detect_langs(output_text)
        # If English is detected with high confidence (> 0.5), it's likely leakage
        for l in langs:
            if l.lang == 'en' and l.prob > 0.8:
                return True
            # If the top language is English and it's much more likely than others
            if l.lang == 'en' and langs[0].lang == 'en' and l.prob > 0.5:
                return True
    except Exception as e:
        logging.debug(f"langdetect failed: {e}")

    # Refined heuristic: Use a more comprehensive list and higher threshold
    # Legal summaries often contain some English names or terms, so we need to be careful
    common_english = [
        " the ", " and ", " of ", " to ", " in ", " is ", " that ", " it ", " for ", " on ", 
        " with ", " as ", " this ", " was ", " are ", " at ", " by ", " be ", " or ", " has "
    ]
    text_lower = " " + re.sub(r'[^\w\s]', ' ', output_text.lower()) + " "
    count = sum(1 for word in common_english if word in text_lower)
    
    # Increased threshold to 8 to avoid false positives on short legal snippets
    return count >= threshold


# ==================== LLM PROMPTS ====================

def build_prompt(safe_text, language):
    """Build prompt for judgment simplification"""
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


def build_retry_prompt(safe_text, language):
    """Build retry prompt when English leakage is detected"""
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


def build_remedies_prompt(judgment_text, language):
    """Build prompt for remedies advisor to analyze legal options"""
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


# ==================== REMEDIES PARSER ====================

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

def _clean_answer(value: str):
    cleaned = re.sub(r"\s+", " ", (value or "")).strip(" -:\t\n")
    return cleaned or ""

def _strip_question_label(key: str, value: str) -> str:
    if not value:
        return ""

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
    return value or ""

def _normalize_yes_no(value: str) -> str:
    if not value:
        return ""
    lower = value.lower()
    if re.search(r"\byes\b", lower) or any(x in lower for x in ["can appeal", "available", "allowed", "has right"]):
        return "yes"
    if re.search(r"\bno\b", lower) or any(x in lower for x in ["cannot appeal", "not available", "no right", "no appeal"]):
        return "no"
    return ""

def _extract_number(value: str) -> str:
    if not value:
        return ""
    match = re.search(r"\b(\d{1,4})\b", value)
    return match.group(1) if match else ""

def _validate_court_name(value: str) -> str:
    if not value:
        return ""
    cleaned = _clean_answer(value)
    if not cleaned:
        return ""

    normalized = cleaned.lower()
    if normalized in KNOWN_COURTS or any(court in normalized for court in KNOWN_COURTS):
        return cleaned
    return ""

def parse_remedies_response(response_text):
    """
    Extract structured info from LLM response using flexible numbered-line parsing.
    Supports multiple separators: . ) : - 
    Handles both 5-section (old) and 7-section (new) formats.
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
        "cost": "", # For backward compatibility
        "first_action": "",
        "deadline": "",
        "appeal_details": "", # For backward compatibility
        "_is_partial": False,
        "_warning": ""
    }

    text = response_text.strip()
    if not text:
        return remedies

    # Use robust marker-based parsing
    marker_pattern = re.compile(r"(?m)^\s*(\d{1,2})\s*[\.|\)|:|-]\s*(.*)$")
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
        section_text = _clean_answer(" ".join(part for part in [inline_text, block_text] if part))
        cleaned = _strip_question_label(key, section_text)
        
        if cleaned:
            remedies[key] = cleaned

    # Normalization & Compatibility
    if remedies["can_appeal"]:
        remedies["can_appeal"] = _normalize_yes_no(remedies["can_appeal"])
    
    if remedies["appeal_days"]:
        remedies["appeal_days"] = _extract_number(remedies["appeal_days"])
    
    if remedies["appeal_court"]:
        remedies["appeal_court"] = _validate_court_name(remedies["appeal_court"])
    
    # Map 'cost_estimate' to 'cost' for backward compatibility
    if remedies["cost_estimate"]:
        remedies["cost"] = remedies["cost_estimate"]
    
    # Track if all main sections are present
    required = ["what_happened", "can_appeal", "appeal_days", "appeal_court", "cost", "first_action", "deadline"]
    missing = [f for f in required if not remedies[f]]
    if missing:
        remedies["_is_partial"] = True
        remedies["_warning"] = "Note: Some information may be incomplete."

    return remedies


def extract_appeal_info(appeal_details_text):
    """Extract structured appeal information from appeal details text using regex"""
    info = {
        "days": "",
        "court": "",
        "cost": ""
    }
    text = appeal_details_text or ""

    # Extract days to appeal
    days_match = re.search(r"(\d+)\s*(?:days?|day)", text.lower())
    if days_match:
        info["days"] = days_match.group(1)

    # Extract appeal court
    court_keywords = ["High Court", "District Court", "Supreme Court", "Lower Court"]
    for keyword in court_keywords:
        if keyword.lower() in text.lower():
            info["court"] = keyword
            break

    # Extract cost
    cost_match = re.search(r"₹?([\d,]+(?:[-–][\d,]+)?)", text.replace(" ", ""))
    if cost_match:
        info["cost"] = cost_match.group(1)

    return info


def get_remedies_advice(judgment_text, language, client=None):
    """Call LLM to get remedies for this judgment"""
    if client is None:
        client = get_client()
    
    if not client:
        return None
    
    prompt = build_remedies_prompt(compress_text(judgment_text), language)
    
    try:
        response = client.chat.completions.create(
            model=get_default_model(),
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful legal advisor. Answer questions about legal remedies in India."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=500,  # Longer for detailed answers
            temperature=0.1,  # Low temp = more consistent
        )
        
        response_text = response.choices[0].message.content.strip()
        remedies = parse_remedies_response(response_text)
        
        if remedies is None:
            return {
                "what_happened": None,
                "can_appeal": None,
                "appeal_days": None,
                "appeal_court": None,
                "cost_estimate": None,
                "cost": None,
                "first_action": None,
                "deadline": None,
            }
        
        return remedies
    except Exception as e:
        logging.error(f"Failed to get remedies advice: {str(e)}")
        return None


# ==================== UI STYLING & CONSTANTS ====================

RETRO_STYLING = """
<style>
    body {
        background-color: #0d0d0f;
        color: #e0e0e0;
        font-family: 'Inter', sans-serif;
    }
    .main {
        background-color: #0d0d0f;
    }
    .stButton>button {
        background: linear-gradient(90deg, #2d2dff, #8a2be2);
        border-radius: 8px;
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.6rem 1.2rem;
    }
    .stSelectbox>div>div {
        background-color: #1a1a1d;
        color: #e0e0e0;
        border-radius: 6px;
    }
    .stTextArea>div>textarea {
        background-color: #121214;
        color: #e0e0e0;
    }
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

LANGUAGES = ["English", "Hindi", "Bengali", "Urdu"]

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
    # Using non-greedy match for content to avoid capturing too much if markers are repeated
    bullet_marker_regex = re.compile(r"^\s*([\-\*\u2022\u25cf]|(\d+[\.\)]))\s*(.*)$")
    
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
