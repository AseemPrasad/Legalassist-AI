"""
Shared utilities for LegalEase AI
- PDF extraction and text processing
- LLM prompts and remedies advisor
- Styling and UI constants
"""

import re
from openai import OpenAI
from pypdf import PdfReader

# ==================== API CLIENT ====================

def get_client():
    """Get OpenRouter API client from Streamlit secrets"""
    import streamlit as st
    try:
        return OpenAI(
            api_key=st.secrets["OPENROUTER_API_KEY"],
            base_url=st.secrets["OPENROUTER_BASE_URL"]
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize OpenRouter client: {str(e)}")


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


def english_leakage_detected(output_text, threshold=5):
    """Detect if English words have leaked into non-English output"""
    common_english = [" the ", " and ", " of ", " to ", " in ", " is ", " that ", " it ", " for ", " on "]
    text_lower = " " + output_text.lower() + " "
    count = sum(1 for word in common_english if word in text_lower)
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

def parse_remedies_response(response_text):
    """
    Extract structured info from LLM response using flexible numbered-line parsing.
    Supports multiple separators: . ) : - 
    Handles both 5-section (old) and 7-section (new) formats.
    """
    remedies = {
        "what_happened": "",
        "can_appeal": "",
        "appeal_days": "",
        "appeal_court": "",
        "cost_estimate": "",
        "cost": "",
        "first_action": "",
        "deadline": "",
        "appeal_details": ""
    }

    text = response_text.strip()
    if not text:
        return remedies

    # Detect all numbered sections (flexible separators: . ) : -)
    # Only match 1-2 digit numbers to avoid matching content like "5000-10000"
    pattern = r'^([1-9]\d?)\s*[.):‐-]\s*(.*?)$'
    sections = {}
    
    for line in text.split('\n'):
        match = re.match(pattern, line.strip())
        if match:
            num = int(match.group(1))
            header = match.group(2).strip()
            sections[num] = {"header": header, "content": ""}
    
    # Extract content for each section
    lines = text.split('\n')
    current_section = None
    
    for line in lines:
        match = re.match(pattern, line.strip())
        if match:
            current_section = int(match.group(1))
        elif current_section is not None and current_section in sections:
            if line.strip():  # Only add non-empty lines
                sections[current_section]["content"] += line.strip() + " "
    
    # Clean up content
    for num in sections:
        sections[num]["content"] = sections[num]["content"].strip()
    
    # Map sections to keys based on count
    is_7section = len(sections) >= 7
    
    if is_7section:
        if 1 in sections:
            remedies["what_happened"] = sections[1]["content"]
        if 2 in sections:
            can_appeal_text = sections[2]["content"].lower()
            remedies["can_appeal"] = "yes" if "yes" in can_appeal_text else "no"
        if 3 in sections:
            # Extract just the number from "30 days"
            appeal_days_text = sections[3]["content"]
            match = re.search(r'\d+', appeal_days_text)
            remedies["appeal_days"] = match.group() if match else appeal_days_text
        if 4 in sections:
            remedies["appeal_court"] = sections[4]["content"]
        if 5 in sections:
            remedies["cost_estimate"] = sections[5]["content"]
            remedies["cost"] = sections[5]["content"]  # Support both keys
        if 6 in sections:
            remedies["first_action"] = sections[6]["content"]
        if 7 in sections:
            remedies["deadline"] = sections[7]["content"]
    else:
        # 5-section format (old)
        if 1 in sections:
            remedies["what_happened"] = sections[1]["content"]
        if 2 in sections:
            remedies["can_appeal"] = sections[2]["content"]
        if 3 in sections:
            remedies["appeal_details"] = sections[3]["content"]
        if 4 in sections:
            remedies["first_action"] = sections[4]["content"]
        if 5 in sections:
            remedies["deadline"] = sections[5]["content"]

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
    
    prompt = build_remedies_prompt(compress_text(judgment_text), language)
    
    response = client.chat.completions.create(
        model="meta-llama/llama-3.1-8b-instruct",
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
    
    return remedies


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
DEFAULT_MODEL = "meta-llama/llama-3.1-8b-instruct"

# Alias for backward compatibility
build_summary_prompt = build_prompt
