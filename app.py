import streamlit as st
from openai import OpenAI
import PyPDF2
import logging
import re
import time

LOGGER = logging.getLogger(__name__)

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

# -----------------------------
# App Config
# -----------------------------
st.set_page_config(
    page_title="LegalEase AI",
    page_icon="⚖",
    layout="centered"
)


# -----------------------------
# Load API Keys (OpenRouter)
# -----------------------------
def get_client():
    return OpenAI(
        api_key=st.secrets["OPENROUTER_API_KEY"],
        base_url=st.secrets["OPENROUTER_BASE_URL"]
    )

try:
    client = get_client()
except:
    client = None

# -----------------------------
# Retro Styling
# -----------------------------
st.markdown("""
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
""", unsafe_allow_html=True)

# -----------------------------
# Helper: PDF to text
# -----------------------------
def extract_text_from_pdf(uploaded_pdf):
    reader = PyPDF2.PdfReader(uploaded_pdf)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

# -----------------------------
# Compress text for token safety
# -----------------------------
def compress_text(text, limit=6000):
    if len(text) <= limit:
        return text
    head = text[:3000]
    tail = text[-3000:]
    return head + "\n\n... [TRUNCATED] ...\n\n" + tail

# -----------------------------
# Detect English leakage
# -----------------------------
def english_leakage_detected(output_text, threshold=5):
    common = [" the ", " and ", " of ", " to ", " in ", " is ", " that ", " it ", " for ", " on "]
    text_lower = " " + output_text.lower() + " "
    count = sum(1 for w in common if w in text_lower)
    return count >= threshold

# -----------------------------
# Build prompts
# -----------------------------
def build_prompt(safe_text, language):
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

# -----------------------------
# Remedies Advisor Functions
# -----------------------------

def build_remedies_prompt(judgment_text, language):
    """
    Ask LLM to analyze what remedies are available
    based on the actual judgment content
    """
    return f"""
You are a Legal Rights Advisor. Read this judgment and answer these questions:

JUDGMENT:
{judgment_text}

Answer ONLY these questions in {language}. Be practical and direct.

1. WHAT HAPPENED?
   - Who won and who lost? (1 sentence)

2. CAN THE LOSER APPEAL?
   - Yes or No? Why/Why not? (1-2 sentences)

3. IF YES TO APPEAL:
   - How many days do they have? (Just the number)
   - Which court should they go to? (Court name only)
   - Rough cost in rupees? (e.g., 5000-15000)

4. WHAT SHOULD THEY DO FIRST?
   - What is the first action to take? (1 sentence)

5. IMPORTANT DATES:
   - What deadline should they remember? (1 sentence)

Format your answer clearly with each question number and answer.
"""


def _clean_answer(value):
    cleaned = re.sub(r"\s+", " ", (value or "")).strip(" -:\t\n")
    return cleaned or None


def _strip_question_label(key, value):
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


def _normalize_yes_no(value):
    if not value:
        return None
    lower = value.lower()
    if re.search(r"\byes\b", lower):
        return "yes"
    if re.search(r"\bno\b", lower):
        return "no"
    return None


def _extract_number(value):
    if not value:
        return None
    match = re.search(r"\b(\d{1,4})\b", value)
    return match.group(1) if match else None


def _validate_court_name(value):
    if not value:
        return None
    cleaned = _clean_answer(value)
    if not cleaned:
        return None

    normalized = cleaned.lower()
    if normalized in KNOWN_COURTS or any(court in normalized for court in KNOWN_COURTS):
        return cleaned
    return None


def parse_remedies_response(response_text):
    """
    Extract structured info from LLM response.
    Returns None if no numbered sections can be parsed.
    """
    text = (response_text or "").strip()
    if not text:
        LOGGER.warning("parse_remedies_response: empty response text")
        return None

    mapping = {
        1: "what_happened",
        2: "can_appeal",
        3: "appeal_days",
        4: "appeal_court",
        5: "cost_estimate",
        6: "first_action",
        7: "deadline",
    }
    remedies = {key: None for key in mapping.values()}

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

    if parsed_sections == 0:
        LOGGER.warning("parse_remedies_response: numbered markers found but no parseable content")
        return None

    normalized_can_appeal = _normalize_yes_no(remedies.get("can_appeal"))
    if remedies.get("can_appeal") and normalized_can_appeal is None:
        LOGGER.warning("parse_remedies_response: invalid can_appeal value=%r", remedies.get("can_appeal"))
    remedies["can_appeal"] = normalized_can_appeal

    normalized_days = _extract_number(remedies.get("appeal_days"))
    if remedies.get("appeal_days") and normalized_days is None:
        LOGGER.warning("parse_remedies_response: invalid appeal_days value=%r", remedies.get("appeal_days"))
    remedies["appeal_days"] = normalized_days

    validated_court = _validate_court_name(remedies.get("appeal_court"))
    if remedies.get("appeal_court") and validated_court is None:
        LOGGER.warning("parse_remedies_response: unknown appeal_court value=%r", remedies.get("appeal_court"))
    remedies["appeal_court"] = validated_court

    LOGGER.info(
        "parse_remedies_response: parsed_sections=%d extracted_keys=%s",
        parsed_sections,
        [key for key, value in remedies.items() if value is not None],
    )
    return remedies


def get_remedies_advice(judgment_text, language):
    """
    Call LLM to get remedies for this judgment
    """
    prompt = build_remedies_prompt(judgment_text, language)
    
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
    if remedies is None:
        LOGGER.warning("get_remedies_advice: remedies parsing failed")
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

    remedies["cost"] = remedies.get("cost_estimate")
    return remedies

# -----------------------------
# UI
# -----------------------------

# -----------------------------
# Main Action Wrapper
# -----------------------------
def main():
    st.title("⚡ LegalEase AI")
    st.subheader("Legal Judgment Simplifier")

    st.markdown("""
    LegalEase AI breaks the Information Barrier in the Judiciary by converting
    complex court judgments into clear, 3-point summaries in your chosen language.
    """)
    st.markdown("---")

    language = st.selectbox("🌐 Select your language", ["English", "Hindi", "Bengali", "Urdu"])
    uploaded_file = st.file_uploader("📄 Upload Judgment PDF", type=["pdf"])
    st.markdown("---")

    if uploaded_file and st.button("🚀 Generate Summary"):
        with st.spinner("Processing judgment…"):
            try:
                raw_text = extract_text_from_pdf(uploaded_file)
                safe_text = compress_text(raw_text)

                prompt = build_prompt(safe_text, language)

                # ⚡ Best multilingual model for Hindi/Bengali/Urdu
                model_id = "meta-llama/llama-3.1-8b-instruct"


                # -----------------------------
                # FIRST ATTEMPT
                # -----------------------------
                response = client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": "You are an expert legal simplification engine."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=280,
                    temperature=0.05,
                )

                summary = response.choices[0].message.content.strip()

                # -----------------------------
                # RETRY IF ENGLISH LEAKAGE
                # -----------------------------
                if language.lower() != "english" and english_leakage_detected(summary):
                    retry_prompt = build_retry_prompt(safe_text, language)

                    response2 = client.chat.completions.create(
                        model=model_id,
                        messages=[
                            {"role": "system", "content": "Strict multilingual rewriting engine."},
                            {"role": "user", "content": retry_prompt}
                        ],
                        max_tokens=260,
                        temperature=0.03,
                    )

                    retry_summary = response2.choices[0].message.content.strip()

                    if len(retry_summary) > 0 and not english_leakage_detected(retry_summary):
                        summary = retry_summary

                if not summary:
                    st.error("The model returned an empty summary. Try a shorter file or switch to English.")
                else:
                    st.markdown("## ✅ Simplified Judgment")
                    st.write(summary)
                    st.success("The judgment has been simplified successfully.")
                    
                    # ===== REMEDIES SECTION =====
                    st.markdown("---")
                    st.markdown("## ⚖️ What Can You Do Now?")
                    
                    with st.spinner("Analyzing your legal options..."):
                        try:
                            remedies = get_remedies_advice(raw_text, language)
                            
                            # Show each answer
                            if remedies.get("what_happened"):
                                st.subheader("What Happened?")
                                st.write(remedies["what_happened"])
                            
                            if remedies.get("can_appeal"):
                                st.subheader("Can You Appeal?")
                                st.write(remedies["can_appeal"])
                                
                                # Only show appeal details if they can appeal
                                if "yes" in remedies["can_appeal"].lower():
                                    st.subheader("Appeal Details")
                                    
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        if remedies.get("appeal_days"):
                                            st.metric("Days to File Appeal", remedies["appeal_days"])
                                        if remedies.get("appeal_court"):
                                            st.write(f"**Appeal to:** {remedies['appeal_court']}")
                                    
                                    with col2:
                                        if remedies.get("cost"):
                                            st.write(f"**Estimated Cost:** {remedies['cost']}")
                            
                            if remedies.get("first_action"):
                                st.subheader("What Should You Do First?")
                                st.write(f"✅ {remedies['first_action']}")
                            
                            if remedies.get("deadline"):
                                st.subheader("⏰ Important Deadline")
                                st.write(remedies["deadline"])
                            
                        except Exception as e:
                            st.error(f"Could not get remedies advice: {str(e)}")
                    
                    # ===== FREE LEGAL HELP SECTION =====
                    st.markdown("---")
                    st.markdown("## 📞 Free Legal Help")
                    
                    help_options = """
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
                    
                    st.info(help_options)

            except Exception as e:
                err = str(e)

                if "402" in err or "credits" in err.lower():
                    st.error("❌ Not enough OpenRouter credits. Please top up.")
                else:
                    st.error(f"An error occurred: {err}")

if __name__ == "__main__":
    main()
