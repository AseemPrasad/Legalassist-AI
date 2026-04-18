import streamlit as st
from openai import OpenAI
from pypdf import PdfReader
import logging
import os

# ==================== Notification System Setup ====================
from database import init_db, SessionLocal, get_db
from scheduler import start_scheduler, stop_scheduler

# Initialize database
init_db()

# Start background scheduler on app startup
if "scheduler_started" not in st.session_state:
    try:
        start_scheduler()
        st.session_state.scheduler_started = True
        logging.info("Background scheduler started")
    except Exception as e:
        logging.error(f"Failed to start scheduler: {str(e)}")
        st.session_state.scheduler_started = False

# ==================== Logging Setup ====================
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# ==================== App Config ====================
st.set_page_config(
    page_title="LegalEase AI",
    page_icon="⚖",
    layout="wide" if st.query_params.get("page") == "deadlines" else "centered"
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
except Exception:
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
    reader = PdfReader(uploaded_pdf)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    if not text.strip():
        raise ValueError("No extractable text found. The PDF may be image-only or empty.")
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


def parse_remedies_response(response_text):
    """
    Extract structured info from LLM response using flexible numbered-line parsing.
    Supports multiple separators: . ) : - 
    Handles both 5-section (old) and 7-section (new) formats.
    """
    import re
    
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


def get_remedies_advice(judgment_text, language):
    """
    Call LLM to get remedies for this judgment
    """
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
                    
                    # ===== ANALYTICS & TRACKING SECTION =====
                    st.markdown("---")
                    st.markdown("## 📊 Track Your Case & See Statistics")
                    
                    st.info("""
                    **Help us build better predictions!**
                    
                    By tracking your case, you help us understand appeal success rates in your jurisdiction.
                    Later, when you know the outcome of your appeal, you can report it back.
                    """)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button("📈 View Analytics", key="view_analytics"):
                            st.session_state.show_analytics = True
                    
                    with col2:
                        if st.button("🎯 Estimate Appeal Chances", key="estimate_chances"):
                            st.session_state.show_estimator = True
                    
                    with col3:
                        if st.button("📝 Report Outcome", key="report_outcome"):
                            st.session_state.show_feedback = True
                    
                    # Show analytics if requested
                    if st.session_state.get("show_analytics"):
                        st.subheader("📊 Quick Analytics Preview")
                        try:
                            from analytics_engine import AnalyticsAggregator
                            from database import SessionLocal, CaseRecord
                            
                            db = SessionLocal()
                            summary = AnalyticsAggregator.get_dashboard_summary(db)
                            
                            if summary["total_cases_processed"] > 0:
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Total Cases Tracked", summary["total_cases_processed"])
                                with col2:
                                    st.metric("Appeals Success Rate", f"{AnalyticsAggregator.get_regional_trends(db)[0]['appeal_success_rate'] if AnalyticsAggregator.get_regional_trends(db) else 'N/A'}%")
                                with col3:
                                    st.metric("Appeals Filed", summary["appeals_filed"])
                                
                                st.write("📌 **Visit Analytics Dashboard for detailed insights** ➡️ [See Full Dashboard]()")
                            else:
                                st.info("Analytics will be available as more cases are tracked.")
                            
                            db.close()
                        except Exception as e:
                            st.info("Analytics module not ready yet.")
                    
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
