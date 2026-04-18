import streamlit as st
from openai import OpenAI
from pypdf import PdfReader
import logging
import os
import re
import core

# ==================== Notification System Setup ====================
from database import init_db, SessionLocal, get_db
from scheduler import start_scheduler, stop_scheduler

# Initialize database
init_db()

# Constants
DEFAULT_MODEL = st.secrets.get("DEFAULT_MODEL", core.DEFAULT_MODEL)

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
client = None  # Global cache for the client

@st.cache_resource

def _initialize_openai_client():
    """
    Internal function to initialize the OpenAI client using Streamlit secrets.
    """
    return OpenAI(
        api_key=st.secrets["OPENROUTER_API_KEY"],
        base_url=st.secrets["OPENROUTER_BASE_URL"]
    )

def get_client():
    """
    Returns the OpenAI client, initializing it only when needed.
    This prevents the application from crashing on import if st.secrets are missing.
    """
    global client
    if client is not None:
        return client

    try:
        client = _initialize_openai_client()
    except (KeyError, FileNotFoundError, RuntimeError, AttributeError) as e:
        # Graceful fallback for environments where secrets are not available (e.g., tests)
        logging.error(f"Failed to initialize OpenAI client: {e}")
        return None
    return client

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
# LLM Interaction Helpers (using core)
# -----------------------------


def get_remedies_advice(judgment_text, language):
    """
    Call LLM to get remedies for this judgment
    """
    client = get_client()
    if not client:
        return None

    prompt = core.build_remedies_prompt(core.compress_text(judgment_text), language)
    
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
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
    remedies = core.parse_remedies_response(response_text)
    
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
        client = get_client()

        if not client:
            st.error("OpenAI client not initialized. Please ensure OPENROUTER_API_KEY and OPENROUTER_BASE_URL are set in .streamlit/secrets.toml")
            return

        with st.spinner("Processing judgment…"):
            try:
                raw_text = core.extract_text_from_pdf(uploaded_file)
                safe_text = core.compress_text(raw_text)

                prompt = core.build_summary_prompt(safe_text, language)

                # ⚡ Best multilingual model for Hindi/Bengali/Urdu
                model_id = DEFAULT_MODEL
                response = client.chat.completions.create(
                    model=DEFAULT_MODEL,
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
                if language.lower() != "english" and core.english_leakage_detected(summary):
                    retry_prompt = core.build_retry_prompt(safe_text, language)

                    response2 = client.chat.completions.create(
                        model=DEFAULT_MODEL,
                        messages=[
                            {"role": "system", "content": "Strict multilingual rewriting engine."},
                            {"role": "user", "content": retry_prompt}
                        ],
                        max_tokens=260,
                        temperature=0.03,
                    )

                    retry_summary = response2.choices[0].message.content.strip()

                    if len(retry_summary) > 0 and not core.english_leakage_detected(retry_summary):
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
                            from database import CaseRecord
                            
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
