"""
LegalEase AI - Main Application Entry Point
Streamlit multi-page app with deadline notification system.

Run with: streamlit run app_integrated.py
"""

import streamlit as st
import logging
import os

# ==================== CONFIGURATION ====================
st.set_page_config(
    page_title="LegalEase AI",
    page_icon="⚖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== LOGGING SETUP ====================
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ==================== DATABASE & SCHEDULER SETUP ====================
from database import init_db
from scheduler import start_scheduler, stop_scheduler

# Initialize database
try:
    init_db()
    logger.info("Database initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize database: {str(e)}")

# Start background scheduler on app startup
if "scheduler_started" not in st.session_state:
    try:
        start_scheduler()
        st.session_state.scheduler_started = True
        logger.info("Background scheduler started")
    except Exception as e:
        logger.error(f"Failed to start scheduler: {str(e)}")
        st.session_state.scheduler_started = False

# ==================== Logging Setup ====================
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# ==================== Import UI Modules ====================
try:
    from notifications_ui import (
        page_notification_preferences,
        page_manage_deadlines,
        page_notification_history,
    )
    # Import original app components
    from app import (
        get_client,
        get_remedies_advice,
        get_default_model,
    )
    import core
    client = None
    all_features_available = True
except ImportError as e:
    logging.error(f"Failed to import UI modules: {e}")
    all_features_available = False
    client = None


# ==================== Main UI ====================
def main():
    # Sidebar navigation
    st.sidebar.markdown("# ⚖️ LegalEase AI")
    st.sidebar.markdown("**Convert Judgments to Simple Language**")
    st.sidebar.divider()
    
    # Display scheduler status
    if st.session_state.get("scheduler_started"):
        st.sidebar.success("✅ Notifications: Active")
    else:
        st.sidebar.warning("⚠️ Notifications: Offline")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        **Need Help?**
        - 📞 National Legal Services: 1800-180-8111
        - 🌐 nalsa.gov.in
        """
    )

    if all_features_available:
        page_options = [
            "Judgment Analysis",
            "Case Deadlines",
            "Notification History",
            "Preferences",
        ]
    else:
        page_options = ["Judgment Analysis"]
        st.sidebar.warning("Some modules failed to load. Showing Judgment Analysis only.")

    page = st.sidebar.radio("Navigate", page_options)
    
    # Route to appropriate page
    if page == "Judgment Analysis":
        show_judgment_analysis()
    elif page == "Case Deadlines":
        page_manage_deadlines()
    elif page == "Notification History":
        page_notification_history()
    elif page == "Preferences":
        page_notification_preferences()


def show_judgment_analysis():
    """Original app UI for judgment analysis"""
    
    # Retro Styling (from original app)
    st.markdown("""
    <style>
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
    </style>
    """, unsafe_allow_html=True)
    
    st.title("⚡ LegalEase AI")
    st.subheader("Legal Judgment Simplifier")

    st.markdown("""
    LegalEase AI breaks the Information Barrier in the Judiciary by converting
    complex court judgments into clear, 3-point summaries in your chosen language.
    """)
    st.markdown("---")

    language = st.selectbox("🌐 Select your language", 
                          ["English", "Hindi", "Bengali", "Urdu"])
    uploaded_file = st.file_uploader("📄 Upload Judgment PDF", type=["pdf"])
    st.markdown("---")

    if uploaded_file and st.button("🚀 Generate Summary", use_container_width=True):
        from app import get_client
        client = get_client()
        with st.spinner("Processing judgment…"):
            try:
                if not client:
                    st.error("❌ OpenRouter client not configured. Check your API keys.")
                    return

                raw_text = core.extract_text_from_pdf(uploaded_file)
                safe_text = core.compress_text(raw_text)

                prompt = core.build_summary_prompt(safe_text, language)

                # First attempt
                response = client.chat.completions.create(
                    model=get_default_model(),
                    messages=[
                        {"role": "system", "content": "You are an expert legal simplification engine."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=280,
                    temperature=0.05,
                )

                summary_raw = response.choices[0].message.content.strip()
                # Structured parser ensures exactly 3 bullets and removes intros
                summary = core.parse_summary_bullets(summary_raw)

                # Retry if English leakage detected
                if language.lower() != "english" and core.english_leakage_detected(summary):
                    retry_prompt = core.build_retry_prompt(safe_text, language)
                    response2 = client.chat.completions.create(
                        model=get_default_model(),
                        messages=[
                            {"role": "system", "content": "Strict multilingual rewriting engine."},
                            {"role": "user", "content": retry_prompt}
                        ],
                        max_tokens=260,
                        temperature=0.03,
                    )
                    retry_summary_raw = response2.choices[0].message.content.strip()
                    if len(retry_summary_raw) > 0 and not core.english_leakage_detected(retry_summary_raw):
                        summary = core.parse_summary_bullets(retry_summary_raw)

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
                            
                            if remedies.get("what_happened"):
                                st.subheader("What Happened?")
                                st.write(remedies["what_happened"])
                            
                            if remedies.get("can_appeal"):
                                st.subheader("Can You Appeal?")
                                st.write(remedies["can_appeal"])
                                
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
