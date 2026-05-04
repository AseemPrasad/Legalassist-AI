import streamlit as st
import openai
from openai import OpenAI
from pypdf import PdfReader
import logging
import os
import re
import json
from pathlib import Path

# ==================== Import Utilities from core.app_utils ====================
from core.app_utils import (
    get_client,
    get_default_model,
    _initialize_openai_client,
    extract_text_from_pdf,
    compress_text,
    english_leakage_detected,
    build_prompt,
    build_summary_prompt,
    build_retry_prompt,
    build_remedies_prompt,
    parse_remedies_response,
    get_remedies_advice,
    parse_summary_bullets,
    validate_pdf_metadata,
)

# ==================== Notification System Setup ====================
from database import init_db, SessionLocal, get_db, DocumentType, db_session
from scheduler import start_scheduler, stop_scheduler
from auth import init_auth_session, require_auth, get_current_user_id, get_current_user_email, logout_user
from case_manager import get_user_cases_summary, upload_case_document, create_new_case, get_case_detail

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

# Using default Streamlit theme

# ==================== File Upload Configuration ====================
MAX_FILE_SIZE_MB = 10

LEGAL_AID_DIRECTORY_PATH = Path(__file__).parent / "legal_aid_directory.json"


@st.cache_data(show_spinner=False)
def load_legal_aid_directory():
    """Load state-wise legal aid directory from JSON data file."""
    if not LEGAL_AID_DIRECTORY_PATH.exists():
        logging.error("legal_aid_directory.json not found")
        return {}

    try:
        with LEGAL_AID_DIRECTORY_PATH.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        return payload.get("states", {})
    except Exception as e:
        logging.error(f"Failed to load legal aid directory: {str(e)}")
        return {}


def render_localized_legal_help():
    """Render state-specific legal help resources."""
    st.markdown("## 📞 Free Legal Help")

    directory = load_legal_aid_directory()
    if not directory:
        st.warning("Localized legal help data is unavailable right now.")
        return

    state_names = sorted(directory.keys())
    selected_state = st.selectbox(
        "Select your state/UT",
        options=state_names,
        key="legal_help_state_selector",
    )

    state_data = directory.get(selected_state, {})
    authority = state_data.get("legal_aid_authority", {})
    colleges = state_data.get("law_colleges", [])
    ngos = state_data.get("ngos", [])
    bar_association = state_data.get("bar_association", {})
    avg_cost = state_data.get("avg_cost", "Not available")

    st.success(f"Showing legal help resources for {selected_state}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### State Legal Aid Authority")
        st.write(f"**Name:** {authority.get('name', 'N/A')}")
        st.write(f"**Phone:** {authority.get('phone', 'N/A')}")
        st.write(f"**Website:** {authority.get('website', 'N/A')}")

        st.markdown("### State Bar Association")
        st.write(f"**Name:** {bar_association.get('name', 'N/A')}")
        st.write(f"**Phone:** {bar_association.get('phone', 'N/A')}")
        st.write(f"**Website:** {bar_association.get('website', 'N/A')}")

        st.markdown("### Average Appeal Cost")
        st.info(avg_cost)

    with col2:
        st.markdown("### Law Colleges with Legal Clinics")
        if colleges:
            for college in colleges:
                clinic_status = "Yes" if college.get("clinic_available") else "No"
                st.write(
                    f"- **{college.get('name', 'N/A')}** ({college.get('city', 'N/A')}) | Clinic: {clinic_status}"
                )
        else:
            st.write("No college records available.")

        st.markdown("### NGOs")
        if ngos:
            for ngo in ngos:
                st.write(
                    f"- **{ngo.get('name', 'N/A')}** | {ngo.get('specialty', 'General legal aid')} | "
                    f"{ngo.get('phone', 'N/A')} | {ngo.get('website', 'N/A')}"
                )
        else:
            st.write("No NGO records available.")

# ==================== UI Helper Components ====================

def render_remedies_section(remedies):
    """
    Renders the legal remedies and options section based on judgment analysis.
    """
    st.markdown("---")
    st.markdown("## ⚖️ What Can You Do Now?")
    
    with st.spinner("Analyzing your legal options..."):
        try:
            # Show warning if data is partial
            if remedies.get("_is_partial"):
                st.warning(remedies.get("_warning", "Note: Some information may be incomplete."))
            
            # Show each answer with clean layout
            if remedies.get("what_happened"):
                st.markdown("### 📝 What Happened?")
                st.info(remedies["what_happened"])
            
            if remedies.get("can_appeal"):
                st.markdown("### 🏛️ Can You Appeal?")
                st.write(remedies["can_appeal"])
                
                # Only show appeal details if they can appeal
                if "yes" in remedies["can_appeal"].lower():
                    st.markdown("#### Appeal Details")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if remedies.get("appeal_days"):
                            st.metric("Days to File Appeal", remedies["appeal_days"], delta_color="inverse")
                        if remedies.get("appeal_court"):
                            st.markdown(f"**Appeal to:** `{remedies['appeal_court']}`")
                    
                    with col2:
                        if remedies.get("cost"):
                            st.markdown(f"**Estimated Cost:** `{remedies['cost']}`")
                        if remedies.get("deadline"):
                            st.markdown(f"**Deadline Note:** {remedies['deadline']}")
            
            if remedies.get("first_action"):
                st.markdown("### 🚀 What Should You Do First?")
                st.success(f"**Action Plan:** {remedies['first_action']}")
            
            if remedies.get("deadline") and not remedies.get("can_appeal"):
                st.markdown("### ⏰ Important Deadline")
                st.warning(remedies["deadline"])
            
        except Exception as e:
            st.error(f"Could not render remedies advice: {str(e)}")
            logging.exception("Remedies rendering failed")


def render_save_to_case_section(user_id, raw_text, summary, remedies):
    """
    Renders the UI for saving the current analysis to a user's case history.
    """
    st.markdown("---")
    st.markdown("## 💾 Save to Case History")
    
    if not require_auth():
        st.info("Log in to save this document, track deadlines, and view timeline history.")
        if st.button("Go to Login", key="login_to_save"):
            st.switch_page("pages/0_Login.py")
        return

    cases = get_user_cases_summary(user_id, include_closed=False)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Existing Cases")
        if cases:
            case_options = {f"{c['case_number']} - {c['title']}": c['id'] for c in cases}
            selected_case_name = st.selectbox("Select Existing Case", options=list(case_options.keys()))
            selected_case_id = case_options[selected_case_name]
            
            if st.button("Save to Selected Case", use_container_width=True):
                with st.spinner("Saving..."):
                    doc = upload_case_document(
                        user_id=user_id,
                        case_id=selected_case_id,
                        document_type=DocumentType.JUDGMENT,
                        document_content=raw_text,
                        summary=summary,
                        remedies=remedies
                    )
                    if doc:
                        st.success("✅ Saved successfully! Deadlines auto-created.")
                        st.session_state.selected_case_id = selected_case_id
        else:
            st.info("No active cases found. Create one to the right.")
        
        if st.session_state.get("selected_case_id"):
            if st.button("🔍 View Case Details", key="view_existing_case", use_container_width=True):
                st.switch_page("pages/2_Case_Details.py")
            
    with col2:
        st.markdown("### New Case")
        with st.expander("➕ Create & Save New Case"):
            new_case_number = st.text_input("Case Number", placeholder="e.g. CA/123/2024")
            new_case_title = st.text_input("Case Title (Optional)", placeholder="e.g. Sharma vs State")
            new_case_type = st.selectbox("Type", ["civil", "criminal", "family", "other"])
            new_jurisdiction = st.text_input("Jurisdiction", placeholder="e.g. Delhi High Court")
            
            if st.button("Create Case & Save Document", use_container_width=True):
                if new_case_number and new_jurisdiction:
                    new_case = create_new_case(
                        user_id=user_id,
                        case_number=new_case_number,
                        case_type=new_case_type,
                        jurisdiction=new_jurisdiction,
                        title=new_case_title
                    )
                    if new_case:
                        doc = upload_case_document(
                            user_id=user_id,
                            case_id=new_case.id,
                            document_type=DocumentType.JUDGMENT,
                            document_content=raw_text,
                            summary=summary,
                            remedies=remedies
                        )
                        if doc:
                            st.success("✅ Case created and document saved!")
                            st.session_state.selected_case_id = new_case.id
                else:
                    st.error("Case Number and Jurisdiction are required.")


def render_analytics_preview_section():
    """
    Renders a premium analytics preview section with robust database session handling.
    Fixes potential connection leaks by using the db_session context manager.
    """
    st.markdown("---")
    st.markdown("## 📊 Case Tracking & Regional Trends")
    
    st.info("""
    **Help us build better predictions!**
    
    By tracking your case, you help us understand appeal success rates in your jurisdiction.
    The more data we have, the more accurately we can estimate your chances.
    """)
    
    # Action buttons for the analytics module
    act_col1, act_col2, act_col3 = st.columns(3)
    with act_col1:
        if st.button("📈 View Stats", key="view_analytics", use_container_width=True):
            st.session_state.show_analytics = True
    with act_col2:
        if st.button("🎯 Est. Chances", key="estimate_chances", use_container_width=True):
            st.switch_page("pages/2_Appeal_Estimator.py")
    with act_col3:
        if st.button("📝 Report Outcome", key="report_outcome", use_container_width=True):
            st.switch_page("pages/3_Report_Outcome.py")
    
    # Detailed analytics preview
    if st.session_state.get("show_analytics"):
        st.markdown("### 📈 Quick Analytics Preview")
        try:
            from analytics_engine import AnalyticsAggregator
            
            with db_session() as db:
                summary = AnalyticsAggregator.get_dashboard_summary(db)
                
                if summary.get("total_cases_processed", 0) > 0:
                    m1, m2, m3 = st.columns(3)
                    
                    with m1:
                        st.metric("Total Cases Tracked", summary["total_cases_processed"])
                    
                    with m2:
                        trends = AnalyticsAggregator.get_regional_trends(db)
                        success_rate = trends[0]['appeal_success_rate'] if trends else 'N/A'
                        st.metric("Appeals Success Rate", f"{success_rate}%")
                        
                        # Visual indicator for success rate
                        if success_rate != 'N/A':
                            try:
                                rate_f = float(success_rate) / 100.0
                                st.progress(rate_f, text=f"Success Rate Intensity")
                            except: pass
                            
                    with m3:
                        st.metric("Appeals Filed", summary.get("appeals_filed", 0))
                    
                    st.markdown("""
                    <div style="background-color: rgba(255, 255, 255, 0.05); padding: 10px; border-radius: 8px; border-left: 4px solid #00c853; margin: 10px 0;">
                        📌 <b>Explore More:</b> Visit the <a href="/Analytics_Dashboard" target="_self">Full Analytics Dashboard</a> 
                        for heatmaps, judge-specific data, and time-to-verdict projections.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("✨ Community statistics will appear here once more users track their cases. Be the first to contribute!")
        
        except Exception as e:
            logging.error(f"Analytics rendering error: {str(e)}")
            st.info("The analytics module is currently being updated. Please try again later.")


# ==================== Main UI Component ====================
def main():
    init_auth_session()
    
    st.sidebar.markdown("# ⚖️ LegalEase AI")
    if require_auth():
        st.sidebar.success(f"Logged in as {get_current_user_email()}")
        if st.sidebar.button("Logout"):
            logout_user()
            st.rerun()
    else:
        st.sidebar.info("Not logged in. Log in to track cases and deadlines.")
        if st.sidebar.button("Go to Login"):
            st.switch_page("pages/0_Login.py")

    st.title("⚡ LegalEase AI")
    st.subheader("Legal Judgment Simplifier")

    st.markdown("""
    LegalEase AI breaks the Information Barrier in the Judiciary by converting
    complex court judgments into clear, 3-point summaries in your chosen language.
    """)
    st.markdown("---")

    language = st.selectbox("🌐 Select your language", ["English", "Hindi", "Bengali", "Urdu"])
    uploaded_file = st.file_uploader("📄 Upload Judgment PDF", type=["pdf"])
    
    # PDF Validation for size and page count
    is_valid_pdf = True
    if uploaded_file:
        # Check file size
        MAX_FILE_SIZE_MB = 25
        WARN_FILE_SIZE_MB = 10
        file_size_mb = (uploaded_file.size or 0) / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            st.error(f"🛑 File too large. Maximum size is {MAX_FILE_SIZE_MB}MB.")
            is_valid_pdf = False
        elif file_size_mb > WARN_FILE_SIZE_MB:
            st.warning("⚠️ This file is quite large. Processing may take longer than usual.")
            
        # Check page count
        try:
            pdf_reader = PdfReader(uploaded_file)
            num_pages = len(pdf_reader.pages)
            if num_pages > 100:
                st.warning(f"⚠️ This document has {num_pages} pages. Summaries of very long judgments may be less precise.")
            if num_pages > 1000:
                st.error("🛑 Extremely large PDF (1000+ pages) detected. Character limits will be exceeded, leading to a very poor summary. Please upload a shorter excerpt.")
                is_valid_pdf = False
        except Exception as e:
            st.error("Could not read PDF metadata. The file might be corrupted.")
            is_valid_pdf = False

    st.markdown("---")

    generate_clicked = st.button("🚀 Generate Summary") if (uploaded_file and is_valid_pdf) else False
    if uploaded_file and generate_clicked:
        st.session_state.processed_file = uploaded_file.name
        st.session_state.last_language = language

    if uploaded_file and st.session_state.get("processed_file") == uploaded_file.name and st.session_state.get("last_language") == language:
        client = get_client()

        if not client:
            st.error("OpenAI client not initialized. Please ensure OPENROUTER_API_KEY and OPENROUTER_BASE_URL are set in .streamlit/secrets.toml")
            return

        with st.spinner("Processing judgment…"):
            try:
                # Only call LLM if we haven't processed this exact file/language combo
                if st.session_state.get("last_processed") != f"{uploaded_file.name}_{language}":
                    raw_text = extract_text_from_pdf(uploaded_file)
                    safe_text = compress_text(raw_text)

                    prompt = build_summary_prompt(safe_text, language)

                    # ⚡ Best multilingual model for Hindi/Bengali/Urdu
                    model_id = get_default_model()
                    
                    # Added a 60-second timeout to prevent the Streamlit app
                    # from spinning indefinitely in case the OpenAI API
                    # hangs or becomes unresponsive.
                    response = client.chat.completions.create(
                        model=model_id,
                        messages=[
                            {"role": "system", "content": "You are an expert legal simplification engine."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=280,
                        temperature=0.05,
                        timeout=60.0,
                    )

                    summary_raw = response.choices[0].message.content.strip()
                    
                    # Use a structured parser to ensure exactly 3 bullet points 
                    # and remove any introductory text like "Here is your summary:"
                    summary = parse_summary_bullets(summary_raw)

                    # -----------------------------
                    # RETRY IF ENGLISH LEAKAGE
                    # -----------------------------
                    if language.lower() != "english" and english_leakage_detected(summary):
                        retry_prompt = build_retry_prompt(safe_text, language)

                        # Added a 60-second timeout to prevent the Streamlit app
                        # from spinning indefinitely in case the OpenAI API
                        # hangs or becomes unresponsive.
                        response2 = client.chat.completions.create(
                            model=model_id,
                            messages=[
                                {"role": "system", "content": "Strict multilingual rewriting engine."},
                                {"role": "user", "content": retry_prompt}
                            ],
                            max_tokens=260,
                            temperature=0.03,
                            timeout=60.0,
                        )
                        retry_summary_raw = response2.choices[0].message.content.strip()

                        if len(retry_summary_raw) > 0 and not english_leakage_detected(retry_summary_raw):
                            # Apply structured parsing to retry summary as well
                            summary = parse_summary_bullets(retry_summary_raw)

                    remedies = get_remedies_advice(raw_text, language)

                    # Save to session
                    st.session_state.raw_text = raw_text
                    st.session_state.summary = summary
                    st.session_state.remedies = remedies
                    st.session_state.last_processed = f"{uploaded_file.name}_{language}"
                else:
                    # Load from session
                    raw_text = st.session_state.raw_text
                    summary = st.session_state.summary
                    remedies = st.session_state.remedies

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
                            
                            # Show warning if data is partial
                            if remedies.get("_is_partial"):
                                st.warning(remedies.get("_warning", "Note: Some information may be incomplete."))
                            
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
                    
                    # ===== SAVE TO CASE SECTION =====
                    st.markdown("---")
                    st.markdown("## 💾 Save to Case History")
                    
                    if not require_auth():
                        st.info("Log in to save this document, track deadlines, and view timeline history.")
                        if st.button("Go to Login", key="login_to_save"):
                            st.switch_page("pages/0_Login.py")
                    else:
                        user_id = get_current_user_id()
                        cases = get_user_cases_summary(user_id, include_closed=False)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if cases:
                                case_options = {f"{c['case_number']} - {c['title']}": c['id'] for c in cases}
                                selected_case_name = st.selectbox("Select Existing Case", options=list(case_options.keys()))
                                selected_case_id = case_options[selected_case_name]
                                
                                if st.button("Save to Selected Case"):
                                    with st.spinner("Saving..."):
                                        doc = upload_case_document(
                                            user_id=user_id,
                                            case_id=selected_case_id,
                                            document_type=DocumentType.JUDGMENT,
                                            document_content=raw_text,
                                            summary=summary,
                                            remedies=remedies
                                        )
                                        if doc:
                                            st.success("✅ Saved successfully! Deadlines auto-created.")
                                            st.session_state.selected_case_id = selected_case_id
                            else:
                                st.info("No active cases found. Create one to the right.")
                            
                            if st.session_state.get("selected_case_id"):
                                if st.button("View Case Details", key="view_existing_case"):
                                    st.switch_page("pages/2_Case_Details.py")
                                
                        with col2:
                            with st.expander("➕ Or Create New Case"):
                                new_case_number = st.text_input("Case Number")
                                new_case_title = st.text_input("Case Title (Optional)")
                                new_case_type = st.selectbox("Type", ["civil", "criminal", "family", "other"])
                                new_jurisdiction = st.text_input("Jurisdiction", placeholder="e.g. Delhi High Court")
                                if st.button("Create & Save"):
                                    if new_case_number and new_jurisdiction:
                                        new_case = create_new_case(
                                            user_id=user_id,
                                            case_number=new_case_number,
                                            case_type=new_case_type,
                                            jurisdiction=new_jurisdiction,
                                            title=new_case_title
                                        )
                                        if new_case:
                                            doc = upload_case_document(
                                                user_id=user_id,
                                                case_id=new_case.id,
                                                document_type=DocumentType.JUDGMENT,
                                                document_content=raw_text,
                                                summary=summary,
                                                remedies=remedies
                                            )
                                            if doc:
                                                st.success("✅ Case created and document saved!")
                                                st.session_state.selected_case_id = new_case.id
                                    else:
                                        st.error("Case Number and Jurisdiction required.")
                    
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
                    render_localized_legal_help()

            except ValueError as e:
                st.error(f"❌ Extraction Error: {str(e)}")
                logging.error(f"Text extraction failed: {str(e)}")

            except openai.APIConnectionError as e:
                st.error("❌ Network Error: Could not connect to the AI service. Please check your internet.")
                logging.error(f"API Connection error: {str(e)}")

            except openai.RateLimitError as e:
                st.error("❌ Rate Limit: Too many requests. Please wait a moment before trying again.")
                logging.error(f"API Rate limit: {str(e)}")

            except openai.AuthenticationError as e:
                st.error("❌ API Key Error: Your OpenRouter/OpenAI key is invalid or not found.")
                logging.error(f"API Auth error: {str(e)}")

            except openai.APIStatusError as e:
                if e.status_code == 402:
                    st.error("❌ Out of Credits: Please top up your OpenRouter account to continue.")
                else:
                    st.error(f"❌ AI Service Error ({e.status_code}): {e.message}")
                logging.error(f"API Status error: {str(e)}")

            except openai.APIError as e:
                st.error(f"❌ AI Service Error: {str(e)}")
                logging.error(f"OpenAI API error: {str(e)}")

            except Exception as e:
                st.error(f"❌ Unexpected Error: {str(e)}")
                logging.exception("An unhandled exception occurred in the main loop")

if __name__ == "__main__":
    main()
