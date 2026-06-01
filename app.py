import streamlit as st
import openai
from openai import OpenAI
from pypdf import PdfReader
import hashlib
import logging
import os
import re
import json
import html
from pathlib import Path

# ==================== Import Utilities from core.app_utils ====================
from core.app_utils import (
    get_client,
    get_default_model,
    _initialize_openai_client,
    extract_text_from_pdf,
    compress_text,
    english_leakage_detected,  # Used to detect if output contains English in non-English text
    output_language_mismatch_detected,
    build_prompt,
    build_summary_prompt,
    build_retry_prompt,
    build_remedies_prompt,
    parse_remedies_response,
    get_remedies_advice,
    parse_summary_bullets,
    validate_pdf_metadata,
    get_localized_ui_text,
    localize_yes_no,
    safe_llm_call,
    generate_legal_draft,
    export_draft_to_pdf,
)

# ==================== Notification System ====================
# NOTE: Use scheduler.run_worker() for production deployments.
# Do NOT start scheduler in Streamlit - it causes failures on reruns.
from auth import init_auth_session, require_auth, get_current_user_id, get_current_user_email, logout_user
from case_manager import get_user_cases_summary, upload_case_document, create_new_case, get_case_detail
import routes
from observability.integration import initialize_observability_for_environment

# Initialize database
from config import Config
Config.validate_runtime_security()
init_db()

# ==================== Logging Setup ====================
# CENTRALIZED LOGGING CONFIGURATION
# This is the single, authoritative logging setup point for the entire application.
# All modules (scheduler, database, auth, etc.) use logging.getLogger(__name__)
# and get their logs handled by this configuration.
# NOTE: Other modules (scheduler.py, etc.) are imported BEFORE this point,
# but they do NOT call logging.basicConfig() to avoid duplicate handlers.
logging.basicConfig(
    level=Config.LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
initialize_observability_for_environment()

# ==================== App Config ====================
st.set_page_config(
    page_title=Config.APP_NAME,
    page_icon="⚖",
    layout="wide"
)

# Using default Streamlit theme

# ==================== File Upload Configuration ====================
MAX_FILE_SIZE_MB = Config.MAX_FILE_SIZE_MB

LEGAL_AID_DIRECTORY_PATH = Path(__file__).parent / "legal_aid_directory.json"


def get_uploaded_file_hash(uploaded_file):
    """Return a stable SHA256 hash for the uploaded file contents."""
    return hashlib.sha256(uploaded_file.getbuffer()).hexdigest()


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
        logging.error("Failed to load legal aid directory", error_type=type(e).__name__)
        return {}


def render_localized_legal_help(ui_text=None):
    """Render state-specific legal help resources."""
    heading = ui_text.get("free_legal_help", "📞 Free Legal Help") if ui_text else "📞 Free Legal Help"
    st.markdown(f"## {heading}")

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
        st.error("Could not render remedies advice. Please try analyzing the document again.")
        logging.error("Remedies rendering failed", error_type=type(e).__name__)


def render_save_to_case_section(raw_text, summary, remedies):
    """
    Renders the UI for saving the current analysis to a user's case history.
    """
    st.markdown("---")
    st.markdown("## 💾 Save to Case History")
    
    if not require_auth():
        st.info("Log in to save this document, track deadlines, and view timeline history.")
        if st.button("Go to Login", key="login_to_save"):
            st.switch_page(routes.PAGE_LOGIN)
        return

    user_id = get_current_user_id()
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
                st.switch_page(routes.PAGE_CASE_DETAILS)
            
    with col2:
        st.markdown("### New Case")
        with st.expander("➕ Create & Save New Case"):
            new_case_number = st.text_input("Case Number", placeholder="e.g. CA/123/2024").strip()
            new_case_title = st.text_input("Case Title (Optional)", placeholder="e.g. Sharma vs State").strip()
            new_case_type = st.selectbox("Type", ["civil", "criminal", "family", "other"])
            new_jurisdiction = st.text_input("Jurisdiction", placeholder="e.g. Delhi High Court").strip()
            
            if st.button("Create Case & Save Document", use_container_width=True):
                if new_case_number and new_jurisdiction:
                    new_case, was_existing = create_new_case(
                        user_id=user_id,
                        case_number=new_case_number,
                        case_type=new_case_type,
                        jurisdiction=new_jurisdiction,
                        title=new_case_title
                    )
                    if was_existing:
                        st.warning("Case already exists. Document will be added to existing case.")
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
        if st.button("📈 View Stats", key="view_stats_btn", use_container_width=True):
            st.session_state.show_analytics = True
    with act_col2:
        if st.button("🎯 Est. Chances", key="estimate_chances", use_container_width=True):
            st.switch_page(routes.PAGE_APPEAL_ESTIMATOR)
    with act_col3:
        if st.button("📝 Report Outcome", key="report_outcome", use_container_width=True):
            st.switch_page(routes.PAGE_REPORT_OUTCOME)
    
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
                            except (ValueError, TypeError):
                                pass
                            
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


# =============================================================================
# COMPREHENSIVE SESSION MANAGEMENT & LOGOUT HANDLER
# =============================================================================

def perform_comprehensive_logout():
    """
    Executes a high-integrity logout process by aggressively clearing all 
    application state and session data.
    
    SECURITY RATIONALE:
    ------------------
    In a Streamlit environment, simple variable assignment (setting keys to None)
    is often insufficient for guaranteed data isolation, especially in scenarios
    where multiple users might share the same physical machine or browser session.
    
    Stale data persistence can lead to "session bleeding" where fragments of 
    a previous user's state (like cached IDs, document metadata, or UI toggles)
    unexpectedly appear in a subsequent user's experience.
    
    IMPLEMENTATION STRATEGY:
    -----------------------
    1. DATABASE REVOCATION: First, we call the standard logout_user() to 
       synchronously revoke the JWT token in the database. This ensures that 
       even if the client somehow retains the token, it is cryptographically 
       invalidated server-side.
    
    2. STATE WIPE: We then iterate through EVERY key currently present in 
       st.session_state and explicitly delete it. This is more robust than 
       st.session_state.clear() in some edge cases involving widget-linked 
       keys, and it ensures that NO information remains in memory.
    
    3. EXECUTION ABORT: Finally, we trigger st.rerun() to kill the current 
       execution thread and start a fresh, pristine render cycle.
    """
    
    logging.info("Initiating comprehensive logout and session wipe...")
    
    try:
        # Step 1: Backend Revocation
        # This revokes the JWT and clears the standard auth keys in session_state.
        logout_user()
        
        # Step 2: Aggressive Session Wipe
        # We iterate through all keys to ensure nothing is missed, including
        # application-specific keys like 'raw_text', 'summary', 'remedies', etc.
        # This prevents stale analysis data from being visible to the next user.
        for key in list(st.session_state.keys()):
            del st.session_state[key]
            
        logging.info("Session state keys cleared successfully.")
        
    except Exception as e:
        logging.error(f"Critical error during logout session wipe: {str(e)}")
        # We still want to attempt a rerun even if a partial failure occurs
        # to force the UI back to a safe, unauthenticated state.
    
    # Step 3: Forced UI Refresh
    # This abandons the current script run and restarts from the top of app.py.
    st.rerun()


# =============================================================================
# GLOBAL SEARCH COMPONENT WITH XSS MITIGATION
# =============================================================================

def render_global_search_bar():
    """
    Renders a comprehensive global search bar across the application.
    
    SECURITY FOCUS & VULNERABILITY MITIGATION:
    -------------------------------------------
    This component implements strict input sanitization to eliminate 
    Cross-Site Scripting (XSS) and HTML injection vulnerabilities.
    
    Previously, the application was vulnerable to XSS because user-submitted 
    input from the search bar was passed directly to the rendering engine. 
    If a user inputted a malicious payload such as `<script>alert(1)</script>` 
    or `<img src=x onerror=alert(1)>`, it could potentially be executed within 
    the DOM, leading to session hijacking, cookie theft, or unauthorized actions 
    performed on behalf of the user.
    
    By utilizing the native `html.escape` library (or an equivalent like `bleach`), 
    we systematically escape HTML control characters:
      - `<` becomes `&lt;`
      - `>` becomes `&gt;`
      - `&` becomes `&amp;`
      - `"` becomes `&quot;`
      - `'` becomes `&#x27;`
      
    This guarantees that whatever string the user inputs is treated strictly as 
    plaintext data by the browser, completely neutralizing any executable scripts 
    while preserving the visible textual content for the user interface.
    
    Additionally, we enforce maximum input length constraints to protect against 
    Denial of Service (DoS) and layout-breaking attacks using extremely long 
    contiguous strings.
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 Global Search")
    
    # User input field for the global search query
    raw_search_query = st.sidebar.text_input(
        "Search Cases, Documents & Precedents", 
        placeholder="Enter keywords, case numbers...",
        key="global_search_input"
    )
    
    if raw_search_query:
        # ==========================================
        # XSS VULNERABILITY MITIGATION - THE FIX
        # ==========================================
        # Sanitize the input to neutralize HTML tags and script payloads.
        sanitized_query = html.escape(raw_search_query)
        
        # Enforce a maximum length to prevent UI disruption or buffer overflow.
        if len(sanitized_query) > 250:
            sanitized_query = sanitized_query[:250] + "..."
            
        # Securely render the sanitized output back to the user
        st.sidebar.markdown(f"**Results for:** `{sanitized_query}`")
        
        # Simulate backend search processing for UI feedback
        with st.sidebar.status("Searching databases...", expanded=False) as status:
            st.write("Querying Active Case History...")
            st.write("Scanning Uploaded Documents...")
            st.write("Checking Legal Aid Directory...")
            st.write("Evaluating Saved Precedents...")
            status.update(label="Search Complete", state="complete", expanded=False)
            
        # Display simulated mock results based on the sanitized query.
        # In a production scenario, this string would be used to execute 
        # parameterized SQL queries to prevent SQL Injection (SQLi) as well.
        st.sidebar.info(f"No exact matches found for '{sanitized_query}'.")
        
        # Advanced Search Options Toggle - Provides granular search functionality
        with st.sidebar.expander("Advanced Search Options & Filters"):
            st.markdown("**Search Scope:**")
            st.checkbox("Include Closed/Archived Cases", value=False, key="adv_search_closed")
            st.checkbox("Enable Fuzzy Match (Typo Tolerance)", value=True, key="adv_search_fuzzy")
            st.checkbox("Search within Document Text", value=False, key="adv_search_deep")
            
            st.markdown("**Filter by Document Type:**")
            st.multiselect(
                "Select Types",
                ["Judgments", "Contracts", "Legal Notices", "Petitions", "Affidavits", "Other"],
                default=["Judgments", "Legal Notices"],
                key="adv_search_types"
            )
            
            st.markdown("**Filter by Date Range:**")
            col1, col2 = st.columns(2)
            with col1:
                st.date_input("From Date", key="adv_search_from")
            with col2:
                st.date_input("To Date", key="adv_search_to")
                
            st.markdown("**Jurisdiction Filter:**")
            st.selectbox(
                "Court Level",
                ["Any", "Supreme Court", "High Court", "District Court", "Tribunal"],
                key="adv_search_jurisdiction"
            )
            
            st.button("Apply Advanced Filters", key="adv_search_apply", use_container_width=True)
            
    else:
        # Helper instructional text when no query is present
        st.sidebar.caption("Use this global search to quickly navigate your workspace.")
        st.sidebar.caption("Supported queries: Case Numbers (e.g., CA/123), Client Names, Legal Keywords.")


def render_sidebar_navigation():
    """
    Renders the premium sidebar navigation and user profile section.
    
    This component handles the visual representation of the user's auth status
     and provides the primary entry point for the logout workflow.
    """
    st.sidebar.markdown("# ⚖️ LegalEase AI")
    
    # Initialize the Global Search Bar component with integrated XSS protection
    render_global_search_bar()
    
    st.sidebar.markdown("---")
    
    if require_auth():
        user_email = get_current_user_email()
        
        # Premium User Profile UI
        st.sidebar.markdown(f"""
        <div style="background-color: rgba(255, 255, 255, 0.05); padding: 15px; border-radius: 10px; border: 1px solid rgba(255, 255, 255, 0.1); margin-bottom: 20px;">
            <p style="margin: 0; font-size: 0.8rem; opacity: 0.7;">SIGNED IN AS</p>
            <p style="margin: 0; font-weight: bold; overflow: hidden; text-overflow: ellipsis;">{user_email}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Secondary navigation links could go here...
        
        st.sidebar.markdown("### Account Management")
        if st.sidebar.button("🔓 Logout", use_container_width=True, help="Securely sign out and clear all local data"):
            perform_comprehensive_logout()
    else:
        st.sidebar.info("🔒 Secure Mode: Not logged in. Please sign in to access case history and tracking features.")
        
        if st.sidebar.button("🚀 Go to Login", use_container_width=True, type="primary"):
            st.switch_page(routes.PAGE_LOGIN)
            
    st.sidebar.markdown("---")
    st.sidebar.caption("v2.4.0 | LegalAssist AI Enterprise")


# ==================== Main UI Component ====================
def main():
    # Initialize the auth state at the very beginning of the run
    init_auth_session()

    # Render the sidebar navigation and auth controls
    render_sidebar_navigation()

    st.title("⚡ LegalEase AI")
    client = get_client()
    current_language = st.session_state.get("judgment_language", "English")
    ui = get_localized_ui_text(current_language, client)

    st.subheader(ui["app_subtitle"])

    st.markdown(ui["app_intro"])
    st.markdown("---")

    language = st.selectbox("🌐 Select your language", ["English", "Hindi", "Bengali", "Urdu"])
    uploaded_file = st.file_uploader("📄 Upload Judgment PDF / Image", type=["pdf", "jpg", "jpeg", "png", "tiff", "tif", "bmp"])
    enable_ocr = False
    ocr_languages = "eng+hin"
    
    # PDF Validation for size and page count
    is_valid_pdf = True
    if uploaded_file:
        file_ext = uploaded_file.name.split('.')[-1].lower()

        if file_ext != "pdf":
            st.info("📷 Image file detected. OCR will be used to extract text.")
            enable_ocr = True
            ocr_languages = "+".join(
                st.multiselect(
                    "Select OCR languages",
                    ["English", "Hindi", "Bengali", "Urdu"],
                    default=["English"],
                )
            ) or "eng"
        else:
            enable_ocr = st.checkbox("Enable OCR for scanned PDFs", value=False)
            if enable_ocr:
                ocr_languages = "+".join(
                    st.multiselect(
                        "Select OCR languages",
                        ["English", "Hindi", "Bengali", "Urdu"],
                        default=["English"],
                    )
                ) or "eng"

        # Check file size
        MAX_FILE_SIZE_MB = Config.MAX_FILE_SIZE_MB
        WARN_FILE_SIZE_MB = Config.WARN_FILE_SIZE_MB
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
        # Build a content-based cache key so re-uploading a modified file with
        # the same name correctly invalidates the cached result.
        file_bytes = uploaded_file.getvalue()
        content_hash = hashlib.md5(file_bytes).hexdigest()
        cache_key = f"{uploaded_file.name}_{content_hash}_{language}"
        st.session_state.processed_file = uploaded_file.name
        st.session_state.processed_file_hash = content_hash
        st.session_state.last_language = language

    current_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest() if uploaded_file else None
    is_same_file = (uploaded_file is not None) and (st.session_state.get("processed_file") == uploaded_file.name) and (st.session_state.get("processed_file_hash") == current_hash)
    if uploaded_file and is_same_file and st.session_state.get("last_language") == language:
        if not client:
            st.error(ui["openrouter_not_configured"])
            return

        with st.spinner(ui["processing"]):
            try:
                # Compute content hash for cache validation
                file_bytes = uploaded_file.getvalue()
                content_hash = hashlib.md5(file_bytes).hexdigest()
                cache_key = f"{uploaded_file.name}_{content_hash}_{language}"

                # Only call LLM if we haven't processed this exact file/content/language combo
                if st.session_state.get("last_processed") != cache_key:
                    raw_text = extract_text_from_pdf(uploaded_file, enable_ocr=enable_ocr, ocr_languages=ocr_languages)
                    safe_text = compress_text(raw_text)

                    prompt = build_summary_prompt(safe_text, language)

                    # ⚡ Best multilingual model for Hindi/Bengali/Urdu
                    model_id = get_default_model()
                    
                    # Use safe_llm_call for robust error handling and retries
                    summary_raw, error = safe_llm_call(
                        client=client,
                        model=model_id,
                        messages=[
                            {"role": "system", "content": f"You are an expert legal simplification engine. Output only in {language}."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=Config.SUMMARY_MAX_TOKENS,
                        temperature=Config.LLM_TEMPERATURE,
                        timeout=Config.LLM_TIMEOUT,
                    )

                    if error:
                        st.error(f"❌ {error}")
                        return
                    
                    # Use a structured parser to ensure exactly 3 bullet points 
                    # and remove any introductory text like "Here is your summary:"
                    summary = parse_summary_bullets(summary_raw)

                    # -----------------------------
                    # RETRY IF OUTPUT IS NOT IN THE SELECTED LANGUAGE
                    # -----------------------------
                    if language.lower() != "english" and output_language_mismatch_detected(summary, language):
                        retry_prompt = build_retry_prompt(safe_text, language)

                        # Use safe_llm_call for retry as well
                        retry_summary_raw, error2 = safe_llm_call(
                            client=client,
                            model=model_id,
                            messages=[
                                {"role": "system", "content": f"Strict multilingual rewriting engine. Output only in {language}."},
                                {"role": "user", "content": retry_prompt}
                            ],
                            max_tokens=Config.SUMMARY_MAX_TOKENS,
                            temperature=Config.LLM_TEMPERATURE,
                            timeout=Config.LLM_TIMEOUT,
                        )

                        if retry_summary_raw and len(retry_summary_raw) > 0 and not english_leakage_detected(retry_summary_raw):
                            # Apply structured parsing to retry summary as well
                            summary = parse_summary_bullets(retry_summary_raw)

                    remedies = get_remedies_advice(raw_text, language, client)

                    # Save to session
                    st.session_state.raw_text = raw_text
                    st.session_state.summary = summary
                    st.session_state.remedies = remedies
                    st.session_state.last_processed = cache_key
                else:
                    # Load from session
                    raw_text = st.session_state.raw_text
                    summary = st.session_state.summary
                    remedies = st.session_state.remedies

                if not summary:
                    st.error(ui["empty_summary"])
                else:
                    st.markdown(f"## {ui['simplified_judgment']}")
                    st.write(summary)
                    st.success(ui["summary_success"])
                    
                    # ===== REMEDIES SECTION =====
                    st.markdown("---")
                    st.markdown(f"## {ui['remedies_title']}")
                    
                    try:
                        # Show warning if data is partial
                        if remedies.get("_is_partial"):
                            st.warning(ui["partial_warning"])
                        
                        # Show each answer
                        if remedies.get("what_happened"):
                            st.subheader(ui["what_happened"])
                            st.write(remedies["what_happened"])
                        
                        if remedies.get("can_appeal"):
                            st.subheader(ui["can_appeal"])
                            can_appeal_value = remedies["can_appeal"]
                            st.write(localize_yes_no(can_appeal_value, ui))
                            
                            # Only show appeal details if they can appeal
                            if can_appeal_value.strip().lower() == "yes":
                                st.subheader(ui["appeal_details"])
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    if remedies.get("appeal_days"):
                                        st.metric(ui["days_to_file_appeal"], remedies["appeal_days"])
                                    if remedies.get("appeal_court"):
                                        st.write(f"**{ui['appeal_to']}:** {remedies['appeal_court']}")
                                
                                with col2:
                                    if remedies.get("cost"):
                                        st.write(f"**{ui['estimated_cost']}:** {remedies['cost']}")
                        
                        if remedies.get("first_action"):
                            st.subheader(ui["first_action"])
                            st.write(f"✅ {remedies['first_action']}")
                        
                        if remedies.get("deadline"):
                            st.subheader(ui["important_deadline"])
                            st.write(remedies["deadline"])
                        
                        # ===== DRAFTING SECTION =====
                        st.markdown("---")
                        st.markdown("## 📝 One-Click Drafting Center")
                        st.info("Based on these remedies, our AI can generate a formal legal notice or appeal draft for you.")
                        
                        if st.button("⚡ Generate Legal Draft", key="generate_draft_btn"):
                            with st.spinner("Drafting your document..."):
                                draft, error = generate_legal_draft(remedies, language, client)
                                if error:
                                    st.error(f"Drafting failed: {error}")
                                else:
                                    st.session_state.current_draft = draft
                                    st.success("✅ Draft generated! You can edit it below.")
                        
                        if st.session_state.get("current_draft"):
                            edited_draft = st.text_area(
                                "Edit your draft (placeholders in [BRACKETS])", 
                                value=st.session_state.current_draft,
                                height=400,
                                key="edited_draft_area"
                            )
                            st.session_state.current_draft = edited_draft
                            
                            pdf_bytes = export_draft_to_pdf(edited_draft)
                            st.download_button(
                                label="📥 Download as PDF",
                                data=pdf_bytes,
                                file_name="Legal_Notice_Draft.pdf",
                                mime="application/pdf"
                            )
                        
                    except Exception as e:
                        st.error(f"{ui['remedies_error']}")
                        logging.error("Remedies generation failed", error_type=type(e).__name__)
                    
                    render_save_to_case_section(raw_text, summary, remedies)

                    st.markdown("---")
                    st.markdown("## 📊 Track Your Case & See Statistics")
                    
                    st.info("""
                    **Help us build better predictions!**
                    
                    By tracking your case, you help us understand appeal success rates in your jurisdiction.
                    Later, when you know the outcome of your appeal, you can report it back.
                    """)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button("📈 View Analytics", key="view_analytics_bottom"):
                            st.session_state.show_analytics = True
                    
                    with col2:
                        if st.button("🎯 Estimate Appeal Chances", key="estimate_chances_main"):
                            st.session_state.show_estimator = True
                    
                    with col3:
                        if st.button("📝 Report Outcome", key="report_outcome_main"):
                            st.session_state.show_feedback = True
                    
                    # Show analytics if requested
                    if st.session_state.get("show_analytics"):
                        st.subheader("📊 Quick Analytics Preview")
                        db = None
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
                                    regional_trends = AnalyticsAggregator.get_regional_trends(db)
                                    appeal_rate = f"{regional_trends[0]['appeal_success_rate']}%" if regional_trends else 'N/A'
                                    st.metric("Appeals Success Rate", appeal_rate)
                                with col3:
                                    st.metric("Appeals Filed", summary["appeals_filed"])
                                
                                if st.button("📊 View Full Dashboard", use_container_width=True, key="full_dashboard"):
                                    st.switch_page(routes.PAGE_ANALYTICS_DASHBOARD)
                            else:
                                st.info("Analytics will be available as more cases are tracked.")
                        except Exception as e:
                            st.info("Analytics module not ready yet.")
                        finally:
                            if db is not None:
                                db.close()
                    
                    # ===== FREE LEGAL HELP SECTION =====
                    st.markdown("---")
                    render_localized_legal_help(ui)

            except ValueError as e:
                st.error("❌ Extraction Error: Could not extract text from the document. Please ensure it is a valid PDF.")
                logging.error("Text extraction failed", error_type=type(e).__name__)

            except openai.APIConnectionError as e:
                st.error("❌ Network Error: Could not connect to the AI service. Please check your internet.")
                logging.error("API Connection error", error_type=type(e).__name__)

            except openai.RateLimitError as e:
                st.error("❌ Rate Limit: Too many requests. Please wait a moment before trying again.")
                logging.error("API Rate limit reached", error_type=type(e).__name__)

            except openai.AuthenticationError as e:
                st.error("❌ API Key Error: Your OpenRouter/OpenAI key is invalid or not found.")
                logging.error("API Authentication error", error_type=type(e).__name__)

            except openai.APIStatusError as e:
                if e.status_code == 402:
                    st.error("❌ Out of Credits: Please top up your OpenRouter account to continue.")
                else:
                    st.error(f"❌ AI Service Error ({e.status_code}). Please try again later.")
                logging.error("API Status error", error_type=type(e).__name__, status_code=e.status_code)

            except openai.APIError as e:
                st.error("❌ AI Service Error: The AI service returned an unexpected response.")
                logging.error("OpenAI API error", error_type=type(e).__name__)

            except Exception as e:
                st.error("❌ Unexpected Error: Something went wrong. Please try again.")
                logging.error("Unhandled exception in main loop", error_type=type(e).__name__)

if __name__ == "__main__":
    main()
