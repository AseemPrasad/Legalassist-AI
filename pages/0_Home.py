"""
Home page - Judgment Analysis (Main page)
This is the primary feature of LegalEase AI

CHANGE: build_judgment_result_text now returns (plain_text, structured_dict).
        render_shareable_result_box accepts the tuple directly — no other changes needed.
"""

import streamlit as st
import logging
import sys
import os
from config import Config

# Add parent directory to sys.path to resolve 'core' module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.app_utils import (
    get_client,
    extract_text_from_pdf,
    compress_text,
    english_leakage_detected,
    output_language_mismatch_detected,
    build_prompt,
    build_retry_prompt,
    get_remedies_advice,
    extract_appeal_info,
    get_localized_ui_text,
    localize_yes_no,
    RETRO_STYLING,
    LANGUAGES,
    parse_summary_bullets,
    validate_pdf_metadata,
    build_judgment_result_text,
    render_shareable_result_box,
    safe_llm_call,
    generate_legal_draft,
    export_draft_to_pdf,
)

st.markdown(RETRO_STYLING, unsafe_allow_html=True)

# FIX: DevTools revealed the winning rule is:
#   .st-emotion-cache-za2i0z h1 { font-size: 2.75rem }
# We cannot hardcode that hash (it changes between Streamlit versions/builds).
# Solution: use st.markdown() to inject a completely custom <div> that is NOT
# an <h1> at all — Streamlit's h1 rules never touch it. We style it ourselves
# with clamp() so it scales with viewport width.
# subheader (h2) has the same problem, so we replace that too.

MOBILE_HEADER_CSS = """
<style>
  .app-title {
    font-size: clamp(1.0rem, 5.5vw, 1.5rem);
    font-weight: 700;
    line-height: 1.2;
    margin: 0.4rem 0 0.1rem;
    color: inherit;
    white-space: nowrap;
  }
  .app-subtitle {
    font-size: clamp(0.85rem, 3.8vw, 1.25rem);
    font-weight: 600;
    line-height: 1.3;
    margin: 0.2rem 0 0.6rem;
    color: inherit;
    white-space: nowrap;
  }
  /* Samsung Galaxy S and similarly narrow devices (360px) */
  @media (max-width: 380px) {
    .app-title    { font-size: 1.05rem !important; }
    .app-subtitle { font-size: 0.85rem !important; }
    /* Reduce Streamlit's default side padding to reclaim horizontal space */
    .block-container,
    div[data-testid="stAppViewBlockContainer"] {
      padding-left: 0.6rem !important;
      padding-right: 0.6rem !important;
    }
  }
  @media (max-width: 340px) {
    .app-title    { font-size: 0.95rem !important; white-space: normal; word-break: keep-all; }
    .app-subtitle { font-size: 0.8rem  !important; white-space: normal; word-break: keep-all; }
  }
</style>
"""


def render_page():
    """Render the judgment analysis page"""
    # Get client early for translation
    client = get_client()
    
    current_language = st.session_state.get("judgment_language", "English")
    ui = get_localized_ui_text(current_language, client)

    st.title("⚡ LegalEase AI")
    st.subheader(ui["app_subtitle"])

    st.markdown(ui["app_intro"])
    st.markdown("---")

    language = st.selectbox(ui["language_label"], LANGUAGES, key="judgment_language")
    ui = get_localized_ui_text(language, client)
    
    input_method = st.radio(
        ui["input_method"],
        [ui["upload_pdf"], ui["paste_text"]],
        horizontal=True,
    )

    is_valid_input = False
    uploaded_file = None
    pasted_text = None

    if input_method == ui["upload_pdf"]:
        uploaded_file = st.file_uploader(ui["upload_label"], type=["pdf"])
        if uploaded_file:
            is_valid_input = True
    else:
        pasted_text = st.text_area(
            ui.get("paste_text", "📋 Paste Text"),
            height=250,
        )
        if pasted_text and pasted_text.strip():
            is_valid_input = True

    st.markdown("---")

    if st.button(ui["generate_summary"], use_container_width=True):
        if not is_valid_input:
            st.error("Please upload a PDF or paste the judgment text to continue.")
        else:
            with st.spinner(ui["processing"]):
                try:
                    try:
                        client = get_client()
                        ui = get_localized_ui_text(language, client)
                    except Exception as e:
                        st.error(f"❌ {ui['api_client_failed']}: {str(e)}")
                        return

                    if input_method == ui["upload_pdf"]:
                        raw_text = extract_text_from_pdf(uploaded_file)
                    else:
                        raw_text = pasted_text
                    
                    # --- NEW RAG STATE ---
                    st.session_state["judgment_raw_text"] = raw_text
                    st.session_state["chat_history"] = []
                    st.session_state["rag_initialized"] = False
                    # ---------------------

                    safe_text = compress_text(raw_text)

                    prompt = build_prompt(safe_text, language)
                    model_id = "meta-llama/llama-3.1-8b-instruct"

                    # Use safe_llm_call for robust error handling and retries
                    summary_raw, error = safe_llm_call(
                        client=client,
                        model=model_id,
                        messages=[
                            {"role": "system", "content": f"You are an expert legal simplification engine. Output only in {language}."},
                            {"role": "user", "content": prompt},
                        ],
                        max_tokens=Config.SUMMARY_MAX_TOKENS,
                        temperature=0.05,
                    )

                    if error:
                        st.error(f"❌ {error}")
                        return
                    
                    summary = summary_raw

                    if language.lower() != "english" and output_language_mismatch_detected(summary, language):
                        retry_prompt = build_retry_prompt(safe_text, language)
                        # Use safe_llm_call for retry as well
                        retry_summary, error2 = safe_llm_call(
                            client=client,
                            model=model_id,
                            messages=[
                                {"role": "system", "content": f"Strict multilingual rewriting engine. Output only in {language}."},
                                {"role": "user", "content": retry_prompt},
                            ],
                            max_tokens=Config.SUMMARY_MAX_TOKENS,
                            temperature=0.03,
                        )
                        if len(retry_summary) > 0 and not output_language_mismatch_detected(retry_summary, language):
                            summary = retry_summary

                    if not summary:
                        st.error(ui["empty_summary"])
                    else:
                        remedies = {}

                        with st.spinner(ui["remedies_spinner"]):
                            try:
                                remedies = get_remedies_advice(raw_text, language, client) or {}
                            except Exception as e:
                                st.error(f"{ui['remedies_error']}: {str(e)}")

                        # build_judgment_result_text now returns (plain_text, structured_dict)
                        result = build_judgment_result_text(summary, remedies, ui)

                        # render_shareable_result_box accepts the tuple directly
                        render_shareable_result_box(result, ui)
                        st.success(ui["summary_success"])

                        # ===== DRAFTING SECTION =====
                        st.markdown("---")
                        st.markdown("## 📝 One-Click Drafting Center")
                        st.info("Based on these remedies, our AI can generate a formal legal notice or appeal draft for you.")
                        
                        if st.button("⚡ Generate Legal Draft", key="generate_home_draft"):
                            with st.spinner("Drafting your document..."):
                                # Remedies is part of the result tuple (summary, remedies, ui) or extracted
                                # In 0_Home.py, remedies is already defined in line 188
                                draft, error = generate_legal_draft(remedies, language, client)
                                if error:
                                    st.error(f"Drafting failed: {error}")
                                else:
                                    st.session_state.current_home_draft = draft
                                    st.success("✅ Draft generated! You can edit it below.")
                        
                        if st.session_state.get("current_home_draft"):
                            edited_draft = st.text_area(
                                "Edit your draft", 
                                value=st.session_state.current_home_draft,
                                height=350,
                                key="home_draft_area"
                            )
                            st.session_state.current_home_draft = edited_draft
                            
                            pdf_bytes = export_draft_to_pdf(edited_draft)
                            st.download_button(
                                label="📥 Download as PDF",
                                data=pdf_bytes,
                                file_name="Legal_Notice_Draft.pdf",
                                mime="application/pdf",
                                key="download_home_draft"
                            )

                        # ===== VOICE ACCESSIBILITY (TTS) =====
                        st.markdown("---")
                        st.markdown("### 🎧 Listen to Summary")
                        plain_text_summary = result[0] if isinstance(result, tuple) else result
                        if st.button("🔊 Generate Audio", key="generate_audio_btn"):
                            with st.spinner("Generating audio..."):
                                from core.audio_utils import generate_audio
                                audio_bytes = generate_audio(plain_text_summary, language)
                                if audio_bytes:
                                    st.audio(audio_bytes, format="audio/mp3")
                                else:
                                    st.error("Audio generation is not supported for this language or failed.")

                        # ===== ANALYTICS & TRACKING SECTION =====
                        st.markdown("---")
                        st.markdown(f"## {ui['track_title']}")
                        st.info(ui["track_info"])

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            if st.button(ui["view_analytics"], key="view_analytics"):
                                st.session_state.show_analytics = True

                        with col2:
                            if st.button(ui["estimate_chances"], key="estimate_chances"):
                                st.session_state.show_estimator = True

                        with col3:
                            if st.button(ui["report_outcome"], key="report_outcome"):
                                st.session_state.show_feedback = True

                        if st.session_state.get("show_analytics"):
                            st.subheader(ui["quick_analytics_preview"])
                            try:
                                from analytics_engine import AnalyticsAggregator
                                from database import SessionLocal

                                db = SessionLocal()
                                summary_data = AnalyticsAggregator.get_dashboard_summary(db)

                                if summary_data.get("total_cases_processed", 0) > 0:
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric(ui["total_cases_tracked"], summary_data["total_cases_processed"])
                                    with col2:
                                        trends = AnalyticsAggregator.get_regional_trends(db)
                                        success_rate = trends[0]['appeal_success_rate'] if trends else 'N/A'
                                        st.metric(ui["appeals_success_rate"], f"{success_rate}%")
                                    with col3:
                                        st.metric(ui["appeals_filed"], summary_data.get("appeals_filed", 0))
                                    st.write(f"📌 **{ui['analytics_link_text']}**")
                                else:
                                    st.info(ui["analytics_empty"])

                                db.close()
                            except Exception as e:
                                st.info(ui["analytics_not_ready"])

                        # ===== FREE LEGAL HELP SECTION =====
                        st.markdown("---")
                        st.markdown(f"## {ui['free_legal_help']}")
                        st.info(ui["legal_help_resources"])

                        # ===== RAG CHAT REDIRECT =====
                        st.markdown("---")
                        st.markdown("## 💬 Chat with Judgment")
                        st.info("Have specific questions about this document? You can ask our AI assistant.")
                        if st.button("💬 Open Interactive Chat", use_container_width=True):
                            st.switch_page("pages/4_Chat.py")

                except Exception as e:
                    err = str(e)
                    if "402" in err or "credits" in err.lower():
                        st.error(ui["not_enough_credits"])
                    else:
                        st.error(ui["generic_error"].format(error=err))
                        logging.error(f"Error in judgment analysis: {err}")


if __name__ == "__main__":
    render_page()