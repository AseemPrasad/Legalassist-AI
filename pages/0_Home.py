"""
Home page - Judgment Analysis (Main page)
This is the primary feature of LegalEase AI
"""

import streamlit as st
import logging

# Import utilities from core
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
)

# Apply styling
st.markdown(RETRO_STYLING, unsafe_allow_html=True)


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

    # Input section
    language = st.selectbox(ui["language_label"], LANGUAGES, key="judgment_language")
    ui = get_localized_ui_text(language, client)
    uploaded_file = st.file_uploader(ui["upload_label"], type=["pdf"])
    st.markdown("---")

    if uploaded_file and st.button(ui["generate_summary"], use_container_width=True):
        with st.spinner(ui["processing"]):
            try:
                # Get client
                try:
                    client = get_client()
                    ui = get_localized_ui_text(language, client)
                except Exception as e:
                    st.error(f"❌ {ui['api_client_failed']}: {str(e)}")
                    return

                # Extract and process text
                raw_text = extract_text_from_pdf(uploaded_file)
                safe_text = compress_text(raw_text)

                prompt = build_prompt(safe_text, language)
                model_id = "meta-llama/llama-3.1-8b-instruct"

                # First attempt
                response = client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": f"You are an expert legal simplification engine. Output only in {language}."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=280,
                    temperature=0.05,
                )

                summary = response.choices[0].message.content.strip()

                # Retry if English leakage detected
                if language.lower() != "english" and output_language_mismatch_detected(summary, language):
                    retry_prompt = build_retry_prompt(safe_text, language)
                    response2 = client.chat.completions.create(
                        model=model_id,
                        messages=[
                            {"role": "system", "content": f"Strict multilingual rewriting engine. Output only in {language}."},
                            {"role": "user", "content": retry_prompt}
                        ],
                        max_tokens=260,
                        temperature=0.03,
                    )
                    retry_summary = response2.choices[0].message.content.strip()
                    if len(retry_summary) > 0 and not output_language_mismatch_detected(retry_summary, language):
                        summary = retry_summary

                if not summary:
                    st.error(ui["empty_summary"])
                else:
                    # Display results
                    st.markdown(f"## {ui['simplified_judgment']}")
                    st.write(summary)
                    st.success(ui["summary_success"])
                    
                    # ===== REMEDIES SECTION =====
                    st.markdown("---")
                    st.markdown(f"## {ui['remedies_title']}")
                    
                    with st.spinner(ui["remedies_spinner"]):
                        try:
                            remedies = get_remedies_advice(raw_text, language, client)
                            
                            if remedies.get("what_happened"):
                                st.subheader(ui["what_happened"])
                                st.write(remedies["what_happened"])
                            
                            if remedies.get("can_appeal"):
                                st.subheader(ui["can_appeal"])
                                can_appeal_value = remedies["can_appeal"]
                                st.write(localize_yes_no(can_appeal_value, ui))
                                
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
                            
                        except Exception as e:
                            st.error(f"{ui['remedies_error']}: {str(e)}")
                    
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
                    
                    # Show analytics if requested
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

            except Exception as e:
                err = str(e)
                if "402" in err or "credits" in err.lower():
                    st.error(ui["not_enough_credits"])
                else:
                    st.error(ui["generic_error"].format(error=err))
                    logging.error(f"Error in judgment analysis: {err}")


if __name__ == "__main__":
    render_page()
