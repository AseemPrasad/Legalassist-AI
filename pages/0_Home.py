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
    build_prompt,
    build_retry_prompt,
    get_remedies_advice,
    extract_appeal_info,
    RETRO_STYLING,
    LEGAL_HELP_RESOURCES,
    LANGUAGES,
)

# Apply styling
st.markdown(RETRO_STYLING, unsafe_allow_html=True)


def render_page():
    """Render the judgment analysis page"""
    st.title("⚡ LegalEase AI")
    st.subheader("Legal Judgment Simplifier")

    st.markdown("""
    LegalEase AI breaks the Information Barrier in the Judiciary by converting
    complex court judgments into clear, 3-point summaries in your chosen language.
    """)
    st.markdown("---")

    # Input section
    language = st.selectbox("🌐 Select your language", LANGUAGES)
    uploaded_file = st.file_uploader("📄 Upload Judgment PDF", type=["pdf"])
    st.markdown("---")

    if uploaded_file and st.button("🚀 Generate Summary", use_container_width=True):
        with st.spinner("Processing judgment…"):
            try:
                # Get client
                try:
                    client = get_client()
                except Exception as e:
                    st.error(f"❌ Failed to initialize API client: {str(e)}")
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
                        {"role": "system", "content": "You are an expert legal simplification engine."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=280,
                    temperature=0.05,
                )

                summary = response.choices[0].message.content.strip()

                # Retry if English leakage detected
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
                    # Display results
                    st.markdown("## ✅ Simplified Judgment")
                    st.write(summary)
                    st.success("The judgment has been simplified successfully.")
                    
                    # ===== REMEDIES SECTION =====
                    st.markdown("---")
                    st.markdown("## ⚖️ What Can You Do Now?")
                    
                    with st.spinner("Analyzing your legal options..."):
                        try:
                            remedies = get_remedies_advice(raw_text, language, client)
                            
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
                            from database import SessionLocal
                            
                            db = SessionLocal()
                            summary_data = AnalyticsAggregator.get_dashboard_summary(db)
                            
                            if summary_data.get("total_cases_processed", 0) > 0:
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Total Cases Tracked", summary_data["total_cases_processed"])
                                with col2:
                                    trends = AnalyticsAggregator.get_regional_trends(db)
                                    success_rate = trends[0]['appeal_success_rate'] if trends else 'N/A'
                                    st.metric("Appeals Success Rate", f"{success_rate}%")
                                with col3:
                                    st.metric("Appeals Filed", summary_data.get("appeals_filed", 0))
                                
                                st.write("📌 **Visit Analytics Dashboard for detailed insights** ➡️ [See Full Dashboard]()")
                            else:
                                st.info("Analytics will be available as more cases are tracked.")
                            
                            db.close()
                        except Exception as e:
                            st.info("Analytics module not ready yet.")
                    
                    # ===== FREE LEGAL HELP SECTION =====
                    st.markdown("---")
                    st.markdown("## 📞 Free Legal Help")
                    st.info(LEGAL_HELP_RESOURCES)

            except Exception as e:
                err = str(e)
                if "402" in err or "credits" in err.lower():
                    st.error("❌ Not enough OpenRouter credits. Please top up.")
                else:
                    st.error(f"An error occurred: {err}")
                    logging.error(f"Error in judgment analysis: {err}")


if __name__ == "__main__":
    render_page()
