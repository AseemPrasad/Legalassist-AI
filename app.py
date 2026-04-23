import streamlit as st
import openai
from openai import OpenAI
from pypdf import PdfReader
import logging
import os
import re

# ==================== Import Utilities ====================
from core.app_utils import (
    get_client,
    get_default_model,
    extract_text_from_pdf,
    compress_text,
    english_leakage_detected,
    build_summary_prompt,
    build_retry_prompt,
    get_remedies_advice,
)

# ==================== SAFE IMPORT (IGNORE AUTH ERRORS) ====================
try:
    from database import init_db, SessionLocal, DocumentType
    from scheduler import start_scheduler
except:
    pass

# ==================== INIT ====================
try:
    init_db()
except:
    pass

if "scheduler_started" not in st.session_state:
    try:
        start_scheduler()
        st.session_state.scheduler_started = True
    except:
        st.session_state.scheduler_started = False

logging.basicConfig(level="INFO")

st.set_page_config(page_title="LegalEase AI", page_icon="⚖")

# ==================== LANGUAGE MAP ====================
LANGUAGE_MAP = {
    "English": "English",
    "Hindi": "Hindi",
    "Bengali": "Bengali",
    "Urdu": "Urdu",
    "Marathi": "Marathi",
    "Tamil": "Tamil",
    "Telugu": "Telugu",
    "Kannada": "Kannada",
    "Gujarati": "Gujarati",
    "Punjabi": "Punjabi"
}

# ==================== MAIN ====================
def main():

    st.sidebar.markdown("# ⚖️ LegalEase AI")
    st.sidebar.info("Auth disabled (dev mode)")

    st.title("⚡ LegalEase AI")
    st.subheader("Legal Judgment Simplifier")

    st.markdown("Convert complex judgments into simple summaries.")

    # ==================== LANGUAGE ====================
    language = st.selectbox("🌐 Select Language", list(LANGUAGE_MAP.keys()))
    target_language = LANGUAGE_MAP.get(language, "English")

    uploaded_file = st.file_uploader("📄 Upload PDF", type=["pdf"])

    if uploaded_file and st.button("🚀 Generate Summary"):

        client = get_client()

        if not client:
            st.error("❌ API not configured")
            return

        with st.spinner("Processing..."):
            try:
                raw_text = extract_text_from_pdf(uploaded_file)
                safe_text = compress_text(raw_text)

                # ==================== PROMPT (UPDATED) ====================
                base_prompt = build_summary_prompt(safe_text, target_language)

                prompt = f"""
You are a legal assistant.

IMPORTANT RULES:
- Answer ONLY in {target_language}
- Do NOT use English unless the selected language is English
- Use simple, easy-to-understand language
- Output must be fully in {target_language}

{base_prompt}
"""

                model_id = get_default_model()

                response = client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": "Legal simplification engine"},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=300,
                    temperature=0.1,
                )

                summary = response.choices[0].message.content.strip()

                # ==================== RETRY IF ENGLISH ====================
                if language != "English" and english_leakage_detected(summary):

                    retry_prompt = f"""
The previous output contained English words.

Rewrite the following STRICTLY in {target_language}:
- No English words at all
- Keep meaning same
- Use simple language

TEXT:
{summary}
"""

                    response2 = client.chat.completions.create(
                        model=model_id,
                        messages=[{"role": "user", "content": retry_prompt}],
                        max_tokens=300,
                        temperature=0.01,
                    )

                    retry_summary = response2.choices[0].message.content.strip()

                    if retry_summary and not english_leakage_detected(retry_summary):
                        summary = retry_summary

                # ==================== OUTPUT ====================
                st.markdown("## ✅ Simplified Judgment")
                st.write(summary)

                # ==================== REMEDIES ====================
                st.markdown("## ⚖️ What Can You Do?")
                try:
                    remedies = get_remedies_advice(raw_text, language)

                    if remedies.get("what_happened"):
                        st.write(remedies["what_happened"])

                    if remedies.get("can_appeal"):
                        st.write(remedies["can_appeal"])

                except:
                    st.info("Remedies module optional (skipped)")

                st.success("Done!")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# ==================== RUN ====================
if __name__ == "__main__":
    main()