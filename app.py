import streamlit as st
from openai import OpenAI
import PyPDF2
import re
import time

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
client = OpenAI(
    api_key=st.secrets["OPENROUTER_API_KEY"],
    base_url=st.secrets["OPENROUTER_BASE_URL"]
)

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
# UI
# -----------------------------
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

# -----------------------------
# Main Action
# -----------------------------
if uploaded_file and st.button("🚀 Generate Summary"):
    with st.spinner("Processing judgment…"):
        try:
            raw_text = extract_text_from_pdf(uploaded_file)
            safe_text = compress_text(raw_text)

            prompt = build_prompt(safe_text, language)

            # ⚡ Best multilingual model for Hindi/Bengali/Urdu
            model_id = "qwen/qwen-2.5-7b-instruct"


            # -----------------------------
            # FIRST ATTEMPT
            # -----------------------------
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": "You are an expert legal simplification engine."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=350,
                temperature=0.12,
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
                    max_tokens=320,
                    temperature=0.05
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

        except Exception as e:
            err = str(e)

            if "402" in err or "credits" in err.lower():
                st.error("❌ Not enough OpenRouter credits. Please top up.")
            else:
                st.error(f"An error occurred: {err}")