import streamlit as st
from openai import OpenAI
from pypdf import PdfReader
import re
import core

DEFAULT_MODEL = st.secrets.get("DEFAULT_MODEL", core.DEFAULT_MODEL)

# -----------------------------
# App Config
# -----------------------------
st.set_page_config(
    page_title="LegalEase AI (Fixed)",
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
    reader = PdfReader(uploaded_pdf)
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
# Prompts
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
# Remedies Advisor (fixed parser)
# -----------------------------

def build_remedies_prompt(judgment_text, language):
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
    remedies = {
        "what_happened": "",
        "can_appeal": "",
        "appeal_days": "",
        "appeal_court": "",
        "cost": "",
        "first_action": "",
        "deadline": "",
        "appeal_details": ""
    }

    text = response_text.strip()
    if not text:
        return remedies

    # Split into numbered chunks (1-7)
    for i, key in [(1, "what_happened"),
                   (2, "can_appeal"),
                   (3, "appeal_days"),
                   (4, "appeal_court"),
                   (5, "cost"),
                   (6, "first_action"),
                   (7, "deadline")]:
        marker = f"{i}."
        if marker not in text:
            continue
        start = text.index(marker) + len(marker)
        end = len(text)
        for j in range(i + 1, 8):
            next_marker = f"{j}."
            if next_marker in text[start:]:
                end = text.index(next_marker, start)
                break
        answer = text[start:end].strip()

        if i == 3:
            remedies["appeal_days"] = answer
            remedies["appeal_details"] += f"{answer} "
        elif i == 4:
            remedies["appeal_court"] = answer
            remedies["appeal_details"] += f"{answer} "
        elif i == 5:
            remedies["cost"] = answer
            remedies["appeal_details"] += f"{answer} "
        else:
            remedies[key] = answer

    return remedies


def extract_appeal_info(appeal_details_text):
    info = {
        "days": "",
        "court": "",
        "cost": ""
    }
    text = appeal_details_text or ""

    days_match = re.search(r"(\d+)\s*(?:days?|day)", text.lower())
    if days_match:
        info["days"] = days_match.group(1)

    court_keywords = ["High Court", "District Court", "Supreme Court", "Lower Court"]
    for keyword in court_keywords:
        if keyword.lower() in text.lower():
            info["court"] = keyword
            break

    cost_match = re.search(r"₹?([\d,]+(?:[-–][\d,]+)?)", text.replace(" ", ""))
    if cost_match:
        info["cost"] = cost_match.group(1)

    return info


def get_remedies_advice(judgment_text, language):
    prompt = build_remedies_prompt(judgment_text, language)

    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful legal advisor. Answer questions about legal remedies in India."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.1,
    )

    response_text = response.choices[0].message.content.strip()
    remedies = parse_remedies_response(response_text)
    return remedies

# -----------------------------
# UI
# -----------------------------
st.title("⚡ LegalEase AI (Fixed)")
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
            model_id = DEFAULT_MODEL

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
                                appeal_info = extract_appeal_info(remedies.get("appeal_details", ""))
                                st.subheader("Appeal Details")
                                col1, col2 = st.columns(2)
                                with col1:
                                    if appeal_info["days"]:
                                        st.metric("Days to File Appeal", f"{appeal_info['days']} days")
                                    if appeal_info["court"]:
                                        st.write(f"**Appeal to:** {appeal_info['court']}")
                                with col2:
                                    if appeal_info["cost"]:
                                        st.write(f"**Estimated Cost:** ₹{appeal_info['cost']}")

                        if remedies.get("first_action"):
                            st.subheader("What Should You Do First?")
                            st.write(f"✅ {remedies['first_action']}")

                        if remedies.get("deadline"):
                            st.subheader("⏰ Important Deadline")
                            st.write(remedies["deadline"])

                    except Exception as e:
                        st.error(f"Could not get remedies advice: {str(e)}")

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
