import pytest
from core.app_utils import (
    build_summary_prompt,
    english_leakage_detected
)

# ================= SAMPLE TEXT =================
SAMPLE_TEXT = """
The court held that the accused is guilty under Section 420 IPC
for cheating and dishonestly inducing delivery of property.
The accused is sentenced to 2 years of imprisonment.
"""

# ================= TEST 1: HINDI =================
def test_hindi_prompt_generation():
    language = "Hindi"
    prompt = build_summary_prompt(SAMPLE_TEXT, language)

    assert language.lower() in prompt.lower()
    assert "summary" in prompt.lower()  # basic sanity check


# ================= TEST 2: MARATHI =================
def test_marathi_prompt_generation():
    language = "Marathi"
    prompt = build_summary_prompt(SAMPLE_TEXT, language)

    assert language.lower() in prompt.lower()
    assert len(prompt) > 50  # ensures meaningful prompt


# ================= TEST 3: TAMIL =================
def test_tamil_prompt_generation():
    language = "Tamil"
    prompt = build_summary_prompt(SAMPLE_TEXT, language)

    assert language.lower() in prompt.lower()
    assert len(prompt) > 50


# ================= TEST 4: ENGLISH LEAKAGE =================
import re

def english_leakage_detected(text: str) -> bool:
    """
    Detects if English words are present in non-English text.
    Flags even small words like 'test', 'case', etc.
    """

    if not text:
        return False

    # Detect any English word (a-z or A-Z, length >=2)
    english_words = re.findall(r"\b[a-zA-Z]{2,}\b", text)

    return len(english_words) > 0