
import pytest
import os
import json
from pypdf import PdfReader
from unittest.mock import MagicMock, patch
from core import app_utils as core

# Mock streamlit secrets and session state for testing
import streamlit as st
st.secrets = {"OPENROUTER_API_KEY": "test_key", "OPENROUTER_BASE_URL": "test_url"}
if "openai_client" not in st.session_state:
    st.session_state.openai_client = MagicMock()

# Load test metadata
def load_test_cases():
    metadata_path = "tests/test_metadata.json"
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    
    # Fallback mock data if file is missing or corrupt
    # This allows tests to load even in fresh environments
    return [
        {
            "path": "tests/samples/criminal/guilty/case_1.pdf",
            "type": "criminal_guilty",
            "expected_verdict": "guilty"
        }
    ]

test_cases = load_test_cases()

def test_pdf_extraction():
    """Test if PDF extraction works for all sample files"""
    if not test_cases:
        pytest.skip("No test cases found in metadata")
        
    files_tested = 0
    for case in test_cases:
        path = case["path"]
        if not os.path.exists(path):
            continue
        
        with open(path, "rb") as f:
            text = core.extract_text_from_pdf(f)
            # More robust check: at least 100 characters and contains legal terminology
            assert len(text) > 100, f"Extraction returned too little text for {path}"
            
            keywords = ["judgment", "judgement", "court", "case", "verdict", "order", "plaintiff", "defendant", "petitioner", "respondent"]
            text_lower = text.lower()
            assert any(kw in text_lower for kw in keywords), f"No legal keywords found in extracted text from {path}"
            files_tested += 1
            
    if files_tested == 0:
        pytest.skip("No sample PDF files found on disk. Run scripts/generate_test_data.py first.")

def test_compress_text():
    """Test text compression logic"""
    long_text = "A" * 10000
    compressed = core.compress_text(long_text, limit=6000)
    assert len(compressed) < 10000
    assert "... [TRUNCATED] ..." in compressed
    
    short_text = "Small text"
    assert core.compress_text(short_text) == short_text

def test_english_leakage():
    """Test English leakage detection"""
    assert core.english_leakage_detected("This is a simple the and of test", threshold=3) == True
    assert core.english_leakage_detected("नमस्ते यह एक परीक्षण है", threshold=3) == False

def test_parse_remedies_response():
    """Test parsing of LLM response for remedies"""
    mock_response = """
    1. What happened?
    The plaintiff won the property dispute case.

    2) Can the loser appeal?
    Yes, they can appeal.

    3: Appeal timeline
    30 days

    4- Appeal court
    High Court

    5. Cost estimate
    5000-10000

    6. First action
    File a certified copy request.

    7. Important deadline
    The 30 day deadline.
    """
    remedies = core.parse_remedies_response(mock_response)
    assert remedies["what_happened"] == "The plaintiff won the property dispute case."
    assert remedies["can_appeal"] == "yes"
    assert remedies["appeal_days"] == "30"
    assert remedies["appeal_court"] == "High Court"
    assert remedies["cost_estimate"] == "5000-10000"
    assert remedies["first_action"] == "File a certified copy request."
    assert remedies["deadline"] == "The 30 day deadline."

@pytest.mark.parametrize("language", ["English", "Hindi", "Bengali", "Urdu"])
def test_all_languages_prompt_building(language):
    """Test prompt building for all supported languages"""
    prompt = core.build_summary_prompt("Sample text", language)
    assert language in prompt
    assert "3 bullet points" in prompt

@patch("core.app_utils.get_client")
def test_get_remedies_advice_flow(mock_get_client):
    """Test the full flow of getting remedies with mocked LLM"""
    # Setup mock
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_openai = mock_client.chat.completions.create
    mock_choice = MagicMock()
    mock_choice.message.content = """
    1. What happened?
    Defendant was convicted.
    2. Can the loser appeal?
    Yes.
    3. Appeal timeline
    30
    4. Appeal court
    Sessions Court
    5. Cost estimate
    5000
    6. What should they do first?
    Apply for bail.
    7. Important deadline
    Next 30 days.
    """
    mock_openai.return_value.choices = [mock_choice]
    
    remedies = core.get_remedies_advice("Some judgment text", "English", mock_client)
    assert remedies["what_happened"] == "Defendant was convicted."
    assert remedies["can_appeal"] == "yes"
    assert remedies["appeal_days"] == "30"
    assert remedies["appeal_court"] == "Sessions Court"
    assert remedies["cost"] == "5000"
    assert mock_openai.called

def test_judgment_summary_quality_manual():
    """
    Placeholder for manual verification. 
    In a real CI, we might use an LLM-as-a-judge or check for 3 bullet points.
    """
    # Just check if our prompt asks for 3 bullets
    prompt = core.build_summary_prompt("test", "Hindi")
    assert "EXACTLY 3 bullet points" in prompt

if __name__ == "__main__":
    pytest.main([__file__])
