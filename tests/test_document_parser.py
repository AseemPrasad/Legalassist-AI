name=tests/test_document_parser.py
"""
Unit tests for document_parser metadata extraction module.

Tests cover:
- Pattern-based extraction of case metadata
- Date format normalization
- Party name extraction
- Confidence scoring
- Edge cases and error handling
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from document_parser import (
    extract_metadata,
    CaseMetadata,
    _extract_case_number,
    _extract_dates,
    _extract_judge,
    _extract_court,
    _extract_parties,
    _normalize_date,
    _calculate_confidence,
)


class TestCaseMetadata:
    """Test CaseMetadata dataclass."""
    
    def test_dataclass_initialization(self):
        metadata = CaseMetadata(
            case_number="2023/123",
            filing_date="2023-01-15",
            court_name="Delhi High Court",
        )
        assert metadata.case_number == "2023/123"
        assert metadata.court_name == "Delhi High Court"
    
    def test_to_dict(self):
        metadata = CaseMetadata(
            case_number="2023/123",
            confidence_score=0.85,
        )
        data = metadata.to_dict()
        assert isinstance(data, dict)
        assert data["case_number"] == "2023/123"
        assert data["confidence_score"] == 0.85
    
    def test_to_json(self):
        metadata = CaseMetadata(case_number="2023/123")
        json_str = metadata.to_json()
        data = json.loads(json_str)
        assert data["case_number"] == "2023/123"


class TestDateNormalization:
    """Test date parsing and normalization."""
    
    def test_dd_mm_yyyy_format(self):
        result = _normalize_date("15-01-2023")
        assert result == "2023-01-15"
    
    def test_dd_slash_mm_slash_yyyy_format(self):
        result = _normalize_date("15/01/2023")
        assert result == "2023-01-15"
    
    def test_word_date_format(self):
        result = _normalize_date("15 January 2023")
        assert result == "2023-01-15"
    
    def test_short_month_format(self):
        result = _normalize_date("15 Jan 2023")
        assert result == "2023-01-15"
    
    def test_invalid_date_returns_original(self):
        result = _normalize_date("invalid-date")
        assert result == "invalid-date"
    
    def test_none_input(self):
        result = _normalize_date(None)
        assert result is None
    
    def test_empty_string(self):
        result = _normalize_date("")
        assert result is None


class TestCaseNumberExtraction:
    """Test case number extraction."""
    
    def test_case_no_pattern(self):
        text = "Case No. 2023-456-789"
        result = _extract_case_number(text)
        assert result is not None
        assert "2023" in result
    
    def test_writ_petition_pattern(self):
        text = "Writ Petition No. 123 of 2023"
        result = _extract_case_number(text)
        assert result is not None
    
    def test_criminal_appeal_pattern(self):
        text = "Criminal Appeal No. 234 of 2023"
        result = _extract_case_number(text)
        assert result is not None
    
    def test_civil_appeal_pattern(self):
        text = "Civil Appeal No. 345 of 2023"
        result = _extract_case_number(text)
        assert result is not None
    
    def test_no_case_number(self):
        text = "This is a judgment text without case number"
        result = _extract_case_number(text)
        assert result is None


class TestDateExtraction:
    """Test date extraction."""
    
    def test_extract_filing_and_judgment_dates(self):
        text = """
        This case was filed on 15-01-2023.
        The judgment was delivered on 20-05-2023.
        """
        filing, judgment = _extract_dates(text)
        assert filing is not None
        assert judgment is not None
    
    def test_extract_only_judgment_date(self):
        text = "The judgment dated 20-05-2023 is hereby delivered."
        filing, judgment = _extract_dates(text)
        assert judgment is not None
    
    def test_extract_dates_with_different_formats(self):
        text = "Filed: 15/01/2023\nJudgment: 20 May 2023"
        filing, judgment = _extract_dates(text)
        assert filing is not None
        assert judgment is not None
    
    def test_no_dates_in_text(self):
        text = "This text contains no dates"
        filing, judgment = _extract_dates(text)
        assert filing is None and judgment is None


class TestJudgeExtraction:
    """Test judge/bench name extraction."""
    
    def test_extract_justice_name(self):
        text = "Hon'ble Justice John Smith delivered this judgment."
        result = _extract_judge(text)
        assert result is not None
        assert "Smith" in result or "John" in result
    
    def test_extract_chief_justice(self):
        text = "Hon'ble Chief Justice Rajesh Kumar presides."
        result = _extract_judge(text)
        assert result is not None
    
    def test_extract_bench(self):
        text = "Bench: Justice A.K. Misra and Justice S.P. Sharma"
        result = _extract_judge(text)
        assert result is not None
    
    def test_no_judge_name(self):
        text = "The court hearing was conducted."
        result = _extract_judge(text)
        assert result is None


class TestCourtExtraction:
    """Test court name extraction."""
    
    def test_supreme_court(self):
        text = "Supreme Court of India judgment"
        result = _extract_court(text)
        assert result is not None
        assert "Supreme Court" in result
    
    def test_high_court(self):
        text = "Delhi High Court judgment"
        result = _extract_court(text)
        assert result is not None
        assert "High Court" in result
    
    def test_district_court(self):
        text = "District Court, Mumbai"
        result = _extract_court(text)
        assert result is not None
        assert "District Court" in result
    
    def test_sessions_court(self):
        text = "Court: Sessions Court, Bangalore"
        result = _extract_court(text)
        assert result is not None
        assert "Sessions Court" in result
    
    def test_no_court_name(self):
        text = "The court heard the case"
        result = _extract_court(text)
        assert result is None


class TestPartyExtraction:
    """Test party name extraction."""
    
    def test_extract_petitioner_and_respondent(self):
        text = """
        Petitioner: John Doe
        Respondent: Jane Smith
        """
        result = _extract_parties(text)
        assert result['petitioner'] is not None
        assert result['respondent'] is not None
    
    def test_extract_plaintiff_defendant(self):
        text = """
        Plaintiff: ABC Corporation
        Defendant: XYZ Ltd.
        """
        result = _extract_parties(text)
        assert len(result['parties']) > 0
    
    def test_extract_appellant_accused(self):
        text = """
        Appellant: State of Maharashtra
        Accused: Ram Kumar Singh
        """
        result = _extract_parties(text)
        assert len(result['parties']) > 0
    
    def test_multiple_parties(self):
        text = """
        Petitioner(s): John Doe and Jane Doe
        Respondent(s): Government of India and Ministry of Law
        """
        result = _extract_parties(text)
        assert len(result['parties']) > 0
    
    def test_no_parties(self):
        text = "This is a general court text"
        result = _extract_parties(text)
        assert result['petitioner'] is None
        assert result['respondent'] is None


class TestConfidenceCalculation:
    """Test confidence score calculation."""
    
    def test_all_fields_filled(self):
        metadata = {
            'case_number': '123',
            'filing_date': '2023-01-01',
            'judgment_date': '2023-06-01',
            'judge_name': 'Smith',
            'court_name': 'High Court',
            'petitioner': 'Doe',
        }
        score = _calculate_confidence(metadata)
        assert score >= 0.9
    
    def test_half_fields_filled(self):
        metadata = {
            'case_number': '123',
            'filing_date': '2023-01-01',
            'judgment_date': None,
            'judge_name': None,
            'court_name': 'High Court',
            'petitioner': None,
        }
        score = _calculate_confidence(metadata)
        assert 0.4 < score < 0.6
    
    def test_no_fields_filled(self):
        metadata = {
            'case_number': None,
            'filing_date': None,
            'judgment_date': None,
            'judge_name': None,
            'court_name': None,
            'petitioner': None,
        }
        score = _calculate_confidence(metadata)
        assert score < 0.1


class TestMetadataExtraction:
    """Test full metadata extraction (with mocked PDF extraction)."""
    
    @patch('document_parser.core.extract_text_from_pdf')
    def test_extract_metadata_from_text(self, mock_extract):
        """Test metadata extraction with mocked PDF reader."""
        sample_text = """
        Case No. 2023/456/789
        Filed on 15-01-2023
        Petitioner: Ram Kumar Singh
        Respondent: Government of India
        Delhi High Court
        Hon'ble Justice Rajesh Sharma
        Judgment delivered on 20-05-2023
        """
        mock_extract.return_value = sample_text
        
        metadata = extract_metadata("dummy.pdf", enable_llm=False, enable_ocr=False)
        
        assert metadata.case_number is not None
        assert metadata.filing_date is not None
        assert metadata.judgment_date is not None
        assert metadata.court_name is not None
        assert metadata.judge_name is not None
        assert metadata.petitioner is not None
        assert metadata.respondent is not None
        assert metadata.confidence_score > 0.5
    
    @patch('document_parser.core.extract_text_from_pdf')
    def test_extract_metadata_partial(self, mock_extract):
        """Test extraction with incomplete metadata."""
        sample_text = "Case No. 2023/123 Delhi High Court"
        mock_extract.return_value = sample_text
        
        metadata = extract_metadata("dummy.pdf")
        
        assert metadata.case_number is not None
        assert metadata.court_name is not None
        assert metadata.confidence_score > 0.0
    
    @patch('document_parser.core.extract_text_from_pdf')
    def test_extract_metadata_empty_text(self, mock_extract):
        """Test extraction with empty text."""
        mock_extract.return_value = ""
        
        metadata = extract_metadata("dummy.pdf")
        
        assert metadata.confidence_score == 0.0
    
    @patch('document_parser.core.extract_text_from_pdf')
    def test_extract_metadata_error_handling(self, mock_extract):
        """Test error handling."""
        mock_extract.side_effect = Exception("PDF read error")
        
        metadata = extract_metadata("dummy.pdf")
        
        assert metadata.confidence_score == 0.0


class TestSampleJudgments:
    """Test with diverse sample judgment texts."""
    
    @patch('document_parser.core.extract_text_from_pdf')
    def test_criminal_judgment(self, mock_extract):
        """Test extraction from criminal case."""
        text = """
        CRIMINAL APPEAL NO. 1234 OF 2023
        Appellant: State of Maharashtra
        Respondent: Raj Kumar
        
        The Supreme Court of India
        Hon'ble Justice A.K. Misra and Justice S.P. Sharma
        
        Date of filing: 10 March 2023
        Date of Judgment: 25 August 2023
        
        The appellant has appealed against the conviction.
        """
        mock_extract.return_value = text
        metadata = extract_metadata("criminal.pdf")
        assert metadata.court_name is not None
        assert "2023" in str(metadata.filing_date or "")
    
    @patch('document_parser.core.extract_text_from_pdf')
    def test_civil_judgment(self, mock_extract):
        """Test extraction from civil case."""
        text = """
        CIVIL APPEAL NO. 5678 OF 2022
        Petitioner: ABC Corporation Limited
        Respondent: XYZ Industries Pvt Ltd
        
        Delhi High Court
        Hon'ble Justice Rajesh Kumar
        
        Case filed: 05/04/2022
        Judgment date: 15-11-2022
        """
        mock_extract.return_value = text
        metadata = extract_metadata("civil.pdf")
        assert metadata.petitioner is not None
        assert metadata.respondent is not None
    
    @patch('document_parser.core.extract_text_from_pdf')
    def test_family_judgment(self, mock_extract):
        """Test extraction from family law case."""
        text = """
        FAMILY COURT CASE NO. 2023/045
        Petitioner: Priya Singh
        Respondent: Rajesh Singh
        
        District Court, Pune
        Hon'ble Judge Mrs. Vandana Sharma
        
        Filed: 20 January 2023
        Decided: 10 June 2023
        """
        mock_extract.return_value = text
        metadata = extract_metadata("family.pdf")
        assert metadata.court_name is not None
        assert "Priya" in str(metadata.petitioner or "")
    
    @patch('document_parser.core.extract_text_from_pdf')
    def test_labor_judgment(self, mock_extract):
        """Test extraction from labor case."""
        text = """
        WRIT PETITION NO. 123 OF 2023
        Petitioner: Workers Union
        Respondent: Factory Management
        
        High Court, Mumbai
        Hon'ble Justice Mehta
        
        Date: 01-02-2023 to 30-07-2023
        """
        mock_extract.return_value = text
        metadata = extract_metadata("labor.pdf")
        assert metadata.case_number is not None
    
    @patch('document_parser.core.extract_text_from_pdf')
    def test_multilingual_judgment(self, mock_extract):
        """Test extraction with multilingual text."""
        text = """
        Case No. 2023/999
        मामला संख्या / కేసు సంఖ్య
        Petitioner/याचिकाकर्ता: John Doe
        Respondent/प्रतिवादी: Jane Doe
        
        Delhi High Court/दिल्ली उच्च न्यायालय
        Judgment: 15-08-2023
        """
        mock_extract.return_value = text
        metadata = extract_metadata("multilingual.pdf")
        assert metadata.case_number is not None
        assert metadata.confidence_score > 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])