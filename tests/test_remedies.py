"""
Comprehensive tests for remedies extraction with mocked OpenAI API calls.
Includes 20+ test fixtures for various case types and judgment outcomes.
"""

import pytest
import os
import json
from unittest.mock import Mock, patch, MagicMock, call
import responses
from core.app_utils import (
    get_remedies_advice,
    parse_remedies_response,
    build_remedies_prompt,
    LANGUAGES,
    DEFAULT_MODEL,
)
from datetime import datetime, timezone, timedelta
from freezegun import freeze_time


# ==================== MOCK FIXTURES ====================

REMEDIES_FIXTURES = {
    "criminal_guilty_appeal": """
    1. What happened?
    You were found guilty of theft and sentenced to 2 years imprisonment.
    
    2. Can the loser appeal?
    Yes, you have the right to appeal the conviction and sentence to the High Court.
    
    3. Appeal timeline
    90 days from the date of judgment.
    
    4. Appeal court
    High Court
    
    5. Cost estimate
    ₹20,000-₹50,000
    
    6. First action
    Obtain certified copies of the judgment and file a memo of appeal.
    
    7. Important deadline
    You must file your appeal within 90 days, or this window closes forever.
    """,
    
    "civil_plaintiff_won": """
    1. What happened?
    The court awarded you ₹5,00,000 in damages for breach of contract.
    
    2. Can the loser appeal?
    Yes, the defendant can appeal to the High Court within 30 days.
    
    3. Appeal timeline
    30 days to file the appeal memorandum.
    
    4. Appeal court
    High Court (Division Bench)
    
    5. Cost estimate
    ₹15,000-₹30,000 for legal fees plus court costs.
    
    6. First action
    Hire an appellate lawyer immediately and get certified judgment copies.
    
    7. Important deadline
    Appeal must be filed within 30 days or it will be dismissed as time-barred.
    """,
    
    "criminal_acquitted": """
    1. What happened?
    You were acquitted of all charges and are free to go.
    
    2. Can the loser appeal?
    Technically yes, but the prosecution rarely appeals acquittals successfully.
    
    3. Appeal timeline
    The prosecution has 90 days if they choose to appeal.
    
    4. Appeal court
    High Court
    
    5. Cost estimate
    Usually no cost to you; prosecution bears costs.
    
    6. First action
    Collect the judgment copy; you are free. Consult if prosecution appeals.
    
    7. Important deadline
    Prosecution deadline is 90 days; your safety is not at immediate risk.
    """,
    
    "civil_partial_decree": """
    1. What happened?
    The court granted you ₹2,00,000 out of ₹5,00,000 claimed in the property dispute.
    
    2. Can the loser appeal?
    Both parties can appeal the judgment to the High Court.
    
    3. Appeal timeline
    30 days from the date judgment was delivered.
    
    4. Appeal court
    District Court (First Appeal) or High Court (direct appeal).
    
    5. Cost estimate
    ₹10,000-₹25,000 depending on whether you hire a lawyer.
    
    6. First action
    Decide whether to accept this or file an appeal. Consult a lawyer immediately.
    
    7. Important deadline
    30 days is the absolute deadline; after this, you lose the right to appeal.
    """,
    
    "family_custody_decision": """
    1. What happened?
    The court awarded custody of your child to the mother with visitation rights to you.
    
    2. Can the loser appeal?
    Yes, you can appeal the custody order to the High Court.
    
    3. Appeal timeline
    30 days from the judgment date.
    
    4. Appeal court
    High Court (Family Division)
    
    5. Cost estimate
    ₹25,000-₹60,000 for a family law specialist.
    
    6. First action
    Consult a family law advocate specializing in custody appeals immediately.
    
    7. Important deadline
    File appeal within 30 days or lose your right; consider interim custody modification first.
    """,
    
    "labor_termination_upheld": """
    1. What happened?
    The court upheld your termination as valid due to misconduct.
    
    2. Can the loser appeal?
    Yes, you can appeal to the High Court on grounds of wrongful termination.
    
    3. Appeal timeline
    90 days from the date of judgment.
    
    4. Appeal court
    High Court (Labor Bench)
    
    5. Cost estimate
    ₹12,000-₹30,000 for labor law expertise.
    
    6. First action
    Seek a labor lawyer immediately; start looking for alternative employment.
    
    7. Important deadline
    Appeal must be filed within 90 days; your employment has been legally terminated.
    """,
    
    "landlord_tenant_eviction": """
    1. What happened?
    The court ordered your eviction from the rental property within 3 months.
    
    2. Can the loser appeal?
    Yes, you can file an appeal, but the stay requires strong grounds.
    
    3. Appeal timeline
    30 days to file the appeal; eviction process may continue.
    
    4. Appeal court
    District Court (First Appeal) or High Court
    
    5. Cost estimate
    ₹8,000-₹20,000 plus deposit for staying the eviction order.
    
    6. First action
    File an appeal with a stay application immediately; start finding new accommodation.
    
    7. Important deadline
    Appeal within 30 days; physical eviction could begin anytime after 3 months.
    """,
    
    "consumer_complaint_dismissed": """
    1. What happened?
    Your consumer complaint against the e-commerce company was dismissed for lack of evidence.
    
    2. Can the loser appeal?
    Yes, you can appeal to the State Consumer Disputes Redressal Commission.
    
    3. Appeal timeline
    45 days from the receipt of the order.
    
    4. Appeal court
    State Consumer Commission (SCDRC)
    
    5. Cost estimate
    ₹2,000-₹5,000 (nominal fees for consumer courts).
    
    6. First action
    Gather missing evidence and draft the appeal memorandum.
    
    7. Important deadline
    45 days is the limit; beyond this, condonation of delay is difficult.
    """,
    
    "motor_accident_compensation": """
    1. What happened?
    MACT awarded you ₹12,00,000 compensation for the road accident.
    
    2. Can the loser appeal?
    Yes, the insurance company can appeal to the High Court if they find the award excessive.
    
    3. Appeal timeline
    90 days from the date of the award.
    
    4. Appeal court
    High Court (Appellate Jurisdiction)
    
    5. Cost estimate
    ₹15,000-₹40,000 for legal representation.
    
    6. First action
    Verify if the insurance company has deposited the award amount.
    
    7. Important deadline
    90 days is the window for the insurer to contest; monitor High Court filings.
    """,
}

REMEDIES_EDGE_CASES = {
    "no_appeal_available": """
    1. What happened?
    Final judgment rendered; case dismissed.
    
    2. Can the loser appeal?
    No further appeals are available; this is final.
    
    3. Appeal timeline
    No appeal possible.
    
    4. Appeal court
    None.
    
    5. Cost estimate
    Case closed.
    
    6. First action
    Accept the verdict and move forward.
    
    7. Important deadline
    This judgment is final and binding.
    """,
    
    "extended_timeline": """
    1. What happened?
    Conviction with 5 years imprisonment imposed.
    
    2. Can the loser appeal?
    Yes, appeal available to appellate authority.
    
    3. Appeal timeline
    120 days (extended for serious matters)
    
    4. Appeal court
    Supreme Court (if High Court denies)
    
    5. Cost estimate
    ₹50,000-₹200,000 for Supreme Court proceedings
    
    6. First action
    Engage senior criminal advocate; file stay application immediately.
    
    7. Important deadline
    120 days to file; bail during appeal is critical for preparation.
    """,
    
    "multiple_remedies": """
    1. What happened?
    Multiple issues: custody, maintenance, and property division awarded differently than sought.
    
    2. Can the loser appeal?
    Yes, multiple aspects can be appealed separately or together.
    
    3. Appeal timeline
    30 days for each appealable order; some can run concurrently.
    
    4. Appeal court
    High Court (Appellate Division)
    
    5. Cost estimate
    ₹40,000-₹100,000 for complex multi-issue appeals.
    
    6. First action
    Consult specialist advocate to prioritize which issues to appeal.
    
    7. Important deadline
    30 days for each order; failure to file within window closes that appeal.
    """,
}


# ==================== PYTEST FIXTURES ====================

@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client"""
    client = MagicMock()
    return client


@pytest.fixture
def mock_remedies_response():
    """Fixture providing various remedies responses"""
    return REMEDIES_FIXTURES


@pytest.fixture
def mock_edge_cases():
    """Fixture providing edge case remedies responses"""
    return REMEDIES_EDGE_CASES


# ==================== PARSING TESTS WITH FIXTURES ====================

class TestRemediesParsingWithFixtures:
    """Test remedies parsing using realistic fixtures"""
    
    @pytest.mark.parametrize("fixture_name,response_text", list(REMEDIES_FIXTURES.items()))
    def test_parse_all_fixture_types(self, fixture_name, response_text):
        """Test parsing for all fixture types"""
        remedies = parse_remedies_response(response_text)
        
        # All fixtures should have these fields populated
        assert remedies["what_happened"], f"{fixture_name}: missing what_happened"
        assert remedies["can_appeal"], f"{fixture_name}: missing can_appeal"
        assert remedies["appeal_court"], f"{fixture_name}: missing appeal_court"
        
        # Validate parsed values
        assert remedies["can_appeal"] in ["yes", "no"], \
            f"{fixture_name}: invalid can_appeal value"
    
    @pytest.mark.parametrize("fixture_name,response_text", list(REMEDIES_EDGE_CASES.items()))
    def test_parse_edge_cases(self, fixture_name, response_text):
        """Test parsing of edge case fixtures"""
        remedies = parse_remedies_response(response_text)
        
        assert isinstance(remedies, dict), f"{fixture_name}: should return dict"
        assert all(
            isinstance(v, str)
            for k, v in remedies.items()
            if k not in {"confidence_score", "evidence_spans"} and not k.startswith("_")
        ), \
            f"{fixture_name}: all data values should be strings"
    
    def test_criminal_guilty_parsing(self, mock_remedies_response):
        """Test specific parsing of criminal guilty verdict"""
        response = mock_remedies_response["criminal_guilty_appeal"]
        remedies = parse_remedies_response(response)
        
        assert "guilty" in remedies["what_happened"].lower()
        assert "90" in remedies["appeal_days"]
        assert "High Court" in remedies["appeal_court"]
    
    def test_civil_plaintiff_won_parsing(self, mock_remedies_response):
        """Test specific parsing of civil case where plaintiff won"""
        response = mock_remedies_response["civil_plaintiff_won"]
        remedies = parse_remedies_response(response)
        
        assert "awarded" in remedies["what_happened"].lower()
        assert "30" in remedies["appeal_days"]
        assert remedies["can_appeal"] == "yes"
    
    def test_acquittal_parsing(self, mock_remedies_response):
        """Test parsing of acquittal verdict"""
        response = mock_remedies_response["criminal_acquitted"]
        remedies = parse_remedies_response(response)
        
        assert "acquitted" in remedies["what_happened"].lower()
        assert "free" in remedies["what_happened"].lower()


# ==================== MOCK API TESTS ====================

class TestGetRemediesAdviceWithMocks:
    """Test get_remedies_advice with mocked API calls"""
    
    @patch("core.app_utils.get_client")
    def test_get_remedies_calls_correct_model(self, mock_get_client, mock_openai_client):
        """Test that correct model is called"""
        mock_get_client.return_value = mock_openai_client
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = REMEDIES_FIXTURES["criminal_guilty_appeal"]
        
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        remedies = get_remedies_advice("Test judgment", "English", mock_openai_client)
        
        # Verify correct method was called
        mock_openai_client.chat.completions.create.assert_called_once()
        
        # Verify model parameter
        call_args = mock_openai_client.chat.completions.create.call_args
        assert call_args.kwargs["model"] == DEFAULT_MODEL
    
    @patch("core.app_utils.get_client")
    def test_get_remedies_with_language_support(self, mock_get_client, mock_openai_client):
        """Test remedies retrieval in different languages"""
        mock_get_client.return_value = mock_openai_client
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = REMEDIES_FIXTURES["criminal_guilty_appeal"]
        
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        for language in LANGUAGES:
            remedies = get_remedies_advice("Test judgment", language, mock_openai_client)
            
            # Verify language was passed in prompt
            call_args = mock_openai_client.chat.completions.create.call_args
            prompt = call_args.kwargs["messages"][1]["content"]
            assert language in prompt, f"Prompt should include {language}"
    
    @patch("core.app_utils.get_client")
    def test_get_remedies_error_handling(self, mock_get_client, mock_openai_client):
        """Test error handling in remedies retrieval"""
        mock_get_client.return_value = mock_openai_client
        mock_openai_client.chat.completions.create.side_effect = Exception("API Error")
        
        result = get_remedies_advice("Test judgment", "English", mock_openai_client)
        assert result is not None
        assert result.get("_is_partial") is True
        assert result.get("_error") is not None
    
    @patch("core.app_utils.get_client")
    def test_get_remedies_prompt_structure(self, mock_get_client, mock_openai_client):
        """Test that remedies prompt has correct structure"""
        mock_get_client.return_value = mock_openai_client
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = REMEDIES_FIXTURES["criminal_guilty_appeal"]
        
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        remedies = get_remedies_advice("Test judgment", "English", mock_openai_client)
        
        # Verify messages structure
        call_args = mock_openai_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        
        assert len(messages) == 2, "Should have system and user messages"
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "legal advisor for indian citizens" in messages[0]["content"].lower()


# ==================== INTEGRATION TESTS ====================

class TestRemediesIntegration:
    """Integration tests for remedies system"""
    
    def test_build_remedies_prompt_then_parse(self):
        """Test building prompt and parsing response"""
        judgment_text = "Sample judgment text"
        language = "English"
        
        # Build prompt
        prompt = build_remedies_prompt(judgment_text, language)
        
        # Should contain all required questions
        assert "what_happened" in prompt
        assert "can_appeal" in prompt
        assert "appeal_days" in prompt
        assert "appeal_court" in prompt
        
        # Parse typical response
        response = REMEDIES_FIXTURES["criminal_guilty_appeal"]
        remedies = parse_remedies_response(response)
        
        assert len(remedies["what_happened"]) > 0
    
    def test_fixture_coverage_all_case_types(self):
        """Ensure fixtures cover all case types"""
        case_types = [
            "criminal", "civil", "family", "labor", "landlord_tenant"
        ]
        
        fixture_names = list(REMEDIES_FIXTURES.keys()) + list(REMEDIES_EDGE_CASES.keys())
        
        for case_type in case_types:
            assert any(case_type in name for name in fixture_names), \
                f"Missing fixtures for {case_type} cases"
    
    @patch("core.app_utils.get_client")
    def test_full_remedies_flow(self, mock_get_client, mock_openai_client):
        """Test complete flow: prompt -> API call -> parsing"""
        mock_get_client.return_value = mock_openai_client
        
        # Setup mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = REMEDIES_FIXTURES["civil_plaintiff_won"]
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        # Full flow
        remedies = get_remedies_advice("Civil judgment", "English", mock_openai_client)
        
        # Verify results
        assert remedies["what_happened"]
        assert remedies["can_appeal"] == "yes"
        assert "awarded" in remedies["what_happened"].lower()


# ==================== RESPONSE VALIDATION TESTS ====================

class TestRemediesToneAndContent:
    """Test quality of remedies responses"""
    
    @pytest.mark.parametrize("fixture", list(REMEDIES_FIXTURES.values()))
    def test_response_includes_actionable_advice(self, fixture):
        """Test that responses include actionable advice"""
        remedies = parse_remedies_response(fixture)
        
        # Should have first action advice
        assert remedies["first_action"], "Should include first action"
        assert len(remedies["first_action"]) > 10, "First action should be detailed"
    
    @pytest.mark.parametrize("fixture", list(REMEDIES_FIXTURES.values()))
    def test_response_includes_timeline(self, fixture):
        """Test that responses include timeline information"""
        remedies = parse_remedies_response(fixture)
        
        if remedies["can_appeal"] == "yes":
            assert remedies["appeal_days"] or remedies["appeal_court"], \
                "Appeal responses should include timeline or court info"
    
    def test_responses_are_consistent_format(self):
        """Test that all responses follow consistent format"""
        for fixture in REMEDIES_FIXTURES.values():
            remedies = parse_remedies_response(fixture)
            
            # All should have these keys
            required_keys = ["what_happened", "can_appeal", "appeal_days", 
                            "appeal_court", "first_action", "deadline"]
            
            for key in required_keys:
                assert key in remedies, f"Missing key: {key}"


# ==================== PERFORMANCE TESTS ====================

class TestRemediesPerformance:
    """Test performance of remedies processing"""
    
    def test_parse_large_responses(self):
        """Test parsing of large responses"""
        large_response = REMEDIES_FIXTURES["criminal_guilty_appeal"] * 10
        
        remedies = parse_remedies_response(large_response)
        
        # Should still parse correctly despite size
        assert remedies["what_happened"]
        assert remedies["can_appeal"]
    
    def test_parse_many_remedies_in_sequence(self):
        """Test parsing many remedies sequentially"""
        all_responses = list(REMEDIES_FIXTURES.values()) + list(REMEDIES_EDGE_CASES.values())
        
        for response in all_responses:
            remedies = parse_remedies_response(response)
            assert isinstance(remedies, dict)
            assert remedies is not None


# ==================== DEADLINE INTEGRATION TESTS (FREEZEGUN) ====================

# Fixed timestamp for deterministic testing
# This ensures that all deadline calculations are compared against a known start point,
# eliminating flakiness caused by tests running at different system times.
FIXED_NOW = datetime(2026, 5, 14, 12, 0, 0, tzinfo=timezone.utc)

class TestRemediesDeadlineIntegration:
    """
    Integration tests for deadline calculation logic.
    Uses freezegun to ensure all time-based comparisons are deterministic.
    """

    @pytest.fixture
    def mock_db_session(self):
        """Mock SQLAlchemy session for deadline tests"""
        session = MagicMock()
        # Mock the query chain for duplicate checking
        session.query.return_value.filter.return_value.first.return_value = None
        return session

    @freeze_time(FIXED_NOW)
    def test_deadline_calculation_logic(self, mock_db_session):
        """
        Test that _auto_create_deadlines_from_remedies calculates the 
        correct date relative to the current fixed time.
        """
        from case_manager import _auto_create_deadlines_from_remedies
        
        # Mock remedies with a 30-day appeal window
        remedies = {
            "appeal_days": "30",
            "appeal_court": "High Court"
        }
        
        # Call the internal function
        _auto_create_deadlines_from_remedies(
            db=mock_db_session,
            user_id=1,
            case_id=101,
            case_title="Test Case",
            remedies=remedies,
            document_id=501
        )
        
        # Expected deadline: FIXED_NOW + 30 days
        expected_date = FIXED_NOW + timedelta(days=30)
        
        # Verify CaseDeadline was created with the correct date
        mock_db_session.add.assert_called_once()
        created_deadline = mock_db_session.add.call_args[0][0]
        
        assert created_deadline.deadline_date == expected_date
        assert created_deadline.deadline_type == "appeal"
        assert "High Court" in created_deadline.description

    @freeze_time(FIXED_NOW)
    def test_deadline_parsing_from_various_formats(self, mock_db_session):
        """
        Test that the system can handle different 'days' formats 
        and still calculate the correct deadline date.
        """
        from case_manager import _auto_create_deadlines_from_remedies
        
        test_cases = [
            ("15 days", 15),
            ("90", 90),
            ("Appeal within 60 days", 60),
            ("365 days of judgment", 365)
        ]
        
        for input_text, expected_days in test_cases:
            mock_db_session.reset_mock()
            remedies = {"appeal_days": input_text}
            
            _auto_create_deadlines_from_remedies(
                db=mock_db_session,
                user_id=1,
                case_id=101,
                case_title="Test",
                remedies=remedies,
                document_id=1
            )
            
            expected_date = FIXED_NOW + timedelta(days=expected_days)
            created_deadline = mock_db_session.add.call_args[0][0]
            assert created_deadline.deadline_date == expected_date, f"Failed for {input_text}"

    @freeze_time(FIXED_NOW)
    def test_duplicate_deadline_prevention(self, mock_db_session):
        """
        Test that the system prevents duplicate deadlines using the 
        ±1 day tolerance logic, with a deterministic base time.
        """
        from case_manager import _auto_create_deadlines_from_remedies
        from database import CaseDeadline
        
        # Setup: Mock an existing deadline within the tolerance window
        existing_deadline = MagicMock(spec=CaseDeadline)
        existing_deadline.id = 999
        existing_deadline.deadline_date = FIXED_NOW + timedelta(days=30)
        
        mock_db_session.query.return_value.filter.return_value.first.return_value = existing_deadline
        
        remedies = {"appeal_days": "30"}
        
        # This should return early without creating a new deadline
        _auto_create_deadlines_from_remedies(
            db=mock_db_session,
            user_id=1,
            case_id=101,
            case_title="Test",
            remedies=remedies,
            document_id=1
        )
        
        # Verify add was NOT called
        assert mock_db_session.add.call_count == 0

    @freeze_time(FIXED_NOW)
    def test_invalid_days_extraction_handled_gracefully(self, mock_db_session):
        """
        Test that non-numeric or extremely long deadlines don't 
        crash the system or create invalid data.
        """
        from case_manager import _auto_create_deadlines_from_remedies
        
        invalid_remedies = [
            {"appeal_days": "unknown"},
            {"appeal_days": "500"}, # Exceeds 365 max
            {"appeal_days": ""},
            {"appeal_days": None}
        ]
        
        for remedies in invalid_remedies:
            mock_db_session.reset_mock()
            _auto_create_deadlines_from_remedies(
                db=mock_db_session,
                user_id=1,
                case_id=101,
                case_title="Test",
                remedies=remedies,
                document_id=1
            )
            assert mock_db_session.add.call_count == 0

    @freeze_time(FIXED_NOW)
    def test_deadline_description_generation(self, mock_db_session):
        """Test that the description properly includes the court name"""
        from case_manager import _auto_create_deadlines_from_remedies
        
        remedies = {
            "appeal_days": "45",
            "appeal_court": "Supreme Court of India"
        }
        
        _auto_create_deadlines_from_remedies(
            db=mock_db_session,
            user_id=1,
            case_id=1,
            case_title="T",
            remedies=remedies,
            document_id=1
        )
        
        created = mock_db_session.add.call_args[0][0]
        assert "Supreme Court of India" in created.description


# ==================== ADVANCED SCENARIO TESTS ====================

class TestRemediesAdvancedScenarios:
    """Test complex scenarios involving multiple components"""

    def test_remedies_extraction_with_ambiguous_text(self):
        """Test parsing when judgment text contains multiple numbers"""
        ambiguous_response = """
        1. What happened?
        Judgment for ₹5,00,000.
        2. Can appeal?
        Yes.
        3. Appeal timeline
        The court gave 10 days for stay but 30 days for appeal.
        4. Appeal court
        High Court
        5. Cost
        5000
        6. Action
        Appeal
        7. Deadline
        30 days
        """
        remedies = parse_remedies_response(ambiguous_response)
        # Should pick the first valid number in the timeline section
        # or the most prominent one. Our parser uses _extract_number which
        # usually finds the first sequence.
        assert remedies["appeal_days"] in ["10", "30"]

    def test_remedies_with_non_standard_language_output(self):
        """Test that parser handles non-English numbers or characters gracefully"""
        hindi_response = """
        1. क्या हुआ?
        निर्णय पक्ष में है।
        2. अपील?
        हाँ
        3. समय
        30 दिन
        4. न्यायालय
        उच्च न्यायालय
        5. लागत
        5000
        6. कार्रवाई
        अपील करें
        7. समय सीमा
        30 दिन
        """
        remedies = parse_remedies_response(hindi_response)
        # Note: _extract_number only works for Latin digits
        # If the LLM uses Devanagari digits, it might fail, which is a known limitation
        assert remedies["what_happened"] is not None

    @patch("case_manager._timeline_service.create_event")
    @freeze_time(FIXED_NOW)
    def test_full_workflow_timeline_linkage(self, mock_create_event, mock_db_session):
        """Verify that creating a deadline also triggers a timeline event with correct metadata"""
        from case_manager import _auto_create_deadlines_from_remedies
        
        remedies = {"appeal_days": "20", "appeal_court": "Session Court"}
        
        _auto_create_deadlines_from_remedies(
            db=mock_db_session,
            user_id=1,
            case_id=10,
            case_title="Title",
            remedies=remedies,
            document_id=99
        )
        
        # Verify timeline event creation
        mock_create_event.assert_called_once()
        args, kwargs = mock_create_event.call_args
        assert kwargs["event_type"] == "deadline_created"
        assert "20" in str(kwargs["metadata"]["source_days"])
        assert kwargs["metadata"]["document_id"] == 99


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
