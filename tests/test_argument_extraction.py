"""
Tests for legal argument extraction (issue #435)
Tests schema validation, LLM extraction, and fallback mechanisms.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock

from core.argument_extraction_engine import ArgumentExtractionEngine
from core.argument_extraction_schema import (
    ExtractedLegalArgument,
    ArgumentType,
    ArgumentExtractionResult,
)


class TestArgumentExtractionSchema:
    """Test Pydantic schema validation"""
    
    def test_valid_argument_creation(self):
        """Test creating a valid argument"""
        arg = ExtractedLegalArgument(
            argument_text="The defendant violated the statute which is clearly established",
            argument_type=ArgumentType.LEGAL_PRINCIPLE,
            reasoning="Statutory violations create liability under law",
            supporting_evidence="Judgment states clear violation at para 5",
            citation_references=["IPC§326"],
            confidence_score=0.85,
        )
        assert arg.argument_text == "The defendant violated the statute which is clearly established"
        assert arg.confidence_score == 0.85
        assert arg.argument_type == "legal_principle"
    
    def test_argument_fails_minimum_text_length(self):
        """Test validation rejects text with fewer than 5 words"""
        with pytest.raises(ValueError, match="meaningful content"):
            ExtractedLegalArgument(
                argument_text="Too short here",
                argument_type=ArgumentType.LEGAL_PRINCIPLE,
                reasoning="Valid reasoning with sufficient length",
                confidence_score=0.8,
            )
    
    def test_argument_fails_confidence_out_of_bounds(self):
        """Test validation rejects confidence score > 1.0"""
        with pytest.raises(ValueError):
            ExtractedLegalArgument(
                argument_text="A valid legal argument with sufficient length content",
                argument_type=ArgumentType.LEGAL_PRINCIPLE,
                reasoning="Valid reasoning here",
                confidence_score=1.5,
            )
    
    def test_argument_fails_confidence_negative(self):
        """Test validation rejects negative confidence"""
        with pytest.raises(ValueError):
            ExtractedLegalArgument(
                argument_text="A valid legal argument with sufficient length content",
                argument_type=ArgumentType.LEGAL_PRINCIPLE,
                reasoning="Valid reasoning here",
                confidence_score=-0.1,
            )
    
    def test_extraction_result_creation(self):
        """Test ArgumentExtractionResult creation"""
        arg = ExtractedLegalArgument(
            argument_text="Valid legal argument with enough content",
            argument_type=ArgumentType.LEGAL_PRINCIPLE,
            reasoning="Legal validity here",
            confidence_score=0.9,
        )
        
        result = ArgumentExtractionResult(
            case_id=1,
            issue_name="criminal",
            arguments=[arg],
            extraction_method="llm",
            total_extracted=1,
            extraction_quality="high",
        )
        
        assert result.case_id == 1
        assert result.total_extracted == 1
        assert result.extraction_quality == "high"


class TestArgumentExtractionEngine:
    """Test LLM-powered extraction engine"""
    
    @pytest.fixture
    def engine_with_mock_client(self):
        """Create engine with mocked OpenAI client"""
        with patch('core.argument_extraction_engine.openai.OpenAI'):
            engine = ArgumentExtractionEngine(use_llm=True)
            engine.client = MagicMock()  # Mock the client
            return engine
    
    def test_llm_extraction_success(self, engine_with_mock_client):
        """Test successful LLM extraction with valid response"""
        mock_response_content = json.dumps([
            {
                "argument_text": "The defendant clearly violated the statutory provision which is well established",
                "argument_type": "legal_principle",
                "reasoning": "Statutory violation creates legal liability",
                "supporting_evidence": "Judgment at page 5 states clear violation",
                "citation_references": ["IPC§326"],
                "confidence_score": 0.9,
                "issue_tags": ["criminal"]
            }
        ])
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=mock_response_content))]
        engine_with_mock_client.client.chat.completions.create.return_value = mock_response
        
        result = engine_with_mock_client.extract_arguments(
            text="Long case text " * 100,
            issue_name="criminal_offense",
            case_id=1
        )
        
        assert result.extraction_method == "llm"
        assert result.total_extracted == 1
        assert result.extraction_quality == "high"
        assert result.arguments[0].confidence_score == 0.9
    
    def test_llm_extraction_with_markdown_code_blocks(self, engine_with_mock_client):
        """Test extracting JSON from markdown code blocks"""
        mock_response_content = """Here's the extracted arguments:
```json
[{
    "argument_text": "The defendant violated the statute which is well established",
    "argument_type": "legal_principle",
    "reasoning": "Clear statutory violation",
    "confidence_score": 0.85,
    "issue_tags": ["criminal"]
}]
```
End of extraction"""
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=mock_response_content))]
        engine_with_mock_client.client.chat.completions.create.return_value = mock_response
        
        result = engine_with_mock_client.extract_arguments(
            text="Long case text " * 100,
            issue_name="issue",
            case_id=1
        )
        
        assert result.extraction_method == "llm"
        assert len(result.arguments) == 1
    
    def test_fallback_extraction_when_no_client(self):
        """Test fallback when LLM client unavailable"""
        with patch('core.argument_extraction_engine.openai.OpenAI', side_effect=Exception("API unavailable")):
            engine = ArgumentExtractionEngine(use_llm=True)
            
            result = engine.extract_arguments(
                text="The defendant violated the statute clearly " * 50,
                issue_name="criminal",
                case_id=1
            )
            
            assert result.extraction_method == "fallback"
            assert len(result.arguments) > 0
            assert result.extraction_quality == "low"
            assert "LLM unavailable" in str(result.errors)
    
    def test_keyword_extraction_fallback(self):
        """Test keyword-based fallback extraction"""
        engine = ArgumentExtractionEngine(use_llm=False)
        
        text = """The statute was clearly violated. 
        The principle of law states liability. 
        Procedurally, the court followed due process. 
        The witness testified to the facts."""
        
        result = engine.extract_arguments(
            text=text,
            issue_name="statutory_violation",
            case_id=1
        )
        
        assert result.extraction_method == "fallback"
        assert len(result.arguments) > 0
        assert all(arg.confidence_score == 0.4 for arg in result.arguments)
        # Check that different argument types are classified
        arg_types = [arg.argument_type for arg in result.arguments]
        assert len(set(arg_types)) > 1  # Multiple argument types
    
    def test_argument_type_classification(self):
        """Test automatic argument type classification"""
        engine = ArgumentExtractionEngine(use_llm=False)
        
        test_cases = [
            ("The statute section 326 clearly applies", ArgumentType.STATUTORY_INTERPRETATION),
            ("As per court precedent in case law", ArgumentType.PRECEDENT_CITATION),
            ("The procedural rule requires jurisdiction", ArgumentType.PROCEDURAL_ARGUMENT),
            ("The witness testified to the facts", ArgumentType.WITNESS_TESTIMONY),
            ("This is a legal principle doctrine established", ArgumentType.LEGAL_PRINCIPLE),
        ]
        
        for sentence, expected_type in test_cases:
            classified = engine._classify_argument_type(sentence)
            assert classified == expected_type
    
    def test_empty_document_handling(self):
        """Test handling of empty or very short documents"""
        engine = ArgumentExtractionEngine(use_llm=False)
        
        result = engine.extract_arguments(
            text="",
            issue_name="issue",
            case_id=1
        )
        
        assert result.total_extracted == 0
        assert result.extraction_quality == "low"
        assert result.extraction_method == "failed"
    
    def test_extraction_respects_max_arguments_limit(self, engine_with_mock_client):
        """Test that extraction respects max_arguments_per_case limit"""
        # Create 20 arguments in response (more than max of 10)
        arguments_data = [
            {
                "argument_text": f"The defendant violated statute section {i} which is established",
                "argument_type": "legal_principle",
                "reasoning": "Statutory violation",
                "confidence_score": 0.8,
                "issue_tags": ["criminal"]
            }
            for i in range(20)
        ]
        
        mock_response_content = json.dumps(arguments_data)
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=mock_response_content))]
        engine_with_mock_client.client.chat.completions.create.return_value = mock_response
        
        result = engine_with_mock_client.extract_arguments(
            text="Case text " * 100,
            issue_name="issue",
            case_id=1
        )
        
        assert result.total_extracted <= 10
    
    def test_invalid_argument_data_skipped(self, engine_with_mock_client):
        """Test that invalid argument data is skipped with logging"""
        mock_response_content = json.dumps([
            {
                "argument_text": "Valid argument with sufficient length content",
                "argument_type": "legal_principle",
                "reasoning": "Valid reasoning",
                "confidence_score": 0.8,
                "issue_tags": []
            },
            {
                "argument_text": "short",  # Too short - will fail validation
                "argument_type": "legal_principle",
                "reasoning": "Invalid",
                "confidence_score": 0.8,
            }
        ])
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=mock_response_content))]
        engine_with_mock_client.client.chat.completions.create.return_value = mock_response
        
        result = engine_with_mock_client.extract_arguments(
            text="Case text " * 100,
            issue_name="issue",
            case_id=1
        )
        
        # Only 1 valid argument should be extracted, invalid one skipped
        assert result.total_extracted == 1
        assert len(result.errors) > 0


class TestIntegrationWithKnowledgeGraph:
    """Test integration with knowledge graph builder"""
    
    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session"""
        return Mock()
    
    def test_knowledge_graph_uses_extraction_engine(self, mock_db_session):
        """Test that KnowledgeGraphBuilder uses ArgumentExtractionEngine"""
        from core.knowledge_graph import KnowledgeGraphBuilder
        
        # Setup mock data
        case = Mock(id=1)
        issue = Mock(id=1, issue_name="criminal")
        doc = Mock(document_content="Case text " * 50)
        
        mock_db_session.query.return_value.filter.return_value.first.side_effect = [
            case,  # Case query
            doc,   # Document query
        ]
        mock_db_session.query.return_value.filter.return_value.all.return_value = [issue]
        
        with patch('core.argument_extraction_engine.ArgumentExtractionEngine') as mock_engine_cls:
            mock_engine = Mock()
            mock_engine_cls.return_value = mock_engine
            
            # Mock extraction result
            mock_arg = Mock(
                argument_text="Test argument with sufficient length",
                argument_type="legal_principle",
                supporting_evidence="Evidence",
                citation_references=["IPC§326"]
            )
            mock_result = Mock(
                arguments=[mock_arg],
                extraction_method="llm",
                total_extracted=1,
                extraction_quality="high",
                errors=None
            )
            mock_engine.extract_arguments.return_value = mock_result
            
            result = KnowledgeGraphBuilder.extract_arguments_from_case(
                mock_db_session, case_id=1
            )
            
            # Verify engine was instantiated and called
            mock_engine_cls.assert_called_once_with(use_llm=True)
            mock_engine.extract_arguments.assert_called_once()
            
            # Verify result is list of CaseArgument objects
            assert isinstance(result, list)
