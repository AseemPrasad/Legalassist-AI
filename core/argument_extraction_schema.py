"""
Pydantic schemas for structured legal argument extraction.
Ensures validated, consistent argument representation.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional
from enum import Enum


class ArgumentType(str, Enum):
    """Legal argument classification types"""
    LEGAL_PRINCIPLE = "legal_principle"
    PRECEDENT_CITATION = "precedent_citation"
    PROCEDURAL_ARGUMENT = "procedural_argument"
    WITNESS_TESTIMONY = "witness_testimony"
    STATUTORY_INTERPRETATION = "statutory_interpretation"
    FACTUAL_ASSERTION = "factual_assertion"


class ExtractedLegalArgument(BaseModel):
    """Structured representation of a legal argument"""
    
    argument_text: str = Field(..., min_length=20, max_length=2000)
    argument_type: ArgumentType = Field(...)
    reasoning: str = Field(..., min_length=10, max_length=500)
    supporting_evidence: Optional[str] = Field(None, max_length=500)
    citation_references: Optional[List[str]] = Field(None, max_items=10)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    issue_tags: List[str] = Field(default_factory=list, max_items=5)
    
    @validator('argument_text')
    def validate_text_quality(cls, v):
        """Ensure argument has substantive content"""
        if v.count(' ') < 5:
            raise ValueError("Argument must contain meaningful content")
        return v
    
    class Config:
        use_enum_values = True


class ArgumentExtractionResult(BaseModel):
    """Result of batch argument extraction"""
    case_id: int
    issue_name: str
    arguments: List[ExtractedLegalArgument]
    extraction_method: str  # 'llm', 'fallback', 'failed'
    total_extracted: int
    extraction_quality: str  # 'high', 'medium', 'low'
    errors: Optional[List[str]] = None
