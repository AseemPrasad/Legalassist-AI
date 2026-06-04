"""
LLM-powered legal argument extraction with fallback mechanisms.
Uses OpenRouter API for semantic extraction, falls back to keyword matching.
"""

import json
import logging
from typing import List, Optional, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential

from config import Config
from core.argument_extraction_schema import (
    ExtractedLegalArgument,
    ArgumentExtractionResult,
    ArgumentType,
)

logger = logging.getLogger(__name__)


class ArgumentExtractionEngine:
    """LLM-powered legal argument extraction with graceful fallback"""
    
    def __init__(self, use_llm: bool = True):
        """
        Initialize extraction engine
        
        Args:
            use_llm: Whether to attempt LLM extraction
        """
        self.use_llm = use_llm
        self.client = self._initialize_client() if use_llm else None
        self.max_arguments_per_case = 10
        
    def _initialize_client(self):
        """Initialize OpenRouter client"""
        try:
            import openai
            return openai.OpenAI(
                api_key=Config.OPENROUTER_API_KEY,
                base_url=Config.OPENROUTER_BASE_URL,
            )
        except Exception as e:
            logger.warning(f"Failed to initialize OpenRouter client: {e}")
            return None
    
    @retry(
        stop=stop_after_attempt(Config.AI_MAX_RETRIES),
        wait=wait_exponential(multiplier=Config.AI_RETRY_BACKOFF_BASE),
    )
    def _call_llm_for_extraction(
        self,
        text: str,
        issue_name: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Call LLM to extract structured arguments"""
        if not self.client:
            return None
        
        system_prompt = """You are a legal expert extracting arguments from cases.
Return ONLY a valid JSON array of extracted arguments."""
        
        user_prompt = f"""Extract legal arguments from this case text about: {issue_name}

Text (first 2500 chars):
{text[:2500]}

Return JSON array with structure:
[
  {{
    "argument_text": "The actual argument",
    "argument_type": "legal_principle",
    "reasoning": "Why this is legally valid",
    "supporting_evidence": "Quote if available",
    "citation_references": ["IPC§326"],
    "confidence_score": 0.85,
    "issue_tags": ["{issue_name}"]
  }}
]

Extract 3-6 most significant arguments only. confidence_score based on evidence quality."""
        
        try:
            response = self.client.chat.completions.create(
                model=Config.DEFAULT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1500,
                timeout=Config.AI_REQUEST_TIMEOUT,
            )
            
            response_text = response.choices[0].message.content
            
            # Try direct JSON parsing
            try:
                arguments = json.loads(response_text)
                return arguments if isinstance(arguments, list) else None
            except json.JSONDecodeError:
                # Try extracting from markdown code blocks
                if "```json" in response_text:
                    json_str = response_text.split("```json")[1].split("```")[0]
                    arguments = json.loads(json_str)
                    return arguments if isinstance(arguments, list) else None
                raise
                
        except Exception as e:
            logger.error(f"LLM extraction failed: {str(e)}")
            return None
    
    def extract_arguments(
        self,
        text: str,
        issue_name: str,
        case_id: int,
    ) -> ArgumentExtractionResult:
        """
        Extract legal arguments with LLM + fallback
        
        Args:
            text: Case document text
            issue_name: Issue to extract for
            case_id: Case ID
            
        Returns:
            ArgumentExtractionResult with extracted arguments
        """
        errors = []
        
        # Try LLM extraction
        if self.use_llm and self.client:
            try:
                raw_arguments = self._call_llm_for_extraction(text, issue_name)
                
                if raw_arguments:
                    validated_args = []
                    for arg_data in raw_arguments[:self.max_arguments_per_case]:
                        try:
                            arg = ExtractedLegalArgument(**arg_data)
                            validated_args.append(arg)
                        except Exception as e:
                            logger.warning(f"Validation failed: {e}")
                            errors.append(f"Validation: {str(e)}")
                    
                    if validated_args:
                        quality = "high" if len(validated_args) >= 3 else "medium"
                        return ArgumentExtractionResult(
                            case_id=case_id,
                            issue_name=issue_name,
                            arguments=validated_args,
                            extraction_method="llm",
                            total_extracted=len(validated_args),
                            extraction_quality=quality,
                            errors=errors if errors else None,
                        )
                        
            except Exception as e:
                logger.warning(f"LLM extraction unavailable, using fallback: {e}")
                errors.append(f"LLM unavailable: {str(e)}")
        
        # Fallback: Keyword-based extraction
        fallback_args = self._keyword_extraction_fallback(text, issue_name)
        
        if fallback_args:
            return ArgumentExtractionResult(
                case_id=case_id,
                issue_name=issue_name,
                arguments=fallback_args,
                extraction_method="fallback",
                total_extracted=len(fallback_args),
                extraction_quality="low",
                errors=errors,
            )
        
        # Complete failure
        return ArgumentExtractionResult(
            case_id=case_id,
            issue_name=issue_name,
            arguments=[],
            extraction_method="failed",
            total_extracted=0,
            extraction_quality="low",
            errors=["No extraction method available"],
        )
    
    def _keyword_extraction_fallback(
        self,
        text: str,
        issue_name: str
    ) -> List[ExtractedLegalArgument]:
        """Fallback keyword-based extraction"""
        arguments = []
        
        # Split into sentences (improved from naive '.' split)
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 30]
        
        for sentence in sentences[:8]:
            # Classify by keywords
            arg_type = self._classify_argument_type(sentence)
            
            try:
                arg = ExtractedLegalArgument(
                    argument_text=sentence[:2000],
                    argument_type=arg_type,
                    reasoning="Extracted via keyword analysis",
                    supporting_evidence=None,
                    citation_references=None,
                    confidence_score=0.4,
                    issue_tags=[issue_name.lower()],
                )
                arguments.append(arg)
            except Exception as e:
                logger.debug(f"Fallback argument validation failed: {e}")
        
        return arguments
    
    @staticmethod
    def _classify_argument_type(sentence: str) -> ArgumentType:
        """Classify argument type by keywords"""
        sentence_lower = sentence.lower()
        
        if any(w in sentence_lower for w in ["statute", "section", "act", "law"]):
            return ArgumentType.STATUTORY_INTERPRETATION
        elif any(w in sentence_lower for w in ["court", "procedure", "rule", "jurisdiction"]):
            return ArgumentType.PROCEDURAL_ARGUMENT
        elif any(w in sentence_lower for w in ["testimony", "witness", "deposed"]):
            return ArgumentType.WITNESS_TESTIMONY
        elif any(w in sentence_lower for w in ["precedent", "case", "decided", "held"]):
            return ArgumentType.PRECEDENT_CITATION
        elif any(w in sentence_lower for w in ["principle", "doctrine", "established"]):
            return ArgumentType.LEGAL_PRINCIPLE
        else:
            return ArgumentType.FACTUAL_ASSERTION
