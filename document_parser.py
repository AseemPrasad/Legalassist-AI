name=document_parser.py
"""
Metadata extraction module for legal judgment documents.

Extracts basic case information including:
- Case number/identifier
- Filing date
- Judgment date
- Presiding judge/bench
- Court/jurisdiction name
- Party names (petitioner/respondent/plaintiff/accused)
"""

import re
import json
import logging
from typing import Dict, Optional, List, Any, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict, field

import core

LOGGER = logging.getLogger(__name__)

# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

@dataclass
class CaseMetadata:
    """Structured representation of extracted case metadata."""
    case_number: Optional[str] = None
    filing_date: Optional[str] = None
    judgment_date: Optional[str] = None
    judge_name: Optional[str] = None
    court_name: Optional[str] = None
    petitioner: Optional[str] = None
    respondent: Optional[str] = None
    parties: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    extraction_method: str = "pattern"  # "pattern" or "llm"
    raw_text_sample: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding internal fields."""
        data = asdict(self)
        return data
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


# ==============================================================================
# REGEX PATTERNS FOR METADATA EXTRACTION
# ==============================================================================

# Pattern for case numbers (various Indian court formats)
CASE_NUMBER_PATTERNS = [
    r'Case No\.?\s*[:/]?\s*(\d{1,4}[-/]\d{1,4}[-/]\d{2,4})',
    r'Writ Petition No\.?\s*[:/]?\s*(\d{1,4}\s*(?:of|OF)\s*\d{4})',
    r'Criminal Appeal No\.?\s*[:/]?\s*(\d{1,4}\s*(?:of|OF)\s*\d{4})',
    r'Civil Appeal No\.?\s*[:/]?\s*(\d{1,4}\s*(?:of|OF)\s*\d{4})',
    r'Application No\.?\s*[:/]?\s*(\d{1,4}\s*(?:of|OF)\s*\d{4})',
    r'(?:Case\s+No|No\.?)\s*[:\-]?\s*(\d+/\d+/\d+)',
]

# Pattern for dates (multiple formats)
DATE_PATTERNS = [
    r'(\d{1,2}[-./]\d{1,2}[-./]\d{2,4})',  # DD-MM-YYYY or DD/MM/YYYY
    r'(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4})',
    r'(\d{4}[-./]\d{1,2}[-./]\d{1,2})',  # YYYY-MM-DD
]

# Pattern for judge names
JUDGE_PATTERNS = [
    r'(?:Hon\'?ble\s+)?(?:Mr\.|Ms\.|Mrs\.)?(?:Justice\s+)([A-Z][a-zA-Z\s\.]+)',
    r'Hon\'?ble\s+(?:Chief\s+)?Justice\s+([A-Z][a-zA-Z\s\.]+)',
]

# Pattern for court names
COURT_PATTERNS = [
    r'(Supreme Court of India)',
    r'([A-Z][a-zA-Z\s]+\s+High Court)',
    r'(District Court[^,]*)',
    r'(Sessions Court[^,]*)',
    r'(Family Court[^,]*)',
    r'(Consumer Court[^,]*)',
]

# Pattern for party names
PARTY_PATTERNS = [
    r'Petitioner(?:\(s\))?\s*[:/]?\s*([^\n,]+)',
    r'Respondent(?:\(s\))?\s*[:/]?\s*([^\n,]+)',
    r'Plaintiff(?:\(s\))?\s*[:/]?\s*([^\n,]+)',
    r'Defendant(?:\(s\))?\s*[:/]?\s*([^\n,]+)',
    r'Appellant(?:\(s\))?\s*[:/]?\s*([^\n,]+)',
    r'Accused(?:\(s\))?\s*[:/]?\s*([^\n,]+)',
]


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def _normalize_date(date_str: str) -> Optional[str]:
    """
    Normalize various date formats to YYYY-MM-DD.
    
    Args:
        date_str: The date string in various formats
        
    Returns:
        Normalized date in YYYY-MM-DD format, or None if parsing fails
    """
    if not date_str:
        return None
    
    date_str = date_str.strip()
    
    # Common date formats to try
    formats = [
        "%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y",
        "%d %B %Y", "%d %b %Y",
        "%Y-%m-%d", "%Y/%m/%d",
        "%d-%m-%y", "%d/%m/%y",
    ]
    
    for fmt in formats:
        try:
            parsed = datetime.strptime(date_str, fmt)
            return parsed.strftime("%Y-%m-%d")
        except ValueError:
            continue
    
    LOGGER.debug(f"Could not parse date: {date_str}")
    return date_str  # Return original if parsing fails


def _extract_with_patterns(text: str, patterns: List[str]) -> List[str]:
    """
    Extract matches using a list of regex patterns.
    
    Args:
        text: The text to search
        patterns: List of regex patterns to try
        
    Returns:
        List of matched strings
    """
    matches = []
    for pattern in patterns:
        try:
            found = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            matches.extend(found)
        except Exception as e:
            LOGGER.debug(f"Pattern matching error: {e}")
            continue
    
    return [m.strip() for m in matches if m and m.strip()]


def _calculate_confidence(metadata: Dict[str, Any], total_fields: int = 6) -> float:
    """
    Calculate confidence score based on number of fields extracted.
    
    Args:
        metadata: Extracted metadata dictionary
        total_fields: Total number of fields expected
        
    Returns:
        Confidence score between 0.0 and 1.0
    """
    filled_fields = sum(1 for v in metadata.values() if v)
    return min(1.0, filled_fields / total_fields)


def _clean_text_for_llm(text: str, max_length: int = 3000) -> str:
    """
    Prepare text for LLM processing by cleaning and truncating.
    
    Args:
        text: Raw extracted text
        max_length: Maximum length for LLM
        
    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Take first portion (usually where metadata is)
    return text[:max_length]


# ==============================================================================
# PATTERN-BASED EXTRACTION
# ==============================================================================

def _extract_case_number(text: str) -> Optional[str]:
    """Extract case number using regex patterns."""
    matches = _extract_with_patterns(text, CASE_NUMBER_PATTERNS)
    return matches[0] if matches else None


def _extract_dates(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract filing and judgment dates.
    
    Returns:
        Tuple of (filing_date, judgment_date)
    """
    dates = _extract_with_patterns(text, DATE_PATTERNS)
    
    filing_date = None
    judgment_date = None
    
    # Heuristic: look for keywords around dates
    for date in dates:
        if not date:
            continue
        
        # Look for context around the date
        idx = text.lower().find(date)
        if idx != -1:
            context = text[max(0, idx - 100):idx + 100].lower()
            
            if 'filed' in context or 'filing' in context:
                filing_date = _normalize_date(date)
            elif 'judgment' in context or 'decided' in context or 'delivered' in context:
                judgment_date = _normalize_date(date)
    
    # If we only found one date, assign it based on context
    if not filing_date and not judgment_date and dates:
        judgment_date = _normalize_date(dates[0])
    elif not judgment_date and dates:
        judgment_date = _normalize_date(dates[-1])
    
    return filing_date, judgment_date


def _extract_judge(text: str) -> Optional[str]:
    """Extract judge/bench name."""
    matches = _extract_with_patterns(text, JUDGE_PATTERNS)
    if matches:
        # Clean up the match
        judge = matches[0].strip()
        judge = re.sub(r'[,\.\s]+$', '', judge)
        return judge
    return None


def _extract_court(text: str) -> Optional[str]:
    """Extract court name."""
    matches = _extract_with_patterns(text, COURT_PATTERNS)
    return matches[0] if matches else None


def _extract_parties(text: str) -> Dict[str, Any]:
    """
    Extract party names (petitioner, respondent, etc.).
    
    Returns:
        Dictionary with petitioner, respondent, and parties list
    """
    parties_dict = {
        'petitioner': None,
        'respondent': None,
        'parties': []
    }
    
    # Extract using party patterns
    matches = _extract_with_patterns(text, PARTY_PATTERNS)
    
    for match in matches:
        # Clean up party name
        party = re.sub(r'\(.*?\)', '', match).strip()
        party = re.sub(r'[,\.\s]+$', '', party)
        
        if party and len(party) > 2:
            parties_dict['parties'].append(party)
    
    # Assign first and second as petitioner/respondent
    if parties_dict['parties']:
        parties_dict['petitioner'] = parties_dict['parties'][0]
        if len(parties_dict['parties']) > 1:
            parties_dict['respondent'] = parties_dict['parties'][1]
    
    return parties_dict


# ==============================================================================
# LLM-BASED EXTRACTION (FALLBACK)
# ==============================================================================

def _extract_metadata_via_llm(text: str) -> Dict[str, Any]:
    """
    Use LLM to extract metadata when pattern matching is insufficient.
    
    Args:
        text: The cleaned judgment text
        
    Returns:
        Dictionary of extracted metadata
    """
    try:
        from cli_client import get_client
        
        prompt = f"""Extract the following metadata from this legal judgment text:
1. Case Number: (e.g., 2023/123, WP-1234-2023)
2. Filing Date: (YYYY-MM-DD format if possible)
3. Judgment Date: (YYYY-MM-DD format if possible)
4. Judge Name: (Presiding Judge/Bench)
5. Court Name: (Which court?)
6. Petitioner/Plaintiff Name: (Who filed the case?)
7. Respondent/Defendant Name: (Against whom?)

Text:
{text}

Return as JSON with these exact keys:
{{
  "case_number": "...",
  "filing_date": "...",
  "judgment_date": "...",
  "judge_name": "...",
  "court_name": "...",
  "petitioner": "...",
  "respondent": "..."
}}
"""
        
        client = get_client()
        
        # Make API call with timeout
        response = client.chat.completions.create(
            model="meta-llama/llama-3.1-8b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            timeout=30,
        )
        
        if response and response.choices:
            content = response.choices[0].message.content
            
            # Parse JSON from response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
    
    except Exception as e:
        LOGGER.debug(f"LLM extraction failed: {e}")
    
    return {}


# ==============================================================================
# MAIN EXTRACTION FUNCTION
# ==============================================================================

def extract_metadata(
    document_input,
    enable_llm: bool = False,
    enable_ocr: bool = False,
) -> CaseMetadata:
    """
    Extract metadata from a legal judgment document.
    
    This function extracts basic case information including case number,
    dates, judge name, court name, and party names from legal documents.
    
    Args:
        document_input: Path to PDF file, text string, or file-like object
        enable_llm: If True, use LLM for complex metadata extraction
        enable_ocr: If True, enable OCR for scanned documents
        
    Returns:
        CaseMetadata object with extracted information
        
    Example:
        >>> metadata = extract_metadata('judgment.pdf')
        >>> print(metadata.case_number)
        >>> print(metadata.to_json())
    """
    metadata_dict = {
        'case_number': None,
        'filing_date': None,
        'judgment_date': None,
        'judge_name': None,
        'court_name': None,
        'petitioner': None,
        'respondent': None,
        'parties': [],
    }
    
    extraction_method = "pattern"
    
    try:
        # Step 1: Extract text from document
        LOGGER.info(f"Extracting text from document: {document_input}")
        text = core.extract_text_from_pdf(
            document_input,
            enable_ocr=enable_ocr,
            ocr_languages="eng+hin",
        )
        
        if not text:
            LOGGER.warning("No text extracted from document")
            metadata = CaseMetadata(**metadata_dict)
            metadata.confidence_score = 0.0
            return metadata
        
        # Store first 500 chars for reference
        metadata_dict['raw_text_sample'] = text[:500]
        
        # Step 2: Pattern-based extraction
        LOGGER.info("Performing pattern-based metadata extraction")
        metadata_dict['case_number'] = _extract_case_number(text)
        filing_date, judgment_date = _extract_dates(text)
        metadata_dict['filing_date'] = filing_date
        metadata_dict['judgment_date'] = judgment_date
        metadata_dict['judge_name'] = _extract_judge(text)
        metadata_dict['court_name'] = _extract_court(text)
        
        parties = _extract_parties(text)
        metadata_dict['petitioner'] = parties['petitioner']
        metadata_dict['respondent'] = parties['respondent']
        metadata_dict['parties'] = parties['parties']
        
        # Step 3: LLM-based extraction (optional)
        if enable_llm:
            LOGGER.info("Performing LLM-based metadata extraction")
            cleaned_text = _clean_text_for_llm(text)
            llm_results = _extract_metadata_via_llm(cleaned_text)
            
            # Fill in missing fields from LLM
            for key in ['case_number', 'filing_date', 'judgment_date', 'judge_name', 'court_name', 'petitioner', 'respondent']:
                if not metadata_dict[key] and key in llm_results:
                    metadata_dict[key] = llm_results[key]
            
            extraction_method = "llm"
        
        # Step 4: Calculate confidence
        confidence = _calculate_confidence(metadata_dict)
        
        # Create result object
        metadata = CaseMetadata(**metadata_dict)
        metadata.confidence_score = confidence
        metadata.extraction_method = extraction_method
        
        LOGGER.info(f"Metadata extraction complete. Confidence: {confidence}")
        return metadata
    
    except Exception as e:
        LOGGER.error(f"Error extracting metadata: {e}", exc_info=True)
        metadata = CaseMetadata(**metadata_dict)
        metadata.confidence_score = 0.0
        return metadata


# ==============================================================================
# BATCH EXTRACTION
# ==============================================================================

def extract_metadata_batch(
    document_paths: List[Path],
    enable_llm: bool = False,
    enable_ocr: bool = False,
) -> List[Dict[str, Any]]:
    """
    Extract metadata from multiple documents.
    
    Args:
        document_paths: List of document paths
        enable_llm: Enable LLM extraction
        enable_ocr: Enable OCR
        
    Returns:
        List of metadata dictionaries with file paths
    """
    results = []
    for doc_path in document_paths:
        try:
            metadata = extract_metadata(doc_path, enable_llm, enable_ocr)
            result = metadata.to_dict()
            result['document_path'] = str(doc_path)
            results.append(result)
        except Exception as e:
            LOGGER.error(f"Failed to extract metadata from {doc_path}: {e}")
            results.append({
                'document_path': str(doc_path),
                'error': str(e),
            })
    
    return results