"""
PDF processing logic for CLI.

This module handles:
- Text extraction from PDFs
- Language detection and normalization
- Summary generation via LLM
- Legal remedy extraction via LLM
- Single PDF processing orchestration
"""

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

from openai import OpenAI
from langdetect import DetectorFactory, LangDetectException, detect
import structlog

import core
from cli_config import CLIConfig
from cli_client import chat_completion, get_usage_tokens, estimate_cost_usd, CLIError
from config import Config

LOGGER = structlog.get_logger(__name__)

# Make language detection deterministic
DetectorFactory.seed = 0

SUPPORTED_LANGUAGES = set(core.LANGUAGE_ALIASES)
LANG_CODE_TO_NAME = core.LANGUAGE_CODE_TO_NAME


def detect_language_name(text: str) -> str:
    """
    Detect the language of the given text.
    
    Uses langdetect with sampling strategy to avoid bias from
    English headers/footers in multilingual documents.
    
    Args:
        text: Text to detect language for
        
    Returns:
        Human-readable language name (e.g., 'Hindi', 'English')
    """
    if not text.strip():
        return "English"

    length = len(text)
    if length <= 3000:
        sample = text
    else:
        # Sample beginning, middle, and end to avoid bias from headers/footers
        parts = [
            text[:1000],
            text[length // 2 - 500 : length // 2 + 500],
            text[-1000:]
        ]
        sample = " ".join(parts)

    try:
        code = detect(sample)
    except LangDetectException:
        return "English"
    return LANG_CODE_TO_NAME.get(code, "English")


def normalize_language(language: str, text_for_auto: str = "") -> str:
    """
    Normalize language specification to a supported language name.
    
    Handles:
    - 'auto' detection based on text
    - Language aliases (e.g., 'en' -> 'English')
    - Fallback to English for unsupported languages
    
    Args:
        language: Language specification (e.g., 'auto', 'en', 'Hindi')
        text_for_auto: Text to detect language from if auto-detection is requested
        
    Returns:
        Normalized language name
    """
    if not language:
        return detect_language_name(text_for_auto)
    lower = language.strip().lower()
    if lower == "auto":
        return detect_language_name(text_for_auto)
    if lower in SUPPORTED_LANGUAGES:
        return core.LANGUAGE_ALIASES[lower]
    return "English"


def generate_summary(
    client: OpenAI,
    config: CLIConfig,
    raw_text: str,
) -> Tuple[str, int, int, int]:
    """
    Generate a legal summary using LLM.
    
    Features:
    - Multilingual support with language mismatch detection
    - Retry with stricter settings if output language doesn't match request
    - Token and cost tracking
    
    Args:
        client: OpenAI client
        config: CLI configuration (includes model, language, max_tokens settings)
        raw_text: Raw text from PDF
        
    Returns:
        Tuple of (summary_text, prompt_tokens, completion_tokens, total_tokens)
        
    Raises:
        CLIError: If model returns empty summary
    """
    safe_text = core.compress_text(raw_text, limit=config.max_chars)
    summary_prompt = core.build_summary_prompt(safe_text, config.language)
    
    resp_summary = chat_completion(
        client=client,
        model=config.model,
        system_prompt="You are an expert legal simplification engine.",
        user_prompt=summary_prompt,
        max_tokens=Config.SUMMARY_MAX_TOKENS,
        temperature=0.05,
    )
    
    summary = (resp_summary.choices[0].message.content or "").strip()
    p_sum, c_sum, t_sum = get_usage_tokens(resp_summary)

    # If non-English, check for language mismatch and retry if needed
    if config.language.lower() != "english" and core.output_language_mismatch_detected(summary, config.language):
        retry_prompt = core.build_retry_prompt(safe_text, config.language)
        resp_retry = chat_completion(
            client=client,
            model=config.model,
            system_prompt="Strict multilingual rewriting engine.",
            user_prompt=retry_prompt,
            max_tokens=Config.SUMMARY_MAX_TOKENS,
            temperature=0.03,
        )
        retry_summary = (resp_retry.choices[0].message.content or "").strip()
        p_ret, c_ret, t_ret = get_usage_tokens(resp_retry)
        p_sum += p_ret
        c_sum += c_ret
        t_sum += t_ret
        if retry_summary and not core.output_language_mismatch_detected(retry_summary, config.language):
            summary = retry_summary

    if not summary:
        raise CLIError("Model returned empty summary.")
        
    return summary, p_sum, c_sum, t_sum


def get_remedies(
    client: OpenAI,
    config: CLIConfig,
    raw_text: str,
    file_name: str = "unknown"
) -> Tuple[Dict[str, Optional[str]], int, int, int]:
    """
    Extract legal remedies from document text using LLM.
    
    Parses LLM response to extract:
    - What happened in the case
    - Whether appeal is possible
    - Days to file appeal
    - Appeal court
    - Cost estimate
    - First action to take
    - Key deadline
    
    Args:
        client: OpenAI client
        config: CLI configuration
        raw_text: Raw text from PDF
        file_name: Filename for logging
        
    Returns:
        Tuple of (remedies_dict, prompt_tokens, completion_tokens, total_tokens)
    """
    remedies_prompt = core.build_remedies_prompt(raw_text, config.language)
    resp_remedies = chat_completion(
        client=client,
        model=config.model,
        system_prompt="You are a helpful legal advisor. Answer questions about legal remedies in India.",
        user_prompt=remedies_prompt,
        max_tokens=Config.REMEDIES_MAX_TOKENS,
        temperature=0.1,
    )
    
    remedies_text = (resp_remedies.choices[0].message.content or "").strip()
    remedies = core.parse_remedies_response(remedies_text)
    
    if remedies is None:
        LOGGER.warning(
            "get_remedies: remedies parsing failed for file=%s",
            file_name,
        )
        remedies = {
            "what_happened": None,
            "can_appeal": None,
            "appeal_days": None,
            "appeal_court": None,
            "cost_estimate": None,
            "first_action": None,
            "deadline": None,
        }
        
    p_rem, c_rem, t_rem = get_usage_tokens(resp_remedies)
    return remedies, p_rem, c_rem, t_rem


def process_one_pdf(
    pdf_path: Path,
    client: OpenAI,
    config: CLIConfig,
) -> Dict[str, object]:
    """
    Process a single PDF file and extract legal insights.
    
    Steps:
    1. Extract text (using OCR if enabled and necessary)
    2. Detect/normalize language
    3. Call LLM for summarization
    4. Call LLM for remedy extraction
    5. Calculate metrics (tokens, cost, duration)
    
    Args:
        pdf_path: Path to PDF file
        client: OpenAI client (or None for extraction-only mode)
        config: CLI configuration with processing parameters
        
    Returns:
        Dictionary with extraction results and metadata:
        - file_name, file_path, status, error
        - language, summary, remedies (what_happened, can_appeal, etc.)
        - prompt_tokens, completion_tokens, total_tokens, api_cost_usd
        - duration_seconds, processed_at
        - extraction_method, ocr_used, extraction_confidence
    """
    started = time.time()
    processed_at = datetime.now(timezone.utc).isoformat()
    
    LOGGER.debug("process_one_pdf_start", file_path=str(pdf_path))

    result: Dict[str, object] = {
        "file_name": pdf_path.name,
        "file_path": str(pdf_path.resolve()),
        "status": "success",
        "error": "",
        "language": "",
        "summary": "",
        "what_happened": "",
        "can_appeal": "",
        "appeal_days": "",
        "appeal_court": "",
        "cost_estimate": "",
        "first_action": "",
        "deadline": "",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "api_cost_usd": 0.0,
        "duration_seconds": 0.0,
        "processed_at": processed_at,
        "extraction_method": "",
        "ocr_enabled": config.enable_ocr,
        "ocr_used": False,
        "extraction_confidence": "",
    }

    try:
        raw_text = ""
        extraction_method = "unknown"
        extraction_confidence = ""
        ocr_used = False

        # Attempt extraction with diagnostics if available
        if hasattr(core, "extract_text_with_diagnostics"):
            LOGGER.debug("extracting_text_with_diagnostics", file=pdf_path.name)
            diagnostics = core.extract_text_with_diagnostics(
                pdf_input=pdf_path,
                enable_ocr=config.enable_ocr,
                ocr_languages=config.ocr_languages,
                ocr_dpi=config.ocr_dpi,
            )
            raw_text = str(diagnostics.get("text", "") or "")
            extraction_method = str(diagnostics.get("method", "") or "unknown")
            ocr_used = bool(diagnostics.get("ocr_used", False))
            conf = diagnostics.get("confidence")
            extraction_confidence = "" if conf is None else str(conf)
            
            LOGGER.debug(
                "extraction_metadata",
                method=extraction_method,
                ocr_used=ocr_used,
                confidence=extraction_confidence,
                text_length=len(raw_text)
            )
        else:
            # Fallback to standard extraction
            LOGGER.debug("extracting_text_standard", file=pdf_path.name)
            raw_text = core.extract_text_from_pdf(
                pdf_path,
                enable_ocr=config.enable_ocr,
                ocr_languages=config.ocr_languages,
                ocr_dpi=config.ocr_dpi,
            )
            extraction_method = "ocr_or_standard"

        if not raw_text:
            raise CLIError("No extractable text found in PDF.")
            
        result["extraction_method"] = extraction_method
        result["ocr_used"] = ocr_used
        result["extraction_confidence"] = extraction_confidence
        
        # If no client, skip LLM processing (extraction-only mode)
        if client is None:
            LOGGER.debug("skipping_llm_processing", reason="no_client_provided")
            return result

        # Language normalization
        language = normalize_language(config.language, text_for_auto=raw_text)
        result["language"] = language
        LOGGER.debug("language_determined", language=language)

        # Update config with detected language for consistent processing
        config_with_language = type('obj', (object,), {**config.__dict__, 'language': language})()

        # Phase 1: Generate Summary
        LOGGER.debug("generating_summary", file=pdf_path.name)
        summary, p_sum, c_sum, t_sum = generate_summary(
            client=client,
            config=config_with_language,
            raw_text=raw_text,
        )

        # Phase 2: Get Remedies
        LOGGER.debug("extracting_remedies", file=pdf_path.name)
        remedies, p_rem, c_rem, t_rem = get_remedies(
            client=client,
            config=config_with_language,
            raw_text=raw_text,
            file_name=pdf_path.name
        )

        # Phase 3: Aggregate Metrics
        prompt_tokens = p_sum + p_rem
        completion_tokens = c_sum + c_rem
        total_tokens = t_sum + t_rem
        
        cost_usd = estimate_cost_usd(
            prompt_tokens,
            completion_tokens,
            prompt_cost_per_1k=config.prompt_cost_per_1k,
            completion_cost_per_1k=config.completion_cost_per_1k,
        )

        # Update result with extracted content and metrics
        result.update(
            {
                "summary": summary,
                "what_happened": remedies.get("what_happened") or "",
                "can_appeal": remedies.get("can_appeal") or "",
                "appeal_days": remedies.get("appeal_days") or "",
                "appeal_court": remedies.get("appeal_court") or "",
                "cost_estimate": remedies.get("cost_estimate") or "",
                "first_action": remedies.get("first_action") or "",
                "deadline": remedies.get("deadline") or "",
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "api_cost_usd": round(cost_usd, 8),
            }
        )
        
        LOGGER.debug("process_one_pdf_success", file=pdf_path.name, cost=result["api_cost_usd"])

    except Exception as exc:
        # Catch errors to allow batch processing to continue
        result["status"] = "error"
        result["error"] = str(exc)
        LOGGER.error("process_one_pdf_failed", file=pdf_path.name, error=str(exc))

    result["duration_seconds"] = round(time.time() - started, 3)
    return result
