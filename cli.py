import argparse
import csv
import hashlib
import json
import logging
import structlog
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pypdf import PdfReader
from langdetect import DetectorFactory, LangDetectException, detect
from openai import OpenAI, RateLimitError
from tqdm import tqdm
try:
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
except ModuleNotFoundError:
    class _ProgressColumn:
        def __init__(self, *args, **kwargs):
            pass

    SpinnerColumn = BarColumn = TextColumn = TimeElapsedColumn = _ProgressColumn

    class Progress:
        """Small tqdm-backed fallback when rich is not installed."""

        def __init__(self, *args, **kwargs):
            self._bar = None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            if self._bar:
                self._bar.close()

        def add_task(self, description, total):
            self._bar = tqdm(total=total, desc=description)
            return 0

        def advance(self, task_id, advance=1):
            if self._bar:
                self._bar.update(advance)

        def update(self, task_id, description=None, **kwargs):
            if self._bar and description:
                self._bar.set_description_str(description)
from logging_config import configure_logging
import core
from config import Config

# Make language detection deterministic.
DetectorFactory.seed = 0

SUPPORTED_LANGUAGES = set(core.LANGUAGE_ALIASES)
LANG_CODE_TO_NAME = core.LANGUAGE_CODE_TO_NAME
SUPPORTED_LANGUAGE_HELP = ", ".join(["auto", *core.LANGUAGES])
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", core.DEFAULT_MODEL)
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
LOGGER = structlog.get_logger(__name__)

KNOWN_COURTS = {
    "supreme court",
    "high court",
    "district court",
    "sessions court",
    "session court",
    "civil court",
    "family court",
    "consumer court",
    "tribunal",
}

# Global semaphore for API concurrency control
_API_SEMAPHORE: Optional[threading.Semaphore] = None
_SEMAPHORE_LOCK = threading.Lock()


def _reinitialize_semaphore(concurrency: int) -> None:
    """Replace the global API semaphore with a new one sized to *concurrency*.

    Calling this before any worker threads are spawned ensures the correct
    limit is applied regardless of whether execution entered through main()
    or directly via process_command / batch_command (e.g. in tests).
    """
    global _API_SEMAPHORE
    with _SEMAPHORE_LOCK:
        _API_SEMAPHORE = threading.Semaphore(concurrency)


def get_api_semaphore() -> threading.Semaphore:
    """Get the API semaphore, initializing it lazily with default concurrency if needed."""
    global _API_SEMAPHORE
    if _API_SEMAPHORE is None:
        with _SEMAPHORE_LOCK:
            if _API_SEMAPHORE is None:
                _API_SEMAPHORE = threading.Semaphore(5)
    return _API_SEMAPHORE


def _reinitialize_semaphore(concurrency: int) -> None:
    """Replace the global API semaphore with a new one sized to *concurrency*.

    Calling this before any worker threads are spawned ensures the correct
    limit is applied regardless of whether execution entered through main()
    or directly via process_command / batch_command (e.g. in tests).
    """
    global API_SEMAPHORE
    API_SEMAPHORE = threading.Semaphore(concurrency)



class CLIError(Exception):
    pass


@dataclass
class CostTracker:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    def add(self, prompt_tokens: int, completion_tokens: int, total_tokens: int, cost_usd: float) -> None:
        with self._lock:
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            self.total_tokens += total_tokens
            self.total_cost_usd += cost_usd

    def snapshot(self) -> Dict[str, float]:
        with self._lock:
            return {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
                "total_cost_usd": round(self.total_cost_usd, 8),
            }


def get_client() -> OpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise CLIError(
            "Missing API key. Set OPENROUTER_API_KEY (preferred) or OPENAI_API_KEY in your environment. "
            "You can also add these to your .env file."
        )

    base_url = os.getenv("OPENROUTER_BASE_URL", DEFAULT_BASE_URL)
    return OpenAI(api_key=api_key, base_url=base_url)


# Prompt building and core logic moved to core.py


def detect_language_name(text: str) -> str:
    if not text.strip():
        return "English"

    length = len(text)
    if length <= 3000:
        sample = text
    else:
        # Sample beginning, middle, and end to avoid bias from English cover pages
        # or administrative footers in local language documents.
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
    if not language:
        return detect_language_name(text_for_auto)
    lower = language.strip().lower()
    if lower == "auto":
        return detect_language_name(text_for_auto)
    if lower in SUPPORTED_LANGUAGES:
        return core.LANGUAGE_ALIASES[lower]
    return "English"


def _usage_tokens(response) -> Tuple[int, int, int]:
    usage = getattr(response, "usage", None)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0)
    return prompt_tokens, completion_tokens, total_tokens


def _estimate_cost_usd(
    prompt_tokens: int,
    completion_tokens: int,
    prompt_cost_per_1k: float,
    completion_cost_per_1k: float,
) -> float:
    return ((prompt_tokens / 1000.0) * prompt_cost_per_1k) + (
        (completion_tokens / 1000.0) * completion_cost_per_1k
    )


def _chat_completion(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
    max_retries: int = 5,
    timeout: float = None,
):
    """
    Internal helper to handle chat completion requests with retries and concurrency control.
    
    This function wraps the OpenAI client call, providing:
    1. Concurrency control via a global semaphore.
    2. Exponential backoff for rate limiting (429 errors).
    3. Timeout management.
    4. Detailed debug logging when verbose mode is enabled.
    """
    if timeout is None:
        # Fallback to default timeout from configuration if not specified.
        timeout = Config.LLM_TIMEOUT
    
    last_err = None
    
    # Debug log the start of the completion request.
    # This helps track how many requests are being sent and to which model.
    LOGGER.debug(
        "chat_completion_start",
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout
    )
    
    for attempt in range(max_retries):
        try:
            # Concurrency control is critical to avoid overwhelming the API provider
            # or exceeding local resource limits.
            with get_api_semaphore():
                # Perform the actual API call.
                # Note: We pass both system and user prompts to provide context.
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=timeout,
                )
                
                # Log success at debug level for auditing.
                LOGGER.debug(
                    "chat_completion_success",
                    model=model,
                    attempt=attempt + 1
                )
                return response
                
        except RateLimitError as e:
            last_err = e
            # If we've exhausted all retries, log the error and propagate.
            if attempt == max_retries - 1:
                LOGGER.error("api_rate_limit_exhausted", attempts=max_retries, error=str(e))
                raise
            
            # Exponential backoff: 2, 4, 8, 16, 32 seconds.
            # This gives the API time to recover from traffic spikes.
            wait_time = 2 ** (attempt + 1)
            LOGGER.warning(
                "api_rate_limited",
                attempt=attempt + 1,
                wait_seconds=wait_time,
                error=str(e)
            )
            time.sleep(wait_time)
        except Exception as e:
            # We don't retry on other errors (like auth or invalid params) 
            # to avoid infinite loops on fatal configuration issues.
            LOGGER.debug("chat_completion_fatal_error", error=str(e), error_type=type(e).__name__)
            raise e
    
    if last_err:
        raise last_err



def generate_summary(
    client: OpenAI,
    model: str,
    raw_text: str,
    language: str,
    max_chars: int,
) -> Tuple[str, int, int, int]:
    """Generates a legal summary with multilingual leakage protection."""
    safe_text = core.compress_text(raw_text, limit=max_chars)
    summary_prompt = core.build_summary_prompt(safe_text, language)
    
    resp_summary = _chat_completion(
        client=client,
        model=model,
        system_prompt="You are an expert legal simplification engine.",
        user_prompt=summary_prompt,
        max_tokens=Config.SUMMARY_MAX_TOKENS,
        temperature=0.05,
    )
    
    summary = (resp_summary.choices[0].message.content or "").strip()
    p_sum, c_sum, t_sum = _usage_tokens(resp_summary)

    if language.lower() != "english" and core.output_language_mismatch_detected(summary, language):
        retry_prompt = core.build_retry_prompt(safe_text, language)
        resp_retry = _chat_completion(
            client=client,
            model=model,
            system_prompt="Strict multilingual rewriting engine.",
            user_prompt=retry_prompt,
            max_tokens=Config.SUMMARY_MAX_TOKENS,
            temperature=0.03,
        )
        retry_summary = (resp_retry.choices[0].message.content or "").strip()
        p_ret, c_ret, t_ret = _usage_tokens(resp_retry)
        p_sum += p_ret
        c_sum += c_ret
        t_sum += t_ret
        if retry_summary and not core.output_language_mismatch_detected(retry_summary, language):
            summary = retry_summary

    if not summary:
        raise CLIError("Model returned empty summary.")
        
    return summary, p_sum, c_sum, t_sum


def get_remedies(
    client: OpenAI,
    model: str,
    raw_text: str,
    language: str,
    file_name: str = "unknown"
) -> Tuple[Dict[str, Optional[str]], int, int, int]:
    """Calls LLM to extract legal remedies and parse the response."""
    remedies_prompt = core.build_remedies_prompt(raw_text, language)
    resp_remedies = _chat_completion(
        client=client,
        model=model,
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
        
    p_rem, c_rem, t_rem = _usage_tokens(resp_remedies)
    return remedies, p_rem, c_rem, t_rem


def process_one_pdf(
    pdf_path: Path,
    client: OpenAI,
    language_arg: str,
    model: str,
    max_chars: int,
    prompt_cost_per_1k: float,
    completion_cost_per_1k: float,
    enable_ocr: bool = False,
    ocr_languages: str = "eng+hin",
    ocr_dpi: int = 300,
) -> Dict[str, object]:
    """
    Core logic to process a single PDF file and extract legal insights.
    
    Steps:
    1. Extract text (using OCR if enabled and necessary).
    2. Detect/Normalize language.
    3. Call LLM for summarization.
    4. Call LLM for remedy extraction.
    5. Calculate metrics (tokens, cost, duration).
    """
    started = time.time()
    processed_at = datetime.now(timezone.utc).isoformat()
    
    # Debug logging to track which file is being handled in which thread.
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
        "ocr_enabled": enable_ocr,
        "ocr_used": False,
        "extraction_confidence": "",
    }

    try:
        raw_text = ""
        extraction_method = "unknown"
        extraction_confidence = ""
        ocr_used = False

        # Attempt to extract text with diagnostics if available in core.
        # Diagnostics provide more metadata about HOW the text was extracted.
        if hasattr(core, "extract_text_with_diagnostics"):
            LOGGER.debug("extracting_text_with_diagnostics", file=pdf_path.name)
            diagnostics = core.extract_text_with_diagnostics(
                pdf_input=pdf_path,
                enable_ocr=enable_ocr,
                ocr_languages=ocr_languages,
                ocr_dpi=ocr_dpi,
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
            # Fallback to standard text extraction if diagnostics are not supported.
            LOGGER.debug("extracting_text_standard", file=pdf_path.name)
            raw_text = core.extract_text_from_pdf(
                pdf_path,
                enable_ocr=enable_ocr,
                ocr_languages=ocr_languages,
                ocr_dpi=ocr_dpi,
            )
            extraction_method = "ocr_or_standard"

        if not raw_text:
            # If no text is found, we cannot proceed with LLM processing.
            raise CLIError("No extractable text found in PDF.")
            
        result["extraction_method"] = extraction_method
        result["ocr_used"] = ocr_used
        result["extraction_confidence"] = extraction_confidence
        
        # If client is None, it means we are in 'dry run' or 'extraction only' mode.
        if client is None:
            LOGGER.debug("skipping_llm_processing", reason="no_client_provided")
            return result

        # Language normalization determines the prompt language and leakage protection.
        language = normalize_language(language_arg, text_for_auto=raw_text)
        result["language"] = language
        LOGGER.debug("language_determined", language=language)

        # Phase 1: Generate Summary
        # This condenses the legal document into a layman-friendly summary.
        LOGGER.debug("generating_summary", file=pdf_path.name)
        summary, p_sum, c_sum, t_sum = generate_summary(
            client=client,
            model=model,
            raw_text=raw_text,
            language=language,
            max_chars=max_chars
        )

        # Phase 2: Get Remedies
        # This extracts specific actionable legal items from the judgment.
        LOGGER.debug("extracting_remedies", file=pdf_path.name)
        remedies, p_rem, c_rem, t_rem = get_remedies(
            client=client,
            model=model,
            raw_text=raw_text,
            language=language,
            file_name=pdf_path.name
        )

        # Phase 3: Aggregate Metrics and Results
        # Tokens and costs are tracked for auditing and budget management.
        prompt_tokens = p_sum + p_rem
        completion_tokens = c_sum + c_rem
        total_tokens = t_sum + t_rem
        
        cost_usd = _estimate_cost_usd(
            prompt_tokens,
            completion_tokens,
            prompt_cost_per_1k=prompt_cost_per_1k,
            completion_cost_per_1k=completion_cost_per_1k,
        )

        # Update result dictionary with extracted content and metrics.
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
        # Catch any unexpected errors to ensure the rest of the batch can continue.
        result["status"] = "error"
        result["error"] = str(exc)
        LOGGER.error("process_one_pdf_failed", file=pdf_path.name, error=str(exc))

    result["duration_seconds"] = round(time.time() - started, 3)
    return result


def load_checkpoint(checkpoint_file: Path, corruption_threshold: float = 0.1) -> List[Dict[str, object]]:
    """Load checkpoint records from a JSONL file.
    
    Args:
        checkpoint_file: Path to the checkpoint file.
        corruption_threshold: Maximum fraction of lines that can be corrupted before failing (0.0-1.0).
                             Defaults to 0.1 (10%).
    
    Returns:
        List of valid checkpoint records.
    
    Raises:
        CLIError: If corruption exceeds the threshold.
    """
    if not checkpoint_file.exists():
        return []

    records: List[Dict[str, object]] = []
    skipped_lines: List[Tuple[int, str]] = []
    line_num = 0
    
    with checkpoint_file.open("r", encoding="utf-8") as f:
        for line in f:
            line_num += 1
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                skipped_lines.append((line_num, str(e)))
                LOGGER.warning(
                    "checkpoint_line_corrupted",
                    line_number=line_num,
                    error=str(e),
                    line_preview=line[:100],
                )
    
    # Check if corruption exceeds threshold
    total_lines = line_num
    if total_lines > 0 and skipped_lines:
        corruption_rate = len(skipped_lines) / total_lines
        if corruption_rate > corruption_threshold:
            error_msg = (
                f"Checkpoint file corruption rate {corruption_rate:.1%} exceeds threshold {corruption_threshold:.1%}. "
                f"Skipped {len(skipped_lines)} out of {total_lines} lines. "
                f"First corrupted line: {skipped_lines[0][0]} ({skipped_lines[0][1]})"
            )
            LOGGER.error("checkpoint_corruption_threshold_exceeded", corruption_rate=corruption_rate, skipped_count=len(skipped_lines))
            raise CLIError(error_msg)
        elif skipped_lines:
            LOGGER.info(
                "checkpoint_partially_corrupted",
                skipped_count=len(skipped_lines),
                total_lines=total_lines,
                corruption_rate=f"{corruption_rate:.1%}",
                recovered_records=len(records),
            )
    
    return records


def dedupe_latest_by_file(records: List[Dict[str, object]]) -> List[Dict[str, object]]:
    latest: Dict[str, Dict[str, object]] = {}
    for rec in records:
        file_path = str(rec.get("file_path", ""))
        if file_path:
            latest[file_path] = rec
    return list(latest.values())


def export_results(records: List[Dict[str, object]], output_path: Path, export_format: str) -> Tuple[Path, Path]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stem = output_path.with_suffix("")

    csv_path = stem.with_suffix(".csv")
    json_path = stem.with_suffix(".json")

    ordered = dedupe_latest_by_file(records)
    ordered.sort(key=lambda x: str(x.get("file_name", "")))

    if export_format in {"csv", "both"}:
        if ordered:
            fieldnames = list(ordered[0].keys())
        else:
            fieldnames = [
                "file_name",
                "file_path",
                "status",
                "error",
                "language",
                "summary",
                "what_happened",
                "can_appeal",
                "appeal_days",
                "appeal_court",
                "cost_estimate",
                "first_action",
                "deadline",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "api_cost_usd",
                "duration_seconds",
                "processed_at",
            ]
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in ordered:
                writer.writerow(row)

    if export_format in {"json", "both"}:
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(ordered, f, ensure_ascii=False, indent=2)

    return csv_path, json_path


def collect_pdf_files(folder: Path, recursive: bool) -> List[Path]:
    pattern = "**/*.pdf" if recursive else "*.pdf"
    return sorted([p for p in folder.glob(pattern) if p.is_file()])


def print_cost_summary(snapshot: Dict[str, float]) -> None:
    LOGGER.info("batch_cost_summary", **snapshot)


def process_command(args: argparse.Namespace) -> int:
    _reinitialize_semaphore(args.concurrency)
    file_path = Path(args.file)
    if not file_path.exists() or file_path.suffix.lower() != ".pdf":
        raise CLIError(f"Invalid PDF file: {file_path}")
    client = get_client()

    result = process_one_pdf(
        pdf_path=file_path,
        client=client,
        language_arg=args.language,
        model=args.model,
        max_chars=args.max_chars,
        prompt_cost_per_1k=args.prompt_cost_per_1k,
        completion_cost_per_1k=args.completion_cost_per_1k,
        enable_ocr=args.enable_ocr,
        ocr_languages=args.ocr_languages,
        ocr_dpi=args.ocr_dpi,
    )

    LOGGER.info("process_result", result=result)

    if args.output:
        out_path = Path(args.output)
        records = [result]
        csv_path, json_path = export_results(records, out_path, args.format)
        if args.format in {"csv", "both"}:
            LOGGER.info("wrote_file", path=str(csv_path), format="csv")
        if args.format in {"json", "both"}:
            LOGGER.info("wrote_file", path=str(json_path), format="json")

    return 0 if result.get("status") == "success" else 1


def batch_command(args: argparse.Namespace) -> int:
    _reinitialize_semaphore(args.concurrency)
    folder = Path(args.folder)
    if not folder.exists() or not folder.is_dir():
        raise CLIError(f"Invalid folder: {folder}")
    client = get_client()

    all_files = collect_pdf_files(folder, recursive=args.recursive)
    if not all_files:
        raise CLIError(f"No PDF files found in folder: {folder}")

    output_path = Path(args.output)
    checkpoint_file = Path(args.checkpoint) if args.checkpoint else output_path.with_suffix(output_path.suffix + ".checkpoint.jsonl")

    # Delete stale checkpoint BEFORE loading so --no-resume truly starts fresh.
    if not args.resume and checkpoint_file.exists():
        checkpoint_file.unlink()

    try:
        existing_records = load_checkpoint(checkpoint_file) if args.resume else []
    except CLIError as e:
        LOGGER.error("checkpoint_load_failed", error=str(e))
        raise

    done_success = {
        str(rec.get("file_path"))
        for rec in existing_records
        if rec.get("status") == "success" and rec.get("file_path")
    }

    to_process = [p for p in all_files if str(p.resolve()) not in done_success]

    LOGGER.info("batch_discovery", total_found=len(all_files), already_completed=len(done_success), pending=len(to_process))

    if not to_process:
        csv_path, json_path = export_results(existing_records, output_path, args.format)
        LOGGER.info("no_pending_files_refresh", msg="No pending files. Export refreshed from checkpoint.")
        if args.format in {"csv", "both"}:
            LOGGER.info("wrote_file", path=str(csv_path), format="csv")
        if args.format in {"json", "both"}:
            LOGGER.info("wrote_file", path=str(json_path), format="json")
        return 0

    tracker = CostTracker()
    run_records: List[Dict[str, object]] = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_one_pdf,
                pdf_path=pdf_path,
                client=client,
                language_arg=args.language,
                model=args.model,
                max_chars=args.max_chars,
                prompt_cost_per_1k=args.prompt_cost_per_1k,
                completion_cost_per_1k=args.completion_cost_per_1k,
                enable_ocr=args.enable_ocr,
                ocr_languages=args.ocr_languages,
                ocr_dpi=args.ocr_dpi,
            ): pdf_path
            for pdf_path in to_process
        }

        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        interrupted = False
        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
        ) as progress, checkpoint_file.open("a", encoding="utf-8") as cp_file:
            task_id = progress.add_task("Processing PDFs", total=len(futures))
            try:
                for future in as_completed(futures):
                    record = future.result()
                    run_records.append(record)

                    # Write to checkpoint immediately and sync to disk to prevent data loss
                    cp_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    cp_file.flush()
                    try:
                        os.fsync(cp_file.fileno())
                    except OSError:
                        pass

                    tracker.add(
                        int(record.get("prompt_tokens", 0) or 0),
                        int(record.get("completion_tokens", 0) or 0),
                        int(record.get("total_tokens", 0) or 0),
                        float(record.get("api_cost_usd", 0.0) or 0.0),
                    )

                    progress.advance(task_id, 1)
                    status = str(record.get("status"))
                    progress.update(task_id, description=f"last={status} cost_usd={tracker.snapshot()['total_cost_usd']:.4f}")
            except KeyboardInterrupt:
                # User pressed Ctrl+C — cancel futures that haven't started yet,
                # then fall through to the finally block for a clean export.
                interrupted = True
                LOGGER.warning("batch_interrupted", completed=len(run_records), pending=len(futures) - len(run_records))
                for f in futures:
                    f.cancel()
            finally:
                # Always flush the checkpoint file before leaving the context so
                # the on-disk state matches run_records regardless of how we exit.
                try:
                    cp_file.flush()
                    os.fsync(cp_file.fileno())
                except OSError:
                    pass

    # Export whatever was completed — covers both normal finish and interruption.
    all_records = existing_records + run_records
    csv_path, json_path = export_results(all_records, output_path, args.format)

    success_count = sum(1 for x in run_records if x.get("status") == "success")
    error_count = len(run_records) - success_count

    if interrupted:
        LOGGER.warning(
            "batch_interrupted_export",
            processed=len(run_records),
            successful=success_count,
            failed=error_count,
            msg="Run was interrupted. Partial results exported. Re-run without --no-resume to continue.",
        )
    else:
        LOGGER.info("batch_summary", processed=len(run_records), successful=success_count, failed=error_count)

    if args.format in {"csv", "both"}:
        LOGGER.info("wrote_file", path=str(csv_path), format="csv")
    if args.format in {"json", "both"}:
        LOGGER.info("wrote_file", path=str(json_path), format="json")

    print_cost_summary(tracker.snapshot())

    if interrupted:
        return 130  # Standard UNIX exit code for Ctrl+C termination

    return 0 if error_count == 0 else 2


def build_parser() -> argparse.ArgumentParser:
    """
    Constructs the CLI argument parser with comprehensive help and options.
    
    The parser supports:
    - process: Single file processing.
    - batch/process_batch: Bulk processing of directories.
    - Global flags like --verbose for debugging.
    """
    parser = argparse.ArgumentParser(
        prog="LegalEase CLI",
        description="CLI for single and batch processing of legal judgment PDFs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Global options that apply to all commands
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output for debugging. Changes logging level from INFO to DEBUG.",
    )

    # Common arguments shared between single and batch processing modes
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--model", 
        default=DEFAULT_MODEL, 
        help="LLM model name for generation (e.g., gpt-4, claude-3-opus)."
    )
    common.add_argument(
        "--language",
        default="auto",
        help=f"Target output language for summaries: {SUPPORTED_LANGUAGE_HELP}. Default: auto",
    )
    common.add_argument(
        "--max-chars",
        type=int,
        default=6000,
        help="Max characters of PDF text to send to the LLM. Prevents context window overflows. Default: 6000",
    )
    common.add_argument(
        "--prompt-cost-per-1k",
        type=float,
        default=0.0,
        help="Estimated USD cost per 1K prompt tokens for cost reporting.",
    )
    common.add_argument(
        "--completion-cost-per-1k",
        type=float,
        default=0.0,
        help="Estimated USD cost per 1K completion tokens for cost reporting.",
    )
    common.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Maximum concurrent API calls allowed at once. Default: 5",
    )
    common.add_argument(
        "--enable-ocr",
        action="store_true",
        help="Enable Tesseract OCR fallback for scanned or image-based PDF documents.",
    )
    common.add_argument(
        "--ocr-languages",
        default="eng+hin",
        help="OCR language codes (e.g., 'eng+hin'). Requires Tesseract language packs.",
    )
    common.add_argument(
        "--ocr-dpi",
        type=int,
        default=300,
        help="DPI resolution for PDF-to-image conversion during OCR. Default: 300",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Single file processing command
    p_process = subparsers.add_parser("process", parents=[common], help="Process a single PDF file.")
    p_process.add_argument("--file", required=True, help="Path to the source PDF file.")
    p_process.add_argument("--output", help="Output file path (e.g., ./result.csv). If omitted, only logs to stdout.")
    p_process.add_argument(
        "--format",
        choices=["csv", "json", "both"],
        default="both",
        help="Desired export format if --output is specified.",
    )
    p_process.set_defaults(func=process_command)

    # Batch processing command
    p_batch = subparsers.add_parser("batch", parents=[common], help="Process multiple PDFs from a folder.")
    p_batch.add_argument("--folder", "--input", dest="folder", required=True, help="Input directory containing PDF files.")
    p_batch.add_argument("--output", required=True, help="Base path for exported results.")
    p_batch.add_argument("--workers", type=int, default=4, help="Number of parallel file processing workers. Default: 4")
    p_batch.add_argument("--recursive", action="store_true", help="Whether to search for PDFs in subdirectories.")
    p_batch.add_argument("--checkpoint", help="Path to the checkpoint file to track progress. Default: <output>.checkpoint.jsonl")
    p_batch.add_argument("--resume", dest="resume", action="store_true", default=True, help="Resume from an existing checkpoint if found.")
    p_batch.add_argument("--no-resume", dest="resume", action="store_false", help="Ignore existing checkpoints and start fresh.")
    p_batch.add_argument(
        "--format",
        choices=["csv", "json", "both"],
        default="both",
        help="Desired export format for batch results.",
    )
    p_batch.set_defaults(func=batch_command)

    # process_batch alias for better UX/compatibility
    p_batch_alias = subparsers.add_parser(
        "process_batch",
        parents=[common],
        help="Alias for the 'batch' command.",
    )
    p_batch_alias.add_argument("--folder", "--input", dest="folder", required=True, help="Input directory.")
    p_batch_alias.add_argument("--output", required=True, help="Output base path.")
    p_batch_alias.add_argument("--workers", type=int, default=4, help="Parallel workers.")
    p_batch_alias.add_argument("--recursive", action="store_true", help="Search subdirectories.")
    p_batch_alias.add_argument("--checkpoint", help="Checkpoint file path.")
    p_batch_alias.add_argument("--resume", dest="resume", action="store_true", default=True, help="Resume progress.")
    p_batch_alias.add_argument("--no-resume", dest="resume", action="store_false", help="Fresh start.")
    p_batch_alias.add_argument(
        "--format",
        choices=["csv", "json", "both"],
        default="both",
        help="Export format.",
    )
    p_batch_alias.set_defaults(func=batch_command)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """
    Main entry point for the LegalEase CLI application.
    
    Responsibilities:
    1. Parse command line arguments.
    2. Configure logging level (dynamic based on --verbose).
    3. Initialize global state (semaphores, clients).
    4. Execute the requested command (process/batch).
    5. Handle global exceptions and exit codes.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    
    # Determine the appropriate logging level.
    # DEBUG level enables detailed tracing of PDF extraction and LLM calls.
    # INFO level is the standard operational mode.
    log_level = logging.DEBUG if args.verbose else logging.INFO
    
    # Configure structured logging and rich output.
    # We attempt to use our custom logging configuration, falling back
    # to standard library basicConfig if it fails (e.g., missing dependencies).
    try:
        configure_logging(level=log_level)
        LOGGER.debug("logging_initialized", level=logging.getLevelName(log_level))
    except Exception as e:
        # Fallback configuration to ensure we don't lose logs if the complex setup fails.
        logging.basicConfig(level=log_level)
        LOGGER.warning("logging_config_fallback", error=str(e))

    # Initialize global semaphore with user-specified concurrency.
    # This prevents the CLI from exceeding API rate limits or system memory.
    _reinitialize_semaphore(args.concurrency)
    LOGGER.debug("semaphore_initialized", concurrency=args.concurrency)

    # Basic validation for worker counts in batch mode.
    if getattr(args, "workers", 1) < 1:
        LOGGER.error("validation_error", detail="--workers must be >= 1")
        raise CLIError("--workers must be >= 1")

    try:
        # Route execution to the function associated with the chosen subcommand.
        return args.func(args)
    except CLIError as exc:
        # Known application errors are logged cleanly without stack traces.
        LOGGER.error("cli_error", error=str(exc))
        return 2
    except Exception as exc:
        # Unexpected errors are logged with full stack traces in verbose mode
        # or simplified logs in standard mode.
        LOGGER.exception("unexpected_error", error=str(exc))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
