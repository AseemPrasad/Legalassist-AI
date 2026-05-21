"""
LegalEase CLI - Command-line interface for legal document processing.

This is the main entry point for the CLI application. It orchestrates:
- Argument parsing
- Logging configuration
- Delegating to command handlers

The actual processing logic is split across focused modules:
- cli_client: API client and concurrency management
- cli_processing: PDF extraction and LLM processing
- cli_checkpoint: Checkpoint loading and deduplication
- cli_export: Results export to CSV/JSON
- cli_config: Shared configuration dataclass
"""

import argparse
import json
import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm
try:
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
except ModuleNotFoundError:
    class _ProgressColumn:
        def __init__(self, *args, **kwargs):
            pass

    SpinnerColumn = BarColumn = TextColumn = TimeElapsedColumn = _ProgressColumn

    class Progress:
        """Fallback progress tracker when rich is not installed."""

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

import structlog

from logging_config import configure_logging
import core
from cli_config import CLIConfig
from cli_client import (
    get_client,
    reinitialize_semaphore,
    CLIError,
)
from cli_processing import process_one_pdf
from cli_checkpoint import (
    load_checkpoint,
    collect_completed_files,
)
from cli_export import (
    export_results,
    collect_pdf_files,
    print_cost_summary,
)

LOGGER = structlog.get_logger(__name__)

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", core.DEFAULT_MODEL)
SUPPORTED_LANGUAGE_HELP = ", ".join(["auto", *core.LANGUAGES])


@dataclass
class CostTracker:
    """Track cumulative token usage and API costs across batch processing."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    def add(self, prompt_tokens: int, completion_tokens: int, total_tokens: int, cost_usd: float) -> None:
        """Add metrics from a single processing result."""
        with self._lock:
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            self.total_tokens += total_tokens
            self.total_cost_usd += cost_usd

    def snapshot(self) -> Dict[str, float]:
        """Get current metrics as a dictionary."""
        with self._lock:
            return {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
                "total_cost_usd": round(self.total_cost_usd, 8),
            }


def process_command(args: argparse.Namespace) -> int:
    """
    Handler for the 'process' command (single PDF processing).
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # Initialize concurrency control
    reinitialize_semaphore(args.concurrency)
    
    # Validate input file
    file_path = Path(args.file)
    if not file_path.exists() or file_path.suffix.lower() != ".pdf":
        raise CLIError(f"Invalid PDF file: {file_path}")
    
    # Create API client
    client = get_client()
    
    # Build config
    config = CLIConfig(
        model=args.model,
        language=args.language,
        max_chars=args.max_chars,
        prompt_cost_per_1k=args.prompt_cost_per_1k,
        completion_cost_per_1k=args.completion_cost_per_1k,
        concurrency=args.concurrency,
        enable_ocr=args.enable_ocr,
        ocr_languages=args.ocr_languages,
        ocr_dpi=args.ocr_dpi,
        export_format=args.format,
    )

    # Process the PDF
    result = process_one_pdf(pdf_path=file_path, client=client, config=config)

    LOGGER.info("process_result", result=result)

    # Export if output path specified
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
    """
    Handler for the 'batch' / 'process_batch' commands (bulk PDF processing).
    
    Supports:
    - Resume from checkpoint
    - Concurrent worker threads
    - Real-time cost tracking
    - Graceful interrupt handling (Ctrl+C)
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success, 2 for errors, 130 for interrupt)
    """
    # Initialize concurrency control
    reinitialize_semaphore(args.concurrency)
    
    # Validate input folder
    folder = Path(args.folder)
    if not folder.exists() or not folder.is_dir():
        raise CLIError(f"Invalid folder: {folder}")
    
    # Create API client
    client = get_client()
    
    # Build config
    config = CLIConfig(
        model=args.model,
        language=args.language,
        max_chars=args.max_chars,
        prompt_cost_per_1k=args.prompt_cost_per_1k,
        completion_cost_per_1k=args.completion_cost_per_1k,
        concurrency=args.concurrency,
        enable_ocr=args.enable_ocr,
        ocr_languages=args.ocr_languages,
        ocr_dpi=args.ocr_dpi,
        export_format=args.format,
    )

    # Collect PDF files
    all_files = collect_pdf_files(folder, recursive=args.recursive)
    if not all_files:
        raise CLIError(f"No PDF files found in folder: {folder}")

    # Set up checkpoint file
    output_path = Path(args.output)
    checkpoint_file = Path(args.checkpoint) if args.checkpoint else output_path.with_suffix(output_path.suffix + ".checkpoint.jsonl")

    # Delete stale checkpoint BEFORE loading so --no-resume truly starts fresh
    if not args.resume and checkpoint_file.exists():
        checkpoint_file.unlink()

    try:
        existing_records = load_checkpoint(checkpoint_file, corruption_threshold=config.checkpoint_corruption_threshold) if args.resume else []
    except CLIError as e:
        LOGGER.error("checkpoint_load_failed", error=str(e))
        raise

    # Determine which files still need processing
    done_success = collect_completed_files(existing_records, status_filter="success")
    to_process = [p for p in all_files if str(p.resolve()) not in done_success]

    LOGGER.info("batch_discovery", total_found=len(all_files), already_completed=len(done_success), pending=len(to_process))

    # If nothing to do, just export existing results
    if not to_process:
        csv_path, json_path = export_results(existing_records, output_path, args.format)
        LOGGER.info("no_pending_files_refresh", msg="No pending files. Export refreshed from checkpoint.")
        if args.format in {"csv", "both"}:
            LOGGER.info("wrote_file", path=str(csv_path), format="csv")
        if args.format in {"json", "both"}:
            LOGGER.info("wrote_file", path=str(json_path), format="json")
        return 0

    # Process files concurrently
    tracker = CostTracker()
    run_records: List[Dict[str, object]] = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_one_pdf,
                pdf_path=pdf_path,
                client=client,
                config=config,
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

                    # Write to checkpoint immediately for progress tracking
                    cp_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    cp_file.flush()
                    try:
                        import os as os_module
                        os_module.fsync(cp_file.fileno())
                    except OSError:
                        pass

                    # Update tracker
                    tracker.add(
                        int(record.get("prompt_tokens", 0) or 0),
                        int(record.get("completion_tokens", 0) or 0),
                        int(record.get("total_tokens", 0) or 0),
                        float(record.get("api_cost_usd", 0.0) or 0.0),
                    )

                    # Update progress display
                    progress.advance(task_id, 1)
                    status = str(record.get("status"))
                    progress.update(task_id, description=f"last={status} cost_usd={tracker.snapshot()['total_cost_usd']:.4f}")
            except KeyboardInterrupt:
                # Handle Ctrl+C gracefully
                interrupted = True
                LOGGER.warning("batch_interrupted", completed=len(run_records), pending=len(futures) - len(run_records))
                for f in futures:
                    f.cancel()
            finally:
                # Always flush checkpoint
                try:
                    cp_file.flush()
                    import os as os_module
                    os_module.fsync(cp_file.fileno())
                except OSError:
                    pass

    # Export combined results
    all_records = existing_records + run_records
    csv_path, json_path = export_results(all_records, output_path, args.format)

    # Calculate summary statistics
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
        return 130  # Standard UNIX exit code for Ctrl+C

    return 0 if error_count == 0 else 2


def build_parser() -> argparse.ArgumentParser:
    """
    Construct the CLI argument parser with all supported commands and options.
    
    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="LegalEase CLI",
        description="CLI for single and batch processing of legal judgment PDFs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Global options
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output for debugging. Changes logging level from INFO to DEBUG.",
    )

    # Common arguments shared between commands
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

    # process_batch alias
    p_batch_alias = subparsers.add_parser(
        "process_batch",
        parents=[common],
        help="Alias for 'batch' command (reuses same implementation).",
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
    Main entry point for the CLI application.
    
    Responsibilities:
    1. Parse command-line arguments
    2. Configure logging
    3. Delegate to command handler
    4. Handle exceptions and return proper exit codes
    
    Args:
        argv: Command-line arguments (default: sys.argv[1:])
        
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    try:
        configure_logging(level=log_level)
        LOGGER.debug("logging_initialized", level=logging.getLevelName(log_level))
    except Exception as e:
        logging.basicConfig(level=log_level)
        LOGGER.warning("logging_config_fallback", error=str(e))

    # Initialize semaphore with user-specified concurrency
    reinitialize_semaphore(args.concurrency)
    LOGGER.debug("semaphore_initialized", concurrency=args.concurrency)

    # Validate workers
    if getattr(args, "workers", 1) < 1:
        LOGGER.error("validation_error", detail="--workers must be >= 1")
        raise CLIError("--workers must be >= 1")

    try:
        # Route to command handler
        return args.func(args)
    except CLIError as exc:
        LOGGER.error("cli_error", error=str(exc))
        return 2
    except Exception as exc:
        LOGGER.exception("unexpected_error", error=str(exc))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
