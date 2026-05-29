"""
LegalEase CLI - Command-line interface for legal document processing.

Entry-point responsibilities:
- Argument parsing
- Logging configuration
- Routing to command handlers

Processing logic lives in focused sibling modules:
- cli_client:     API client and concurrency management
- cli_processing: PDF extraction and LLM processing
- cli_checkpoint: Checkpoint loading and deduplication
- cli_export:     Results export to CSV / JSON
- cli_config:     Shared CLIConfig dataclass
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Generator, List, Optional, Sequence

import structlog

from logging_config import configure_logging
import core
from cli_config import CLIConfig
from cli_client import CLIError, get_client, reinitialize_semaphore
from cli_processing import process_one_pdf
from cli_checkpoint import collect_completed_files, load_checkpoint
from cli_export import collect_pdf_files, export_results, print_cost_summary

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

LOGGER = structlog.get_logger(__name__)

DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", core.DEFAULT_MODEL)
SUPPORTED_LANGUAGE_HELP: str = ", ".join(["auto", *core.LANGUAGES])

# Standard UNIX exit code for SIGINT / Ctrl+C
_EXIT_INTERRUPTED = 130


# ---------------------------------------------------------------------------
# Progress display
# ---------------------------------------------------------------------------

try:
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    def _make_progress() -> Progress:
        return Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
        )

except ModuleNotFoundError:  # pragma: no cover
    from tqdm import tqdm  # type: ignore[assignment]

    class _TqdmProgress:
        """Minimal Progress-compatible shim backed by tqdm."""

        def __init__(self) -> None:
            self._bar: Optional[tqdm] = None  # type: ignore[type-arg]

        def __enter__(self) -> "_TqdmProgress":
            return self

        def __exit__(self, *_: object) -> None:
            if self._bar:
                self._bar.close()

        def add_task(self, description: str, total: int) -> int:
            self._bar = tqdm(total=total, desc=description)
            return 0

        def advance(self, _task_id: int, advance: int = 1) -> None:
            if self._bar:
                self._bar.update(advance)

        def update(self, _task_id: int, description: Optional[str] = None, **_: object) -> None:
            if self._bar and description:
                self._bar.set_description_str(description)

    def _make_progress() -> "_TqdmProgress":  # type: ignore[misc]
        return _TqdmProgress()


# ---------------------------------------------------------------------------
# Cost tracking
# ---------------------------------------------------------------------------

@dataclass
class CostTracker:
    """Accumulate token usage and API costs across concurrent processing."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def add(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        cost_usd: float,
    ) -> None:
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


# ---------------------------------------------------------------------------
# Shared command setup helpers
# ---------------------------------------------------------------------------

def _build_config(args: argparse.Namespace) -> CLIConfig:
    """Construct a CLIConfig from parsed arguments (common to all commands)."""
    return CLIConfig(
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


def _log_export_paths(fmt: str, csv_path: Path, json_path: Path) -> None:
    """Emit log entries for whichever export formats were requested."""
    if fmt in {"csv", "both"}:
        LOGGER.info("wrote_file", path=str(csv_path), format="csv")
    if fmt in {"json", "both"}:
        LOGGER.info("wrote_file", path=str(json_path), format="json")


@contextmanager
def _open_checkpoint(path: Path) -> Generator[object, None, None]:
    """
    Open *path* in append mode and guarantee an fsync on exit.

    Yields the open file object so callers can write records line by line.
    Swallows OSError from fsync (e.g. on pseudo-filesystems) but always
    flushes the stdio buffer.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        try:
            yield fh
        finally:
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass


def _fsync_record(fh: object, record: dict) -> None:  # type: ignore[type-arg]
    """Append *record* as a JSON line and sync to disk."""
    fh.write(json.dumps(record, ensure_ascii=False) + "\n")  # type: ignore[union-attr]
    fh.flush()
    try:
        os.fsync(fh.fileno())  # type: ignore[union-attr]
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def process_command(args: argparse.Namespace) -> int:
    """
    'process' command — process a single PDF file.

    Returns:
        0 on success, 1 on processing failure.
    """
    file_path = Path(args.file)
    if not file_path.exists() or file_path.suffix.lower() != ".pdf":
        raise CLIError(f"Invalid PDF file: {file_path}")

    config = _build_config(args)
    client = get_client()
    result = process_one_pdf(pdf_path=file_path, client=client, config=config)
    LOGGER.info("process_result", result=result)

    if args.output:
        csv_path, json_path = export_results([result], Path(args.output), args.format)
        _log_export_paths(args.format, csv_path, json_path)

    return 0 if result.get("status") == "success" else 1


def batch_command(args: argparse.Namespace) -> int:
    """
    'batch' / 'process_batch' command — process a folder of PDFs concurrently.

    Features:
    - Resume from checkpoint (default) or start fresh with --no-resume
    - Parallel worker threads
    - Real-time cost tracking
    - Graceful Ctrl+C handling

    Returns:
        0  all files succeeded
        2  one or more files failed
        130 interrupted by the user (SIGINT)
    """
    folder = Path(args.folder)
    if not folder.exists() or not folder.is_dir():
        raise CLIError(f"Invalid folder: {folder}")

    config = _build_config(args)
    client = get_client()

    # ── Collect files ────────────────────────────────────────────────────────
    all_files = collect_pdf_files(folder, recursive=args.recursive)
    if not all_files:
        raise CLIError(f"No PDF files found in: {folder}")

    # ── Checkpoint resolution ────────────────────────────────────────────────
    output_path = Path(args.output)
    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else output_path.with_suffix(output_path.suffix + ".checkpoint.jsonl")
    )

    # Wipe stale checkpoint *before* loading so --no-resume truly starts fresh
    if not args.resume and checkpoint_path.exists():
        checkpoint_path.unlink()

    try:
        existing_records = (
            load_checkpoint(
                checkpoint_path,
                corruption_threshold=config.checkpoint_corruption_threshold,
            )
            if args.resume
            else []
        )
    except CLIError:
        LOGGER.error("checkpoint_load_failed", path=str(checkpoint_path))
        raise

    done_paths = collect_completed_files(existing_records, status_filter="success")
    pending = [p for p in all_files if str(p.resolve()) not in done_paths]

    LOGGER.info(
        "batch_discovery",
        total_found=len(all_files),
        already_completed=len(done_paths),
        pending=len(pending),
    )

    # ── Nothing left to do ───────────────────────────────────────────────────
    if not pending:
        csv_path, json_path = export_results(existing_records, output_path, args.format)
        LOGGER.info("no_pending_files_refresh", msg="No pending files. Export refreshed from checkpoint.")
        _log_export_paths(args.format, csv_path, json_path)
        return 0

    # ── Concurrent processing ────────────────────────────────────────────────
    tracker = CostTracker()
    run_records: List[Dict[str, object]] = []
    interrupted = False

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_one_pdf, pdf_path=p, client=client, config=config): p
            for p in pending
        }

        with _make_progress() as progress, _open_checkpoint(checkpoint_path) as cp_file:
            task_id = progress.add_task("Processing PDFs", total=len(futures))
            try:
                for future in as_completed(futures):
                    record = future.result()
                    run_records.append(record)
                    _fsync_record(cp_file, record)

                    tracker.add(
                        int(record.get("prompt_tokens", 0) or 0),
                        int(record.get("completion_tokens", 0) or 0),
                        int(record.get("total_tokens", 0) or 0),
                        float(record.get("api_cost_usd", 0.0) or 0.0),
                    )

                    progress.advance(task_id)
                    cost = tracker.snapshot()["total_cost_usd"]
                    progress.update(
                        task_id,
                        description=f"last={record.get('status')}  cost=${cost:.4f}",
                    )

            except KeyboardInterrupt:
                interrupted = True
                for f in futures:
                    f.cancel()
                LOGGER.warning(
                    "batch_interrupted",
                    completed=len(run_records),
                    pending=len(futures) - len(run_records),
                )

    # ── Export ───────────────────────────────────────────────────────────────
    all_records = existing_records + run_records
    csv_path, json_path = export_results(all_records, output_path, args.format)

    success_count = sum(1 for r in run_records if r.get("status") == "success")
    error_count = len(run_records) - success_count

    if interrupted:
        LOGGER.warning(
            "batch_interrupted_export",
            processed=len(run_records),
            successful=success_count,
            failed=error_count,
            msg="Run interrupted. Partial results exported. Re-run to continue.",
        )
    else:
        LOGGER.info(
            "batch_summary",
            processed=len(run_records),
            successful=success_count,
            failed=error_count,
        )

    _log_export_paths(args.format, csv_path, json_path)
    print_cost_summary(tracker.snapshot())

    if interrupted:
        return _EXIT_INTERRUPTED
    return 0 if error_count == 0 else 2


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _add_batch_args(parser: argparse.ArgumentParser) -> None:
    """Register the arguments shared by both 'batch' and 'process_batch'."""
    parser.add_argument(
        "--folder", "--input",
        dest="folder",
        required=True,
        help="Input directory containing PDF files.",
    )
    parser.add_argument("--output", required=True, help="Base path for exported results.")
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel file-processing workers.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories when collecting PDFs.",
    )
    parser.add_argument(
        "--checkpoint",
        help="Checkpoint file path. Default: <output>.checkpoint.jsonl",
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Resume from an existing checkpoint (default).",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Ignore any existing checkpoint and start fresh.",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "json", "both"],
        default="both",
        help="Export format for batch results.",
    )
    parser.set_defaults(func=batch_command)


def build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="LegalEase CLI",
        description="Process legal judgment PDFs individually or in bulk.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Set log level to DEBUG.",
    )

    # Arguments shared across all sub-commands
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", default=DEFAULT_MODEL, help="LLM model name.")
    common.add_argument(
        "--language",
        default="auto",
        help=f"Output language for summaries: {SUPPORTED_LANGUAGE_HELP}.",
    )
    common.add_argument(
        "--max-chars",
        type=int,
        default=6000,
        help="Max PDF characters sent to the LLM per document.",
    )
    common.add_argument(
        "--prompt-cost-per-1k",
        type=float,
        default=0.0,
        help="USD cost per 1 K prompt tokens (cost reporting only).",
    )
    common.add_argument(
        "--completion-cost-per-1k",
        type=float,
        default=0.0,
        help="USD cost per 1 K completion tokens (cost reporting only).",
    )
    common.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Max concurrent API calls.",
    )
    common.add_argument(
        "--enable-ocr",
        action="store_true",
        help="Enable Tesseract OCR for scanned / image-based PDFs.",
    )
    common.add_argument(
        "--ocr-languages",
        default="eng+hin",
        help="Tesseract language codes (e.g. 'eng+hin').",
    )
    common.add_argument(
        "--ocr-dpi",
        type=int,
        default=300,
        help="DPI for PDF-to-image conversion during OCR.",
    )

    subs = parser.add_subparsers(dest="command", required=True)

    # ── process (single file) ────────────────────────────────────────────────
    p_process = subs.add_parser(
        "process",
        parents=[common],
        help="Process a single PDF file.",
    )
    p_process.add_argument("--file", required=True, help="Path to the source PDF.")
    p_process.add_argument(
        "--output",
        help="Output file path. If omitted, results are logged to stdout only.",
    )
    p_process.add_argument(
        "--format",
        choices=["csv", "json", "both"],
        default="both",
        help="Export format (only used when --output is given).",
    )
    p_process.set_defaults(func=process_command)

    # ── batch & process_batch (folder) ───────────────────────────────────────
    for name, help_text in [
        ("batch", "Process multiple PDFs from a folder."),
        ("process_batch", "Alias for 'batch'."),
    ]:
        _add_batch_args(subs.add_parser(name, parents=[common], help=help_text))

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def check_cli_directory_permissions(path: str) -> bool:
    """
    Return True if *path* is (or can be created as) a readable/writable directory.

    Call this before attempting file output or database dump operations.
    """
    target = Path(path)
    if not target.exists():
        try:
            target.mkdir(parents=True, exist_ok=True)
        except OSError:
            return False
    return os.access(target, os.W_OK | os.R_OK)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    CLI entry point.

    1. Parse arguments
    2. Configure logging
    3. Initialise the API concurrency semaphore
    4. Validate universal constraints
    5. Delegate to the appropriate command handler

    Returns:
        0  success
        2  CLI / validation error
        3  unexpected exception
        130 interrupted by the user
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # ── Logging ──────────────────────────────────────────────────────────────
    log_level = logging.DEBUG if args.verbose else logging.INFO
    try:
        configure_logging(level=log_level)
    except Exception as exc:  # pragma: no cover
        logging.basicConfig(level=log_level)
        LOGGER.warning("logging_config_fallback", error=str(exc))
    LOGGER.debug("logging_initialized", level=logging.getLevelName(log_level))

    # ── Concurrency semaphore (single call — command handlers must not repeat) ─
    reinitialize_semaphore(args.concurrency)
    LOGGER.debug("semaphore_initialized", concurrency=args.concurrency)

    # ── Universal validation ─────────────────────────────────────────────────
    if getattr(args, "workers", 1) < 1:
        LOGGER.error("validation_error", detail="--workers must be >= 1")
        raise CLIError("--workers must be >= 1")

    # ── Dispatch ─────────────────────────────────────────────────────────────
    try:
        return args.func(args)
    except CLIError as exc:
        LOGGER.error("cli_error", error=str(exc))
        return 2
    except Exception as exc:
        LOGGER.exception("unexpected_error", error=str(exc))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())