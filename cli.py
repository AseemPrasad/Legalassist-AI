import argparse
import csv
import json
import logging
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
from openai import OpenAI
from tqdm import tqdm
import core

# Make language detection deterministic.
DetectorFactory.seed = 0

SUPPORTED_LANGUAGES = {"english", "hindi", "bengali", "urdu"}
LANG_CODE_TO_NAME = {
    "en": "English",
    "hi": "Hindi",
    "bn": "Bengali",
    "ur": "Urdu",
}
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", core.DEFAULT_MODEL)
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
LOGGER = logging.getLogger(__name__)

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
            "Missing API key. Set OPENROUTER_API_KEY (preferred) or OPENAI_API_KEY in your environment."
        )

    base_url = os.getenv("OPENROUTER_BASE_URL", DEFAULT_BASE_URL)
    return OpenAI(api_key=api_key, base_url=base_url)


# Prompt building and core logic moved to core.py


def detect_language_name(text: str) -> str:
    if not text.strip():
        return "English"
    sample = text[:3000]
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
        return lower.capitalize()
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
):
    return client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )


def process_one_pdf(
    pdf_path: Path,
    client: OpenAI,
    language_arg: str,
    model: str,
    max_chars: int,
    prompt_cost_per_1k: float,
    completion_cost_per_1k: float,
) -> Dict[str, object]:
    started = time.time()
    processed_at = datetime.now(timezone.utc).isoformat()

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
    }

    try:
        raw_text = core.extract_text_from_pdf(pdf_path)
        if not raw_text:
            raise CLIError("No extractable text found in PDF.")

        language = normalize_language(language_arg, text_for_auto=raw_text)
        result["language"] = language

        safe_text = core.compress_text(raw_text, limit=max_chars)

        summary_prompt = core.build_summary_prompt(safe_text, language)
        resp_summary = _chat_completion(
            client=client,
            model=model,
            system_prompt="You are an expert legal simplification engine.",
            user_prompt=summary_prompt,
            max_tokens=280,
            temperature=0.05,
        )
        summary = (resp_summary.choices[0].message.content or "").strip()

        p1, c1, t1 = _usage_tokens(resp_summary)

        if language.lower() != "english" and core.english_leakage_detected(summary):
            retry_prompt = core.build_retry_prompt(safe_text, language)
            resp_retry = _chat_completion(
                client=client,
                model=model,
                system_prompt="Strict multilingual rewriting engine.",
                user_prompt=retry_prompt,
                max_tokens=260,
                temperature=0.03,
            )
            retry_summary = (resp_retry.choices[0].message.content or "").strip()
            p2, c2, t2 = _usage_tokens(resp_retry)
            p1 += p2
            c1 += c2
            t1 += t2
            if retry_summary and not core.english_leakage_detected(retry_summary):
                summary = retry_summary

        if not summary:
            raise CLIError("Model returned empty summary.")

        remedies_prompt = core.build_remedies_prompt(raw_text, language)
        resp_remedies = _chat_completion(
            client=client,
            model=model,
            system_prompt="You are a helpful legal advisor. Answer questions about legal remedies in India.",
            user_prompt=remedies_prompt,
            max_tokens=500,
            temperature=0.1,
        )
        remedies_text = (resp_remedies.choices[0].message.content or "").strip()
        remedies = core.parse_remedies_response(remedies_text)
        if remedies is None:
            LOGGER.warning(
                "process_one_pdf: remedies parsing failed for file=%s",
                pdf_path.name,
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

        p3, c3, t3 = _usage_tokens(resp_remedies)
        prompt_tokens = p1 + p3
        completion_tokens = c1 + c3
        total_tokens = t1 + t3
        cost_usd = _estimate_cost_usd(
            prompt_tokens,
            completion_tokens,
            prompt_cost_per_1k=prompt_cost_per_1k,
            completion_cost_per_1k=completion_cost_per_1k,
        )

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

    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)

    result["duration_seconds"] = round(time.time() - started, 3)
    return result


def load_checkpoint(checkpoint_file: Path) -> List[Dict[str, object]]:
    if not checkpoint_file.exists():
        return []

    records: List[Dict[str, object]] = []
    with checkpoint_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def append_checkpoint(checkpoint_file: Path, record: Dict[str, object]) -> None:
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


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
    print("\n=== Batch Cost Summary ===")
    print(f"Prompt tokens: {snapshot['prompt_tokens']}")
    print(f"Completion tokens: {snapshot['completion_tokens']}")
    print(f"Total tokens: {snapshot['total_tokens']}")
    print(f"Estimated API cost (USD): ${snapshot['total_cost_usd']:.6f}")


def process_command(args: argparse.Namespace) -> int:
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
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.output:
        out_path = Path(args.output)
        records = [result]
        csv_path, json_path = export_results(records, out_path, args.format)
        if args.format in {"csv", "both"}:
            print(f"Wrote CSV: {csv_path}")
        if args.format in {"json", "both"}:
            print(f"Wrote JSON: {json_path}")

    return 0 if result.get("status") == "success" else 1


def batch_command(args: argparse.Namespace) -> int:
    folder = Path(args.folder)
    if not folder.exists() or not folder.is_dir():
        raise CLIError(f"Invalid folder: {folder}")
    client = get_client()

    all_files = collect_pdf_files(folder, recursive=args.recursive)
    if not all_files:
        raise CLIError(f"No PDF files found in folder: {folder}")

    output_path = Path(args.output)
    checkpoint_file = Path(args.checkpoint) if args.checkpoint else output_path.with_suffix(output_path.suffix + ".checkpoint.jsonl")

    existing_records = load_checkpoint(checkpoint_file) if args.resume else []
    done_success = {
        str(rec.get("file_path"))
        for rec in existing_records
        if rec.get("status") == "success" and rec.get("file_path")
    }

    to_process = [p for p in all_files if str(p.resolve()) not in done_success]

    if not args.resume and checkpoint_file.exists():
        checkpoint_file.unlink()

    print(f"Total PDFs found: {len(all_files)}")
    print(f"Already completed (resume): {len(done_success)}")
    print(f"Pending for this run: {len(to_process)}")

    if not to_process:
        csv_path, json_path = export_results(existing_records, output_path, args.format)
        print("No pending files. Export refreshed from checkpoint.")
        if args.format in {"csv", "both"}:
            print(f"Wrote CSV: {csv_path}")
        if args.format in {"json", "both"}:
            print(f"Wrote JSON: {json_path}")
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
            ): pdf_path
            for pdf_path in to_process
        }

        progress = tqdm(total=len(futures), desc="Processing PDFs", unit="file")
        try:
            for future in as_completed(futures):
                record = future.result()
                run_records.append(record)
                append_checkpoint(checkpoint_file, record)

                tracker.add(
                    int(record.get("prompt_tokens", 0) or 0),
                    int(record.get("completion_tokens", 0) or 0),
                    int(record.get("total_tokens", 0) or 0),
                    float(record.get("api_cost_usd", 0.0) or 0.0),
                )

                progress.update(1)
                status = str(record.get("status"))
                progress.set_postfix({"last": status, "cost_usd": f"{tracker.snapshot()['total_cost_usd']:.4f}"})
        finally:
            progress.close()

    all_records = existing_records + run_records
    csv_path, json_path = export_results(all_records, output_path, args.format)

    success_count = sum(1 for x in run_records if x.get("status") == "success")
    error_count = len(run_records) - success_count

    print("\n=== Batch Processing Summary ===")
    print(f"Processed this run: {len(run_records)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {error_count}")

    if args.format in {"csv", "both"}:
        print(f"Wrote CSV: {csv_path}")
    if args.format in {"json", "both"}:
        print(f"Wrote JSON: {json_path}")

    print_cost_summary(tracker.snapshot())

    return 0 if error_count == 0 else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="LegalEase CLI",
        description="CLI for single and batch processing of legal judgment PDFs.",
    )

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", default=DEFAULT_MODEL, help="Model name for generation.")
    common.add_argument(
        "--language",
        default="auto",
        help="Output language: auto, English, Hindi, Bengali, Urdu. Default: auto",
    )
    common.add_argument(
        "--max-chars",
        type=int,
        default=6000,
        help="Max characters to send for summary prompt. Default: 6000",
    )
    common.add_argument(
        "--prompt-cost-per-1k",
        type=float,
        default=0.0,
        help="Estimated USD cost per 1K prompt tokens (for cost reporting).",
    )
    common.add_argument(
        "--completion-cost-per-1k",
        type=float,
        default=0.0,
        help="Estimated USD cost per 1K completion tokens (for cost reporting).",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    p_process = subparsers.add_parser("process", parents=[common], help="Process a single PDF file.")
    p_process.add_argument("--file", required=True, help="Path to a PDF file.")
    p_process.add_argument("--output", help="Output base path, e.g. ./result.csv")
    p_process.add_argument(
        "--format",
        choices=["csv", "json", "both"],
        default="both",
        help="Export format when --output is provided.",
    )
    p_process.set_defaults(func=process_command)

    p_batch = subparsers.add_parser("batch", parents=[common], help="Process all PDFs from a folder.")
    p_batch.add_argument("--folder", "--input", dest="folder", required=True, help="Folder containing PDFs.")
    p_batch.add_argument("--output", required=True, help="Output base path, e.g. ./results.csv")
    p_batch.add_argument("--workers", type=int, default=4, help="Parallel workers. Default: 4")
    p_batch.add_argument("--recursive", action="store_true", help="Scan folder recursively for PDFs.")
    p_batch.add_argument("--checkpoint", help="Checkpoint file path. Default: <output>.checkpoint.jsonl")
    p_batch.add_argument("--resume", dest="resume", action="store_true", default=True, help="Resume from checkpoint (default).")
    p_batch.add_argument("--no-resume", dest="resume", action="store_false", help="Start a fresh run and overwrite checkpoint.")
    p_batch.add_argument(
        "--format",
        choices=["csv", "json", "both"],
        default="both",
        help="Export format.",
    )
    p_batch.set_defaults(func=batch_command)

    # Alias for requested command style.
    p_batch_alias = subparsers.add_parser(
        "process_batch",
        parents=[common],
        help="Alias of batch command.",
    )
    p_batch_alias.add_argument("--folder", "--input", dest="folder", required=True, help="Folder containing PDFs.")
    p_batch_alias.add_argument("--output", required=True, help="Output base path, e.g. ./results.csv")
    p_batch_alias.add_argument("--workers", type=int, default=4, help="Parallel workers. Default: 4")
    p_batch_alias.add_argument("--recursive", action="store_true", help="Scan folder recursively for PDFs.")
    p_batch_alias.add_argument("--checkpoint", help="Checkpoint file path. Default: <output>.checkpoint.jsonl")
    p_batch_alias.add_argument("--resume", dest="resume", action="store_true", default=True, help="Resume from checkpoint (default).")
    p_batch_alias.add_argument("--no-resume", dest="resume", action="store_false", help="Start a fresh run and overwrite checkpoint.")
    p_batch_alias.add_argument(
        "--format",
        choices=["csv", "json", "both"],
        default="both",
        help="Export format.",
    )
    p_batch_alias.set_defaults(func=batch_command)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if getattr(args, "workers", 1) < 1:
        raise CLIError("--workers must be >= 1")

    try:
        return args.func(args)
    except CLIError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # Defensive catch to keep CLI stable.
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
