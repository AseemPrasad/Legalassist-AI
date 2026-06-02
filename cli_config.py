"""
Shared configuration dataclass for CLI processing.

This module defines CLIConfig which consolidates all processing parameters
and passes them through to processing functions, eliminating the need for
many function parameters.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class CLIConfig:
    """Configuration for CLI operations.
    
    Attributes:
        model: LLM model name (e.g., 'gpt-4', 'claude-3-opus')
        language: Target language for summaries ('auto' to auto-detect)
        max_chars: Max characters to send to LLM (prevents context overflow)
        prompt_cost_per_1k: Cost per 1K prompt tokens (USD)
        completion_cost_per_1k: Cost per 1K completion tokens (USD)
        concurrency: Max concurrent API calls allowed
        enable_ocr: Whether to enable OCR fallback for scanned PDFs
        ocr_languages: OCR language codes (e.g., 'eng+hin')
        ocr_dpi: DPI resolution for PDF-to-image conversion during OCR
        export_format: Export format ('csv', 'json', or 'both')
        checkpoint_corruption_threshold: Max corruption rate before failing (0.0-1.0)
    """
    model: str
    language: str = "auto"
    max_chars: int = 6000
    prompt_cost_per_1k: float = 0.0
    completion_cost_per_1k: float = 0.0
    concurrency: int = 5
    enable_ocr: bool = False
    ocr_languages: str = "eng+hin"
    ocr_dpi: int = 300
    export_format: str = "both"
    checkpoint_corruption_threshold: float = 0.1

    def to_dict(self):
        """Convert config to dictionary for logging/debugging."""
        return {
            "model": self.model,
            "language": self.language,
            "max_chars": self.max_chars,
            "prompt_cost_per_1k": self.prompt_cost_per_1k,
            "completion_cost_per_1k": self.completion_cost_per_1k,
            "concurrency": self.concurrency,
            "enable_ocr": self.enable_ocr,
            "ocr_languages": self.ocr_languages,
            "ocr_dpi": self.ocr_dpi,
            "export_format": self.export_format,
            "checkpoint_corruption_threshold": self.checkpoint_corruption_threshold,
        }
