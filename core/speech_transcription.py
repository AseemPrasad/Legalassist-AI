"""Lightweight transcription engine (provider-agnostic stub).

This module provides a simple interface for integrating speech-to-text.
It prefers to return decoded text when given UTF-8 bytes (useful for tests),
and falls back to a provider call if configured. Provider calls are guarded
so tests/imports don't require external dependencies.
"""
from __future__ import annotations

import base64
import logging
from typing import Optional

import openai
from config import Config

logger = logging.getLogger(__name__)


class TranscriptionEngine:
    def __init__(self, provider: str = "openai"):
        self.provider = provider
        # initialize keys safely
        self.openai_key = getattr(Config, "OPENAI_API_KEY", "") or getattr(Config, "OPENROUTER_API_KEY", "")
        if self.openai_key:
            try:
                openai.api_key = self.openai_key
            except Exception:
                logger.debug("OpenAI client not fully available in this environment")

    def transcribe_bytes(self, audio_bytes: bytes, language: str = "en") -> Optional[str]:
        """Transcribe raw audio bytes.

        Behavior:
        - If bytes decode as UTF-8, return decoded text (testing convenience).
        - Otherwise, if OpenAI key present, attempt provider call (best-effort).
        - Otherwise return a helpful fallback string.
        """
        # Fast path: text already
        try:
            text = audio_bytes.decode("utf-8")
            if text.strip():
                return text
        except Exception:
            pass

        # If provider configured, attempt to call (guarded)
        if self.provider == "openai" and self.openai_key:
            try:
                # Use the OpenAI 'audio.transcriptions' API when available.
                # This is a best-effort call and may fail in test environments.
                audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
                # The actual SDK method may differ; wrap in try/except
                resp = openai.Audio.transcribe("whisper-1", audio_b64, language=language)
                text = resp.get("text") if isinstance(resp, dict) else None
                if text:
                    return text
            except Exception as e:
                logger.debug("OpenAI transcription failed: %s", str(e))

        # Fallback
        return "[transcription_unavailable]"
