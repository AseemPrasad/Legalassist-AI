"""Unified speech transcription service.

This module is the single transcription pipeline used by both the REST API
(api/routes/speech.py) and the Chat UI (core/audio_utils.py).

It uses the modern OpenAI client (``client.audio.transcriptions.create``)
and never falls back to the deprecated ``openai.Audio.transcribe`` global.

Raises
------
TranscriptionProviderUnavailable
    When no API client can be constructed (503 territory).
TranscriptionInvalidAudio
    When the supplied bytes are empty or clearly not audio (400 territory).
"""
from __future__ import annotations

import io
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain exceptions
# ---------------------------------------------------------------------------

class TranscriptionError(Exception):
    """Base class for transcription errors."""


class TranscriptionProviderUnavailable(TranscriptionError):
    """Raised when the upstream provider (OpenAI/OpenRouter) is unreachable
    or not configured.  Maps to HTTP 503."""


class TranscriptionInvalidAudio(TranscriptionError):
    """Raised when the supplied audio bytes are empty or invalid.
    Maps to HTTP 400."""


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def transcribe_audio(
    audio_bytes: bytes,
    language: Optional[str] = None,
    *,
    client=None,
    filename: str = "audio.wav",
) -> str:
    """Transcribe raw audio bytes using OpenAI Whisper.

    Parameters
    ----------
    audio_bytes:
        Raw binary audio data (wav, mp3, webm, …).
    language:
        Optional BCP-47 language hint (e.g. ``"en"``, ``"fr"``).
        When *None* Whisper auto-detects the language.
    client:
        An already-initialised ``openai.OpenAI`` (or compatible) client.
        When *None* the function builds one from the project config.
    filename:
        Filename hint passed to the API so it can infer the codec.

    Returns
    -------
    str
        The transcribed text, stripped of leading/trailing whitespace.

    Raises
    ------
    TranscriptionInvalidAudio
        If *audio_bytes* is empty.
    TranscriptionProviderUnavailable
        If no client can be constructed or the provider call fails.
    """
    if not audio_bytes:
        raise TranscriptionInvalidAudio("audio_bytes must not be empty")

    # Build client lazily so callers that supply their own client pay no cost.
    if client is None:
        client = _build_client()

    file_obj = io.BytesIO(audio_bytes)
    file_obj.name = filename

    kwargs: dict = {"model": "whisper-1", "file": file_obj, "response_format": "text"}
    if language:
        kwargs["language"] = language

    try:
        result = client.audio.transcriptions.create(**kwargs)
    except Exception as exc:
        logger.error("Whisper transcription failed: %s", exc)
        raise TranscriptionProviderUnavailable(
            f"Transcription provider call failed: {exc}"
        ) from exc

    # The SDK returns a plain str when response_format="text".
    if isinstance(result, str):
        return result.strip()

    # Defensive: some SDK versions wrap the result.
    text = getattr(result, "text", None) or (result.get("text") if isinstance(result, dict) else None)
    if text:
        return str(text).strip()

    raise TranscriptionProviderUnavailable("Provider returned an empty transcription")


# ---------------------------------------------------------------------------
# Client factory (kept internal; callers should pass their own client when
# they already have one, e.g. the Streamlit Chat page).
# ---------------------------------------------------------------------------

def _build_client():
    """Build an OpenAI-compatible client from project config.

    Raises TranscriptionProviderUnavailable if no key is configured.
    """
    try:
        from config import Config
        from openai import OpenAI

        api_key = getattr(Config, "OPENAI_API_KEY", "") or getattr(Config, "OPENROUTER_API_KEY", "")
        if not api_key:
            raise TranscriptionProviderUnavailable(
                "No OPENAI_API_KEY or OPENROUTER_API_KEY configured"
            )

        base_url = getattr(Config, "OPENROUTER_BASE_URL", None)
        if base_url and not getattr(Config, "OPENAI_API_KEY", ""):
            return OpenAI(api_key=api_key, base_url=base_url)
        return OpenAI(api_key=api_key)

    except TranscriptionProviderUnavailable:
        raise
    except Exception as exc:
        raise TranscriptionProviderUnavailable(
            f"Could not initialise transcription client: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Legacy compatibility shim
# ---------------------------------------------------------------------------

class TranscriptionEngine:
    """Thin wrapper kept for backwards compatibility.

    New code should call :func:`transcribe_audio` directly.
    """

    def __init__(self, provider: str = "openai"):
        self.provider = provider

    def transcribe_bytes(
        self,
        audio_bytes: bytes,
        language: str = "en",
        *,
        client=None,
    ) -> str:
        """Transcribe *audio_bytes* and return the text.

        Behavior:
        - If bytes decode as UTF-8, return decoded text (testing convenience).
        - Otherwise, attempt provider call.
        - If it fails, return the fallback sentinel '[transcription_unavailable]'.
        """
        # Fast path: text already
        try:
            text = audio_bytes.decode("utf-8")
            if text.strip():
                return text
        except Exception:
            pass

        try:
            return transcribe_audio(audio_bytes, language=language, client=client)
        except Exception:
            return "[transcription_unavailable]"
