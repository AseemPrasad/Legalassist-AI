"""Tests for the unified speech transcription pipeline.

All tests use a mocked OpenAI client so no network calls are made.
The suite verifies:
- Valid audio bytes are forwarded to Whisper and the text is returned.
- Empty bytes raise TranscriptionInvalidAudio (400 territory).
- Provider failures raise TranscriptionProviderUnavailable (503 territory).
- The REST API endpoint maps exceptions to the correct HTTP status codes.
- The legacy TranscriptionEngine shim still works.
- core.audio_utils.transcribe_audio delegates to the unified pipeline.
"""
from __future__ import annotations

import base64
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from core.speech_transcription import (
    TranscriptionEngine,
    TranscriptionInvalidAudio,
    TranscriptionProviderUnavailable,
    transcribe_audio,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client(text: str = "Hello from Whisper") -> MagicMock:
    """Return a mock OpenAI client whose Whisper endpoint returns *text*."""
    client = MagicMock()
    client.audio.transcriptions.create.return_value = text
    return client


# ---------------------------------------------------------------------------
# core.speech_transcription.transcribe_audio
# ---------------------------------------------------------------------------

class TestTranscribeAudio:
    def test_returns_transcription_for_valid_audio(self):
        client = _make_client("Dictated note")
        result = transcribe_audio(b"\x00\x01\x02\x03", client=client)
        assert result == "Dictated note"

    def test_passes_language_to_provider(self):
        client = _make_client("Bonjour")
        transcribe_audio(b"\x00\x01", language="fr", client=client)
        call_kwargs = client.audio.transcriptions.create.call_args.kwargs
        assert call_kwargs.get("language") == "fr"

    def test_omits_language_when_none(self):
        client = _make_client("Auto-detected")
        transcribe_audio(b"\x00\x01", language=None, client=client)
        call_kwargs = client.audio.transcriptions.create.call_args.kwargs
        assert "language" not in call_kwargs

    def test_strips_whitespace_from_result(self):
        client = _make_client("  trimmed  ")
        result = transcribe_audio(b"\x00\x01", client=client)
        assert result == "trimmed"

    def test_raises_invalid_audio_for_empty_bytes(self):
        client = _make_client()
        with pytest.raises(TranscriptionInvalidAudio):
            transcribe_audio(b"", client=client)

    def test_raises_provider_unavailable_on_api_error(self):
        client = MagicMock()
        client.audio.transcriptions.create.side_effect = RuntimeError("connection refused")
        with pytest.raises(TranscriptionProviderUnavailable):
            transcribe_audio(b"\x00\x01", client=client)

    def test_raises_provider_unavailable_when_no_key_configured(self):
        """When no client is supplied and no API key is set, raise 503."""
        with patch("core.speech_transcription._build_client") as mock_build:
            mock_build.side_effect = TranscriptionProviderUnavailable("no key")
            with pytest.raises(TranscriptionProviderUnavailable):
                transcribe_audio(b"\x00\x01")

    def test_accepts_dict_response_from_sdk(self):
        """Some SDK versions return a dict instead of a plain string."""
        client = MagicMock()
        client.audio.transcriptions.create.return_value = {"text": "dict response"}
        result = transcribe_audio(b"\x00\x01", client=client)
        assert result == "dict response"

    def test_accepts_object_with_text_attribute(self):
        client = MagicMock()
        client.audio.transcriptions.create.return_value = SimpleNamespace(text="object response")
        result = transcribe_audio(b"\x00\x01", client=client)
        assert result == "object response"


# ---------------------------------------------------------------------------
# Legacy TranscriptionEngine shim
# ---------------------------------------------------------------------------

class TestTranscriptionEngine:
    def test_transcribe_bytes_returns_text(self):
        client = _make_client("Engine result")
        engine = TranscriptionEngine()
        result = engine.transcribe_bytes(b"\x00\x01", client=client)
        assert result == "Engine result"

    def test_transcribe_bytes_raises_on_empty(self):
        engine = TranscriptionEngine()
        with pytest.raises(TranscriptionInvalidAudio):
            engine.transcribe_bytes(b"", client=_make_client())

    def test_transcribe_bytes_raises_on_provider_failure(self):
        client = MagicMock()
        client.audio.transcriptions.create.side_effect = Exception("boom")
        engine = TranscriptionEngine()
        with pytest.raises(TranscriptionProviderUnavailable):
            engine.transcribe_bytes(b"\x00\x01", client=client)


# ---------------------------------------------------------------------------
# REST API endpoint
# ---------------------------------------------------------------------------

class TestSpeechAPIEndpoint:
    """Integration-style tests for POST /api/transcribe."""

    def _make_app(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from api.routes.speech import router
        from api.auth import get_current_user

        app = FastAPI()
        app.include_router(router)
        # Bypass auth for these tests
        app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id="test-user")
        return TestClient(app)

    def _b64(self, data: bytes) -> str:
        return base64.b64encode(data).decode()

    def test_returns_200_with_transcription(self):
        tc = self._make_app()
        with patch("api.routes.speech.transcribe_audio", return_value="Hello world"):
            resp = tc.post("/api/transcribe", json={"audio_base64": self._b64(b"\x00\x01"), "language": "en"})
        assert resp.status_code == 200
        assert resp.json() == {"transcription": "Hello world"}

    def test_returns_400_for_invalid_audio(self):
        tc = self._make_app()
        with patch("api.routes.speech.transcribe_audio", side_effect=TranscriptionInvalidAudio("empty")):
            resp = tc.post("/api/transcribe", json={"audio_base64": self._b64(b""), "language": "en"})
        assert resp.status_code == 400

    def test_returns_503_when_provider_unavailable(self):
        tc = self._make_app()
        with patch("api.routes.speech.transcribe_audio", side_effect=TranscriptionProviderUnavailable("no key")):
            resp = tc.post("/api/transcribe", json={"audio_base64": self._b64(b"\x00\x01"), "language": "en"})
        assert resp.status_code == 503

    def test_returns_400_for_bad_base64(self):
        tc = self._make_app()
        resp = tc.post("/api/transcribe", json={"audio_base64": "!!!not-base64!!!", "language": "en"})
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# core.audio_utils.transcribe_audio delegation
# ---------------------------------------------------------------------------

class TestAudioUtilsDelegation:
    def test_delegates_to_unified_pipeline(self):
        """audio_utils.transcribe_audio must call core.speech_transcription.transcribe_audio."""
        from core.audio_utils import transcribe_audio as au_transcribe

        mock_client = _make_client("Delegated result")
        with patch("core.speech_transcription.transcribe_audio", return_value="Delegated result") as mock_fn:
            result = au_transcribe(b"\x00\x01", client=mock_client)
        mock_fn.assert_called_once()
        assert result == "Delegated result"

    def test_returns_empty_string_on_failure(self):
        """audio_utils should swallow exceptions and return empty string."""
        from core.audio_utils import transcribe_audio as au_transcribe

        with patch("core.speech_transcription.transcribe_audio", side_effect=TranscriptionProviderUnavailable("no key")):
            result = au_transcribe(b"\x00\x01", client=_make_client())
        assert result == ""

    def test_returns_empty_string_for_empty_bytes(self):
        from core.audio_utils import transcribe_audio as au_transcribe

        result = au_transcribe(b"")
        assert result == ""
