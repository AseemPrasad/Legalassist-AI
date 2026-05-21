import base64

from core.speech_transcription import TranscriptionEngine


def test_transcribe_text_bytes():
    engine = TranscriptionEngine()
    sample = "This is a dictated note."
    out = engine.transcribe_bytes(sample.encode("utf-8"))
    assert out == sample


def test_transcribe_non_text_bytes_fallback():
    engine = TranscriptionEngine()
    # random non-decodable bytes
    audio = bytes([0x00, 0xFF, 0xAA, 0x10])
    out = engine.transcribe_bytes(audio)
    assert out in ("[transcription_unavailable]",)


def test_api_transcribe_roundtrip(monkeypatch):
    # ensure API path using base64 input decodes and returns transcription
    engine = TranscriptionEngine()
    sample = "Quick note"
    b64 = base64.b64encode(sample.encode("utf-8")).decode("ascii")
    # Direct call: decode & transcribe
    audio_bytes = base64.b64decode(b64)
    out = engine.transcribe_bytes(audio_bytes)
    assert out == sample
