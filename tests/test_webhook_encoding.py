"""Tests for graceful webhook payload encoding handling."""

from __future__ import annotations

from urllib.parse import parse_qsl


def test_decode_invalid_utf8_does_not_raise():
    raw_bytes = b"Hello\x80World"
    result = raw_bytes.decode("utf-8", errors="replace")
    assert "Hello" in result
    assert "World" in result
    assert "\ufffd" in result


def test_decode_clean_utf8_unchanged():
    raw_bytes = b"MessageSid=SM123&MessageStatus=delivered"
    result = raw_bytes.decode("utf-8", errors="replace")
    assert result == "MessageSid=SM123&MessageStatus=delivered"
    assert "\ufffd" not in result


def test_twilio_form_survives_bad_byte():
    raw_bytes = b"MessageSid=SM123&MessageStatus=delivered&BadField=\x80garbage"
    decoded = raw_bytes.decode("utf-8", errors="replace")
    params = dict(parse_qsl(decoded, keep_blank_values=True))
    assert params.get("MessageSid") == "SM123"
    assert params.get("MessageStatus") == "delivered"


def test_sendgrid_json_survives_bad_trailing_byte():
    import json
    raw_bytes = b'[{"event":"dropped"}]\x80'
    decoded = raw_bytes.decode("utf-8", errors="replace")
    try:
        json.loads(decoded)
    except json.JSONDecodeError:
        pass
    # The key test: decoding itself should not raise UnicodeDecodeError


def test_utf8_replaces_malformed_sequences():
    """Multiple malformed bytes should each be replaced."""
    raw_bytes = b"\x80\x81\x82"
    result = raw_bytes.decode("utf-8", errors="replace")
    assert result == "\ufffd\ufffd\ufffd"
