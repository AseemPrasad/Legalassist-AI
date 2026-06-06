from core.log_redaction import sanitize_log_text, sanitize_log_value


def test_sanitize_aadhaar():
    text = "My Aadhaar number is 1234 5678 9012 and PAN is ABCDE1234F."
    redacted = sanitize_log_text(text)
    
    assert "1234 5678 9012" not in redacted
    assert "ABCDE1234F" not in redacted
    assert "[redacted-aadhaar]" in redacted
    assert "[redacted-pan]" in redacted


def test_sanitize_sensitive_keys():
    res = sanitize_log_value("1234-5678-9012", key="aadhaar")
    assert res == "[redacted]"
    
    res2 = sanitize_log_value("ABCDE1234F", key="pan_card")
    assert res2 == "[redacted]"
