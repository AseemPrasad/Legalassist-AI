import base64
from core.efiling import EfilingClient, clear_submissions


def setup_function():
    clear_submissions()


def test_submit_and_status_flow():
    payload = base64.b64encode(b"%PDF-1.4 fake pdf").decode("utf-8")
    res = EfilingClient.submit("SUPREME", payload, metadata={"filename": "file.pdf"})
    assert res["tracking_id"]
    assert res["status"] in {"accepted", "pending"}

    status = EfilingClient.get_status(res["tracking_id"])
    assert status["tracking_id"] == res["tracking_id"]
    assert status["status"] == "accepted"


def test_invalid_court_rejected():
    payload = base64.b64encode(b"%PDF-1.4 fake pdf").decode("utf-8")
    try:
        EfilingClient.submit("UNKNOWN", payload, metadata={"filename": "file.pdf"})
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "Unsupported court" in str(exc)


def test_invalid_format_rejected():
    payload = base64.b64encode(b"plain text").decode("utf-8")
    try:
        EfilingClient.submit("HIGH", payload, metadata={"filename": "file.txt"})
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "Invalid or unsupported document format" in str(exc)
