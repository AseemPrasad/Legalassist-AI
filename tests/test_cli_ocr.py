from pathlib import Path
from unittest.mock import MagicMock

import cli


def _mock_client() -> MagicMock:
    client = MagicMock()
    first = MagicMock()
    first.choices = [MagicMock(message=MagicMock(content="- point 1\n- point 2\n- point 3"))]
    first.usage = MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    second = MagicMock()
    second.choices = [MagicMock(message=MagicMock(content="1. Plaintiff won\n2. Yes\n3. 30\n4. High Court\n5. 5000\n6. File appeal\n7. 30 days"))]
    second.usage = MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    client.chat.completions.create.side_effect = [first, second]
    return client


def test_parser_accepts_enable_ocr_flags():
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "process",
            "--file",
            "judgment.pdf",
            "--enable-ocr",
            "--ocr-languages",
            "eng+hin",
            "--ocr-dpi",
            "350",
        ]
    )
    assert args.enable_ocr is True
    assert args.ocr_languages == "eng+hin"
    assert args.ocr_dpi == 350


def test_process_one_pdf_uses_diagnostics(monkeypatch):
    mock_core = MagicMock()
    mock_core.extract_text_with_diagnostics.return_value = {
        "text": "Sample judgment text",
        "method": "ocr_tesseract",
        "ocr_used": True,
        "confidence": 88.2,
    }
    monkeypatch.setattr(cli, "core", mock_core)

    result = cli.process_one_pdf(
        pdf_path=Path("sample.pdf"),
        client=_mock_client(),
        language_arg="English",
        model="test-model",
        max_chars=5000,
        prompt_cost_per_1k=0.0,
        completion_cost_per_1k=0.0,
        enable_ocr=True,
        ocr_languages="eng+hin",
        ocr_dpi=300,
    )

    assert result["status"] == "success"
    assert result["extraction_method"] == "ocr_tesseract"
    assert result["ocr_used"] is True
    assert result["ocr_enabled"] is True
    assert result["extraction_confidence"] == "88.2"
    mock_core.extract_text_with_diagnostics.assert_called_once()
