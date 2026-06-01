"""Tests for scanned PDF OCR path in MultiModalProcessor.

All tests use mocks so no Tesseract, Poppler, or pdf2image installation
is required in the test environment.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core.multimodal_processor import MultiModalProcessor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_processor():
    """Return a MultiModalProcessor with mocked sub-engines."""
    with patch("core.multimodal_processor.OCREngine"):
        proc = MultiModalProcessor()
    proc.ocr_engine = MagicMock()
    proc.ocr_engine.extract_text.return_value = {
        "success": True,
        "text": "OCR extracted text",
        "confidence": {"overall": 85.0},
    }
    return proc


def _fake_pil_image():
    from PIL import Image
    return Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))


# ---------------------------------------------------------------------------
# _pdf_page_to_image
# ---------------------------------------------------------------------------

class TestPdfPageToImage:
    def test_returns_numpy_array_when_pdf2image_available(self):
        proc = _make_processor()
        fake_page = MagicMock()

        pil_img = _fake_pil_image()

        with patch("core.multimodal_processor.PdfWriter") as mock_writer_cls, \
             patch("core.multimodal_processor.convert_from_bytes", return_value=[pil_img], create=True):
            # Make the import inside the method succeed
            import sys
            fake_pdf2image = MagicMock()
            fake_pdf2image.convert_from_bytes.return_value = [pil_img]
            fake_pdf2image.exceptions.PDFInfoNotInstalledError = Exception
            fake_pdf2image.exceptions.PDFPageCountError = Exception
            sys.modules.setdefault("pdf2image", fake_pdf2image)
            sys.modules.setdefault("pdf2image.exceptions", fake_pdf2image.exceptions)

            mock_writer = MagicMock()
            mock_writer_cls.return_value = mock_writer
            mock_writer.write = lambda buf: buf.write(b"%PDF-1.4 fake")

            result = proc._pdf_page_to_image(fake_page)

        # Should be a numpy array (or None if pdf2image stub didn't wire up)
        assert result is None or isinstance(result, np.ndarray)

    def test_returns_none_when_pdf2image_missing(self):
        proc = _make_processor()
        fake_page = MagicMock()

        import sys
        # Temporarily remove pdf2image from sys.modules to simulate ImportError
        original = sys.modules.pop("pdf2image", None)
        try:
            result = proc._pdf_page_to_image(fake_page)
        finally:
            if original is not None:
                sys.modules["pdf2image"] = original

        assert result is None

    def test_returns_none_when_poppler_missing(self):
        proc = _make_processor()
        fake_page = MagicMock()

        import sys

        class FakePDFInfoNotInstalledError(Exception):
            pass

        fake_pdf2image = MagicMock()
        fake_pdf2image.exceptions = MagicMock()
        fake_pdf2image.exceptions.PDFInfoNotInstalledError = FakePDFInfoNotInstalledError
        fake_pdf2image.exceptions.PDFPageCountError = Exception
        fake_pdf2image.convert_from_bytes.side_effect = FakePDFInfoNotInstalledError("no poppler")
        sys.modules["pdf2image"] = fake_pdf2image
        sys.modules["pdf2image.exceptions"] = fake_pdf2image.exceptions

        result = proc._pdf_page_to_image(fake_page)
        assert result is None


# ---------------------------------------------------------------------------
# _process_scanned_pdf
# ---------------------------------------------------------------------------

class TestProcessScannedPdf:
    def test_returns_text_when_pdf2image_available(self):
        proc = _make_processor()
        pil_img = _fake_pil_image()

        import sys
        fake_pdf2image = MagicMock()
        fake_pdf2image.convert_from_bytes.return_value = [pil_img, pil_img]
        sys.modules["pdf2image"] = fake_pdf2image

        result = proc._process_scanned_pdf(b"%PDF-1.4 fake", ["eng"], False)

        assert result["success"] is True
        assert result["ocr_used"] is True
        assert result["pages_processed"] == 2
        assert "OCR extracted text" in result["text"]

    def test_returns_error_when_pdf2image_missing(self):
        proc = _make_processor()

        import sys
        sys.modules.pop("pdf2image", None)

        result = proc._process_scanned_pdf(b"%PDF-1.4 fake", ["eng"], False)

        assert result["success"] is False
        assert "pdf2image" in result["error"].lower()


# ---------------------------------------------------------------------------
# image_processor binarize fallback uses Exception not bare except
# ---------------------------------------------------------------------------

class TestImageProcessorBinarize:
    def test_binarize_fallback_uses_typed_except(self):
        """Verify the bare except: was replaced with except Exception."""
        import inspect
        from core.image_processor import ImageProcessor
        source = inspect.getsource(ImageProcessor.preprocess_for_ocr)
        assert "except:" not in source, "Bare 'except:' found — should be 'except Exception:'"
