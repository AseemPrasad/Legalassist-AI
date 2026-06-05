import sys
from unittest.mock import MagicMock

# Create a dummy exception class to mock HTTP/Request exceptions
class DummyException(Exception):
    pass

# Dynamic Mocking layer for missing third-party dependencies
class DummyModule(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        if name in ('RequestException', 'Timeout', 'ConnectionError', 'HTTPError'):
            return DummyException
        return MagicMock()

# Inject dummy modules into sys.modules before importing anything from core
for module_name in [
    'openai', 'pypdf', 'langdetect', 'pdfplumber', 'config',
    'opentelemetry', 'opentelemetry.trace', 'fpdf', 'streamlit',
    'fpdf.enums', 'streamlit.components.v1', 'pytesseract', 'pdf2image',
    'pytz', 'requests', 'httpx', 'backoff', 'celery', 'apscheduler',
    'sqlalchemy', 'alembic', 'redis', 'prometheus_client', 'jaeger_client',
    'requests.exceptions'
]:
    if module_name == 'requests.exceptions':
        req_exc = DummyModule()
        req_exc.RequestException = DummyException
        req_exc.Timeout = DummyException
        req_exc.ConnectionError = DummyException
        req_exc.HTTPError = DummyException
        sys.modules[module_name] = req_exc
    else:
        sys.modules[module_name] = DummyModule()

import pytest
import numpy as np
import cv2
from unittest.mock import patch
from core.image_processor import ImageProcessor


def test_preprocess_for_ocr_basic():
    """Test that basic preprocessing works and returns a grayscale image."""
    # Create a simple 3-channel dummy image (100x100 RGB)
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.putText(dummy_image, "Test", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Preprocess
    processed = ImageProcessor.preprocess_for_ocr(dummy_image, aggressive=False)

    # Check output properties
    assert processed is not None
    assert isinstance(processed, np.ndarray)
    # The output should be 2D (grayscale/binary)
    assert len(processed.shape) == 2


def test_binarize_image_otsu_fallback(monkeypatch):
    """Test that binarization falls back to adaptive thresholding when Otsu fails."""
    dummy_image = np.zeros((100, 100), dtype=np.uint8)
    cv2.putText(dummy_image, "Test", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)

    # Track how many times binarize_image is called and with what arguments
    calls = []
    original_binarize = ImageProcessor.binarize_image

    def mock_binarize_image(img, method='adaptive'):
        calls.append(method)
        if method == 'otsu':
            # Raise a cv2.error to trigger the fallback logic
            raise cv2.error("Simulated OpenCV threshold error")
        return original_binarize(img, method)

    # Apply mock
    monkeypatch.setattr(ImageProcessor, "binarize_image", mock_binarize_image)

    # Preprocess
    processed = ImageProcessor.preprocess_for_ocr(dummy_image, aggressive=False)

    # Verify fallback happened: otsu was tried first, then adaptive
    assert "otsu" in calls
    assert "adaptive" in calls
    assert processed is not None


def test_keyboard_interrupt_not_swallowed():
    """Ensure KeyboardInterrupt is not swallowed by the preprocessing error handler."""
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)

    with patch('core.image_processor.ImageProcessor.denoise_image', side_effect=KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            ImageProcessor.preprocess_for_ocr(dummy_image)


def test_system_exit_not_swallowed():
    """Ensure SystemExit is not swallowed by the preprocessing error handler."""
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)

    with patch('core.image_processor.ImageProcessor.denoise_image', side_effect=SystemExit):
        with pytest.raises(SystemExit):
            ImageProcessor.preprocess_for_ocr(dummy_image)
