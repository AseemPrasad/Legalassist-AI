"""
Tests for API input validation and request size limits
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import UploadFile, status
from io import BytesIO

from api.validation import (
    ValidationConfig,
    ValidationError,
    PayloadTooLargeError,
    validate_file_upload,
    validate_text_input,
    validate_batch_size,
    validate_query_string,
    validate_json_payload,
)


class TestValidationConfig:
    """Test validation configuration"""
    
    def test_default_config_values(self):
        """Test default config values are set correctly"""
        assert ValidationConfig.MAX_UPLOAD_SIZE == 500 * 1024 * 1024
        assert ValidationConfig.MAX_TEXT_LENGTH == 10 * 1024 * 1024
        assert ".pdf" in ValidationConfig.ALLOWED_EXTENSIONS
        assert "application/pdf" in ValidationConfig.ALLOWED_MIME_TYPES
    
    def test_from_settings(self):
        """Test loading config from settings object"""
        mock_settings = MagicMock()
        mock_settings.UPLOAD_MAX_SIZE = 100 * 1024 * 1024
        mock_settings.UPLOAD_EXTENSIONS = [".pdf", ".txt"]
        
        result = ValidationConfig.from_settings(mock_settings)
        
        assert result.MAX_UPLOAD_SIZE == 100 * 1024 * 1024
        assert ".pdf" in result.ALLOWED_EXTENSIONS


class TestFileUploadValidation:
    """Test file upload validation"""
    
    def test_validate_file_upload_success(self):
        """Test successful file validation"""
        file = MagicMock(spec=UploadFile)
        file.filename = "document.pdf"
        file.content_type = "application/pdf"
        file.size = 1024 * 1024  # 1 MB
        
        # Should not raise
        validate_file_upload(file)
    
    def test_validate_file_upload_invalid_mime_type(self):
        """Test file with invalid MIME type"""
        file = MagicMock(spec=UploadFile)
        file.filename = "malware.exe"
        file.content_type = "application/x-msdownload"
        file.size = 1024
        
        with pytest.raises(ValidationError) as exc_info:
            validate_file_upload(file)
        
        assert "not allowed" in str(exc_info.value.detail).lower()
    
    def test_validate_file_upload_invalid_extension(self):
        """Test file with invalid extension"""
        file = MagicMock(spec=UploadFile)
        file.filename = "document.xyz"
        file.content_type = "application/pdf"
        file.size = 1024
        
        with pytest.raises(ValidationError) as exc_info:
            validate_file_upload(file)
        
        assert "not allowed" in str(exc_info.value.detail).lower()
    
    def test_validate_file_upload_exceeds_size(self):
        """Test file exceeding max size"""
        file = MagicMock(spec=UploadFile)
        file.filename = "large.pdf"
        file.content_type = "application/pdf"
        file.size = 1000 * 1024 * 1024  # 1000 MB
        
        with pytest.raises(PayloadTooLargeError) as exc_info:
            validate_file_upload(file, max_size=500 * 1024 * 1024)
        
        assert "exceeds maximum" in str(exc_info.value.detail).lower()
        assert exc_info.value.status_code == status.HTTP_413_CONTENT_TOO_LARGE
    
    def test_validate_file_upload_custom_limits(self):
        """Test file validation with custom limits"""
        file = MagicMock(spec=UploadFile)
        file.filename = "doc.txt"
        file.content_type = "text/plain"
        file.size = 100 * 1024  # 100 KB
        
        # Should succeed with custom extensions
        validate_file_upload(
            file,
            max_size=1000 * 1024,
            allowed_extensions={".txt"},
            allowed_mime_types={"text/plain"}
        )


class TestTextValidation:
    """Test text input validation"""
    
    def test_validate_text_input_success(self):
        """Test successful text validation"""
        text = "This is a valid legal document text."
        validate_text_input(text)
    
    def test_validate_text_input_exceeds_limit(self):
        """Test text input exceeding max length"""
        text = "A" * (11 * 1024 * 1024)  # 11 MB
        
        with pytest.raises(PayloadTooLargeError) as exc_info:
            validate_text_input(text, max_length=10 * 1024 * 1024)
        
        assert "exceeds maximum" in str(exc_info.value.detail).lower()
    
    def test_validate_text_input_utf8_encoding(self):
        """Test text validation with UTF-8 characters"""
        text = "हिंदी text " * 1000  # Repeated Hindi text
        text_bytes = len(text.encode("utf-8"))
        
        # Should fail if exceeds limit
        with pytest.raises(PayloadTooLargeError):
            validate_text_input(text, max_length=text_bytes - 100)
        
        # Should pass if within limit
        validate_text_input(text, max_length=text_bytes + 1000)


class TestBatchValidation:
    """Test batch size validation"""
    
    def test_validate_batch_size_success(self):
        """Test valid batch size"""
        items = [{"id": i} for i in range(10)]
        validate_batch_size(items, max_items=100)
    
    def test_validate_batch_size_exceeds_limit(self):
        """Test batch exceeding max size"""
        items = [{"id": i} for i in range(150)]
        
        with pytest.raises(ValidationError) as exc_info:
            validate_batch_size(items, max_items=100)
        
        assert "exceeds maximum" in str(exc_info.value.detail).lower()
    
    def test_validate_batch_size_default_limit(self):
        """Test batch validation with default limit"""
        items = [{"id": i} for i in range(ValidationConfig.MAX_BATCH_SIZE + 10)]
        
        with pytest.raises(ValidationError):
            validate_batch_size(items)


class TestQueryStringValidation:
    """Test query string validation"""
    
    def test_validate_query_string_success(self):
        """Test valid query string"""
        query = "?case_id=123&year=2025"
        validate_query_string(query, max_length=2048)
    
    def test_validate_query_string_exceeds_limit(self):
        """Test query string exceeding max length"""
        query = "?q=" + ("A" * 3000)
        
        with pytest.raises(ValidationError) as exc_info:
            validate_query_string(query, max_length=2048)
        
        assert "too long" in str(exc_info.value.detail).lower()


class TestJsonPayloadValidation:
    """Test JSON payload validation"""
    
    def test_validate_json_payload_success(self):
        """Test valid JSON payload size"""
        payload_size = 5 * 1024 * 1024  # 5 MB
        validate_json_payload(payload_size, max_size=10 * 1024 * 1024)
    
    def test_validate_json_payload_exceeds_limit(self):
        """Test JSON payload exceeding max size"""
        payload_size = 15 * 1024 * 1024  # 15 MB
        
        with pytest.raises(PayloadTooLargeError) as exc_info:
            validate_json_payload(payload_size, max_size=10 * 1024 * 1024)
        
        assert "exceeds maximum" in str(exc_info.value.detail).lower()


@pytest.mark.asyncio
async def test_validate_file_upload_streaming():
    """Test file size validation during streaming"""
    from api.validation import validate_file_upload_streaming
    
    # Create mock async file
    file = AsyncMock(spec=UploadFile)
    file.filename = "test.pdf"
    
    # Simulate reading in 1MB chunks, total 5 MB
    chunk_data = b"X" * (1024 * 1024)
    read_calls = [chunk_data, chunk_data, chunk_data, chunk_data, chunk_data, b""]
    file.read.side_effect = read_calls
    file.seek = AsyncMock()
    
    # Should succeed
    bytes_read = await validate_file_upload_streaming(file, max_size=10 * 1024 * 1024)
    assert bytes_read == 5 * 1024 * 1024


@pytest.mark.asyncio
async def test_validate_file_upload_streaming_exceeds_limit():
    """Test file streaming validation when exceeding limit"""
    from api.validation import validate_file_upload_streaming
    
    file = AsyncMock(spec=UploadFile)
    file.filename = "toolarge.pdf"
    
    # Simulate reading chunks that exceed limit
    chunk_data = b"X" * (10 * 1024 * 1024)
    file.read.side_effect = [chunk_data, chunk_data, b""]
    file.seek = AsyncMock()
    
    # Should raise PayloadTooLargeError
    with pytest.raises(PayloadTooLargeError) as exc_info:
        await validate_file_upload_streaming(file, max_size=15 * 1024 * 1024)
    
    assert "exceeded" in str(exc_info.value.detail).lower()
