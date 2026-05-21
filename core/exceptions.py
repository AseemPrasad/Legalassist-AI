"""
Custom exceptions for the LegalAssist-AI application.

This module defines a standardized hierarchy of exceptions to be used across
the application, replacing raw Exceptions and None-returns with descriptive,
catchable error types.
"""

class LegalAssistError(Exception):
    """
    Base exception class for all errors within LegalAssist-AI.
    
    Attributes:
        message -- explanation of the error
        original_exception -- the underlying exception that triggered this error, if any
    """
    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message)
        self.message = message
        self.original_exception = original_exception

    def __str__(self):
        if self.original_exception:
            return f"{self.message} (Caused by: {repr(self.original_exception)})"
        return self.message


class InputReadingError(LegalAssistError):
    """
    Raised when an error occurs while reading input data.
    
    This includes file system errors, stream errors, or invalid input formats
    that prevent the application from accessing the raw source data.
    """
    pass


class PDFProcessingError(LegalAssistError):
    """
    Raised when an error occurs during PDF parsing or text extraction.
    
    This covers failures in pdfplumber, pypdf, or other PDF-specific
    libraries used to process legal documents.
    """
    pass


class OCRDependencyError(LegalAssistError):
    """
    Raised when OCR processing is requested but required dependencies are missing.
    
    This includes missing Python packages (pytesseract, pdf2image) or missing
    system binaries (Tesseract OCR, poppler).
    """
    pass


class OCRProcessingError(LegalAssistError):
    """
    Raised when OCR execution fails or produces unreadable results.
    """
    pass


class LLMResponseParsingError(LegalAssistError):
    """
    Raised when the output from an AI model does not match the expected format.
    
    This is used when structured data (like remedy details) cannot be reliably
    extracted from the model's natural language response.
    """
    pass


class ConfigurationError(LegalAssistError):
    """
    Raised when there is an issue with the application's configuration.
    """
    pass


class ModelInferenceError(LegalAssistError):
    """
    Raised when a call to an AI model fails (e.g., timeout, API error).
    """
    pass
