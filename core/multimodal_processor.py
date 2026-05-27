"""
Multi-Modal Document Processor
Unified interface for processing PDFs, images, and scanned documents
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from PIL import Image
import io

from core.image_processor import ImageProcessor
from core.ocr_engine import OCREngine
from pypdf import PdfReader

logger = logging.getLogger(__name__)


class MultiModalProcessor:
    """Unified processor for multi-modal document analysis"""
    
    def __init__(self, tesseract_path: Optional[str] = None):
        """Initialize multi-modal processor
        
        Args:
            tesseract_path: Custom path to Tesseract executable
        """
        self.image_processor = ImageProcessor()
        self.ocr_engine = OCREngine(tesseract_path)
        logger.info("MultiModalProcessor initialized")
    
    def process_document(
        self,
        file_data: Union[bytes, str, Path],
        file_type: Optional[str] = None,
        languages: List[str] = ['eng'],
        enable_ocr: bool = True,
        aggressive_ocr: bool = False
    ) -> Dict[str, Any]:
        """Process any document type (PDF, image, scanned PDF)
        
        Args:
            file_data: File bytes, path, or Path object
            file_type: Explicit file type ('pdf', 'image', 'scanned_pdf')
            languages: List of language codes for OCR
            enable_ocr: Enable OCR for scanned documents
            aggressive_ocr: Use aggressive OCR preprocessing
            
        Returns:
            Dictionary with extracted text and metadata
        """
        # Determine file type if not provided
        if file_type is None:
            file_type = self._detect_file_type(file_data)
        
        logger.info(f"Processing document as type: {file_type}")
        
        # Route to appropriate processor
        if file_type == 'pdf':
            return self._process_pdf(file_data, languages, enable_ocr, aggressive_ocr)
        elif file_type == 'image':
            return self._process_image(file_data, languages, aggressive_ocr)
        elif file_type == 'scanned_pdf':
            return self._process_scanned_pdf(file_data, languages, aggressive_ocr)
        else:
            logger.error(f"Unsupported file type: {file_type}")
            return {
                'text': '',
                'success': False,
                'error': f'Unsupported file type: {file_type}',
                'file_type': file_type
            }
    
    def _detect_file_type(self, file_data: Union[bytes, str, Path]) -> str:
        """Detect file type from data or extension
        
        Args:
            file_data: File bytes, path, or Path object
            
        Returns:
            Detected file type
        """
        if isinstance(file_data, (str, Path)):
            file_path = Path(file_data)
            extension = file_path.suffix.lower()
            
            if extension == '.pdf':
                return 'pdf'
            elif extension in ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp']:
                return 'image'
            else:
                return 'unknown'
        
        elif isinstance(file_data, bytes):
            # Try to detect from magic bytes
            if file_data.startswith(b'%PDF'):
                return 'pdf'
            elif file_data.startswith(b'\x89PNG'):
                return 'image'
            elif file_data.startswith(b'\xff\xd8\xff'):
                return 'image'  # JPEG
            else:
                return 'unknown'
        
        return 'unknown'
    
    def _process_pdf(
        self,
        file_data: Union[bytes, str, Path],
        languages: List[str],
        enable_ocr: bool,
        aggressive_ocr: bool
    ) -> Dict[str, Any]:
        """Process standard PDF with extractable text
        
        Args:
            file_data: PDF file data or path
            languages: Language codes for OCR if needed
            enable_ocr: Enable OCR fallback
            aggressive_ocr: Use aggressive OCR
            
        Returns:
            Dictionary with extracted text and metadata
        """
        try:
            # Load PDF
            if isinstance(file_data, (str, Path)):
                with open(file_data, 'rb') as f:
                    pdf_reader = PdfReader(f)
            else:
                pdf_reader = PdfReader(io.BytesIO(file_data))
            
            # Extract text from all pages
            text_pages = []
            total_pages = len(pdf_reader.pages)
            
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    text_pages.append(page_text)
                elif enable_ocr:
                    # Fallback to OCR for this page
                    logger.info(f"Page {page_num + 1} has no extractable text, using OCR")
                    page_image = self._pdf_page_to_image(page)
                    if page_image is not None:
                        ocr_result = self.ocr_engine.extract_text(
                            page_image,
                            languages,
                            preprocess=aggressive_ocr
                        )
                        if ocr_result['success']:
                            text_pages.append(ocr_result['text'])
            
            full_text = '\n\n'.join(text_pages)
            
            # Check if extraction was successful
            if len(full_text.strip()) < 50 and enable_ocr:
                logger.warning("PDF has little extractable text, treating as scanned")
                return self._process_scanned_pdf(file_data, languages, aggressive_ocr)
            
            return {
                'text': full_text,
                'success': True,
                'file_type': 'pdf',
                'pages_processed': len(text_pages),
                'total_pages': total_pages,
                'ocr_used': False,
                'confidence': 100.0  # High confidence for extractable text
            }
            
        except Exception as e:
            logger.error(f"PDF processing failed: {str(e)}")
            if enable_ocr:
                logger.info("Falling back to OCR processing")
                return self._process_scanned_pdf(file_data, languages, aggressive_ocr)
            return {
                'text': '',
                'success': False,
                'error': str(e),
                'file_type': 'pdf'
            }
    
    def _process_scanned_pdf(
        self,
        file_data: Union[bytes, str, Path],
        languages: List[str],
        aggressive_ocr: bool
    ) -> Dict[str, Any]:
        """Process scanned PDF using OCR
        
        Args:
            file_data: PDF file data or path
            languages: Language codes for OCR
            aggressive_ocr: Use aggressive OCR preprocessing
            
        Returns:
            Dictionary with extracted text and metadata
        """
        try:
            # Convert PDF to images
            from pdf2image import convert_from_bytes
            
            if isinstance(file_data, (str, Path)):
                with open(file_data, 'rb') as f:
                    pdf_bytes = f.read()
            else:
                pdf_bytes = file_data
            
            # Convert PDF pages to images
            images = convert_from_bytes(pdf_bytes, dpi=300)
            
            # Process each page with OCR
            text_pages = []
            confidences = []
            
            for i, image in enumerate(images):
                # Convert PIL image to numpy array
                image_array = np.array(image)
                
                # Extract text using OCR
                ocr_result = self.ocr_engine.extract_text(
                    image_array,
                    languages,
                    preprocess=aggressive_ocr
                )
                
                if ocr_result['success']:
                    text_pages.append(ocr_result['text'])
                    confidences.append(ocr_result['confidence']['overall'])
                else:
                    logger.warning(f"OCR failed for page {i + 1}")
            
            full_text = '\n\n'.join(text_pages)
            avg_confidence = np.mean(confidences) if confidences else 0
            
            return {
                'text': full_text,
                'success': True,
                'file_type': 'scanned_pdf',
                'pages_processed': len(text_pages),
                'total_pages': len(images),
                'ocr_used': True,
                'confidence': avg_confidence,
                'languages_used': languages
            }
            
        except ImportError:
            logger.error("pdf2image not installed. Install with: pip install pdf2image")
            return {
                'text': '',
                'success': False,
                'error': 'pdf2image not installed. Install with: pip install pdf2image',
                'file_type': 'scanned_pdf'
            }
        except Exception as e:
            logger.error(f"Scanned PDF processing failed: {str(e)}")
            return {
                'text': '',
                'success': False,
                'error': str(e),
                'file_type': 'scanned_pdf'
            }
    
    def _process_image(
        self,
        file_data: Union[bytes, str, Path],
        languages: List[str],
        aggressive_ocr: bool
    ) -> Dict[str, Any]:
        """Process image file with OCR
        
        Args:
            file_data: Image file data or path
            languages: Language codes for OCR
            aggressive_ocr: Use aggressive OCR preprocessing
            
        Returns:
            Dictionary with extracted text and metadata
        """
        try:
            # Load image
            if isinstance(file_data, (str, Path)):
                image = self.image_processor.load_image(str(file_data))
            else:
                image = self.image_processor.load_image_from_bytes(file_data)
            
            if image is None:
                return {
                    'text': '',
                    'success': False,
                    'error': 'Failed to load image',
                    'file_type': 'image'
                }
            
            # Assess image quality
            quality_metrics = self.image_processor.assess_image_quality(image)
            
            # Determine if aggressive preprocessing is needed
            use_aggressive = aggressive_ocr or not quality_metrics['ocr_suitable']
            
            # Extract text with OCR
            ocr_result = self.ocr_engine.extract_text(
                image,
                languages,
                preprocess=True,
                config=None
            )
            
            # Add quality metrics to result
            ocr_result['file_type'] = 'image'
            ocr_result['quality_metrics'] = quality_metrics
            ocr_result['aggressive_preprocessing_used'] = use_aggressive
            
            return ocr_result
            
        except Exception as e:
            logger.error(f"Image processing failed: {str(e)}")
            return {
                'text': '',
                'success': False,
                'error': str(e),
                'file_type': 'image'
            }
    
    def _pdf_page_to_image(self, page, dpi: int = 300) -> Optional[np.ndarray]:
        """Convert a single PDF page to a numpy image array for OCR.

        Uses pdf2image (which requires Poppler) to rasterize the page.
        Falls back gracefully with a logged error if either dependency is
        missing.

        Args:
            page: pypdf Page object (used only to obtain the page index and
                  the parent PdfReader so we can re-render via pdf2image).
            dpi:  Rasterization resolution.  300 DPI is the OCR sweet-spot.

        Returns:
            Image as a numpy array (BGR, uint8) or None on failure.
        """
        try:
            from pdf2image import convert_from_bytes
            from pdf2image.exceptions import PDFInfoNotInstalledError, PDFPageCountError
        except ImportError:
            logger.error(
                "pdf2image is not installed. "
                "Install it with: pip install pdf2image  "
                "(Poppler must also be available on PATH)"
            )
            return None

        try:
            # Reconstruct the raw bytes for just this page so we can hand
            # them to pdf2image without needing the original file handle.
            import io as _io
            from pypdf import PdfWriter

            writer = PdfWriter()
            writer.add_page(page)
            buf = _io.BytesIO()
            writer.write(buf)
            single_page_bytes = buf.getvalue()

            images = convert_from_bytes(single_page_bytes, dpi=dpi)
            if not images:
                logger.warning("pdf2image returned no images for page")
                return None

            pil_image = images[0]
            return np.array(pil_image)

        except (PDFInfoNotInstalledError, PDFPageCountError) as exc:
            logger.error(
                "Poppler is not installed or not on PATH — required for PDF "
                "rasterization. Install Poppler and ensure it is accessible. "
                "Error: %s", exc
            )
            return None
        except Exception as exc:
            logger.error("_pdf_page_to_image failed: %s", exc)
            return None
    
    def extract_handwriting(
        self,
        file_data: Union[bytes, str, Path],
        languages: List[str] = ['eng']
    ) -> Dict[str, Any]:
        """Extract handwritten text from image
        
        Args:
            file_data: Image file data or path
            languages: Language codes for OCR
            
        Returns:
            Dictionary with extracted handwriting and confidence
        """
        try:
            # Load image
            if isinstance(file_data, (str, Path)):
                image = self.image_processor.load_image(str(file_data))
            else:
                image = self.image_processor.load_image_from_bytes(file_data)
            
            if image is None:
                return {
                    'text': '',
                    'success': False,
                    'error': 'Failed to load image',
                    'is_handwriting': True
                }
            
            # Extract handwriting
            result = self.ocr_engine.extract_handwriting(image, languages)
            result['file_type'] = 'image'
            
            return result
            
        except Exception as e:
            logger.error(f"Handwriting extraction failed: {str(e)}")
            return {
                'text': '',
                'success': False,
                'error': str(e),
                'is_handwriting': True
            }
    
    def batch_process(
        self,
        file_paths: List[Union[str, Path]],
        languages: List[str] = ['eng'],
        enable_ocr: bool = True,
        callback=None
    ) -> List[Dict[str, Any]]:
        """Process multiple files in batch
        
        Args:
            file_paths: List of file paths
            languages: Language codes for OCR
            enable_ocr: Enable OCR for scanned documents
            callback: Optional callback for progress updates
            
        Returns:
            List of processing results
        """
        results = []
        total = len(file_paths)
        
        for i, file_path in enumerate(file_paths):
            try:
                result = self.process_document(
                    file_path,
                    languages=languages,
                    enable_ocr=enable_ocr
                )
                result['file'] = str(file_path)
                results.append(result)
                
                if callback:
                    callback(i + 1, total, result)
                    
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {str(e)}")
                results.append({
                    'file': str(file_path),
                    'success': False,
                    'error': str(e)
                })
        
        return results
    
    def get_processing_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary of batch processing results
        
        Args:
            results: List of processing results
            
        Returns:
            Summary statistics
        """
        total = len(results)
        successful = sum(1 for r in results if r.get('success', False))
        failed = total - successful
        
        file_types = {}
        for r in results:
            ft = r.get('file_type', 'unknown')
            file_types[ft] = file_types.get(ft, 0) + 1
        
        ocr_used = sum(1 for r in results if r.get('ocr_used', False))
        
        avg_confidence = 0
        confidences = [r.get('confidence', 0) for r in results if r.get('confidence')]
        if confidences:
            avg_confidence = np.mean(confidences)
        
        return {
            'total_files': total,
            'successful': successful,
            'failed': failed,
            'success_rate': (successful / total * 100) if total > 0 else 0,
            'file_types': file_types,
            'ocr_used_count': ocr_used,
            'average_confidence': avg_confidence
        }


# Convenience function for quick document processing
def process_document(
    file_data: Union[bytes, str, Path],
    languages: List[str] = ['eng'],
    enable_ocr: bool = True
) -> str:
    """Quick document processing
    
    Args:
        file_data: File data or path
        languages: Language codes for OCR
        enable_ocr: Enable OCR for scanned documents
        
    Returns:
        Extracted text
    """
    processor = MultiModalProcessor()
    result = processor.process_document(file_data, languages, enable_ocr)
    return result.get('text', '')
