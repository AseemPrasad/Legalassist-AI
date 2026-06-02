"""
OCR Engine Module for Multi-Modal Document Processing
Handles text extraction from images and scanned documents using Tesseract
"""

import logging
import pytesseract
import numpy as np
from typing import Dict, Any, Optional
from config import Config

from core.image_processor import ImageProcessor

logger = logging.getLogger(__name__)


class OCREngine:
    """Advanced OCR engine with multi-language support and confidence scoring"""
    
    # Language codes for Tesseract
    LANGUAGE_CODES = {
        'English': 'eng',
        'Hindi': 'hin',
        'Bengali': 'ben',
        'Urdu': 'urd',
        'Sanskrit': 'san',
        'Tamil': 'tam',
        'Telugu': 'tel',
        'Marathi': 'mar',
        'Gujarati': 'guj',
        'Punjabi': 'pan',
    }
    
    def __init__(self, tesseract_path: Optional[str] = None):
        """Initialize OCR engine
        
        Args:
            tesseract_path: Custom path to tesseract executable
        """
        self.tesseract_path = tesseract_path or Config.OCR_LANGUAGES
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        # Verify Tesseract installation
        try:
            pytesseract.get_tesseract_version()
            logger.info("Tesseract OCR engine initialized successfully")
        except Exception as e:
            logger.error(f"Tesseract not found: {str(e)}")
            logger.warning("OCR functionality will be limited")
    
    def extract_text(
        self,
        image: np.ndarray,
        languages: list = ['eng'],
        preprocess: bool = True,
        config: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extract text from image using OCR
        
        Args:
            image: Input image as numpy array
            languages: List of language codes (e.g., ['eng', 'hin'])
            preprocess: Apply image preprocessing before OCR
            config: Custom Tesseract configuration
            
        Returns:
            Dictionary with extracted text and metadata
        """
        try:
            # Preprocess image if requested
            if preprocess:
                processed_image = ImageProcessor.preprocess_for_ocr(image)
            else:
                processed_image = image
            
            # Assess image quality
            quality_metrics = ImageProcessor.assess_image_quality(image)
            
            # Configure Tesseract
            if config is None:
                config = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,;:!?()[]{}"\'- '
            
            # Build language string
            lang_string = '+'.join(languages)
            
            # Perform OCR
            logger.info(f"Starting OCR with languages: {lang_string}")
            text = pytesseract.image_to_string(
                processed_image,
                lang=lang_string,
                config=config
            )
            
            # Get detailed OCR data with confidence scores
            ocr_data = pytesseract.image_to_data(
                processed_image,
                lang=lang_string,
                config=config,
                output_type=pytesseract.Output.DICT
            )
            
            # Calculate confidence metrics
            confidence_metrics = self._calculate_confidence(ocr_data)
            
            result = {
                'text': text.strip(),
                'confidence': confidence_metrics,
                'quality_metrics': quality_metrics,
                'languages_used': languages,
                'word_count': len(text.split()),
                'char_count': len(text),
                'success': True
            }
            
            logger.info(f"OCR completed. Confidence: {confidence_metrics['overall']:.2f}%")
            return result
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {str(e)}")
            return {
                'text': '',
                'confidence': {'overall': 0, 'word_level': 0},
                'quality_metrics': ImageProcessor.assess_image_quality(image),
                'languages_used': languages,
                'word_count': 0,
                'char_count': 0,
                'success': False,
                'error': str(e)
            }
    
    def extract_text_with_layout(
        self,
        image: np.ndarray,
        languages: list = ['eng']
    ) -> Dict[str, Any]:
        """Extract text with layout information (paragraphs, lines, words)
        
        Args:
            image: Input image
            languages: List of language codes
            
        Returns:
            Dictionary with structured text and layout info
        """
        try:
            processed_image = ImageProcessor.preprocess_for_ocr(image)
            lang_string = '+'.join(languages)
            
            # Get detailed OCR data
            ocr_data = pytesseract.image_to_data(
                processed_image,
                lang=lang_string,
                config='--oem 3 --psm 6',
                output_type=pytesseract.Output.DICT
            )
            
            # Structure the data
            structured_text = self._structure_ocr_data(ocr_data)
            
            return {
                'structured_text': structured_text,
                'full_text': ' '.join([w['text'] for w in structured_text['words'] if w['text'].strip()]),
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Layout extraction failed: {str(e)}")
            return {
                'structured_text': {'paragraphs': [], 'lines': [], 'words': []},
                'full_text': '',
                'success': False,
                'error': str(e)
            }
    
    def extract_handwriting(
        self,
        image: np.ndarray,
        languages: list = ['eng']
    ) -> Dict[str, Any]:
        """Extract handwritten text using specialized OCR configuration
        
        Args:
            image: Input image
            languages: List of language codes
            
        Returns:
            Dictionary with extracted handwriting and confidence
        """
        try:
            # Enhanced preprocessing for handwriting
            processed = ImageProcessor.preprocess_for_ocr(image, aggressive=True)
            
            # Handwriting-specific configuration
            config = '--oem 3 --psm 6 -c tessedit_do_invert=0'
            
            # Extract text
            result = self.extract_text(processed, languages, preprocess=False, config=config)
            
            # Add handwriting-specific metadata
            result['is_handwriting'] = True
            result['handwriting_confidence'] = result['confidence']['overall'] * 0.8  # Conservative estimate
            
            logger.info(f"Handwriting extraction completed. Confidence: {result['handwriting_confidence']:.2f}%")
            return result
            
        except Exception as e:
            logger.error(f"Handwriting extraction failed: {str(e)}")
            return {
                'text': '',
                'confidence': {'overall': 0},
                'is_handwriting': True,
                'handwriting_confidence': 0,
                'success': False,
                'error': str(e)
            }
    
    def batch_process_images(
        self,
        image_paths: list,
        languages: list = ['eng'],
        callback=None
    ) -> list:
        """Process multiple images in batch
        
        Args:
            image_paths: List of image file paths
            languages: List of language codes
            callback: Optional callback function for progress updates
            
        Returns:
            List of OCR results
        """
        results = []
        total = len(image_paths)
        
        for i, image_path in enumerate(image_paths):
            try:
                image = ImageProcessor.load_image(image_path)
                if image is None:
                    results.append({
                        'file': image_path,
                        'success': False,
                        'error': 'Failed to load image'
                    })
                    continue
                
                result = self.extract_text(image, languages)
                result['file'] = image_path
                results.append(result)
                
                if callback:
                    callback(i + 1, total, result)
                    
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {str(e)}")
                results.append({
                    'file': image_path,
                    'success': False,
                    'error': str(e)
                })
        
        return results
    
    def _calculate_confidence(self, ocr_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate confidence metrics from OCR data
        
        Args:
            ocr_data: Raw OCR data from Tesseract
            
        Returns:
            Dictionary with confidence metrics
        """
        try:
            confidences = []
            for conf in ocr_data.get('conf', []):
                try:
                    conf_val = float(conf)
                    if conf_val > 0:  # Ignore -1 (no text)
                        confidences.append(conf_val)
                except (ValueError, TypeError):
                    continue
            
            if not confidences:
                return {'overall': 0.0, 'word_level': 0.0, 'min': 0.0, 'max': 0.0}
            
            overall = np.mean(confidences)
            word_level = np.mean(confidences) if confidences else 0
            min_conf = np.min(confidences)
            max_conf = np.max(confidences)
            
            return {
                'overall': float(overall),
                'word_level': float(word_level),
                'min': float(min_conf),
                'max': float(max_conf)
            }
        except Exception as e:
            logger.error(f"Confidence calculation failed: {str(e)}")
            return {'overall': 0.0, 'word_level': 0.0, 'min': 0.0, 'max': 0.0}
    
    def _structure_ocr_data(self, ocr_data: Dict[str, Any]) -> Dict[str, Any]:
        """Structure raw OCR data into paragraphs, lines, and words
        
        Args:
            ocr_data: Raw OCR data from Tesseract
            
        Returns:
            Structured text data
        """
        try:
            words = []
            lines = {}
            paragraphs = {}
            
            for i in range(len(ocr_data.get('text', []))):
                text = ocr_data['text'][i].strip()
                if not text:
                    continue
                
                word_info = {
                    'text': text,
                    'confidence': float(ocr_data['conf'][i]) if ocr_data['conf'][i] != '-1' else 0,
                    'bbox': (
                        ocr_data['left'][i],
                        ocr_data['top'][i],
                        ocr_data['left'][i] + ocr_data['width'][i],
                        ocr_data['top'][i] + ocr_data['height'][i]
                    )
                }
                words.append(word_info)
                
                # Group by line number
                line_num = ocr_data['line_num'][i]
                if line_num not in lines:
                    lines[line_num] = []
                lines[line_num].append(word_info)
            
            # Convert lines to list
            structured_lines = []
            for line_num in sorted(lines.keys()):
                line_words = lines[line_num]
                line_text = ' '.join([w['text'] for w in line_words])
                line_conf = np.mean([w['confidence'] for w in line_words])
                
                structured_lines.append({
                    'line_number': line_num,
                    'text': line_text,
                    'confidence': float(line_conf),
                    'words': line_words
                })
            
            return {
                'words': words,
                'lines': structured_lines,
                'paragraphs': structured_lines  # Simplified: treat lines as paragraphs
            }
            
        except Exception as e:
            logger.error(f"OCR data structuring failed: {str(e)}")
            return {'words': [], 'lines': [], 'paragraphs': []}
    
    def detect_language(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect the primary language in an image
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with detected language and confidence
        """
        try:
            # Test common languages
            languages_to_test = ['eng', 'hin', 'ben', 'urd']
            results = {}
            
            for lang in languages_to_test:
                result = self.extract_text(image, [lang], preprocess=True)
                if result['success'] and result['word_count'] > 5:
                    results[lang] = {
                        'confidence': result['confidence']['overall'],
                        'word_count': result['word_count']
                    }
            
            if not results:
                return {
                    'detected_language': 'unknown',
                    'confidence': 0,
                    'alternatives': []
                }
            
            # Find best match
            best_lang = max(results.items(), key=lambda x: x[1]['confidence'])
            
            return {
                'detected_language': best_lang[0],
                'confidence': best_lang[1]['confidence'],
                'alternatives': [
                    {'language': lang, 'confidence': data['confidence']}
                    for lang, data in sorted(results.items(), key=lambda x: x[1]['confidence'], reverse=True)[1:4]
                ]
            }
            
        except Exception as e:
            logger.error(f"Language detection failed: {str(e)}")
            return {
                'detected_language': 'unknown',
                'confidence': 0,
                'alternatives': []
            }
    
    def improve_ocr_quality(
        self,
        image: np.ndarray,
        languages: list = ['eng'],
        iterations: int = 3
    ) -> Dict[str, Any]:
        """Iteratively improve OCR quality by trying different preprocessing methods
        
        Args:
            image: Input image
            languages: List of language codes
            iterations: Number of preprocessing iterations to try
            
        Returns:
            Best OCR result from all iterations
        """
        best_result = None
        best_confidence = 0
        
        preprocessing_methods = [
            lambda img: ImageProcessor.preprocess_for_ocr(img, aggressive=False),
            lambda img: ImageProcessor.preprocess_for_ocr(img, aggressive=True),
            lambda img: ImageProcessor.binarize_image(ImageProcessor.enhance_contrast(ImageProcessor.convert_to_grayscale(img)), method='adaptive'),
            lambda img: ImageProcessor.binarize_image(ImageProcessor.enhance_contrast(ImageProcessor.convert_to_grayscale(img)), method='otsu'),
        ]
        
        for i, method in enumerate(preprocessing_methods[:iterations]):
            try:
                processed = method(image)
                result = self.extract_text(processed, languages, preprocess=False)
                
                if result['success'] and result['confidence']['overall'] > best_confidence:
                    best_confidence = result['confidence']['overall']
                    best_result = result
                    best_result['preprocessing_method'] = i
                    
            except Exception as e:
                logger.warning(f"Preprocessing method {i} failed: {str(e)}")
                continue
        
        if best_result is None:
            # Fallback to basic extraction
            best_result = self.extract_text(image, languages)
            best_result['preprocessing_method'] = 'fallback'
        
        return best_result


# Convenience function for quick OCR extraction
def extract_text_from_image(
    image: np.ndarray,
    languages: list = ['eng'],
    preprocess: bool = True
) -> str:
    """Quick text extraction from image
    
    Args:
        image: Input image
        languages: List of language codes
        preprocess: Apply preprocessing
        
    Returns:
        Extracted text
    """
    ocr = OCREngine()
    result = ocr.extract_text(image, languages, preprocess)
    return result.get('text', '')


# Async non-blocking wrappers
import asyncio


async def extract_text_from_image_async(
    image: np.ndarray,
    languages: list = ['eng'],
    preprocess: bool = True
) -> str:
    """Async wrapper for quick text extraction that offloads work to a thread.

    This keeps async event-loops responsive when performing OCR using
    blocking C-extensions like Tesseract and OpenCV.
    """
    ocr = OCREngine()
    result = await asyncio.to_thread(ocr.extract_text, image, languages, preprocess)
    return result.get('text', '')


class AsyncOCREngine(OCREngine):
    """Provides async equivalents for heavy OCR operations."""

    async def extract_text_async(self, image: np.ndarray, languages: list = ['eng'], preprocess: bool = True, config: Optional[str] = None) -> Dict[str, Any]:
        return await asyncio.to_thread(self.extract_text, image, languages, preprocess, config)

    async def extract_text_with_layout_async(self, image: np.ndarray, languages: list = ['eng']) -> Dict[str, Any]:
        return await asyncio.to_thread(self.extract_text_with_layout, image, languages)

    async def extract_handwriting_async(self, image: np.ndarray, languages: list = ['eng']) -> Dict[str, Any]:
        return await asyncio.to_thread(self.extract_handwriting, image, languages)

    async def batch_process_images_async(self, image_paths: list, languages: list = ['eng'], callback=None) -> list:
        """Process multiple images concurrently without blocking the event loop.

        It uses a threadpool for CPU-bound image processing and OCR.
        """
        loop = asyncio.get_running_loop()
        tasks = []
        results = []

        async def _process(path):
            try:
                image = await loop.run_in_executor(None, ImageProcessor.load_image, path)
                if image is None:
                    return {'file': path, 'success': False, 'error': 'Failed to load image'}
                res = await asyncio.to_thread(self.extract_text, image, languages)
                res['file'] = path
                return res
            except Exception as e:
                return {'file': path, 'success': False, 'error': str(e)}

        for path in image_paths:
            tasks.append(asyncio.create_task(_process(path)))

        for coro in asyncio.as_completed(tasks):
            r = await coro
            results.append(r)
            if callback:
                try:
                    callback(len(results), len(image_paths), r)
                except Exception:
                    pass

        return results

    async def detect_language_async(self, image: np.ndarray) -> Dict[str, Any]:
        return await asyncio.to_thread(self.detect_language, image)

    async def improve_ocr_quality_async(self, image: np.ndarray, languages: list = ['eng'], iterations: int = 3) -> Dict[str, Any]:
        return await asyncio.to_thread(self.improve_ocr_quality, image, languages, iterations)
