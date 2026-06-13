"""
Image Processing Module for Multi-Modal Document Processing
Handles image preprocessing, enhancement, and quality assessment
"""

import logging
import numpy as np
import cv2
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Advanced image preprocessing for OCR and document analysis"""
    
    @staticmethod
    def load_image(image_path: str) -> Optional[np.ndarray]:
        """Load image from file path
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image as numpy array or None if failed
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return None
            return image
        except Exception as e:
            logger.error(f"Error loading image: {str(e)}")
            return None
    
    @staticmethod
    def load_image_from_bytes(image_bytes: bytes) -> Optional[np.ndarray]:
        """Load image from bytes
        
        Args:
            image_bytes: Image data as bytes
            
        Returns:
            Image as numpy array or None if failed
        """
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                logger.error("Failed to decode image from bytes")
                return None
            return image
        except Exception as e:
            logger.error(f"Error loading image from bytes: {str(e)}")
            return None
    
    @staticmethod
    def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale
        
        Args:
            image: Input image
            
        Returns:
            Grayscale image
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    @staticmethod
    def denoise_image(image: np.ndarray) -> np.ndarray:
        """Remove noise from image using bilateral filter
        
        Args:
            image: Input image
            
        Returns:
            Denoised image
        """
        return cv2.bilateralFilter(image, 9, 75, 75)
    
    @staticmethod
    def enhance_contrast(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:
        """Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        
        Args:
            image: Input grayscale image
            clip_limit: Threshold for contrast limiting
            tile_grid_size: Size of grid for histogram equalization
            
        Returns:
            Contrast-enhanced image
        """
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image)
    
    @staticmethod
    def binarize_image(image: np.ndarray, method: str = 'adaptive') -> np.ndarray:
        """Convert image to binary (black and white)
        
        Args:
            image: Input grayscale image
            method: Binarization method ('adaptive' or 'otsu')
            
        Returns:
            Binary image
        """
        if method == 'adaptive':
            return cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
        elif method == 'otsu':
            _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return binary
        else:
            # Simple threshold
            _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
            return binary
    
    @staticmethod
    def deskew_image(image: np.ndarray) -> np.ndarray:
        """Correct image skew/rotation
        
        Args:
            image: Input image
            
        Returns:
            Deskewed image
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Detect edges
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Find lines
            lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
            
            if lines is not None:
                # Calculate average angle filtering outliers
                angles = []
                for line in lines:
                    rho, theta = line[0]
                    angle = np.degrees(theta)
                    # Normalize angles to identify the page skew offset from horizontal/vertical axes
                    skew = (angle - 90) % 180
                    if skew > 90:
                        skew -= 180
                    if abs(skew) < 45:
                        angles.append(skew)
                
                if angles:
                    median_angle = np.median(angles)
                    
                    # Rotate image
                    (h, w) = image.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                    rotated = cv2.warpAffine(image, M, (w, h), 
                                           flags=cv2.INTER_CUBIC, 
                                           borderMode=cv2.BORDER_REPLICATE)
                    return rotated
            
            return image
        except Exception as e:
            logger.warning(f"Deskewing failed: {str(e)}")
            return image
    
    @staticmethod
    def remove_shadows(image: np.ndarray) -> np.ndarray:
        """Remove shadows from image using morphological operations
        
        Args:
            image: Input image
            
        Returns:
            Image with shadows removed
        """
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge back
            lab = cv2.merge([l, a, b])
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            return result
        except Exception as e:
            logger.warning(f"Shadow removal failed: {str(e)}")
            return image
    
    @staticmethod
    def sharpen_image(image: np.ndarray) -> np.ndarray:
        """Sharpen image using unsharp masking
        
        Args:
            image: Input image
            
        Returns:
            Sharpened image
        """
        try:
            kernel = np.array([[-1, -1, -1],
                             [-1,  9, -1],
                             [-1, -1, -1]])
            sharpened = cv2.filter2D(image, -1, kernel)
            return sharpened
        except Exception as e:
            logger.warning(f"Sharpening failed: {str(e)}")
            return image
    
    @staticmethod
    def assess_image_quality(image: np.ndarray) -> Dict[str, Any]:
        """Assess image quality for OCR suitability
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with quality metrics
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Calculate metrics
            metrics = {}
            
            # 1. Blur detection using Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            metrics['blur_score'] = float(laplacian_var)
            metrics['is_blurry'] = laplacian_var < 100
            
            # 2. Contrast assessment
            metrics['contrast'] = float(gray.std())
            metrics['low_contrast'] = gray.std() < 50
            
            # 3. Brightness assessment
            metrics['brightness'] = float(gray.mean())
            metrics['is_dark'] = gray.mean() < 80
            metrics['is_bright'] = gray.mean() > 200
            
            # 4. Noise estimation
            metrics['noise_level'] = float(np.std(gray - cv2.GaussianBlur(gray, (5, 5), 0)))
            
            # 5. Overall quality score (0-100)
            quality_score = 100
            if metrics['is_blurry']:
                quality_score -= 30
            if metrics['low_contrast']:
                quality_score -= 20
            if metrics['is_dark'] or metrics['is_bright']:
                quality_score -= 15
            if metrics['noise_level'] > 20:
                quality_score -= 20
            
            metrics['overall_quality'] = max(0, quality_score)
            metrics['ocr_suitable'] = quality_score >= 60
            
            return metrics
        except Exception as e:
            logger.error(f"Quality assessment failed: {str(e)}")
            return {
                'blur_score': 0,
                'is_blurry': True,
                'contrast': 0,
                'low_contrast': True,
                'brightness': 0,
                'is_dark': True,
                'is_bright': False,
                'noise_level': 100,
                'overall_quality': 0,
                'ocr_suitable': False
            }
    
    @staticmethod
    def preprocess_for_ocr(image: np.ndarray, aggressive: bool = False) -> np.ndarray:
        """Complete preprocessing pipeline for OCR
        
        Args:
            image: Input image
            aggressive: Use aggressive preprocessing for poor quality images
            
        Returns:
            Preprocessed image optimized for OCR
        """
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                processed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                processed = image.copy()
            
            # Remove shadows if color image
            if len(image.shape) == 3:
                processed = cv2.cvtColor(ImageProcessor.remove_shadows(image), 
                                         cv2.COLOR_BGR2GRAY)
            
            # Denoise
            processed = ImageProcessor.denoise_image(processed)
            
            # Enhance contrast
            processed = ImageProcessor.enhance_contrast(processed)
            
            # Deskew
            processed = ImageProcessor.deskew_image(processed)
            
            # Binarize
            if aggressive:
                processed = ImageProcessor.binarize_image(processed, method='adaptive')
            else:
                # Try Otsu first, fallback to adaptive
                try:
                    processed = ImageProcessor.binarize_image(processed, method='otsu')
                except cv2.error as e:
                    logger.warning(f"Otsu binarization failed, falling back to adaptive thresholding: {str(e)}")
                    processed = ImageProcessor.binarize_image(processed, method='adaptive')
                except Exception as e:
                    logger.warning(f"Unexpected error during Otsu binarization, falling back to adaptive: {str(e)}")
                    processed = ImageProcessor.binarize_image(processed, method='adaptive')
            
            # Sharpen if aggressive
            if aggressive:
                processed = ImageProcessor.sharpen_image(processed)
            
            return processed
        except Exception as e:
            logger.error(f"OCR preprocessing failed: {str(e)}")
            return image
    
    @staticmethod
    def resize_image(image: np.ndarray, max_width: int = 2000) -> np.ndarray:
        """Resize image while maintaining aspect ratio
        
        Args:
            image: Input image
            max_width: Maximum width in pixels
            
        Returns:
            Resized image
        """
        try:
            h, w = image.shape[:2]
            if w <= max_width:
                return image
            
            ratio = max_width / w
            new_h = int(h * ratio)
            resized = cv2.resize(image, (max_width, new_h), interpolation=cv2.INTER_AREA)
            return resized
        except Exception as e:
            logger.warning(f"Image resize failed: {str(e)}")
            return image
    
    @staticmethod
    def save_image(image: np.ndarray, output_path: str) -> bool:
        """Save image to file
        
        Args:
            image: Image to save
            output_path: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cv2.imwrite(output_path, image)
            return True
        except Exception as e:
            logger.error(f"Failed to save image: {str(e)}")
            return False
