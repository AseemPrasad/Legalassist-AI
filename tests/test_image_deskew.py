import numpy as np
import cv2
from core.image_processor import ImageProcessor


def test_enhance_contrast():
    # Create a low-contrast dummy grayscale image
    img = np.ones((100, 100), dtype=np.uint8) * 128
    img[20:80, 20:80] = 135
    
    enhanced = ImageProcessor.enhance_contrast(img, clip_limit=3.0, tile_grid_size=(4, 4))
    assert enhanced.shape == img.shape
    # Enhanced image should have broader distribution (greater variance / standard deviation)
    assert enhanced.std() >= img.std()


def test_deskew_image_straight():
    # Straight horizontal black bar in a white background
    img = np.ones((200, 200), dtype=np.uint8) * 255
    img[90:110, :] = 0
    
    deskewed = ImageProcessor.deskew_image(img)
    # Straight image should not rotate or should remain equivalent
    assert deskewed.shape == img.shape


def test_deskew_image_skewed():
    # Slightly rotated horizontal bar
    img = np.ones((200, 200), dtype=np.uint8) * 255
    # Create diagonal line representing skew
    for i in range(200):
        j = min(199, max(0, 100 + int(i * 0.05)))
        img[j-5:j+5, i] = 0
        
    deskewed = ImageProcessor.deskew_image(img)
    assert deskewed.shape == img.shape
