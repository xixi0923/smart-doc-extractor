"""
Image enhancement pipeline for document preprocessing.
Handles denoising, binarization, deskewing, contrast enhancement, and border removal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from config import PreprocessConfig
from src.utils.logger import get_logger

logger = get_logger("preprocessing")


@dataclass
class TextRegion:
    """Detected text region with bounding box and metadata."""

    x: int
    y: int
    width: int
    height: int
    confidence: float = 1.0
    region_type: str = "unknown"  # title | body | table | seal | header | footer

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        return self.width / max(self.height, 1)

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)


class ImageEnhancer:
    """
    Document image enhancement pipeline.

    Applies a sequence of preprocessing steps to improve
    text readability and OCR accuracy:
        1. Border removal
        2. Denoising (Gaussian + Median)
        3. CLAHE contrast enhancement
        4. Binarization (Otsu / Adaptive / Sauvola)
        5. Deskewing (skew correction)
    """

    def __init__(self, config: Optional[PreprocessConfig] = None):
        self.config = config or PreprocessConfig()

    def enhance(self, image: np.ndarray) -> np.ndarray:
        """
        Apply full enhancement pipeline to an image.

        Args:
            image: Input image (RGB or grayscale).

        Returns:
            Tuple of (binary_image, grayscale_image).
            binary_image: for text detection and layout analysis.
            grayscale_image: for OCR (OCR works better with grayscale than binary).
        """
        logger.info(f"Starting enhancement pipeline (input shape: {image.shape})")

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # Step 1: Remove borders
        if self.config.remove_borders:
            gray = self._remove_borders(gray)

        # Step 2: Denoise
        if self.config.denoise:
            gray = self._denoise(gray)

        # Step 3: CLAHE contrast enhancement
        if self.config.clahe:
            gray = self._apply_clahe(gray)

        # Save grayscale version for OCR (OCR works better with grayscale)
        grayscale_for_ocr = gray.copy()

        # Step 4: Binarize
        if self.config.binarize:
            binary = self._binarize(gray)
        else:
            binary = gray

        # Step 5: Deskew
        if self.config.deskew:
            binary = self._deskew(binary)

        logger.info("Enhancement pipeline completed")
        return binary, grayscale_for_ocr

    def _remove_borders(self, image: np.ndarray) -> np.ndarray:
        """Remove dark borders and artifacts from document edges."""
        h, w = image.shape[:2]

        # Detect border by finding the largest connected white region
        _, thresh = cv2.threshold(
            image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Morphological operations to find border region
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 50, 3), max(h // 50, 3)))
        dilated = cv2.dilate(thresh, kernel, iterations=3)

        # Find contours and keep the largest (document body)
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, bw, bh = cv2.boundingRect(largest)
            # Add small padding
            pad = 5
            x = max(0, x - pad)
            y = max(0, y - pad)
            bw = min(w - x, bw + 2 * pad)
            bh = min(h - y, bh + 2 * pad)
            image = image[y:y + bh, x:x + bw]
            logger.debug(f"Border removed: crop ({x},{y},{bw},{bh})")

        return image

    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """Apply denoising using Gaussian blur followed by median filter."""
        ksize = self.config.denoise_ksize

        # Gaussian blur
        blurred = cv2.GaussianBlur(image, (ksize, ksize), 0)

        # Median filter (good for salt-and-pepper noise)
        denoised = cv2.medianBlur(blurred, ksize)

        return denoised

    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply Contrast Limited Adaptive Histogram Equalization."""
        clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=(8, 8)
        )
        enhanced = clahe.apply(image)
        return enhanced

    def _binarize(self, image: np.ndarray) -> np.ndarray:
        """
        Binarize image using configured method.
        Supports: otsu, adaptive, sauvola.
        """
        method = self.config.binarize_method.lower()

        if method == "otsu":
            _, binary = cv2.threshold(
                image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

        elif method == "adaptive":
            block = self.config.binarize_block_size
            # Ensure block size is odd
            if block % 2 == 0:
                block += 1
            binary = cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, block, 10
            )

        elif method == "sauvola":
            # Niblack/Sauvola-like implementation
            block = self.config.binarize_block_size
            if block % 2 == 0:
                block += 1
            k = 0.34  # Sauvola parameter
            mean = cv2.GaussianBlur(image, (block, block), 0)
            mean_sq = cv2.GaussianBlur(image * image, (block, block), 0)
            std = np.sqrt(np.maximum(mean_sq - mean * mean, 0))
            threshold = mean * (1 + k * (std / 128 - 1))
            binary = np.where(image > threshold, 255, 0).astype(np.uint8)

        else:
            logger.warning(f"Unknown binarize method '{method}', using Otsu")
            _, binary = cv2.threshold(
                image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

        return binary

    def _deskew(self, image: np.ndarray) -> np.ndarray:
        """Detect and correct page skew angle."""
        # Use Hough line transform to detect skew
        min_line_length = min(image.shape[0], image.shape[1]) // 4

        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=100,
            minLineLength=min_line_length,
            maxLineGap=10,
        )

        if lines is None or len(lines) == 0:
            logger.debug("No lines detected for deskew, skipping")
            return image

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            # Normalize to [-45, 45]
            if angle < -45:
                angle += 90
            elif angle > 45:
                angle -= 90
            angles.append(angle)

        if not angles:
            return image

        median_angle = float(np.median(angles))

        if abs(median_angle) < 0.5:
            logger.debug(f"Skew angle {median_angle:.2f} is negligible, skipping")
            return image

        logger.info(f"Correcting skew angle: {median_angle:.2f} degrees")

        # Rotate image
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)

        # Compute new image size to avoid clipping
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        rotation_matrix[0, 2] += (new_w - w) / 2
        rotation_matrix[1, 2] += (new_h - h) / 2

        rotated = cv2.warpAffine(
            image, rotation_matrix, (new_w, new_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

        return rotated
