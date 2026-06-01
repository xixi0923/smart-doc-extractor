"""
Text region detection module.
Detects individual text regions (lines/words) in document images
using MSER and contour analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from config import DetectionConfig
from src.preprocessing.image_enhancer import TextRegion
from src.utils.logger import get_logger

logger = get_logger("detection")


@dataclass
class DetectionResult:
    """Result of text detection on a single image."""

    regions: List[TextRegion] = field(default_factory=list)
    detection_time_ms: float = 0.0
    image_size: Tuple[int, int] = (0, 0)
    method: str = ""

    @property
    def total_regions(self) -> int:
        return len(self.regions)

    def filter_by_confidence(self, min_conf: float = 0.5) -> List[TextRegion]:
        return [r for r in self.regions if r.confidence >= min_conf]


class TextDetector:
    """
    Detects text regions in document images.

    Uses a combination of:
    - MSER (Maximally Stable Extremal Regions)
    - Contour analysis
    - Multi-scale detection
    - Non-maximum suppression (NMS)

    The detector is designed to work on preprocessed (binary or enhanced)
    document images for optimal text region localization.
    """

    def __init__(self, config: Optional[DetectionConfig] = None):
        self.config = config or DetectionConfig()

    def detect(self, image: np.ndarray) -> DetectionResult:
        """
        Detect text regions in the input image.

        Args:
            image: Preprocessed document image (grayscale or binary).

        Returns:
            DetectionResult with all detected text regions.
        """
        import time

        t_start = time.time()
        h, w = image.shape[:2]
        result = DetectionResult(image_size=(h, w), method=self.config.method)

        logger.info(
            f"Detecting text regions (method: {self.config.method}, "
            f"image: {w}x{h})"
        )

        method = self.config.method.lower()

        if method == "mser_contour":
            regions = self._detect_mser_contour(image)
        elif method == "contour_only":
            regions = self._detect_contours(image)
        elif method == "mser_only":
            regions = self._detect_mser(image)
        else:
            logger.warning(f"Unknown method '{method}', using mser_contour")
            regions = self._detect_mser_contour(image)

        # Filter by size constraints
        regions = self._filter_by_size(regions)

        # Apply non-maximum suppression
        regions = self._nms(regions, self.config.nms_threshold)

        # Sort by position (top-to-bottom, left-to-right)
        regions.sort(key=lambda r: (r.y, r.x))

        result.regions = regions
        result.detection_time_ms = (time.time() - t_start) * 1000

        logger.info(
            f"Detected {len(regions)} text regions "
            f"in {result.detection_time_ms:.1f}ms"
        )

        return result

    def _detect_mser_contour(
        self, image: np.ndarray
    ) -> List[TextRegion]:
        """Combined MSER + contour detection for robust text localization."""
        mser_regions = self._detect_mser(image)
        contour_regions = self._detect_contours(image)

        # Merge results (deduplication happens in NMS)
        all_regions = mser_regions + contour_regions
        return all_regions

    def _detect_mser(self, image: np.ndarray) -> List[TextRegion]:
        """Detect text using MSER algorithm."""
        regions = []

        # MSER works on grayscale
        gray = image.copy() if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        mser = cv2.MSER_create()
        mser.setMinArea(self.config.min_region_area)
        mser.setMaxArea(self.config.max_region_area)

        try:
            bboxes, _ = mser.detectRegions(gray)
        except cv2.error:
            logger.debug("MSER detection failed (likely empty image)")
            return regions

        if bboxes is None:
            return regions

        for bbox in bboxes:
            x, y, bw, bh = self._int_rect(bbox)
            if bw > 0 and bh > 0:
                ar = bw / max(bh, 1)
                if self.config.min_aspect_ratio <= ar <= self.config.max_aspect_ratio:
                    regions.append(TextRegion(
                        x=x, y=y, width=bw, height=bh,
                        confidence=0.7,
                    ))

        logger.debug(f"MSER found {len(regions)} candidate regions")
        return regions

    def _detect_contours(
        self, image: np.ndarray
    ) -> List[TextRegion]:
        """Detect text using contour analysis on binary image."""
        regions = []

        # Ensure binary
        if len(image.shape) == 2:
            _, binary = cv2.threshold(
                image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            _, binary = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

        # Invert (text is dark on light background)
        inverted = cv2.bitwise_not(binary)

        # Dilate to connect nearby characters into text lines
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (30, 5)
        )
        dilated = cv2.dilate(inverted, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.config.min_region_area:
                continue
            if area > self.config.max_region_area * 5:
                continue

            x, y, bw, bh = cv2.boundingRect(contour)
            ar = bw / max(bh, 1)

            if self.config.min_aspect_ratio <= ar <= self.config.max_aspect_ratio:
                regions.append(TextRegion(
                    x=x, y=y, width=bw, height=bh,
                    confidence=0.6,
                ))

        logger.debug(f"Contour analysis found {len(regions)} candidate regions")
        return regions

    def _filter_by_size(self, regions: List[TextRegion]) -> List[TextRegion]:
        """Filter regions by area and aspect ratio constraints."""
        filtered = []
        for r in regions:
            if self.config.min_region_area <= r.area <= self.config.max_region_area:
                ar = r.aspect_ratio
                if self.config.min_aspect_ratio <= ar <= self.config.max_aspect_ratio:
                    filtered.append(r)
        return filtered

    def _nms(
        self,
        regions: List[TextRegion],
        threshold: float = 0.3,
    ) -> List[TextRegion]:
        """
        Non-maximum suppression to remove overlapping regions.

        Args:
            regions: List of text regions to suppress.
            threshold: IoU threshold for suppression.

        Returns:
            Suppressed list of non-overlapping regions.
        """
        if not regions:
            return []

        # Sort by confidence (highest first)
        sorted_regions = sorted(
            regions, key=lambda r: r.confidence, reverse=True
        )

        keep = []
        while sorted_regions:
            best = sorted_regions.pop(0)
            keep.append(best)

            remaining = []
            for region in sorted_regions:
                iou = self._compute_iou(best, region)
                if iou < threshold:
                    remaining.append(region)

            sorted_regions = remaining

        logger.debug(f"NMS: {len(regions)} -> {len(keep)} regions")
        return keep

    @staticmethod
    def _compute_iou(a: TextRegion, b: TextRegion) -> float:
        """Compute Intersection over Union between two regions."""
        ax1, ay1 = a.x, a.y
        ax2, ay2 = a.x + a.width, a.y + a.height
        bx1, by1 = b.x, b.y
        bx2, by2 = b.x + b.width, b.y + b.height

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = a.area + b.area - intersection

        if union <= 0:
            return 0.0

        return intersection / union

    @staticmethod
    def _int_rect(points: np.ndarray) -> Tuple[int, int, int, int]:
        """Convert float bounding box points to int tuple."""
        x, y, w, h = cv2.boundingRect(points.reshape(-1, 1, 2))
        return int(x), int(y), int(w), int(h)
