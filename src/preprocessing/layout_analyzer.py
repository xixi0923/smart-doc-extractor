"""
Layout analysis module.
Analyzes document structure: text lines, header/body/footer separation,
table region detection, seal/stamp detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.preprocessing.image_enhancer import TextRegion
from src.utils.logger import get_logger

logger = get_logger("layout_analyzer")


class RegionType(str, Enum):
    """Classification of document regions."""

    TITLE = "title"
    HEADER = "header"
    BODY = "body"
    TABLE = "table"
    SEAL = "seal"
    FOOTER = "footer"
    SIDEBAR = "sidebar"
    UNKNOWN = "unknown"


@dataclass
class LayoutRegion:
    """A region identified during layout analysis."""

    x: int
    y: int
    width: int
    height: int
    region_type: RegionType
    confidence: float = 1.0
    content_lines: int = 0

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def area(self) -> int:
        return self.width * self.height


class LayoutAnalyzer:
    """
    Analyzes document layout structure.

    Detects:
    - Title region (large text at top)
    - Header / Footer (top/bottom margins)
    - Table regions (grid patterns)
    - Seal / Stamp regions (red/circular)
    - Body text areas
    """

    def __init__(self, min_table_lines: int = 3):
        """
        Args:
            min_table_lines: Minimum horizontal lines to detect a table.
        """
        self.min_table_lines = min_table_lines

    def analyze(self, image: np.ndarray) -> List[LayoutRegion]:
        """
        Analyze the full document layout.

        Args:
            image: Preprocessed (binary) document image.

        Returns:
            List of LayoutRegion objects describing document structure.
        """
        h, w = image.shape[:2]
        regions = []

        logger.info(f"Analyzing layout (image size: {w}x{h})")

        # Detect each region type
        title = self._detect_title(image)
        if title:
            regions.append(title)

        header = self._detect_header(image)
        if header:
            regions.append(header)

        footer = self._detect_footer(image)
        if footer:
            regions.append(footer)

        tables = self._detect_tables(image)
        regions.extend(tables)

        seals = self._detect_seals(image)
        regions.extend(seals)

        logger.info(f"Layout analysis found {len(regions)} regions")
        return regions

    def _detect_title(self, image: np.ndarray) -> Optional[LayoutRegion]:
        """
        Detect title region — typically large text in the upper portion.
        Looks for a dense text block in the top 20% of the document.
        """
        h, w = image.shape[:2]
        top_region = image[:h // 5, :]

        # Find text contours in top region
        contours = self._find_text_contours(top_region)
        if not contours:
            return None

        # Find the bounding rect of all contours
        all_points = np.vstack(contours)
        x, y, bw, bh = cv2.boundingRect(all_points)

        # Title usually spans wide
        if bw < w * 0.3:
            return None

        return LayoutRegion(
            x=x, y=y, width=bw, height=bh,
            region_type=RegionType.TITLE,
            confidence=0.8,
        )

    def _detect_header(self, image: np.ndarray) -> Optional[LayoutRegion]:
        """Detect header region in the top margin."""
        h, w = image.shape[:2]
        margin = max(h // 20, 20)

        top_strip = image[:margin, :]
        white_ratio = np.sum(top_strip == 255) / top_strip.size

        # If there's significant content in the top strip
        if white_ratio < 0.95:
            return LayoutRegion(
                x=0, y=0, width=w, height=margin,
                region_type=RegionType.HEADER,
                confidence=0.6,
            )
        return None

    def _detect_footer(self, image: np.ndarray) -> Optional[LayoutRegion]:
        """Detect footer region in the bottom margin."""
        h, w = image.shape[:2]
        margin = max(h // 20, 20)

        bottom_strip = image[h - margin:, :]
        white_ratio = np.sum(bottom_strip == 255) / bottom_strip.size

        if white_ratio < 0.95:
            return LayoutRegion(
                x=0, y=h - margin, width=w, height=margin,
                region_type=RegionType.FOOTER,
                confidence=0.6,
            )
        return None

    def _detect_tables(self, image: np.ndarray) -> List[LayoutRegion]:
        """
        Detect table regions by finding grid patterns.
        Tables are characterized by intersecting horizontal and vertical lines.
        """
        h, w = image.shape[:2]
        regions = []

        # Detect horizontal lines
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 3, 1))
        horiz_lines = cv2.morphologyEx(
            cv2.bitwise_not(image), cv2.MORPH_OPEN, horiz_kernel, iterations=2
        )

        # Detect vertical lines
        vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 3))
        vert_lines = cv2.morphologyEx(
            cv2.bitwise_not(image), cv2.MORPH_OPEN, vert_kernel, iterations=2
        )

        # Intersection = table structure
        table_mask = cv2.bitwise_and(horiz_lines, vert_lines)

        contours, _ = cv2.findContours(
            table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1000:
                continue

            x, y, bw, bh = cv2.boundingRect(contour)

            # Table should have reasonable aspect ratio
            if bw < w * 0.1 or bh < h * 0.05:
                continue

            regions.append(LayoutRegion(
                x=x, y=y, width=bw, height=bh,
                region_type=RegionType.TABLE,
                confidence=0.7,
            ))

        logger.debug(f"Detected {len(regions)} table region(s)")
        return regions

    def _detect_seals(self, image: np.ndarray) -> List[LayoutRegion]:
        """
        Detect seal/stamp regions.
        Seals are typically red, circular, and located in specific positions.
        Works on RGB images.
        """
        regions = []

        # This works best on the original RGB image, not binary.
        # If we only have binary, skip seal detection.
        if len(image.shape) == 2:
            logger.debug("Binary image provided, skipping seal detection")
            return regions

        # Convert to HSV and detect red regions
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Red hue in HSV (wraps around, need two ranges)
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 70, 50])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(
            red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 200 or area > 50000:
                continue

            x, y, bw, bh = cv2.boundingRect(contour)

            # Seals are roughly square/circular
            aspect = bw / max(bh, 1)
            if 0.5 < aspect < 2.0:
                regions.append(LayoutRegion(
                    x=x, y=y, width=bw, height=bh,
                    region_type=RegionType.SEAL,
                    confidence=0.8,
                ))

        logger.debug(f"Detected {len(regions)} seal region(s)")
        return regions

    def _find_text_contours(self, image: np.ndarray) -> List[np.ndarray]:
        """Find contours representing text regions in a binary image."""
        inverted = cv2.bitwise_not(image)
        contours, _ = cv2.findContours(
            inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return contours

    def classify_region(
        self,
        region: LayoutRegion,
        image_shape: Tuple[int, int],
    ) -> RegionType:
        """
        Classify an unknown region based on position and size heuristics.

        Args:
            region: Region to classify.
            image_shape: (height, width) of the full document.

        Returns:
            Classified RegionType.
        """
        h, w = image_shape
        cx, cy = region.center

        # Position-based heuristics
        if cy < h * 0.15:
            return RegionType.HEADER
        if cy > h * 0.85:
            return RegionType.FOOTER

        # Size-based heuristics
        aspect = region.width / max(region.height, 1)
        area_ratio = region.area / (w * h)

        if aspect < 0.2 or aspect > 5.0:
            return RegionType.SIDEBAR

        if area_ratio > 0.3:
            return RegionType.TABLE

        return RegionType.BODY
