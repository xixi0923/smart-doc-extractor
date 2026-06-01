"""
Region classification module.
Classifies detected text regions into semantic categories
(title, body text, table, seal, header, footer, etc.)
based on spatial and visual heuristics.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from src.preprocessing.image_enhancer import TextRegion
from src.preprocessing.layout_analyzer import LayoutRegion, LayoutAnalyzer, RegionType
from src.utils.logger import get_logger

logger = get_logger("region_classifier")


class RegionClassifier:
    """
    Classifies detected text regions by combining:
    1. Layout analysis results (if available)
    2. Position-based heuristics
    3. Size and aspect ratio features
    """

    def __init__(self, layout_analyzer: Optional[LayoutAnalyzer] = None):
        self.layout_analyzer = layout_analyzer or LayoutAnalyzer()

    def classify(
        self,
        regions: List[TextRegion],
        image_shape: Tuple[int, int],
        layout_regions: Optional[List[LayoutRegion]] = None,
    ) -> List[TextRegion]:
        """
        Classify all detected text regions.

        Args:
            regions: Detected text regions from TextDetector.
            image_shape: (height, width) of the document image.
            layout_regions: Optional pre-computed layout regions.

        Returns:
            Regions with updated region_type field.
        """
        h, w = image_shape

        for region in regions:
            # Check overlap with layout regions first
            if layout_regions:
                region.region_type = self._match_layout_region(
                    region, layout_regions
                )
            else:
                region.region_type = self._classify_by_position(
                    region, h, w
                )

        logger.debug(
            f"Classified {len(regions)} regions: "
            f"{self._count_types(regions)}"
        )

        return regions

    def _match_layout_region(
        self,
        region: TextRegion,
        layout_regions: List[LayoutRegion],
    ) -> str:
        """Match a text region against known layout regions by overlap."""
        best_type = "body"
        best_overlap = 0.0

        rx1, ry1 = region.x, region.y
        rx2, ry2 = region.x + region.width, region.y + region.height

        for lr in layout_regions:
            lx1, ly1 = lr.x, lr.y
            lx2, ly2 = lr.x + lr.width, lr.y + lr.height

            ix1 = max(rx1, lx1)
            iy1 = max(ry1, ly1)
            ix2 = min(rx2, lx2)
            iy2 = min(ry2, ly2)

            intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            overlap = intersection / max(region.area, 1)

            if overlap > best_overlap:
                best_overlap = overlap
                best_type = lr.region_type.value

        return best_type

    def _classify_by_position(
        self,
        region: TextRegion,
        img_h: int,
        img_w: int,
    ) -> str:
        """
        Classify region using position-based heuristics.

        Rules:
        - Top 15% → title or header
        - Bottom 10% → footer
        - Very wide region → title or separator
        - Normal → body
        """
        y_center = region.y + region.height / 2
        y_ratio = y_center / img_h

        # Title/Header detection
        if y_ratio < 0.15:
            # Large font = title, small = header
            if region.height > img_h * 0.03:
                return "title"
            return "header"

        # Footer detection
        if y_ratio > 0.90:
            return "footer"

        # Wide regions in top half might be titles
        if y_ratio < 0.4 and region.width > img_w * 0.5:
            return "title"

        return "body"

    @staticmethod
    def _count_types(regions: List[TextRegion]) -> dict:
        """Count regions by type."""
        counts = {}
        for r in regions:
            counts[r.region_type] = counts.get(r.region_type, 0) + 1
        return counts
