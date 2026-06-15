"""
End-to-end document extraction pipeline.
Orchestrates all stages: preprocessing → detection → OCR → extraction.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import AppConfig
from src.detection.region_classifier import RegionClassifier
from src.detection.text_detector import DetectionResult, TextDetector
from src.extraction.field_extractor import ExtractionResult, FieldExtractor
from src.preprocessing.image_enhancer import ImageEnhancer
from src.preprocessing.layout_analyzer import LayoutAnalyzer
from src.recognition.ocr_engine import OcrPageResult, create_ocr_engine
from src.recognition.text_postprocess import TextPostProcessor
from src.utils.image_utils import load_image, save_image
from src.utils.logger import get_logger

logger = get_logger("pipeline")


@dataclass
class PipelineResult:
    """Complete result of the document extraction pipeline."""

    # Input info
    source_path: str = ""
    image_size: Tuple[int, int] = (0, 0)

    # Stage results
    enhanced_image: Optional[np.ndarray] = None
    grayscale_image: Optional[np.ndarray] = None
    original_rgb: Optional[np.ndarray] = None
    detection_result: Optional[DetectionResult] = None
    ocr_result: Optional[OcrPageResult] = None
    extraction_result: Optional[ExtractionResult] = None

    # Performance metrics
    stage_timings: Dict[str, float] = field(default_factory=dict)
    total_time_ms: float = 0.0

    # Status
    success: bool = True
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        result = {
            "source": self.source_path,
            "image_size": list(self.image_size),
            "success": self.success,
            "total_time_ms": round(self.total_time_ms, 1),
            "stage_timings": {
                k: round(v, 1) for k, v in self.stage_timings.items()
            },
            "extraction": self.extraction_result.to_dict() if self.extraction_result else None,
            "ocr": {
                "full_text": self.ocr_result.full_text if self.ocr_result else "",
                "avg_confidence": round(self.ocr_result.avg_confidence, 3) if self.ocr_result else 0,
                "engine": self.ocr_result.engine_name if self.ocr_result else "",
                "region_count": len(self.ocr_result.results) if self.ocr_result else 0,
            } if self.ocr_result else None,
            "detection": {
                "region_count": self.detection_result.total_regions if self.detection_result else 0,
                "method": self.detection_result.method if self.detection_result else "",
            } if self.detection_result else None,
        }

        if self.error:
            result["error"] = self.error
        if self.warnings:
            result["warnings"] = self.warnings

        return result


class DocumentExtractor:
    """
    End-to-end document information extraction pipeline.

    Stages:
    1. Image loading and preprocessing (ImageEnhancer)
    2. Text region detection (TextDetector)
    3. Region classification (RegionClassifier)
    4. OCR recognition (OcrEngine)
    5. Text post-processing (TextPostProcessor)
    6. Field extraction (FieldExtractor)
    """

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or AppConfig()

        # Initialize pipeline components
        self.enhancer = ImageEnhancer(self.config.preprocess)
        self.detector = TextDetector(self.config.detection)
        self.layout_analyzer = LayoutAnalyzer()
        self.classifier = RegionClassifier(self.layout_analyzer)

        # Lazy-init OCR engine (can be heavy)
        self._ocr_engine = None
        self._postprocessor = TextPostProcessor()
        self._field_extractor = FieldExtractor(self.config.extraction)

    @property
    def ocr_engine(self):
        """Lazy initialization of OCR engine."""
        if self._ocr_engine is None:
            try:
                self._ocr_engine = create_ocr_engine(self.config.ocr)
            except ImportError as e:
                logger.error(f"Failed to init OCR engine: {e}")
                raise
        return self._ocr_engine

    def process(self, image_path: str) -> PipelineResult:
        """
        Process a single document image through the full pipeline.

        Args:
            image_path: Path to the input document image.

        Returns:
            PipelineResult with all stage outputs and final extraction.
        """
        t_total = time.time()
        result = PipelineResult(source_path=str(image_path))

        try:
            # Stage 1: Load image
            logger.info(f"Processing: {image_path}")
            image = load_image(
                image_path,
                color_mode=self.config.image.color_mode,
                max_dimension=self.config.image.max_dimension,
            )
            result.image_size = image.shape[:2]
            t1 = time.time()
            result.stage_timings["load"] = (t1 - t_total) * 1000

            # Stage 2: Preprocess / Enhance
            t2 = time.time()
            # Keep original RGB for seal detection (requires color info)
            if len(image.shape) == 3:
                result.original_rgb = image.copy()
            else:
                result.original_rgb = image.copy()
            # enhance returns both grayscale (for OCR) and binary (for detection)
            enhanced, grayscale = self.enhancer.enhance(image)
            result.enhanced_image = enhanced
            result.grayscale_image = grayscale
            result.stage_timings["preprocess"] = (time.time() - t2) * 1000
            logger.info("Stage complete: preprocessing")

            # Stage 3: Text detection (uses binary image)
            t3 = time.time()
            detection = self.detector.detect(enhanced)
            result.detection_result = detection
            result.stage_timings["detection"] = (time.time() - t3) * 1000
            logger.info("Stage complete: detection")

            # Stage 4: Layout analysis + region classification (uses original RGB for seals)
            t4 = time.time()
            layout_regions = self.layout_analyzer.analyze(result.original_rgb)
            classified_regions = self.classifier.classify(
                detection.regions, enhanced.shape[:2], layout_regions
            )
            detection.regions = classified_regions
            result.stage_timings["layout"] = (time.time() - t4) * 1000
            logger.info("Stage complete: layout + classification")

            # Stage 5: OCR recognition (uses grayscale, not binary)
            t5 = time.time()
            ocr_result = self.ocr_engine.recognize(grayscale)
            # Post-process OCR output
            ocr_result = self._postprocessor.process_page(ocr_result)
            result.ocr_result = ocr_result
            result.stage_timings["ocr"] = (time.time() - t5) * 1000
            logger.info(
                f"Stage complete: OCR ({ocr_result.engine_name}, "
                f"{len(ocr_result.results)} regions, "
                f"avg_conf={ocr_result.avg_confidence:.2f})"
            )

            # Stage 6: Field extraction
            t6 = time.time()
            extraction = self._field_extractor.extract(
                ocr_result.results,
                ocr_result.full_text,
                self.config.extraction.document_type,
            )
            result.extraction_result = extraction
            result.stage_timings["extraction"] = (time.time() - t6) * 1000
            logger.info("Stage complete: extraction")

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            result.success = False
            result.error = str(e)

        result.total_time_ms = (time.time() - t_total) * 1000
        logger.info(
            f"Pipeline finished in {result.total_time_ms:.1f}ms "
            f"(success={result.success})"
        )

        return result

    def process_batch(
        self, paths: List[str], callback=None
    ) -> List[PipelineResult]:
        """
        Process multiple document images.

        Args:
            paths: List of image file paths.
            callback: Optional callback(processed_count, total_count, result).

        Returns:
            List of PipelineResult for each image.
        """
        results = []
        total = len(paths)

        logger.info(f"Starting batch processing: {total} images")

        for i, path in enumerate(paths):
            logger.info(f"Processing {i + 1}/{total}: {path}")
            result = self.process(path)
            results.append(result)

            if callback:
                callback(i + 1, total, result)

        success_count = sum(1 for r in results if r.success)
        logger.info(
            f"Batch complete: {success_count}/{total} successful"
        )

        return results
