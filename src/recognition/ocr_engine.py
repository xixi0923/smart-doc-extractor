"""
Pluggable OCR engine abstraction and implementations.
Supports Tesseract OCR and EasyOCR as backends.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import OcrConfig
from src.preprocessing.image_enhancer import TextRegion
from src.utils.logger import get_logger

logger = get_logger("ocr")


@dataclass
class OcrResult:
    """Result of OCR recognition on a single text region."""

    text: str
    confidence: float
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    region_type: str = "unknown"

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


@dataclass
class OcrPageResult:
    """Result of full-page OCR."""

    results: List[OcrResult] = field(default_factory=list)
    full_text: str = ""
    total_time_ms: float = 0.0
    avg_confidence: float = 0.0
    engine_name: str = ""

    def get_non_empty_results(self) -> List[OcrResult]:
        return [r for r in self.results if not r.is_empty]


class BaseOcrEngine(ABC):
    """Abstract base class for OCR engines."""

    @abstractmethod
    def recognize(self, image: np.ndarray) -> OcrPageResult:
        """Run OCR on the full image."""
        ...

    @abstractmethod
    def recognize_region(
        self, image: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> OcrResult:
        """Run OCR on a specific region of the image."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


class TesseractEngine(BaseOcrEngine):
    """
    Tesseract OCR engine wrapper.
    Uses pytesseract for Python integration.
    """

    def __init__(self, config: Optional[OcrConfig] = None):
        self.config = config or OcrConfig()
        self._tesseract = None
        self._initialize()

    def _initialize(self) -> None:
        """Initialize Tesseract with lazy import."""
        try:
            import pytesseract
            self._tesseract = pytesseract
            logger.info(
                f"Tesseract initialized (lang={self.config.language}, "
                f"psm_config={self.config.tesseract_config})"
            )
        except ImportError:
            logger.warning(
                "pytesseract not installed. "
                "Install with: pip install pytesseract"
            )
            raise

    @property
    def name(self) -> str:
        return "tesseract"

    def recognize(self, image: np.ndarray) -> OcrPageResult:
        """Run Tesseract on the full image."""
        t_start = time.time()

        # Convert RGB to grayscale for Tesseract
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # Ensure proper dtype
        gray = self._ensure_u8(gray)

        try:
            data = self._tesseract.image_to_data(
                gray,
                lang=self.config.language,
                config=self.config.tesseract_config,
                output_type=self._tesseract.Output.DICT,
            )

            results = []
            texts = []
            confidences = []

            n = len(data["text"])
            for i in range(n):
                text = data["text"][i].strip()
                conf = float(data["conf"][i])

                if not text or conf < 0:
                    continue

                x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]

                if w <= 0 or h <= 0:
                    continue

                results.append(OcrResult(
                    text=text,
                    confidence=conf / 100.0,  # Tesseract uses 0-100
                    bbox=(x, y, w, h),
                ))
                texts.append(text)
                confidences.append(conf / 100.0)

            full_text = "\n".join(texts)
            avg_conf = float(np.mean(confidences)) if confidences else 0.0

            elapsed = (time.time() - t_start) * 1000

            return OcrPageResult(
                results=results,
                full_text=full_text,
                total_time_ms=elapsed,
                avg_confidence=avg_conf,
                engine_name=self.name,
            )

        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            return OcrPageResult(
                engine_name=self.name,
                total_time_ms=(time.time() - t_start) * 1000,
            )

    def recognize_region(
        self, image: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> OcrResult:
        """Run Tesseract on a cropped region."""
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            return OcrResult(text="", confidence=0.0, bbox=bbox)

        # Add padding
        pad = 3
        img_h, img_w = image.shape[:2]
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img_w, x + w + pad)
        y2 = min(img_h, y + h + pad)

        crop = image[y1:y2, x1:x2]

        if len(crop.shape) == 3:
            crop = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        crop = self._ensure_u8(crop)

        try:
            text = self._tesseract.image_to_string(
                crop,
                lang=self.config.language,
                config=self.config.tesseract_config,
            ).strip()

            # Get confidence
            conf_data = self._tesseract.image_to_data(
                crop,
                lang=self.config.language,
                config=self.config.tesseract_config,
                output_type=self._tesseract.Output.DICT,
            )
            confs = [float(c) for c in conf_data["conf"] if float(c) >= 0]
            avg_conf = float(np.mean(confs)) / 100.0 if confs else 0.0

            return OcrResult(
                text=text,
                confidence=avg_conf,
                bbox=bbox,
            )
        except Exception as e:
            logger.error(f"Tesseract region OCR failed: {e}")
            return OcrResult(text="", confidence=0.0, bbox=bbox)

    @staticmethod
    def _ensure_u8(image: np.ndarray) -> np.ndarray:
        """Ensure image is uint8."""
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        return image


class EasyOcrEngine(BaseOcrEngine):
    """
    EasyOCR engine wrapper.
    Supports 80+ languages with GPU acceleration.
    """

    def __init__(self, config: Optional[OcrConfig] = None):
        self.config = config or OcrConfig()
        self._reader = None
        self._initialize()

    def _initialize(self) -> None:
        """Initialize EasyOCR with lazy import."""
        try:
            import easyocr
            langs = self.config.language.split("+")
            self._reader = easyocr.Reader(
                langs,
                gpu=self.config.easyocr_gpu,
                verbose=False,
            )
            logger.info(
                f"EasyOCR initialized (langs={langs}, gpu={self.config.easyocr_gpu})"
            )
        except ImportError:
            logger.warning(
                "easyocr not installed. "
                "Install with: pip install easyocr"
            )
            raise

    @property
    def name(self) -> str:
        return "easyocr"

    def recognize(self, image: np.ndarray) -> OcrPageResult:
        """Run EasyOCR on the full image."""
        t_start = time.time()

        try:
            raw_results = self._reader.readtext(image)

            results = []
            texts = []
            confidences = []

            for bbox, text, conf in raw_results:
                # Convert EasyOCR bbox (4 corners) to rect
                pts = np.array(bbox)
                x = int(pts[:, 0].min())
                y = int(pts[:, 1].min())
                w = int(pts[:, 0].max()) - x
                h = int(pts[:, 1].max()) - y

                results.append(OcrResult(
                    text=text,
                    confidence=conf,
                    bbox=(x, y, w, h),
                ))
                texts.append(text)
                confidences.append(conf)

            full_text = "\n".join(texts)
            avg_conf = float(np.mean(confidences)) if confidences else 0.0

            return OcrPageResult(
                results=results,
                full_text=full_text,
                total_time_ms=(time.time() - t_start) * 1000,
                avg_confidence=avg_conf,
                engine_name=self.name,
            )
        except Exception as e:
            logger.error(f"EasyOCR failed: {e}")
            return OcrPageResult(
                engine_name=self.name,
                total_time_ms=(time.time() - t_start) * 1000,
            )

    def recognize_region(
        self, image: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> OcrResult:
        """Run EasyOCR on a cropped region."""
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            return OcrResult(text="", confidence=0.0, bbox=bbox)

        pad = 3
        img_h, img_w = image.shape[:2]
        crop = image[
            max(0, y - pad):min(img_h, y + h + pad),
            max(0, x - pad):min(img_w, x + w + pad),
        ]

        try:
            raw_results = self._reader.readtext(crop)
            if raw_results:
                text = " ".join(r[1] for r in raw_results)
                conf = float(np.mean([r[2] for r in raw_results]))
            else:
                text = ""
                conf = 0.0
            return OcrResult(text=text, confidence=conf, bbox=bbox)
        except Exception as e:
            logger.error(f"EasyOCR region OCR failed: {e}")
            return OcrResult(text="", confidence=0.0, bbox=bbox)


def create_ocr_engine(config: Optional[OcrConfig] = None) -> BaseOcrEngine:
    """
    Factory function to create an OCR engine based on config.

    Args:
        config: OCR configuration. Uses Tesseract by default.

    Returns:
        An instance of BaseOcrEngine subclass.

    Raises:
        ImportError: If the required OCR backend is not installed.
        ValueError: If the specified engine is not supported.
    """
    config = config or OcrConfig()

    if config.engine == "tesseract":
        return TesseractEngine(config)
    elif config.engine == "easyocr":
        return EasyOcrEngine(config)
    else:
        raise ValueError(
            f"Unsupported OCR engine: {config.engine}. "
            f"Supported: tesseract, easyocr"
        )
