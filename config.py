"""
Smart Doc Extractor — Configuration
======================================
Central configuration for the document information extraction pipeline.
All tunable parameters are managed here via dataclass configs.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class ImageConfig:
    """Input image handling parameters."""

    target_dpi: int = 300              # resample target
    max_dimension: int = 4096          # downscale if larger
    color_mode: str = "rgb"            # rgb | gray
    allowed_formats: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")


@dataclass
class PreprocessConfig:
    """Image preprocessing pipeline parameters."""

    denoise: bool = True               # apply Gaussian/Median denoise
    denoise_ksize: int = 3             # kernel size for denoise
    binarize: bool = True              # convert to binary
    binarize_method: str = "adaptive"  # otsu | adaptive | sauvola
    binarize_block_size: int = 31       # block size for adaptive threshold
    deskew: bool = True                # correct page skew
    clahe: bool = True                 # contrast limited adaptive histogram equalization
    clahe_clip_limit: float = 2.0      # CLAHE clip limit
    remove_borders: bool = True        # remove dark border artifacts


@dataclass
class DetectionConfig:
    """Text region detection parameters."""

    method: str = "mser_contour"       # mser_contour | contour_only | mser_only
    min_region_area: int = 100          # minimum text region area (pixels)
    max_region_area: int = 50000       # maximum text region area
    min_aspect_ratio: float = 0.1       # filter too thin regions
    max_aspect_ratio: float = 10.0      # filter too wide regions
    merge_padding: int = 5              # padding for merging nearby regions
    nms_threshold: float = 0.3          # non-maximum suppression IoU threshold
    scale_range: tuple = (0.5, 2.0)    # multi-scale detection range


@dataclass
class OcrConfig:
    """OCR recognition engine parameters."""

    engine: str = "tesseract"          # tesseract | easyocr
    language: str = "chi_sim+eng"      # language(s) for OCR
    tesseract_config: str = "--psm 6"  # tesseract page segmentation mode
    tesseract_oem: int = 3             # OEM_LSTM_ONLY
    easyocr_gpu: bool = False          # use GPU for EasyOCR
    confidence_threshold: float = 0.5   # discard low-confidence OCR results


@dataclass
class ExtractionConfig:
    """Field extraction parameters."""

    document_type: str = "auto"         # auto | invoice | receipt | bank_statement | contract
    templates_dir: str = "templates"    # directory for document type templates
    confidence_threshold: float = 0.6   # minimum confidence for extracted fields
    enable_fuzzy_match: bool = True     # enable fuzzy string matching
    fuzzy_threshold: int = 85           # fuzzy matching threshold (0-100)


@dataclass
class OutputConfig:
    """Output and reporting parameters."""

    format: str = "json"               # json | csv | both
    output_dir: str = "output"         # default output directory
    save_annotated_image: bool = True   # draw boxes on image and save
    save_intermediate: bool = False     # save intermediate processing images
    generate_html_report: bool = True  # generate HTML visualization report


@dataclass
class ApiConfig:
    """REST API server parameters."""

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    max_upload_size_mb: int = 50
    cors_origins: tuple = ("*",)


@dataclass
class LogConfig:
    """Logging configuration."""

    level: str = "INFO"                # DEBUG | INFO | WARNING | ERROR
    format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    log_file: Optional[str] = None     # None = console only


@dataclass
class AppConfig:
    """Root configuration aggregating all sub-configs."""

    image: ImageConfig = field(default_factory=ImageConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    ocr: OcrConfig = field(default_factory=OcrConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    api: ApiConfig = field(default_factory=ApiConfig)
    log: LogConfig = field(default_factory=LogConfig)

    def to_dict(self) -> dict:
        """Serialize config to dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize config to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save(self, path: str | Path) -> None:
        """Save config to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: str | Path) -> AppConfig:
        """Load config from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        def _build(dataclass_cls, sub_data):
            if sub_data is None:
                return dataclass_cls()
            valid_fields = {f.name for f in dataclass_cls.__dataclass_fields__.values()}
            filtered = {k: v for k, v in sub_data.items() if k in valid_fields}
            return dataclass_cls(**filtered)

        return AppConfig(
            image=_build(ImageConfig, data.get("image")),
            preprocess=_build(PreprocessConfig, data.get("preprocess")),
            detection=_build(DetectionConfig, data.get("detection")),
            ocr=_build(OcrConfig, data.get("ocr")),
            extraction=_build(ExtractionConfig, data.get("extraction")),
            output=_build(OutputConfig, data.get("output")),
            api=_build(ApiConfig, data.get("api")),
            log=_build(LogConfig, data.get("log")),
        )
