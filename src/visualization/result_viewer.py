"""
Result visualization module.
Generates annotated images and HTML reports showing extraction results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.pipeline.extractor import PipelineResult
from src.utils.image_utils import gray_to_rgb, save_image
from src.utils.logger import get_logger

logger = get_logger("visualization")


# Color scheme for different region types
REGION_COLORS: Dict[str, Tuple[int, int, int]] = {
    "title": (0, 165, 255),        # Orange
    "header": (255, 165, 0),      # Blue
    "body": (0, 255, 0),           # Green
    "table": (255, 0, 255),        # Magenta
    "seal": (0, 0, 255),           # Red
    "footer": (128, 128, 128),     # Gray
    "sidebar": (255, 255, 0),      # Cyan
    "unknown": (200, 200, 200),    # Light gray
}

# Confidence color (red=low, yellow=mid, green=high)
def confidence_color(confidence: float) -> Tuple[int, int, int]:
    """Return color based on confidence score (0-1)."""
    if confidence >= 0.8:
        return (0, 200, 0)    # Green
    elif confidence >= 0.6:
        return (0, 165, 255)  # Orange
    else:
        return (0, 0, 255)    # Red


class ResultViewer:
    """
    Visualizes document extraction results.

    Features:
    - Draw bounding boxes on original image
    - Annotate extracted fields with labels
    - Generate confidence-coded visual output
    """

    def __init__(self, font_scale: float = 0.6, line_thickness: int = 2):
        self.font_scale = font_scale
        self.line_thickness = line_thickness

    def annotate_image(
        self,
        original_image: np.ndarray,
        pipeline_result: PipelineResult,
    ) -> np.ndarray:
        """
        Draw detection and extraction results on the original image.

        Args:
            original_image: Original document image (RGB).
            pipeline_result: Pipeline result with detection and extraction data.

        Returns:
            Annotated image (RGB) with bounding boxes and labels.
        """
        # Work on a copy
        annotated = original_image.copy()

        # Ensure RGB
        if len(annotated.shape) == 2:
            annotated = gray_to_rgb(annotated)

        # Draw detected text regions
        if pipeline_result.detection_result:
            for region in pipeline_result.detection_result.regions:
                color = REGION_COLORS.get(
                    region.region_type,
                    REGION_COLORS["unknown"],
                )
                x, y, w, h = region.bbox
                cv2.rectangle(
                    annotated, (x, y), (x + w, y + h),
                    color, self.line_thickness,
                )

        # Draw extracted fields with labels
        if pipeline_result.extraction_result:
            for extracted_field in pipeline_result.extraction_result.fields:
                bbox = extracted_field.bbox
                if bbox == (0, 0, 0, 0):
                    continue

                x, y, w, h = bbox
                color = confidence_color(extracted_field.confidence)

                # Draw thicker border for extracted fields
                cv2.rectangle(
                    annotated, (x, y), (x + w, y + h),
                    color, self.line_thickness + 1,
                )

                # Add label
                label = f"{extracted_field.display_name}: {extracted_field.value[:20]}"
                label_size, _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX,
                    self.font_scale * 0.8, 1,
                )

                # Label background
                label_y = max(y - 5, label_size[1] + 5)
                cv2.rectangle(
                    annotated,
                    (x, label_y - label_size[1] - 2),
                    (x + label_size[0] + 4, label_y + 2),
                    color, -1,
                )

                # Label text
                cv2.putText(
                    annotated, label,
                    (x + 2, label_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.font_scale * 0.8,
                    (255, 255, 255), 1, cv2.LINE_AA,
                )

        return annotated

    def save_annotated(
        self,
        original_image: np.ndarray,
        pipeline_result: PipelineResult,
        output_path: str,
    ) -> str:
        """
        Save annotated image to file.

        Returns:
            Path to the saved file.
        """
        annotated = self.annotate_image(original_image, pipeline_result)
        save_image(annotated, output_path)
        logger.info(f"Saved annotated image: {output_path}")
        return output_path

    def generate_html_report(
        self,
        pipeline_result: PipelineResult,
        output_path: str,
        annotated_image_path: Optional[str] = None,
    ) -> str:
        """
        Generate an HTML report for the extraction result.

        Args:
            pipeline_result: Pipeline result to report.
            output_path: Where to save the HTML file.
            annotated_image_path: Optional path to annotated image.

        Returns:
            Path to the generated HTML file.
        """
        extraction = pipeline_result.extraction_result
        ocr = pipeline_result.ocr_result

        # Build fields table rows
        field_rows = ""
        if extraction:
            for f in extraction.fields:
                conf_pct = round(f.confidence * 100)
                conf_color = "#28a745" if conf_pct >= 80 else "#fd7e14" if conf_pct >= 60 else "#dc3545"
                field_rows += f"""
                <tr>
                    <td><code>{f.field_type.value}</code></td>
                    <td>{f.display_name}</td>
                    <td>{f.value}</td>
                    <td style="color: {conf_color}; font-weight: 500;">{conf_pct}%</td>
                    <td>{f.category}</td>
                </tr>"""

        ocr_text_preview = (ocr.full_text[:2000] + "...") if ocr and len(ocr.full_text) > 2000 else (ocr.full_text if ocr else "")

        # Image tag
        img_tag = ""
        if annotated_image_path and Path(annotated_image_path).exists():
            img_src = Path(annotated_image_path).name
            img_tag = f'<div class="image-section"><h2>Annotated Image</h2><img src="{img_src}" alt="Extraction Result" class="result-image"></div>'

        # Stage timings
        timing_rows = ""
        for stage, ms in pipeline_result.stage_timings.items():
            timing_rows += f"<tr><td>{stage}</td><td>{ms:.1f}ms</td></tr>"

        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document Extraction Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f5f5f5; color: #333; line-height: 1.6; }}
        .container {{ max-width: 960px; margin: 0 auto; padding: 20px; }}
        h1 {{ font-size: 24px; margin-bottom: 4px; }}
        .subtitle {{ color: #666; font-size: 14px; margin-bottom: 20px; }}
        .card {{ background: white; border-radius: 8px; padding: 20px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .card h2 {{ font-size: 16px; margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px solid #eee; }}
        .metrics {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }}
        .metric {{ background: #f8f9fa; border-radius: 6px; padding: 12px; text-align: center; }}
        .metric .value {{ font-size: 24px; font-weight: 600; }}
        .metric .label {{ font-size: 12px; color: #666; margin-top: 4px; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f8f9fa; font-weight: 600; font-size: 12px; text-transform: uppercase; }}
        .result-image {{ max-width: 100%; border-radius: 4px; border: 1px solid #ddd; }}
        .ocr-text {{ background: #f8f9fa; padding: 12px; border-radius: 4px; font-family: monospace; font-size: 12px; white-space: pre-wrap; max-height: 300px; overflow-y: auto; }}
        .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 500; }}
        .badge-success {{ background: #d4edda; color: #155724; }}
        .badge-warning {{ background: #fff3cd; color: #856404; }}
        .badge-danger {{ background: #f8d7da; color: #721c24; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Document Extraction Report</h1>
        <p class="subtitle">Source: {pipeline_result.source_path} | Total time: {pipeline_result.total_time_ms:.0f}ms</p>

        <div class="card">
            <h2>Summary</h2>
            <div class="metrics">
                <div class="metric">
                    <div class="value">{extraction.total_fields if extraction else 0}</div>
                    <div class="label">Extracted Fields</div>
                </div>
                <div class="metric">
                    <div class="value">{extraction.high_confidence_fields if extraction else 0}</div>
                    <div class="label">High Confidence</div>
                </div>
                <div class="metric">
                    <div class="value">{ocr.avg_confidence * 100:.0f}%</div>
                    <div class="label">OCR Confidence</div>
                </div>
                <div class="metric">
                    <div class="value">{pipeline_result.image_size[1]}x{pipeline_result.image_size[0]}</div>
                    <div class="label">Image Size</div>
                </div>
            </div>
        </div>

        {img_tag}

        <div class="card">
            <h2>Extracted Fields</h2>
            <table>
                <thead>
                    <tr><th>Field</th><th>Name</th><th>Value</th><th>Confidence</th><th>Category</th></tr>
                </thead>
                <tbody>{field_rows}</tbody>
            </table>
        </div>

        <div class="card">
            <h2>OCR Full Text</h2>
            <div class="ocr-text">{ocr_text_preview}</div>
        </div>

        <div class="card">
            <h2>Pipeline Performance</h2>
            <table>
                <thead><tr><th>Stage</th><th>Time</th></tr></thead>
                <tbody>{timing_rows}</tbody>
            </table>
        </div>
    </div>
</body>
</html>"""

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info(f"Generated HTML report: {output_path}")
        return str(output_path)
