"""
Field extraction engine.
Extracts structured information from OCR text using regex patterns,
template matching, and rule-based approaches.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import ExtractionConfig
from src.extraction.field_types import (
    FIELD_METADATA,
    ExtractedField,
    FieldType,
    get_fields_by_category,
)
from src.recognition.ocr_engine import OcrResult
from src.utils.logger import get_logger
from src.utils.text_utils import compute_text_similarity

logger = get_logger("extraction")


@dataclass
class ExtractionResult:
    """Result of field extraction from a document."""

    fields: List[ExtractedField] = field(default_factory=list)
    document_type: str = "unknown"
    extraction_time_ms: float = 0.0
    total_fields: int = 0
    high_confidence_fields: int = 0

    def get_field(self, field_type: FieldType) -> Optional[ExtractedField]:
        """Get a specific extracted field by type."""
        for f in self.fields:
            if f.field_type == field_type:
                return f
        return None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "document_type": self.document_type,
            "total_fields": self.total_fields,
            "high_confidence_fields": self.high_confidence_fields,
            "extraction_time_ms": round(self.extraction_time_ms, 1),
            "fields": {
                f.field_type.value: f.to_dict() for f in self.fields
            },
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


class FieldExtractor:
    """
    Extracts structured fields from OCR text using:
    1. Regex pattern matching (from field type definitions)
    2. Template-based extraction (JSON/YAML document templates)
    3. Context-aware field mapping (position-based hints)
    4. Confidence scoring
    """

    def __init__(self, config: Optional[ExtractionConfig] = None):
        self.config = config or ExtractionConfig()
        self._templates: Dict[str, dict] = {}
        self._load_templates()

    def extract(
        self,
        ocr_results: List[OcrResult],
        full_text: str = "",
        document_type: str = "auto",
    ) -> ExtractionResult:
        """
        Extract fields from OCR results.

        Args:
            ocr_results: List of OCR results with text and bounding boxes.
            full_text: Full page text (concatenated).
            document_type: Document type hint (auto, invoice, receipt, etc.).

        Returns:
            ExtractionResult with all extracted fields.
        """
        import time
        t_start = time.time()

        result = ExtractionResult()
        lines = [r.text for r in ocr_results if r.text.strip()]

        # Build combined text for multi-line matching
        combined_text = full_text or "\n".join(lines)

        logger.info(
            f"Extracting fields (type={document_type}, "
            f"lines={len(lines)}, chars={len(combined_text)})"
        )

        # Determine document type if auto
        if document_type == "auto":
            result.document_type = self._detect_document_type(combined_text)
            document_type = result.document_type
        else:
            result.document_type = document_type

        logger.info(f"Detected document type: {document_type}")

        # Get field types to extract based on document type
        field_types = self._get_field_types_for_doc_type(document_type)

        # Extract each field type
        for ft in field_types:
            extracted = self._extract_field(
                ft, combined_text, lines, ocr_results
            )
            if extracted:
                result.fields.append(extracted)

        # Sort by confidence (high first)
        result.fields.sort(key=lambda f: f.confidence, reverse=True)
        result.total_fields = len(result.fields)
        result.high_confidence_fields = sum(
            1 for f in result.fields
            if f.confidence >= self.config.confidence_threshold
        )

        result.extraction_time_ms = (time.time() - t_start) * 1000

        logger.info(
            f"Extraction complete: {result.total_fields} fields "
            f"({result.high_confidence_fields} high confidence) "
            f"in {result.extraction_time_ms:.1f}ms"
        )

        return result

    def _extract_field(
        self,
        field_type: FieldType,
        combined_text: str,
        lines: List[str],
        ocr_results: List[OcrResult],
    ) -> Optional[ExtractedField]:
        """Extract a single field using regex patterns."""
        metadata = FIELD_METADATA[field_type]
        patterns = metadata.get("patterns", [])

        if not patterns:
            return None

        best_match = None
        best_confidence = 0.0
        best_source = ""
        best_line_idx = -1

        for pattern in patterns:
            # Search in combined text
            match = re.search(pattern, combined_text, re.IGNORECASE | re.MULTILINE)
            if match:
                value = match.group(1) if match.groups() else match.group()
                value = value.strip()

                # Calculate confidence based on match quality
                conf = self._calculate_confidence(
                    value, pattern, field_type, combined_text
                )

                # Find source line
                line_idx = self._find_source_line(value, lines)

                if conf > best_confidence:
                    best_match = value
                    best_confidence = conf
                    best_source = match.group(0)
                    best_line_idx = line_idx

        if best_match is None:
            return None

        # Get bounding box from OCR result if available
        bbox = (0, 0, 0, 0)
        if 0 <= best_line_idx < len(ocr_results):
            bbox = ocr_results[best_line_idx].bbox

        return ExtractedField(
            field_type=field_type,
            value=best_match,
            confidence=best_confidence,
            source_text=best_source,
            source_line=best_line_idx,
            bbox=bbox,
        )

    def _calculate_confidence(
        self,
        value: str,
        pattern: str,
        field_type: FieldType,
        context: str,
    ) -> float:
        """
        Calculate extraction confidence score.

        Factors:
        - Pattern specificity (more specific = higher)
        - Value format match
        - Context quality (surrounding text relevance)
        """
        confidence = 0.5  # base confidence

        # Pattern with labels (e.g., "发票号码：") is more specific
        if re.search(r"[\u4e00-\u9fff]", pattern):
            confidence += 0.2

        # Value length reasonableness
        if len(value) >= 3:
            confidence += 0.1

        # Validation check
        validation = FIELD_METADATA[field_type].get("validation")
        if validation and re.match(validation, value):
            confidence += 0.2

        # Cap at 1.0
        return min(confidence, 1.0)

    def _find_source_line(
        self, value: str, lines: List[str]
    ) -> int:
        """Find which line contains the extracted value."""
        for i, line in enumerate(lines):
            if value in line:
                return i
        return -1

    def _detect_document_type(self, text: str) -> str:
        """Auto-detect document type from text content."""
        text_lower = text.lower()

        # Detection keywords for each document type
        type_keywords = {
            "invoice": [
                "发票", "增值税", "invoice", "税额", "纳税人识别号",
                "价税合计", "开票日期",
            ],
            "receipt": [
                "收据", "receipt", "收款", "付款", "已收",
            ],
            "bank_statement": [
                "银行流水", "对账单", "statement", "交易记录",
                "账户", "余额",
            ],
            "contract": [
                "合同", "协议", "contract", "甲方", "乙方",
                "签署", "条款",
            ],
            "quotation": [
                "报价", "quotation", "报价单", "单价",
            ],
        }

        scores = {}
        for doc_type, keywords in type_keywords.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            scores[doc_type] = score

        best_type = max(scores, key=scores.get)
        if scores[best_type] > 0:
            return best_type

        return "unknown"

    def _get_field_types_for_doc_type(
        self, doc_type: str
    ) -> List[FieldType]:
        """Get relevant field types based on document type."""
        if doc_type == "invoice":
            return [
                FieldType.INVOICE_NUMBER,
                FieldType.INVOICE_CODE,
                FieldType.INVOICE_DATE,
                FieldType.SELLER_NAME,
                FieldType.BUYER_NAME,
                FieldType.SELLER_TAX_ID,
                FieldType.BUYER_TAX_ID,
                FieldType.AMOUNT_BEFORE_TAX,
                FieldType.TAX_AMOUNT,
                FieldType.AMOUNT_TOTAL,
                FieldType.RECEIVER_NAME,
                FieldType.RECEIVER_PHONE,
                FieldType.REMARKS,
            ]
        elif doc_type == "receipt":
            return [
                FieldType.INVOICE_DATE,
                FieldType.AMOUNT_TOTAL,
                FieldType.PAYMENT_METHOD,
                FieldType.RECEIVER_NAME,
                FieldType.RECEIVER_PHONE,
                FieldType.REMARKS,
            ]
        elif doc_type == "bank_statement":
            return [
                FieldType.BANK_NAME,
                FieldType.BANK_ACCOUNT,
                FieldType.AMOUNT_TOTAL,
                FieldType.RECEIVER_NAME,
            ]
        elif doc_type == "contract":
            return [
                FieldType.SELLER_NAME,
                FieldType.BUYER_NAME,
                FieldType.INVOICE_DATE,
                FieldType.AMOUNT_TOTAL,
                FieldType.RECEIVER_PHONE,
                FieldType.RECEIVER_ADDRESS,
            ]
        else:
            # Extract all common fields for unknown types
            return [
                FieldType.INVOICE_NUMBER,
                FieldType.INVOICE_DATE,
                FieldType.AMOUNT_TOTAL,
                FieldType.SELLER_NAME,
                FieldType.BUYER_NAME,
                FieldType.RECEIVER_NAME,
                FieldType.RECEIVER_PHONE,
                FieldType.REMARKS,
            ]

    def _load_templates(self) -> None:
        """Load document type templates from the templates directory."""
        templates_dir = Path(self.config.templates_dir)
        if not templates_dir.exists():
            logger.debug(
                f"Templates directory not found: {templates_dir}"
            )
            return

        for template_file in templates_dir.glob("*.json"):
            try:
                with open(template_file, "r", encoding="utf-8") as f:
                    template = json.load(f)
                doc_type = template.get("document_type", template_file.stem)
                self._templates[doc_type] = template
                logger.info(f"Loaded template: {doc_type}")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load template {template_file}: {e}")
