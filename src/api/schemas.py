"""
Pydantic schemas for the REST API request/response models.
"""

from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class ExtractRequest(BaseModel):
    """Request model for single document extraction."""

    document_type: str = Field(
        default="auto",
        description="Document type hint: auto, invoice, receipt, bank_statement, contract",
    )
    output_format: str = Field(
        default="json",
        description="Output format: json, annotated_image",
    )
    save_intermediate: bool = Field(
        default=False,
        description="Save intermediate processing images",
    )


class ExtractResponse(BaseModel):
    """Response model for extraction result."""

    success: bool
    source: str = ""
    document_type: str = "unknown"
    total_fields: int = 0
    high_confidence_fields: int = 0
    total_time_ms: float = 0.0
    fields: Dict[str, dict] = Field(default_factory=dict)
    ocr_text: str = ""
    ocr_confidence: float = 0.0
    error: Optional[str] = None


class BatchExtractRequest(BaseModel):
    """Request model for batch extraction."""

    document_type: str = Field(default="auto")
    output_format: str = Field(default="json")


class BatchExtractResponse(BaseModel):
    """Response model for batch extraction."""

    total: int = 0
    successful: int = 0
    failed: int = 0
    total_time_ms: float = 0.0
    results: List[dict] = Field(default_factory=list)


class TemplateInfo(BaseModel):
    """Information about a document template."""

    document_type: str
    name: str
    description: str = ""
    field_types: List[str] = Field(default_factory=list)


class TemplatesResponse(BaseModel):
    """Response model for listing available templates."""

    templates: List[TemplateInfo] = Field(default_factory=list)
    total: int = 0


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    version: str = ""
    ocr_engine: str = ""
    ocr_available: bool = True
