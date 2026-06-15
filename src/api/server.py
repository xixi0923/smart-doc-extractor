"""
REST API server for Smart Doc Extractor.
Provides HTTP endpoints for document extraction.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from config import AppConfig
from src import __version__
from src.api.schemas import (
    BatchExtractResponse,
    ExtractRequest,
    ExtractResponse,
    HealthResponse,
    TemplatesResponse,
)
from src.pipeline.extractor import DocumentExtractor
from src.utils.logger import get_logger, setup_logging

logger = get_logger("api")


def create_app(config: Optional[AppConfig] = None) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        config: Optional application config. Uses defaults if None.

    Returns:
        Configured FastAPI app instance.
    """
    config = config or AppConfig()
    app = FastAPI(
        title="Smart Doc Extractor",
        description="Intelligent document information extraction API. "
                    "Extracts structured fields from invoice, receipt, "
                    "bank statement and contract images.",
        version=__version__,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(config.api.cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Serve static files (web console)
    static_dir = Path(__file__).parent / "static"
    if static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def web_console():
        """Serve the web debugging console."""
        index_path = static_dir / "index.html"
        if index_path.is_file():
            return FileResponse(str(index_path), media_type="text/html")
        return HTMLResponse("<h1>Smart Doc Extractor</h1><p>Web console not found.</p>")

    # Storage for extractor instance (lazy init)
    _extractor_holder: dict = {"extractor": None}

    def get_extractor() -> DocumentExtractor:
        if _extractor_holder["extractor"] is None:
            _extractor_holder["extractor"] = DocumentExtractor(config)
        return _extractor_holder["extractor"]

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Check API health and OCR engine availability."""
        ocr_available = False
        ocr_engine_name = config.ocr.engine
        try:
            extractor = get_extractor()
            ocr_engine_name = extractor.ocr_engine.name
            ocr_available = True
        except Exception:
            pass

        return HealthResponse(
            status="healthy",
            version=__version__,
            ocr_engine=ocr_engine_name,
            ocr_available=ocr_available,
        )

    @app.post("/extract", response_model=ExtractResponse)
    async def extract_document(
        file: UploadFile = File(...),
        document_type: str = Form(default="auto"),
    ):
        """
        Extract information from a single document image.

        Upload an image file (JPG, PNG, BMP, TIFF, WebP).
        Returns structured extraction results as JSON.
        """
        if not file.content_type:
            file.content_type = "image/jpeg"

        # Save uploaded file to temp
        suffix = Path(file.filename).suffix if file.filename else ".png"
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix
        ) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            extractor = get_extractor()
            result = extractor.process(tmp_path)

            if not result.success:
                return ExtractResponse(
                    success=False,
                    error=result.error,
                )

            return ExtractResponse(
                success=True,
                source=result.source_path,
                document_type=result.extraction_result.document_type if result.extraction_result else "unknown",
                total_fields=result.extraction_result.total_fields if result.extraction_result else 0,
                high_confidence_fields=result.extraction_result.high_confidence_fields if result.extraction_result else 0,
                total_time_ms=result.total_time_ms,
                fields=result.extraction_result.to_dict()["fields"] if result.extraction_result else {},
                ocr_text=result.ocr_result.full_text[:500] if result.ocr_result else "",
                ocr_confidence=result.ocr_result.avg_confidence if result.ocr_result else 0.0,
            )

        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @app.post("/extract/batch", response_model=BatchExtractResponse)
    async def extract_batch(
        files: List[UploadFile] = File(...),
        document_type: str = Form(default="auto"),
    ):
        """
        Extract information from multiple document images.

        Upload multiple image files at once.
        Returns aggregated results for all documents.
        """
        t_start = time.time()
        results = []
        success_count = 0

        extractor = get_extractor()

        for file in files:
            suffix = Path(file.filename).suffix if file.filename else ".png"
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix
            ) as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = tmp.name

            try:
                result = extractor.process(tmp_path)
                results.append(result.to_dict())
                if result.success:
                    success_count += 1
            except Exception as e:
                results.append({"success": False, "error": str(e)})
            finally:
                Path(tmp_path).unlink(missing_ok=True)

        total_time = (time.time() - t_start) * 1000

        return BatchExtractResponse(
            total=len(files),
            successful=success_count,
            failed=len(files) - success_count,
            total_time_ms=total_time,
            results=results,
        )

    @app.post("/extract/visual", response_model=ExtractResponse)
    async def extract_with_visual(
        file: UploadFile = File(...),
        document_type: str = Form(default="auto"),
    ):
        """
        Extract information and return annotated image as base64.

        Returns all extraction fields plus an annotated image
        with bounding boxes and field labels for the web console.
        """
        import base64
        import cv2

        if not file.content_type:
            file.content_type = "image/jpeg"

        suffix = Path(file.filename).suffix if file.filename else ".png"
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix
        ) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            extractor = get_extractor()
            result = extractor.process(tmp_path)

            if not result.success:
                return ExtractResponse(
                    success=False,
                    error=result.error,
                )

            # Generate annotated image
            from src.utils.image_utils import load_image
            from src.visualization.result_viewer import ResultViewer

            viewer = ResultViewer()
            original = load_image(tmp_path)
            annotated = viewer.annotate_image(original, result)

            # Encode annotated image as JPEG base64
            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
            annotated_b64 = base64.b64encode(buf).decode("utf-8")

            response = ExtractResponse(
                success=True,
                source=result.source_path,
                document_type=result.extraction_result.document_type if result.extraction_result else "unknown",
                total_fields=result.extraction_result.total_fields if result.extraction_result else 0,
                high_confidence_fields=result.extraction_result.high_confidence_fields if result.extraction_result else 0,
                total_time_ms=result.total_time_ms,
                fields=result.extraction_result.to_dict()["fields"] if result.extraction_result else {},
                ocr_text=result.ocr_result.full_text[:2000] if result.ocr_result else "",
                ocr_confidence=result.ocr_result.avg_confidence if result.ocr_result else 0.0,
                annotated_image=f"data:image/jpeg;base64,{annotated_b64}",
            )

            return response

        except Exception as e:
            logger.error(f"Visual extraction failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @app.get("/templates", response_model=TemplatesResponse)
    async def list_templates():
        """List available document type templates."""
        from src.extraction.field_types import FIELD_METADATA, FieldType

        templates = [
            {
                "document_type": "invoice",
                "name": "增值税发票",
                "description": "Chinese VAT invoice extraction",
                "field_types": [ft.value for ft in [
                    FieldType.INVOICE_NUMBER, FieldType.INVOICE_CODE,
                    FieldType.INVOICE_DATE, FieldType.SELLER_NAME,
                    FieldType.BUYER_NAME, FieldType.AMOUNT_TOTAL,
                ]],
            },
            {
                "document_type": "receipt",
                "name": "收据",
                "description": "Receipt extraction",
                "field_types": [ft.value for ft in [
                    FieldType.INVOICE_DATE, FieldType.AMOUNT_TOTAL,
                    FieldType.PAYMENT_METHOD,
                ]],
            },
            {
                "document_type": "bank_statement",
                "name": "银行流水",
                "description": "Bank statement extraction",
                "field_types": [ft.value for ft in [
                    FieldType.BANK_NAME, FieldType.BANK_ACCOUNT,
                    FieldType.AMOUNT_TOTAL,
                ]],
            },
            {
                "document_type": "contract",
                "name": "合同",
                "description": "Contract extraction",
                "field_types": [ft.value for ft in [
                    FieldType.SELLER_NAME, FieldType.BUYER_NAME,
                    FieldType.INVOICE_DATE, FieldType.AMOUNT_TOTAL,
                ]],
            },
        ]

        return TemplatesResponse(templates=templates, total=len(templates))

    return app


def run_server(config: Optional[AppConfig] = None):
    """Run the API server with uvicorn."""
    import uvicorn

    config = config or AppConfig()

    # Setup logging
    setup_logging(
        level=config.log.level,
        log_format=config.log.format,
        log_file=config.log.log_file,
    )

    logger.info(
        f"Starting Smart Doc Extractor API on {config.api.host}:{config.api.port}"
    )

    app = create_app(config)
    uvicorn.run(
        app,
        host=config.api.host,
        port=config.api.port,
        workers=config.api.workers,
        log_level=config.log.level.lower(),
    )
