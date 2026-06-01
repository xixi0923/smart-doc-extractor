# Smart Doc Extractor

智能文档信息提取系统 — 从发票、收据、银行流水、合同等文档图像中自动提取结构化字段信息。

## Features

- **多文档类型支持**: 发票 (VAT Invoice)、收据 (Receipt)、银行流水 (Bank Statement)、合同 (Contract)
- **智能版面分析**: 自动检测标题、正文、表格、印章等区域
- **可插拔 OCR 引擎**: Tesseract OCR / EasyOCR，灵活切换
- **正则+规则抽取引擎**: 20+ 预定义字段类型，支持自定义模板扩展
- **置信度评分**: 每个字段附带置信度分数，低置信度标记需人工复核
- **REST API 服务**: FastAPI 提供批量处理、异步任务、结果查询
- **可视化结果**: 在原图上标注提取框 + HTML 报告

## Architecture

```
Image Input → Preprocessing → Text Detection → OCR Recognition → Field Extraction → JSON Output
     │              │              │                │                  │
     │         denoise          MSER +         Tesseract/         regex +
     │         binarize         contour          EasyOCR           rules
     │         deskew           NMS           postprocess
     │         CLAHE         classification
     │      border removal    layout analysis
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Image Processing | OpenCV |
| OCR Engine | Tesseract / EasyOCR |
| REST API | FastAPI + Uvicorn |
| Config | Python dataclass |
| Validation | JSON Schema |

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/xixi0923/smart-doc-extractor.git
cd smart-doc-extractor

# Install dependencies
pip install opencv-python pytesseract numpy fastapi uvicorn pydantic

# Optional: Install EasyOCR as alternative OCR backend
pip install easyocr

# Install Tesseract OCR system binary (required for tesseract backend)
# Ubuntu/Debian: sudo apt install tesseract-ocr tesseract-ocr-chi-sim
# macOS: brew install tesseract
# Windows: download installer from https://github.com/UB-Mannheim/tesseract/wiki
```

### CLI Usage

```bash
# Extract from a single image (auto-detect document type)
python main.py extract samples/invoice.jpg

# Specify document type
python main.py extract samples/invoice.jpg --type invoice --output results/

# Batch process a directory
python main.py batch samples/ --type invoice

# Start REST API server
python main.py serve --host 0.0.0.0 --port 8000

# Show configuration
python main.py config show

# Save configuration to file
python main.py config save --path my_config.json
```

### REST API

```bash
# Start server
python main.py serve

# Health check
curl http://localhost:8000/health

# Extract from a single document
curl -X POST http://localhost:8000/extract \
  -F "file=@invoice.jpg" \
  -F "document_type=invoice"

# Batch extract
curl -X POST http://localhost:8000/extract/batch \
  -F "files=@doc1.jpg" \
  -F "files=@doc2.png"

# List document templates
curl http://localhost:8000/templates
```

## Project Structure

```
smart-doc-extractor/
├── config.py                          # Central configuration (dataclass)
├── main.py                            # CLI entry point
├── templates/
│   ├── invoice.json                   # Invoice field template
│   └── receipt.json                  # Receipt field template
├── src/
│   ├── __init__.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── image_enhancer.py          # Denoise, binarize, deskew, CLAHE
│   │   └── layout_analyzer.py         # Title, header, table, seal detection
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── text_detector.py           # MSER + contour text detection
│   │   └── region_classifier.py        # Region type classification
│   ├── recognition/
│   │   ├── __init__.py
│   │   ├── ocr_engine.py              # Pluggable OCR (Tesseract/EasyOCR)
│   │   └── text_postprocess.py        # Text cleanup and correction
│   ├── extraction/
│   │   ├── __init__.py
│   │   ├── field_types.py             # 20+ field type definitions
│   │   └── field_extractor.py         # Regex + rule extraction engine
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── extractor.py              # End-to-end pipeline orchestration
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── result_viewer.py           # Annotated images + HTML reports
│   ├── api/
│   │   ├── __init__.py
│   │   ├── server.py                  # FastAPI REST server
│   │   └── schemas.py                 # Pydantic request/response models
│   └── utils/
│       ├── logger.py                  # Logging setup
│       ├── image_utils.py             # Image I/O and utilities
│       └── text_utils.py              # Text normalization functions
├── samples/                           # Sample document images
├── output/                            # Default output directory
├── tests/                             # Unit tests
├── .gitignore
├── LICENSE
└── README.md
```

## Configuration

All parameters are managed in `config.py` via dataclass configs:

| Module | Config Class | Key Parameters |
|--------|-------------|----------------|
| Image I/O | `ImageConfig` | target_dpi, max_dimension, allowed_formats |
| Preprocessing | `PreprocessConfig` | denoise, binarize_method, deskew, clahe |
| Detection | `DetectionConfig` | method, min_region_area, nms_threshold |
| OCR | `OcrConfig` | engine, language, tesseract_config |
| Extraction | `ExtractionConfig` | document_type, confidence_threshold |
| API | `ApiConfig` | host, port, workers, cors_origins |

### Custom Configuration

```python
from config import AppConfig, OcrConfig

config = AppConfig()
config.ocr.engine = "easyocr"          # Use EasyOCR instead of Tesseract
config.ocr.language = "chi_sim+eng"    # Chinese + English
config.preprocess.binarize_method = "adaptive"
config.output.save_annotated_image = True

# Save to file
config.save("my_config.json")

# Load from file
config = AppConfig.load("my_config.json")
```

## Supported Document Types

| Type | Description | Key Fields |
|------|------------|------------|
| `invoice` | 增值税发票 | 发票号码, 发票代码, 开票日期, 价税合计 |
| `receipt` | 收据 | 日期, 金额, 收款人, 备注 |
| `bank_statement` | 银行流水 | 银行名称, 银行账号, 余额 |
| `contract` | 合同 | 甲方, 乙方, 日期, 金额 |

## Field Types (20+)

Identification: `invoice_number`, `invoice_code`, `seller_tax_id`, `buyer_tax_id`, `bank_account`, `page_number`

Temporal: `invoice_date`, `due_date`

Entity: `seller_name`, `buyer_name`, `receiver_name`, `bank_name`

Monetary: `amount_before_tax`, `tax_amount`, `amount_total`

Contact: `receiver_phone`, `receiver_address`

Metadata: `currency`, `payment_method`, `remarks`, `document_type`

## Pipeline Performance

| Stage | Typical Time | Notes |
|-------|-------------|-------|
| Image Load | 10-50ms | Depends on image size |
| Preprocessing | 50-200ms | Denoise + binarize + deskew |
| Text Detection | 100-500ms | MSER + contour analysis |
| OCR Recognition | 500-3000ms | Depends on engine and content |
| Field Extraction | 10-50ms | Regex matching |

## License

This project is licensed under the MIT License.
