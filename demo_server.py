"""
智能文档提取系统 —— 真实 OCR 服务器
====================================
集成 RapidOCR（基于 PaddleOCR + ONNX，模型内嵌无需联网下载）
对上传图片进行真实文字识别，并通过正则+规则引擎提取结构化字段。

使用方式:
    python demo_server.py          # 默认 http://localhost:8000
    python demo_server.py --port 9000
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
import sys
import time
import tempfile
import traceback
from pathlib import Path
from typing import Optional, List, Dict, Tuple

# ========== 依赖检查 ==========
try:
    import uvicorn
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
except ImportError:
    print("[错误] 缺少 FastAPI/uvicorn，请先运行：")
    print("  pip install fastapi uvicorn python-multipart")
    sys.exit(1)

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("[警告] 未找到 opencv-python，标注图功能将不可用")

# ========== 路径设置 ==========
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "src" / "api" / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

# 确保输出目录存在
(BASE_DIR / "output").mkdir(exist_ok=True)

# ========== 全局 OCR 引擎（延迟初始化） ==========
_rapidocr_engine = None


def get_ocr_engine():
    """获取或初始化 RapidOCR 引擎（基于 PaddleOCR + ONNX，模型内嵌）。"""
    global _rapidocr_engine
    if _rapidocr_engine is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            print("[初始化] 正在加载 RapidOCR 引擎（模型内嵌，无需联网）...")
            _rapidocr_engine = RapidOCR()
            print("[初始化] RapidOCR 引擎加载完成")
        except ImportError:
            print("[错误] 未安装 rapidocr-onnxruntime，请运行：")
            print("  pip install rapidocr-onnxruntime")
            raise
        except Exception as e:
            print(f"[错误] RapidOCR 初始化失败: {e}")
            raise
    return _rapidocr_engine


# ========== 图像预处理 ==========
def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    对图像进行预处理以提升 OCR 识别效果。
    - 缩放至合理尺寸（最长边不超过 2000px）
    - CLAHE 对比度增强
    """
    if not HAS_CV2:
        return image

    # 如果图片太大，缩放到合理尺寸
    h, w = image.shape[:2]
    max_dim = 2000
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # CLAHE 对比度增强（对低对比度图片有效）
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        image = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    return image


# ========== OCR 识别 ==========
def run_ocr(image: np.ndarray) -> Tuple[List[dict], str, float]:
    """
    使用 RapidOCR 对图片进行文字识别。

    Returns:
        (results, full_text, avg_confidence)
        - results: 列表，每项 {"text": str, "confidence": float, "bbox": (x, y, w, h)}
        - full_text: 完整识别文本
        - avg_confidence: 平均置信度
    """
    t_start = time.time()

    engine = get_ocr_engine()

    # 运行 RapidOCR 识别
    result, elapse = engine(image)

    results = []
    texts = []
    confidences = []

    if result:
        for item in result:
            bbox_pts = item[0]  # 4 个角点 [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            text = item[1]
            conf = float(item[2])  # RapidOCR 返回字符串格式的置信度

            if not text.strip():
                continue

            # 转换 4 角点为 (x, y, w, h) 矩形
            pts = np.array(bbox_pts, dtype=np.int32)
            x = int(pts[:, 0].min())
            y = int(pts[:, 1].min())
            w = int(pts[:, 0].max()) - x
            h = int(pts[:, 1].max()) - y

            results.append({
                "text": text,
                "confidence": conf,
                "bbox": (x, y, w, h),
            })
            texts.append(text)
            confidences.append(conf)

    full_text = "\n".join(texts)
    avg_conf = float(np.mean(confidences)) if confidences else 0.0
    elapsed = (time.time() - t_start) * 1000

    print(f"  [OCR] 识别到 {len(results)} 个文本区域，平均置信度 {avg_conf:.2f}，耗时 {elapsed:.0f}ms")

    return results, full_text, avg_conf


# ========== 文档类型自动检测 ==========
def detect_document_type(text: str) -> str:
    """根据 OCR 文本内容自动判断文档类型。"""
    type_keywords = {
        "invoice": [
            "发票", "增值税", "invoice", "税额", "纳税人识别号",
            "价税合计", "开票日期", "发票号码", "发票代码",
            "销售方", "购买方", "税率",
        ],
        "receipt": [
            "收据", "receipt", "收款", "已收", "付款",
            "微信支付", "支付宝", "刷卡", "现金",
            "商户", "消费", "小票",
        ],
        "bank_statement": [
            "银行流水", "对账单", "statement", "交易记录",
            "账户", "余额", "借方", "贷方", "转账",
            "存入", "支出", "汇款",
        ],
        "contract": [
            "合同", "协议", "contract", "甲方", "乙方",
            "签署", "条款", "履行", "违约", "约定",
        ],
    }

    scores = {}
    for doc_type, keywords in type_keywords.items():
        score = sum(1 for kw in keywords if kw in text)
        scores[doc_type] = score

    best_type = max(scores, key=scores.get)
    if scores[best_type] > 0:
        return best_type

    return "unknown"


# ========== 字段提取（正则+规则引擎） ==========
FIELD_DEFINITIONS = {
    "invoice_number": {
        "display_name": "发票号码",
        "patterns": [
            r"(?:发票号码|No\.?|编号)[：:\s]*(\d{8,20})",
            r"号码[：:\s]*(\d{8,20})",
        ],
        "category": "identification",
    },
    "invoice_code": {
        "display_name": "发票代码",
        "patterns": [
            r"(?:发票代码)[：:\s]*(\d{10,12})",
        ],
        "category": "identification",
    },
    "invoice_date": {
        "display_name": "开票日期",
        "patterns": [
            r"(?:开票日期|日期|Date|时间)[：:\s]*(\d{4}[\s年/\-.]\d{1,2}[\s月/\-.]\d{1,2}日?)",
            r"(\d{4}[\s年/\-.]\d{1,2}[\s月/\-.]\d{1,2})",
        ],
        "category": "temporal",
    },
    "seller_name": {
        "display_name": "销售方名称",
        "patterns": [
            r"(?:销方|销售方|卖方|Seller)[：:\s]*([\u4e00-\u9fa5A-Za-z\s]+(?:公司|企业|厂|商店|事务所|银行|中心|Ltd|Inc))",
            r"(?:名称)[：:\s]*([\u4e00-\u9fa5]+(?:公司|企业|厂|商店|银行))",
        ],
        "category": "entity",
    },
    "buyer_name": {
        "display_name": "购买方名称",
        "patterns": [
            r"(?:购方|购买方|买方|Buyer|客户)[：:\s]*([\u4e00-\u9fa5A-Za-z\s]+(?:公司|企业|厂|商店|事务所|银行|中心|Ltd|Inc))",
        ],
        "category": "entity",
    },
    "seller_tax_id": {
        "display_name": "销售方税号",
        "patterns": [
            r"(?:纳税人识别号|税号)[：:\s]*([A-Z0-9]{15,20})",
        ],
        "category": "identification",
    },
    "buyer_tax_id": {
        "display_name": "购买方税号",
        "patterns": [
            r"(?:购方.*?纳税人识别号|买方.*?税号)[：:\s]*([A-Z0-9]{15,20})",
        ],
        "category": "identification",
    },
    "amount_pretax": {
        "display_name": "金额(不含税)",
        "patterns": [
            r"(?:金额|不含税|Amount)[：:\s]*[¥￥$]?\s*([\d,]+\.?\d*)",
        ],
        "category": "monetary",
    },
    "tax_amount": {
        "display_name": "税额",
        "patterns": [
            r"(?:税额|Tax)[：:\s]*[¥￥$]?\s*([\d,]+\.?\d*)",
        ],
        "category": "monetary",
    },
    "amount_total": {
        "display_name": "价税合计",
        "patterns": [
            r"(?:价税合计|总计|总金额|合计|Total)[：:\s]*[¥￥$]?\s*([\d,]+\.?\d*)",
            r"[¥￥]\s*([\d,]+\.?\d*)",
        ],
        "category": "monetary",
    },
    "tax_rate": {
        "display_name": "税率",
        "patterns": [
            r"(?:税率)[：:\s]*(\d+\.?\d*%)",
        ],
        "category": "monetary",
    },
    "payment_method": {
        "display_name": "支付方式",
        "patterns": [
            r"(?:支付方式|付款方式|Payment)[：:\s]*(\S+)",
        ],
        "category": "metadata",
    },
    "bank_name": {
        "display_name": "开户银行",
        "patterns": [
            r"(?:开户行|开户银行|Bank)[：:\s]*([\u4e00-\u9fa5A-Za-z]+(?:银行|支行))",
        ],
        "category": "entity",
    },
    "bank_account": {
        "display_name": "银行账号",
        "patterns": [
            r"(?:账号|银行账号|Account)[：:\s]*(\d{10,30})",
        ],
        "category": "identification",
    },
    "receiver_name": {
        "display_name": "收款人",
        "patterns": [
            r"(?:收款人|收款单位|Payee)[：:\s]*([\u4e00-\u9fa5A-Za-z]+)",
        ],
        "category": "entity",
    },
    "phone": {
        "display_name": "联系电话",
        "patterns": [
            r"(?:电话|联系电话|Tel|Phone)[：:\s]*([\d\-\s]{10,15})",
        ],
        "category": "contact",
    },
    "remarks": {
        "display_name": "备注",
        "patterns": [
            r"(?:备注|Remark|Note)[：:\s]*(.+)",
        ],
        "category": "metadata",
    },
    "goods_description": {
        "display_name": "货物/劳务名称",
        "patterns": [
            r"(?:货物|劳务|商品|名称)[：:\s]*([\u4e00-\u9fa5A-Za-z\s]+)",
        ],
        "category": "metadata",
    },
    "contract_number": {
        "display_name": "合同编号",
        "patterns": [
            r"(?:合同编号|编号)[：:\s]*(\S+)",
        ],
        "category": "identification",
    },
    "merchant_name": {
        "display_name": "商户名称",
        "patterns": [
            r"(?:商户|门店|店名|Merchant)[：:\s]*([\u4e00-\u9fa5A-Za-z\s]+)",
        ],
        "category": "entity",
    },
}

DOC_TYPE_FIELDS = {
    "invoice": [
        "invoice_number", "invoice_code", "invoice_date", "seller_name",
        "buyer_name", "seller_tax_id", "buyer_tax_id", "amount_pretax",
        "tax_amount", "amount_total", "tax_rate", "goods_description",
    ],
    "receipt": [
        "invoice_date", "amount_total", "payment_method", "merchant_name",
        "goods_description",
    ],
    "bank_statement": [
        "bank_name", "bank_account", "amount_total", "invoice_date",
        "seller_name",
    ],
    "contract": [
        "contract_number", "seller_name", "buyer_name",
        "amount_total", "invoice_date",
    ],
    "unknown": [
        "invoice_number", "invoice_date", "amount_total", "seller_name",
        "buyer_name", "phone", "remarks",
    ],
}


def extract_fields(text: str, document_type: str) -> Tuple[Dict[str, dict], str]:
    """从 OCR 文本中提取结构化字段。"""
    if document_type == "auto":
        document_type = detect_document_type(text)

    field_keys = DOC_TYPE_FIELDS.get(document_type, DOC_TYPE_FIELDS["unknown"])
    extracted = {}

    for key in field_keys:
        field_def = FIELD_DEFINITIONS.get(key)
        if not field_def:
            continue

        patterns = field_def.get("patterns", [])
        best_value = None
        best_conf = 0.0

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                value = match.group(1) if match.groups() else match.group()
                value = value.strip()

                # 计算置信度
                conf = 0.5
                if re.search(r"[\u4e00-\u9fff]", pattern):
                    conf += 0.25
                if len(value) >= 2:
                    conf += 0.1
                validation = field_def.get("validation")
                if validation and re.match(validation, value):
                    conf += 0.15
                conf = min(conf, 0.99)

                if conf > best_conf:
                    best_value = value
                    best_conf = conf

        if best_value is not None:
            extracted[key] = {
                "value": best_value,
                "confidence": round(best_conf, 3),
                "type": key,
            }

    return extracted, document_type


# ========== 生成标注图 ==========
def generate_annotated_image(image: np.ndarray, ocr_results: List[dict],
                              extracted_fields: Dict[str, dict]) -> Optional[bytes]:
    """在原图上绘制 OCR 检测框和提取字段标注，返回 JPEG bytes。"""
    if not HAS_CV2 or image is None:
        return None

    img = image.copy()
    h, w = img.shape[:2]

    # 颜色映射（BGR 格式）
    color_map = {
        "monetary": (0, 0, 220),
        "identification": (220, 120, 0),
        "entity": (0, 180, 0),
        "temporal": (200, 0, 200),
        "contact": (0, 200, 200),
        "metadata": (128, 128, 128),
    }
    default_color = (79, 70, 229)

    # 绘制所有 OCR 检测框（浅色底框）
    for item in ocr_results:
        x, y, bw, bh = item["bbox"]
        conf = item["confidence"]
        # 置信度越高，框越明显
        alpha = 0.3 + conf * 0.5
        color = tuple(int(c * alpha) for c in (180, 180, 180))
        cv2.rectangle(img, (x, y), (x + bw, y + bh), color, 1)

    # 高亮标注已提取的字段
    text_to_bbox = {}
    for item in ocr_results:
        text_to_bbox[item["text"].strip()] = item["bbox"]

    for field_key, field_data in extracted_fields.items():
        value = field_data["value"]
        conf = field_data["confidence"]
        field_def = FIELD_DEFINITIONS.get(field_key, {})
        category = field_def.get("category", "metadata")
        bgr_color = color_map.get(category, default_color)

        # 尝试找到该字段值对应的 bbox
        matched_bbox = None
        for ocr_text, bbox in text_to_bbox.items():
            if value in ocr_text or ocr_text in value:
                matched_bbox = bbox
                break

        if matched_bbox:
            x, y, bw, bh = matched_bbox
            thickness = 3 if conf >= 0.8 else 2
            cv2.rectangle(img, (x, y), (x + bw, y + bh), bgr_color, thickness)

            # 绘制标签
            label = field_def.get("display_name", field_key)
            font_scale = max(0.35, min(0.6, bw / 200))
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            label_y = max(y - 6, th + 4)
            cv2.rectangle(img, (x, label_y - th - 4), (x + tw + 8, label_y + 2), bgr_color, -1)
            cv2.putText(img, label, (x + 3, label_y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return bytes(buf)


# ========== FastAPI 应用 ==========
app = FastAPI(
    title="智能文档提取系统 API",
    description="基于 RapidOCR 的智能文档信息提取服务，支持发票、收据、银行流水、合同等文档类型自动识别和字段提取。",
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ========== 路由 ==========

@app.get("/", response_class=HTMLResponse)
async def index():
    """前端调试控制台主页。"""
    idx = STATIC_DIR / "index.html"
    if idx.is_file():
        return FileResponse(str(idx), media_type="text/html")
    return HTMLResponse("<h1>智能文档提取系统</h1><p>前端文件未找到。</p>")


@app.get("/health")
async def health():
    """健康检查。"""
    ocr_ok = False
    ocr_engine_name = "unknown"
    try:
        engine = get_ocr_engine()
        ocr_ok = True
        ocr_engine_name = "rapidocr"
    except Exception:
        pass

    return {
        "status": "healthy" if ocr_ok else "degraded",
        "version": "0.2.0",
        "mode": "real-ocr",
        "ocr_engine": ocr_engine_name,
        "ocr_available": ocr_ok,
        "message": "RapidOCR 真实识别模式" if ocr_ok else "OCR 引擎不可用",
    }


@app.get("/templates")
async def list_templates():
    """列出支持的文档类型模板。"""
    return {
        "templates": [
            {
                "document_type": "invoice",
                "name": "增值税发票",
                "description": "自动提取发票号码、代码、日期、买卖双方、金额、税率等字段",
                "field_types": DOC_TYPE_FIELDS.get("invoice", []),
            },
            {
                "document_type": "receipt",
                "name": "收据小票",
                "description": "自动提取日期、金额、支付方式、商户名称等字段",
                "field_types": DOC_TYPE_FIELDS.get("receipt", []),
            },
            {
                "document_type": "bank_statement",
                "name": "银行流水",
                "description": "自动提取银行名称、账号、余额、统计期间等字段",
                "field_types": DOC_TYPE_FIELDS.get("bank_statement", []),
            },
            {
                "document_type": "contract",
                "name": "合同",
                "description": "自动提取合同编号、甲乙方、金额、签订日期等字段",
                "field_types": DOC_TYPE_FIELDS.get("contract", []),
            },
        ],
        "total": 4,
    }


def _process_image(image_bytes: bytes, document_type: str) -> dict:
    """
    核心处理逻辑：对图片进行完整的 OCR + 字段提取。

    阶段：加载图像 → 预处理 → OCR识别 → 文档类型检测 → 字段提取 → 标注图生成
    """
    t_total = time.time()

    # ---- 阶段 1: 加载图像 ----
    t_load = time.time()
    image_array = None
    if HAS_CV2:
        try:
            arr = np.frombuffer(image_bytes, dtype=np.uint8)
            image_array = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception as e:
            return {"success": False, "error": f"图像加载失败: {e}"}

    if image_array is None:
        return {"success": False, "error": "无法解析上传的图片，请确认文件格式正确（支持 JPG/PNG/BMP/TIFF）"}

    load_ms = (time.time() - t_load) * 1000
    print(f"  [加载] 图像尺寸 {image_array.shape[1]}x{image_array.shape[0]}，耗时 {load_ms:.0f}ms")

    # ---- 阶段 2: 图像预处理 ----
    t_preprocess = time.time()
    processed = preprocess_image(image_array)
    preprocess_ms = (time.time() - t_preprocess) * 1000

    # ---- 阶段 3: OCR 识别 ----
    t_ocr = time.time()
    try:
        ocr_results, full_text, avg_confidence = run_ocr(processed)
    except Exception as e:
        print(f"  [OCR 错误] {e}")
        traceback.print_exc()
        return {"success": False, "error": f"OCR 识别失败: {e}"}
    ocr_ms = (time.time() - t_ocr) * 1000

    if not ocr_results:
        return {
            "success": True,
            "source": "[上传文件]",
            "document_type": "unknown",
            "total_fields": 0,
            "high_confidence_fields": 0,
            "total_time_ms": (time.time() - t_total) * 1000,
            "fields": {},
            "ocr_text": "",
            "ocr_confidence": 0.0,
            "annotated_image": "",
            "stage_timings": {
                "图像加载": round(load_ms, 1),
                "图像预处理": round(preprocess_ms, 1),
                "OCR识别": round(ocr_ms, 1),
            },
            "mode": "real-ocr",
            "warning": "未识别到任何文字，请检查图片是否清晰、是否包含文字内容",
        }

    # ---- 阶段 4: 文档类型检测 ----
    t_detect = time.time()
    if document_type == "auto":
        document_type = detect_document_type(full_text)
    detect_ms = (time.time() - t_detect) * 1000

    doc_names = {
        "invoice": "增值税发票",
        "receipt": "收据小票",
        "bank_statement": "银行流水",
        "contract": "合同",
        "unknown": "未知类型",
    }

    print(f"  [类型检测] 文档类型: {document_type}（{doc_names.get(document_type, document_type)}），耗时 {detect_ms:.0f}ms")

    # ---- 阶段 5: 字段提取 ----
    t_extract = time.time()
    fields, detected_type = extract_fields(full_text, document_type)
    extract_ms = (time.time() - t_extract) * 1000

    if document_type == "auto":
        document_type = detected_type

    high_conf = sum(1 for f in fields.values() if f["confidence"] >= 0.7)

    print(f"  [字段提取] 提取到 {len(fields)} 个字段（{high_conf} 个高置信度），耗时 {extract_ms:.0f}ms")

    # ---- 阶段 6: 生成标注图 ----
    t_annotate = time.time()
    annotated_b64 = ""
    try:
        annotated_bytes = generate_annotated_image(processed, ocr_results, fields)
        if annotated_bytes:
            annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(annotated_bytes).decode()
    except Exception as e:
        print(f"  [标注图] 生成失败: {e}")
    annotate_ms = (time.time() - t_annotate) * 1000

    total_ms = (time.time() - t_total) * 1000

    stage_timings = {
        "图像加载": round(load_ms, 1),
        "图像预处理": round(preprocess_ms, 1),
        "OCR识别": round(ocr_ms, 1),
        "文档类型检测": round(detect_ms, 1),
        "字段提取": round(extract_ms, 1),
        "标注图生成": round(annotate_ms, 1),
    }

    return {
        "success": True,
        "source": "[上传文件]",
        "document_type": document_type,
        "document_type_name": doc_names.get(document_type, document_type),
        "total_fields": len(fields),
        "high_confidence_fields": high_conf,
        "total_time_ms": round(total_ms, 1),
        "fields": fields,
        "ocr_text": full_text,
        "ocr_confidence": round(avg_confidence, 3),
        "annotated_image": annotated_b64,
        "stage_timings": stage_timings,
        "mode": "real-ocr",
    }


@app.post("/extract")
async def extract_document(
    file: UploadFile = File(...),
    document_type: str = Form(default="auto"),
):
    """单张文档提取（不含标注图，响应更快）。"""
    content = await file.read()
    resp = _process_image(content, document_type)
    resp.pop("annotated_image", None)
    return JSONResponse(content=resp)


@app.post("/extract/visual")
async def extract_with_visual(
    file: UploadFile = File(...),
    document_type: str = Form(default="auto"),
):
    """单张文档提取（含标注图 base64）——前端调试控制台使用此端点。"""
    content = await file.read()
    resp = _process_image(content, document_type)
    return JSONResponse(content=resp)


@app.post("/extract/batch")
async def extract_batch(
    files: List[UploadFile] = File(...),
    document_type: str = Form(default="auto"),
):
    """批量文档提取。"""
    t_start = time.time()
    results = []
    for f in files:
        content = await f.read()
        r = _process_image(content, document_type)
        r.pop("annotated_image", None)
        results.append(r)

    total_ms = (time.time() - t_start) * 1000
    return {
        "total": len(files),
        "successful": sum(1 for r in results if r.get("success")),
        "failed": sum(1 for r in results if not r.get("success")),
        "total_time_ms": round(total_ms, 1),
        "results": results,
    }


# ========== 入口 ==========

def main():
    parser = argparse.ArgumentParser(description="智能文档提取系统 —— RapidOCR 真实识别服务")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址 (默认: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="监听端口 (默认: 8000)")
    parser.add_argument("--reload", action="store_true", help="开发模式热重载")
    args = parser.parse_args()

    # 预加载 OCR 模型
    print("=" * 60)
    print("  智能文档提取系统 —— RapidOCR 真实识别服务")
    print("=" * 60)
    print("  正在初始化 OCR 引擎...")
    try:
        get_ocr_engine()
        print("  RapidOCR 引擎就绪（基于 PaddleOCR + ONNX，中英文支持）")
    except Exception as e:
        print(f"  OCR 引擎加载失败: {e}")
        print("  将在首次请求时重试加载")

    print(f"\n  前端控制台:  http://localhost:{args.port}/")
    print(f"  API 文档:    http://localhost:{args.port}/docs")
    print(f"  健康检查:    http://localhost:{args.port}/health")
    print("  模式:        真实 OCR（RapidOCR 中英文识别）")
    print("=" * 60)

    uvicorn.run(
        "demo_server:app" if args.reload else app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
