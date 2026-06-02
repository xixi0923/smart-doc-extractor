"""
智能文档提取系统 —— 演示服务器
=================================
无需安装 Tesseract/EasyOCR，内置 Mock OCR 模块，
可直接运行用于前后端联调和界面演示。

使用方式:
    python demo_server.py          # 默认 http://localhost:8000
    python demo_server.py --port 9000
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import random
import sys
import time
import tempfile
import traceback
from pathlib import Path
from typing import Optional, List

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
    print("[警告] 未找到 opencv-python，图像标注功能将退化为纯色占位图")

# ========== 路径设置 ==========
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "src" / "api" / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

# 确保输出目录存在
(BASE_DIR / "output").mkdir(exist_ok=True)

# ========== Mock OCR 模拟数据 ==========

MOCK_FIELDS_INVOICE = {
    "invoice_number": {"value": "91120241031000001", "confidence": 0.98, "type": "invoice_number"},
    "invoice_code": {"value": "1100241430", "confidence": 0.96, "type": "invoice_code"},
    "invoice_date": {"value": "2024-10-31", "confidence": 0.97, "type": "invoice_date"},
    "seller_name": {"value": "北京科技有限公司", "confidence": 0.95, "type": "seller_name"},
    "seller_tax_id": {"value": "91110000MA01XXXXX", "confidence": 0.93, "type": "seller_tax_id"},
    "buyer_name": {"value": "上海贸易有限公司", "confidence": 0.94, "type": "buyer_name"},
    "buyer_tax_id": {"value": "91310000MA1AXXXXX", "confidence": 0.92, "type": "buyer_tax_id"},
    "amount_pretax": {"value": "¥10,000.00", "confidence": 0.99, "type": "amount_pretax"},
    "tax_rate": {"value": "13%", "confidence": 0.98, "type": "tax_rate"},
    "tax_amount": {"value": "¥1,300.00", "confidence": 0.99, "type": "tax_amount"},
    "amount_total": {"value": "¥11,300.00", "confidence": 0.99, "type": "amount_total"},
    "goods_description": {"value": "技术服务费", "confidence": 0.91, "type": "goods_description"},
}

MOCK_FIELDS_RECEIPT = {
    "invoice_date": {"value": "2024-11-15 14:32:00", "confidence": 0.97, "type": "invoice_date"},
    "amount_total": {"value": "¥286.50", "confidence": 0.99, "type": "amount_total"},
    "payment_method": {"value": "微信支付", "confidence": 0.95, "type": "payment_method"},
    "merchant_name": {"value": "全家便利店（朝阳门店）", "confidence": 0.93, "type": "merchant_name"},
    "goods_description": {"value": "便利店零售", "confidence": 0.88, "type": "goods_description"},
}

MOCK_FIELDS_BANK = {
    "bank_name": {"value": "中国工商银行", "confidence": 0.98, "type": "bank_name"},
    "bank_account": {"value": "6222 **** **** 1234", "confidence": 0.97, "type": "bank_account"},
    "amount_total": {"value": "¥50,000.00", "confidence": 0.99, "type": "amount_total"},
    "invoice_date": {"value": "2024-11-01 ~ 2024-11-30", "confidence": 0.95, "type": "invoice_date"},
    "seller_name": {"value": "张三", "confidence": 0.92, "type": "seller_name"},
}

MOCK_FIELDS_CONTRACT = {
    "seller_name": {"value": "甲方：北京科技有限公司", "confidence": 0.96, "type": "seller_name"},
    "buyer_name": {"value": "乙方：上海贸易有限公司", "confidence": 0.95, "type": "buyer_name"},
    "amount_total": {"value": "¥100,000.00", "confidence": 0.98, "type": "amount_total"},
    "invoice_date": {"value": "2024-10-01", "confidence": 0.94, "type": "invoice_date"},
    "contract_number": {"value": "HT-2024-10-0001", "confidence": 0.97, "type": "contract_number"},
}

MOCK_OCR_TEXTS = {
    "invoice": "发票代码: 1100241430\n发票号码: 91120241031000001\n开票日期: 2024年10月31日\n"
               "购买方名称: 上海贸易有限公司\n纳税人识别号: 91310000MA1AXXXXX\n"
               "销售方名称: 北京科技有限公司\n纳税人识别号: 91110000MA01XXXXX\n"
               "货物或应税劳务名称: 技术服务费\n金额: 10,000.00\n税率: 13%\n税额: 1,300.00\n"
               "价税合计（大写）: 壹万壹仟叁佰元整 (小写): ¥11,300.00",
    "receipt": "全家便利店（朝阳门店）\n日期: 2024-11-15 14:32\n"
               "商品明细:\n  饮料 ×2    ¥12.00\n  零食 ×3    ¥24.50\n  日用品 ×1  ¥250.00\n"
               "合计: ¥286.50\n支付方式: 微信支付\n谢谢惠顾！",
    "bank_statement": "中国工商银行 对账单\n账号: 6222 **** **** 1234\n户名: 张三\n"
                      "统计期间: 2024-11-01 至 2024-11-30\n期末余额: ¥50,000.00\n"
                      "收入合计: ¥80,000.00  支出合计: ¥30,000.00",
    "contract": "技术服务合同\n合同编号: HT-2024-10-0001\n"
                "甲方: 北京科技有限公司\n乙方: 上海贸易有限公司\n"
                "合同金额: 壹拾万元整（¥100,000.00）\n合同日期: 2024年10月01日",
}


def detect_doc_type_from_image(image_array) -> str:
    """根据图片的色彩特征简单判断文档类型（Demo 用）。"""
    if image_array is None:
        return "invoice"
    # 随机模拟，真实场景应用关键词检测
    return random.choice(["invoice", "receipt", "bank_statement", "contract"])


def generate_annotated_image(image_array, doc_type: str) -> Optional[bytes]:
    """在图片上绘制模拟标注框，返回 JPEG bytes。"""
    if not HAS_CV2 or image_array is None:
        return None

    img = image_array.copy()
    h, w = img.shape[:2]

    # 颜色映射
    colors = {
        "invoice_number": (220, 38, 38),
        "invoice_date": (37, 99, 235),
        "seller_name": (5, 150, 105),
        "buyer_name": (124, 58, 237),
        "amount_total": (220, 38, 38),
        "tax_amount": (234, 88, 12),
        "default": (79, 70, 229),
    }

    # 生成若干模拟文本框
    boxes = []
    row_h = max(28, h // 20)
    x_start = max(20, w // 10)
    x_end = min(w - 20, w * 9 // 10)
    for i in range(min(12, h // (row_h + 8))):
        y = 30 + i * (row_h + 8)
        box_w = random.randint(w // 3, x_end - x_start)
        boxes.append((x_start, y, box_w, row_h))

    labels = list(MOCK_FIELDS_INVOICE.keys()) if doc_type == "invoice" else list(MOCK_FIELDS_RECEIPT.keys())

    for i, (bx, by, bw, bh) in enumerate(boxes):
        label_key = labels[i % len(labels)]
        color = colors.get(label_key, colors["default"])
        # BGR for cv2
        bgr = (color[2], color[1], color[0])
        cv2.rectangle(img, (bx, by), (bx + bw, by + bh), bgr, 2)
        # 文字标注
        label = label_key.replace("_", " ")
        font_scale = 0.45
        thickness = 1
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        # 背景
        cv2.rectangle(img, (bx, by - th - 4), (bx + tw + 6, by), bgr, -1)
        cv2.putText(img, label, (bx + 3, by - 3), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # 标注右上角文档类型
    doc_label = {"invoice": "INVOICE", "receipt": "RECEIPT",
                 "bank_statement": "BANK STMT", "contract": "CONTRACT"}.get(doc_type, "DOCUMENT")
    cv2.rectangle(img, (w - 130, 4), (w - 4, 28), (0, 0, 0), -1)
    cv2.putText(img, doc_label, (w - 125, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 128), 1, cv2.LINE_AA)

    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return bytes(buf)


# ========== FastAPI 应用 ==========

app = FastAPI(
    title="智能文档提取系统 API",
    description="智能文档信息提取演示服务，支持发票、收据、银行流水、合同等文档类型自动识别和字段提取。",
    version="0.1.0",
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
    return HTMLResponse("<h1>智能文档提取系统</h1><p>前端文件未找到，请检查 src/api/static/index.html。</p>")


@app.get("/health")
async def health():
    """健康检查。"""
    return {
        "status": "healthy",
        "version": "0.1.0",
        "mode": "demo",
        "ocr_engine": "mock",
        "ocr_available": True,
        "message": "演示模式运行中，使用 Mock OCR 模拟真实结果",
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
                "field_types": list(MOCK_FIELDS_INVOICE.keys()),
            },
            {
                "document_type": "receipt",
                "name": "收据小票",
                "description": "自动提取日期、金额、支付方式、商户名称等字段",
                "field_types": list(MOCK_FIELDS_RECEIPT.keys()),
            },
            {
                "document_type": "bank_statement",
                "name": "银行流水",
                "description": "自动提取银行名称、账号、余额、统计期间等字段",
                "field_types": list(MOCK_FIELDS_BANK.keys()),
            },
            {
                "document_type": "contract",
                "name": "合同",
                "description": "自动提取合同编号、甲乙方、金额、签订日期等字段",
                "field_types": list(MOCK_FIELDS_CONTRACT.keys()),
            },
        ],
        "total": 4,
    }


def _build_mock_response(image_bytes: bytes, document_type: str) -> dict:
    """
    根据文档类型构建 Mock 提取响应。
    同时生成带标注框的图像。
    """
    t_start = time.time()

    # 加载图像
    image_array = None
    if HAS_CV2:
        try:
            arr = np.frombuffer(image_bytes, dtype=np.uint8)
            image_array = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            pass

    # 自动检测文档类型
    if document_type == "auto":
        document_type = detect_doc_type_from_image(image_array)

    # 选择字段和 OCR 文本
    fields_map = {
        "invoice": MOCK_FIELDS_INVOICE,
        "receipt": MOCK_FIELDS_RECEIPT,
        "bank_statement": MOCK_FIELDS_BANK,
        "contract": MOCK_FIELDS_CONTRACT,
    }
    fields = fields_map.get(document_type, MOCK_FIELDS_INVOICE)
    ocr_text = MOCK_OCR_TEXTS.get(document_type, "文档内容识别中...")

    # 加少量随机扰动使每次结果略有不同（模拟真实 OCR 差异）
    perturbed_fields = {}
    for k, v in fields.items():
        perturbed_fields[k] = {
            "value": v["value"],
            "confidence": min(1.0, v["confidence"] + random.uniform(-0.03, 0.03)),
            "type": v["type"],
        }

    # 模拟各阶段耗时
    base_time = random.uniform(0.05, 0.12)
    stage_timings = {
        "图像预处理": round(base_time * 100, 1),
        "文本检测": round(base_time * 180, 1),
        "版面分析": round(base_time * 60, 1),
        "OCR识别": round(base_time * 900, 1),
        "字段抽取": round(base_time * 120, 1),
    }

    # 生成标注图
    annotated_b64 = ""
    if HAS_CV2 and image_array is not None:
        try:
            annotated_bytes = generate_annotated_image(image_array, document_type)
            if annotated_bytes:
                annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(annotated_bytes).decode()
        except Exception as e:
            print(f"[警告] 标注图生成失败: {e}")

    total_ms = (time.time() - t_start) * 1000 + sum(stage_timings.values())
    high_conf = sum(1 for f in perturbed_fields.values() if f["confidence"] >= 0.9)

    doc_names = {
        "invoice": "增值税发票",
        "receipt": "收据小票",
        "bank_statement": "银行流水",
        "contract": "合同",
    }

    return {
        "success": True,
        "source": "[上传文件]",
        "document_type": document_type,
        "document_type_name": doc_names.get(document_type, document_type),
        "total_fields": len(perturbed_fields),
        "high_confidence_fields": high_conf,
        "total_time_ms": round(total_ms, 1),
        "fields": perturbed_fields,
        "ocr_text": ocr_text,
        "ocr_confidence": round(random.uniform(0.88, 0.96), 3),
        "annotated_image": annotated_b64,
        "stage_timings": stage_timings,
        "mode": "demo",
    }


@app.post("/extract")
async def extract_document(
    file: UploadFile = File(...),
    document_type: str = Form(default="auto"),
):
    """单张文档提取（不含标注图）。"""
    content = await file.read()
    resp = _build_mock_response(content, document_type)
    resp.pop("annotated_image", None)
    return JSONResponse(content=resp)


@app.post("/extract/visual")
async def extract_with_visual(
    file: UploadFile = File(...),
    document_type: str = Form(default="auto"),
):
    """单张文档提取（含标注图 base64）——前端调试控制台使用此端点。"""
    content = await file.read()
    resp = _build_mock_response(content, document_type)
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
        r = _build_mock_response(content, document_type)
        r.pop("annotated_image", None)
        results.append(r)

    total_ms = (time.time() - t_start) * 1000
    return {
        "total": len(files),
        "successful": len(results),
        "failed": 0,
        "total_time_ms": round(total_ms, 1),
        "results": results,
    }


# ========== 入口 ==========

def main():
    parser = argparse.ArgumentParser(description="智能文档提取系统演示服务")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址 (默认: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="监听端口 (默认: 8000)")
    parser.add_argument("--reload", action="store_true", help="开发模式热重载")
    args = parser.parse_args()

    print("=" * 55)
    print("  智能文档提取系统 —— 演示服务器")
    print("=" * 55)
    print(f"  前端控制台:  http://localhost:{args.port}/")
    print(f"  API 文档:    http://localhost:{args.port}/docs")
    print(f"  健康检查:    http://localhost:{args.port}/health")
    print("  模式:        Demo（Mock OCR，无需 Tesseract）")
    print("=" * 55)

    uvicorn.run(
        "demo_server:app" if args.reload else app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
