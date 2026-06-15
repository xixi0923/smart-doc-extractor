# Smart Doc Extractor - 智能文档信息提取系统

从发票、收据、银行流水、合同等文档图像中自动提取结构化字段信息的 Python 系统。

## 功能特性

- **多文档类型支持**：增值税发票、收据、银行流水、合同，支持自动类型检测
- **5 级流水线架构**：图像预处理 → 文本检测 → 版面分析 → OCR 识别 → 字段提取
- **可插拔 OCR 引擎**：Tesseract / EasyOCR / RapidOCR，灵活切换
- **智能版面分析**：自动检测标题、正文、表格、印章等区域，印章检测基于 RGB 颜色空间
- **22+ 预定义字段类型**：正则 + 规则引擎抽取，支持自定义模板扩展
- **置信度评分**：每个字段附带置信度分数，低置信度标记需人工复核
- **中文标注渲染**：基于 PIL/Pillow 渲染中文字符，标注图无乱码
- **REST API 服务**：FastAPI 提供单张/批量处理、可视化标注等接口
- **可视化结果**：在原图上标注提取框 + HTML 报告

## 架构

```
┌──────────┐   ┌───────────┐   ┌───────────┐   ┌──────────┐   ┌──────────┐
│  图像输入  │──▶│  图像预处理  │──▶│ 文本检测&  │──▶│ OCR 识别  │──▶│ 字段提取  │
│          │   │          │   │ 版面分析   │   │          │   │          │
└──────────┘   └───────────┘   └───────────┘   └──────────┘   └──────────┘
               去噪/二值化     MSER + 轮廓    Tesseract/     正则 + 规则
               CLAHE/纠偏     NMS/分类       EasyOCR/       置信度评分
               去边框         印章检测(RGB)   RapidOCR       模板匹配
```

> **关键设计**：预处理阶段同时输出二值图（用于文本检测）和灰度图（用于 OCR），印章检测使用原始 RGB 图像以保留颜色信息。

## 技术栈

| 组件 | 技术 |
|------|------|
| 图像处理 | OpenCV, NumPy |
| OCR 引擎 | Tesseract / EasyOCR / RapidOCR (ONNX) |
| REST API | FastAPI + Uvicorn |
| 数据校验 | Pydantic |
| 中文渲染 | Pillow (PIL) + 系统中文字体 |
| 配置管理 | Python dataclass |
| 文本渲染 | Pillow (支持中文) |

## 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/xixi0923/smart-doc-extractor.git
cd smart-doc-extractor

# 安装核心依赖
pip install opencv-python pytesseract numpy fastapi uvicorn pydantic pillow

# 可选：安装其他 OCR 后端
pip install easyocr                # EasyOCR 后端
pip install rapidocr-onnxruntime   # RapidOCR 后端（推荐，内嵌模型无需联网）

# 安装 Tesseract OCR 系统二进制（使用 Tesseract 后端时必需）
# Ubuntu/Debian: sudo apt install tesseract-ocr tesseract-ocr-chi-sim
# macOS: brew install tesseract tesseract-lang
# Windows: 从 https://github.com/UB-Mannheim/tesseract/wiki 下载安装器
```

### CLI 使用

```bash
# 从单张图片提取（自动检测文档类型）
python main.py extract samples/invoice.jpg

# 指定文档类型
python main.py extract samples/invoice.jpg --type invoice --output results/

# 批量处理目录
python main.py batch samples/ --type invoice

# 启动 REST API 服务
python main.py serve --host 0.0.0.0 --port 8000

# 使用 RapidOCR 模式启动演示服务
python demo_server.py --port 8000
```

### 使用 RapidOCR 快速体验

```bash
pip install rapidocr-onnxruntime fastapi uvicorn python-multipart opencv-python
python demo_server.py
# 访问 http://localhost:8000/ 打开前端调试控制台
```

## API 文档

启动服务后访问 `http://localhost:8000/docs` 查看完整 Swagger 文档。

### 健康检查

```bash
curl http://localhost:8000/health
```

### 单张文档提取

```bash
curl -X POST http://localhost:8000/extract \
  -F "file=@invoice.jpg" \
  -F "document_type=invoice"
```

### 可视化提取（含标注图）

```bash
curl -X POST http://localhost:8000/extract/visual \
  -F "file=@invoice.jpg" \
  -F "document_type=auto"
```

返回结果中包含 `annotated_image` 字段（base64 编码的 JPEG 标注图）。

### 批量提取

```bash
curl -X POST http://localhost:8000/extract/batch \
  -F "files=@doc1.jpg" \
  -F "files=@doc2.png" \
  -F "document_type=auto"
```

### 查询文档模板

```bash
curl http://localhost:8000/templates
```

## 项目结构

```
smart-doc-extractor/
├── config.py                          # 中心配置（dataclass）
├── main.py                            # CLI 入口
├── demo_server.py                     # RapidOCR 演示服务
├── templates/
│   ├── invoice.json                   # 发票字段模板
│   └── receipt.json                   # 收据字段模板
├── src/
│   ├── __init__.py
│   ├── preprocessing/
│   │   ├── image_enhancer.py          # 去噪、二值化、纠偏、CLAHE、返回灰度+二值
│   │   └── layout_analyzer.py         # 标题、页眉页脚、表格、印章(RGB)检测
│   ├── detection/
│   │   ├── text_detector.py           # MSER + 轮廓文本检测
│   │   └── region_classifier.py       # 区域类型分类
│   ├── recognition/
│   │   ├── ocr_engine.py             # 可插拔 OCR（Tesseract/EasyOCR/RapidOCR）
│   │   └── text_postprocess.py        # 文本清理和校正
│   ├── extraction/
│   │   ├── field_types.py             # 22+ 字段类型定义
│   │   └── field_extractor.py         # 正则 + 规则提取引擎
│   ├── pipeline/
│   │   └── extractor.py              # 端到端流水线编排
│   ├── visualization/
│   │   └── result_viewer.py           # 标注图（PIL 中文渲染）+ HTML 报告
│   ├── api/
│   │   ├── server.py                  # FastAPI REST 服务
│   │   └── schemas.py                 # Pydantic 请求/响应模型
│   └── utils/
│       ├── logger.py                  # 日志配置
│       ├── image_utils.py             # 图像 I/O 工具
│       └── text_utils.py              # 文本归一化函数
├── samples/                           # 示例文档图片
├── output/                            # 默认输出目录
├── tests/                             # 单元测试
├── .gitignore
├── LICENSE
└── README.md
```

## 支持的文档类型

| 类型 | 说明 | 关键字段 |
|------|------|----------|
| `invoice` | 增值税发票 | 发票号码、发票代码、开票日期、销售方、购买方、价税合计 |
| `receipt` | 收据 | 日期、金额、收款人、备注 |
| `bank_statement` | 银行流水 | 银行名称、银行账号、金额合计 |
| `contract` | 合同 | 销售方、购买方、日期、金额 |
| `auto` | 自动检测 | 系统自动判断文档类型 |

## 字段类型

### 标识字段（Identification）

| 字段 | 中文名 | 说明 |
|------|--------|------|
| `invoice_number` | 发票号码 | 8-20位数字 |
| `invoice_code` | 发票代码 | 10-12位数字 |
| `seller_tax_id` | 销售方税号 | 15-20位 |
| `buyer_tax_id` | 购买方税号 | 15-20位 |
| `bank_account` | 银行账号 | 10-30位数字 |
| `page_number` | 页码 | 数字 |

### 时间字段（Temporal）

| 字段 | 中文名 |
|------|--------|
| `invoice_date` | 开票日期 |
| `due_date` | 到期日期 |

### 实体字段（Entity）

| 字段 | 中文名 |
|------|--------|
| `seller_name` | 销售方名称 |
| `buyer_name` | 购买方名称 |
| `receiver_name` | 收款人 |
| `sender_name` | 寄件人 |
| `bank_name` | 开户银行 |

### 金额字段（Monetary）

| 字段 | 中文名 |
|------|--------|
| `amount_before_tax` | 金额(不含税) |
| `tax_amount` | 税额 |
| `amount_total` | 价税合计 |
| `table_total` | 表格合计 |

### 联系字段（Contact）

| 字段 | 中文名 |
|------|--------|
| `receiver_phone` | 联系电话 |
| `receiver_address` | 地址 |

### 元数据字段（Metadata）

| 字段 | 中文名 |
|------|--------|
| `currency` | 币种 |
| `payment_method` | 支付方式 |
| `remarks` | 备注 |
| `document_type` | 文档类型 |

## 优化变更日志

### P0 关键修复

| # | 文件 | 问题 | 修复 |
|---|------|------|------|
| 1 | `field_types.py` | `RECEIVER_ADDRESS` 在 `FIELD_METADATA` 中定义两次，后者覆盖前者且正则更差 | 删除重复定义，保留第一个更好的正则 |
| 2 | `extractor.py` / `image_enhancer.py` | 二值图传入 OCR，OCR 对灰度图识别效果更佳 | `enhance()` 方法同时返回二值图和灰度图，OCR 使用灰度图 |
| 3 | `templates/invoice.json` / `receipt.json` | 文件以 Python docstring 注释开头，不是合法 JSON | 移除注释，确保 JSON 格式合法 |
| 4 | `extractor.py` / `layout_analyzer.py` | `_detect_seals` 需要 RGB 输入但收到二值图，导致印章检测永不执行 | 保存原始 RGB 图像，传给 `layout_analyzer.analyze()` |

### P1 重要修复

| # | 文件 | 问题 | 修复 |
|---|------|------|------|
| 5 | `field_types.py` | `SENDER_NAME` 和 `TABLE_TOTAL` 在枚举中定义但 `FIELD_METADATA` 中缺失，导致 KeyError | 添加 `SENDER_NAME` 和 `TABLE_TOTAL` 元数据条目 |
| 6 | `field_extractor.py` | `REMARKS` 在发票字段列表中出现两次 | 删除重复条目 |
| 7 | `ocr_engine.py` | `TesseractEngine.recognize_region` 调用 `image_to_string` + `image_to_data` 两次 OCR | 只使用 `image_to_data` 并从中提取文本和置信度 |
| 8 | `ocr_engine.py` | `demo_server.py` 使用 RapidOCR 但 `ocr_engine.py` 不支持 | 新增 `RapidOcrEngine` 类，工厂函数支持 `rapidocr` 选项 |
| 9 | `result_viewer.py` | `cv2.FONT_HERSHEY_SIMPLEX` 不支持中文，标注图乱码 | 改用 PIL/Pillow 渲染中文文字，自动加载系统中文字体 |
| 10 | `schemas.py` | `/extract/visual` 返回 `annotated_image` 但 `ExtractResponse` 无此字段 | 添加 `annotated_image: str = ""` 字段 |
| 11 | `server.py` | `/extract/visual` 返回 dict 而非 Pydantic 模型 | 直接返回 `ExtractResponse` 模型实例 |

## License

MIT License
