"""
Field type definitions for document information extraction.
Defines field types, their patterns, validation rules, and display names.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class FieldType(Enum):
    """Enumeration of extractable field types."""

    INVOICE_NUMBER = "invoice_number"
    INVOICE_CODE = "invoice_code"
    INVOICE_DATE = "invoice_date"
    DUE_DATE = "due_date"
    SELLER_NAME = "seller_name"
    BUYER_NAME = "buyer_name"
    SELLER_TAX_ID = "seller_tax_id"
    BUYER_TAX_ID = "buyer_tax_id"
    AMOUNT_BEFORE_TAX = "amount_before_tax"
    TAX_AMOUNT = "tax_amount"
    AMOUNT_TOTAL = "amount_total"
    CURRENCY = "currency"
    PAYMENT_METHOD = "payment_method"
    BANK_NAME = "bank_name"
    BANK_ACCOUNT = "bank_account"
    REMARKS = "remarks"
    RECEIVER_NAME = "receiver_name"
    RECEIVER_PHONE = "receiver_phone"
    RECEIVER_ADDRESS = "receiver_address"
    SENDER_NAME = "sender_name"
    TABLE_TOTAL = "table_total"         # sum of table line items
    LINE_ITEMS = "line_items"           # individual table line items
    DOCUMENT_TYPE = "document_type"     # auto-detected document type
    PAGE_NUMBER = "page_number"


# Field metadata: display names, regex patterns, validation info
FIELD_METADATA: Dict[FieldType, Dict] = {
    FieldType.INVOICE_NUMBER: {
        "display_name": "发票号码",
        "display_name_en": "Invoice Number",
        "patterns": [
            r"(?:发票号码|No\.?|编号)[：:\s]*(\d{8,20})",
            r"\b(\d{8,20})\b",  # standalone long number
        ],
        "validation": r"^\d{8,20}$",
        "category": "identification",
    },
    FieldType.INVOICE_CODE: {
        "display_name": "发票代码",
        "display_name_en": "Invoice Code",
        "patterns": [
            r"(?:发票代码)[：:\s]*(\d{10,12})",
        ],
        "validation": r"^\d{10,12}$",
        "category": "identification",
    },
    FieldType.INVOICE_DATE: {
        "display_name": "开票日期",
        "display_name_en": "Invoice Date",
        "patterns": [
            r"(?:开票日期|日期|Date)[：:\s]*(\d{4}[\s年/\-.]\d{1,2}[\s月/\-.]\d{1,2})",
            r"(\d{4}[\s/\-.]\d{1,2}[\s/\-.]\d{1,2})",
        ],
        "validation": None,
        "category": "temporal",
    },
    FieldType.DUE_DATE: {
        "display_name": "到期日期",
        "display_name_en": "Due Date",
        "patterns": [
            r"(?:到期|截止|Due)[：:\s]*(\d{4}[\s年/\-.]\d{1,2}[\s月/\-.]\d{1,2})",
        ],
        "validation": None,
        "category": "temporal",
    },
    FieldType.SELLER_NAME: {
        "display_name": "销售方名称",
        "display_name_en": "Seller Name",
        "patterns": [
            r"(?:销方|销售方|卖方|Seller)[：:\s]*([\u4e00-\u9fa5A-Za-z\s]+(?:公司|企业|厂|商店|事务所|银行))",
            r"(?:名称)[：:\s]*([\u4e00-\u9fa5]+(?:公司|企业|厂|商店))",
        ],
        "validation": None,
        "category": "entity",
    },
    FieldType.BUYER_NAME: {
        "display_name": "购买方名称",
        "display_name_en": "Buyer Name",
        "patterns": [
            r"(?:购方|购买方|买方|Buyer|客户)[：:\s]*([\u4e00-\u9fa5A-Za-z\s]+(?:公司|企业|厂|商店|事务所|银行))",
        ],
        "validation": None,
        "category": "entity",
    },
    FieldType.SELLER_TAX_ID: {
        "display_name": "销售方税号",
        "display_name_en": "Seller Tax ID",
        "patterns": [
            r"(?:销方.*?纳税人识别号|卖方.*?税号)[：:\s]*(\w{15,20})",
            r"(?:纳税人识别号)[：:\s]*(\w{15,20})",
        ],
        "validation": r"^[\w]{15,20}$",
        "category": "identification",
    },
    FieldType.BUYER_TAX_ID: {
        "display_name": "购买方税号",
        "display_name_en": "Buyer Tax ID",
        "patterns": [
            r"(?:购方.*?纳税人识别号|买方.*?税号)[：:\s]*(\w{15,20})",
        ],
        "validation": r"^[\w]{15,20}$",
        "category": "identification",
    },
    FieldType.AMOUNT_BEFORE_TAX: {
        "display_name": "金额(不含税)",
        "display_name_en": "Amount (before tax)",
        "patterns": [
            r"(?:金额|不含税|Amount)[：:\s]*[¥￥$]?\s*([\d,]+\.?\d*)",
            r"(?:合计|小计|Subtotal)[：:\s]*[¥￥$]?\s*([\d,]+\.?\d*)",
        ],
        "validation": r"^[\d,.]+$",
        "category": "monetary",
    },
    FieldType.TAX_AMOUNT: {
        "display_name": "税额",
        "display_name_en": "Tax Amount",
        "patterns": [
            r"(?:税额|Tax)[：:\s]*[¥￥$]?\s*([\d,]+\.?\d*)",
        ],
        "validation": r"^[\d,.]+$",
        "category": "monetary",
    },
    FieldType.AMOUNT_TOTAL: {
        "display_name": "价税合计",
        "display_name_en": "Total Amount",
        "patterns": [
            r"(?:价税合计|总计|总金额|合计|Total)[：:\s]*[¥￥$]?\s*([\d,]+\.?\d*)",
            r"(?:大写.*?)[（(]([¥￥$]?[\d,]+\.?\d*)[）)]",
        ],
        "validation": r"^[\d,.]+$",
        "category": "monetary",
    },
    FieldType.CURRENCY: {
        "display_name": "币种",
        "display_name_en": "Currency",
        "patterns": [
            r"(?:币种|Currency)[：:\s]*(\w+)",
        ],
        "validation": None,
        "category": "metadata",
    },
    FieldType.PAYMENT_METHOD: {
        "display_name": "支付方式",
        "display_name_en": "Payment Method",
        "patterns": [
            r"(?:支付方式|付款方式|Payment)[：:\s]*(\S+)",
        ],
        "validation": None,
        "category": "metadata",
    },
    FieldType.BANK_NAME: {
        "display_name": "开户银行",
        "display_name_en": "Bank Name",
        "patterns": [
            r"(?:开户行|开户银行|Bank)[：:\s]*([\u4e00-\u9fa5A-Za-z]+(?:银行|Branch))",
        ],
        "validation": None,
        "category": "entity",
    },
    FieldType.BANK_ACCOUNT: {
        "display_name": "银行账号",
        "display_name_en": "Bank Account",
        "patterns": [
            r"(?:账号|银行账号|Account)[：:\s]*(\d{10,30})",
        ],
        "validation": r"^\d{10,30}$",
        "category": "identification",
    },
    FieldType.RECEIVER_NAME: {
        "display_name": "收款人",
        "display_name_en": "Receiver Name",
        "patterns": [
            r"(?:收款人|收款单位|Payee)[：:\s]*([\u4e00-\u9fa5A-Za-z]+)",
        ],
        "validation": None,
        "category": "entity",
    },
    FieldType.RECEIVER_PHONE: {
        "display_name": "联系电话",
        "display_name_en": "Phone Number",
        "patterns": [
            r"(?:电话|联系电话|Tel|Phone)[：:\s]*([\d\-\s]{10,15})",
            r"(\d{3}[-\s]?\d{4}[-\s]?\d{4})",
        ],
        "validation": r"^[\d\-\s]{10,15}$",
        "category": "contact",
    },
    FieldType.RECEIVER_ADDRESS: {
        "display_name": "地址",
        "display_name_en": "Address",
        "patterns": [
            r"(?:地址|Address)[：:\s]*([\u4e00-\u9fa5\w\s]+(?:路|街|道|号|室|楼|层|栋|幢|座))",
        ],
        "validation": None,
        "category": "contact",
    },
    FieldType.REMARKS: {
        "display_name": "备注",
        "display_name_en": "Remarks",
        "patterns": [
            r"(?:备注|Remark|Note)[：:\s]*(.+)",
        ],
        "validation": None,
        "category": "metadata",
    },
    FieldType.SENDER_NAME: {
        "display_name": "寄件人",
        "display_name_en": "Sender Name",
        "patterns": [
            r"(?:寄件人|发件人|Sender|寄方)[：:\s]*([\u4e00-\u9fa5A-Za-z]+)",
        ],
        "validation": None,
        "category": "entity",
    },
    FieldType.TABLE_TOTAL: {
        "display_name": "表格合计",
        "display_name_en": "Table Total",
        "patterns": [
            r"(?:表格合计|明细合计|Item Total)[：:\s]*[¥￥$]?\s*([\d,]+\.?\d*)",
        ],
        "validation": r"^[\d,.]+$",
        "category": "monetary",
    },
    FieldType.DOCUMENT_TYPE: {
        "display_name": "文档类型",
        "display_name_en": "Document Type",
        "patterns": [],  # Auto-detected, not regex-extracted
        "validation": None,
        "category": "metadata",
    },
    FieldType.PAGE_NUMBER: {
        "display_name": "页码",
        "display_name_en": "Page Number",
        "patterns": [
            r"(?:第\s*)(\d+)(?:\s*页)",
            r"(?:Page\s*)(\d+)",
        ],
        "validation": r"^\d+$",
        "category": "metadata",
    },
}


@dataclass
class ExtractedField:
    """A single field extracted from a document."""

    field_type: FieldType
    value: str
    confidence: float
    source_text: str = ""          # original text where the field was found
    source_line: int = -1          # line number in OCR output
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)

    @property
    def display_name(self) -> str:
        return FIELD_METADATA[self.field_type]["display_name"]

    @property
    def category(self) -> str:
        return FIELD_METADATA[self.field_type]["category"]

    def to_dict(self) -> dict:
        return {
            "field_type": self.field_type.value,
            "display_name": self.display_name,
            "value": self.value,
            "confidence": round(self.confidence, 3),
            "source_text": self.source_text,
            "category": self.category,
        }


def get_fields_by_category(category: str) -> List[FieldType]:
    """Get all field types belonging to a category."""
    return [
        ft for ft, meta in FIELD_METADATA.items()
        if meta["category"] == category
    ]
