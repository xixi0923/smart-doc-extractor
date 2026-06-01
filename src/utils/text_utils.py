"""
Text utility functions.
Text cleaning, normalization, and formatting for OCR post-processing.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple


def clean_ocr_text(text: str) -> str:
    """
    Clean raw OCR text by removing common artifacts.

    - Remove stray whitespace
    - Fix common OCR misreads (O→0, l→1, etc.)
    - Normalize unicode
    """
    if not text:
        return ""

    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    return text


def normalize_amount(text: str) -> Optional[float]:
    """
    Extract and normalize a monetary amount from text.

    Handles formats like:
        ¥1,234.56  |  1234.56  |  ￥1,234.56元  |  CNY 1234.56
    Returns float or None if no amount found.
    """
    # Remove currency symbols and labels
    cleaned = re.sub(r"[¥￥$€＄]|元|圆|CNY|RMB", "", text, flags=re.IGNORECASE)
    cleaned = cleaned.strip()

    # Find number pattern (with optional commas and decimals)
    match = re.search(r"[\d,]+\.?\d*", cleaned)
    if not match:
        return None

    amount_str = match.group().replace(",", "")
    try:
        return float(amount_str)
    except ValueError:
        return None


def normalize_date(text: str) -> Optional[str]:
    """
    Normalize a date string to ISO format (YYYY-MM-DD).

    Handles formats:
        2024-01-15  |  2024/01/15  |  2024年01月15日  |  20240115
    Returns ISO date string or None.
    """
    # Chinese date format
    m = re.search(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日", text)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"

    # Slash/dash separated
    m = re.search(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})", text)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"

    # Compact format (YYYYMMDD)
    m = re.search(r"(\d{4})(\d{2})(\d{2})", text)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    return None


def normalize_phone(text: str) -> Optional[str]:
    """Extract and normalize a phone number."""
    digits = re.sub(r"\D", "", text)
    if len(digits) >= 10 and len(digits) <= 13:
        return digits
    return None


def extract_chinese_company_name(text: str) -> Optional[str]:
    """
    Extract a Chinese company name from text.
    Looks for patterns ending with common company suffixes.
    """
    patterns = [
        r"([\u4e00-\u9fa5]+(?:公司|有限公司|股份公司|集团|企业|工厂|商店|事务所|银行))",
        r"([\u4e00-\u9fa5]+(?:Co\.|Ltd\.|LLC|Inc\.|Corp\.))",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    return None


def compute_text_similarity(text_a: str, text_b: str) -> int:
    """
    Compute similarity score between two strings (0-100).
    Uses a simple character-level ratio.
    """
    if not text_a or not text_b:
        return 0

    set_a = set(text_a.lower())
    set_b = set(text_b.lower())
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)

    if union == 0:
        return 100

    return int(intersection / union * 100)


def split_text_lines(text: str) -> List[str]:
    """Split OCR text into non-empty lines."""
    lines = text.split("\n")
    return [line.strip() for line in lines if line.strip()]


def merge_adjacent_texts(
    regions: List[dict],
    horizontal_gap: int = 20,
    vertical_gap: int = 5,
) -> List[str]:
    """
    Merge text regions that are on the same line (similar y-coordinate)
    into full text lines, sorted left-to-right.
    """
    if not regions:
        return []

    # Sort by y then x
    sorted_regions = sorted(regions, key=lambda r: (r["bbox"][1], r["bbox"][0]))

    lines = []
    current_line = [sorted_regions[0]]
    current_y = sorted_regions[0]["bbox"][1]

    for region in sorted_regions[1:]:
        y = region["bbox"][1]
        if abs(y - current_y) <= vertical_gap:
            current_line.append(region)
        else:
            # Finish current line
            current_line.sort(key=lambda r: r["bbox"][0])
            merged = " ".join(r["text"] for r in current_line if r.get("text"))
            lines.append(merged)
            current_line = [region]
            current_y = y

    # Last line
    if current_line:
        current_line.sort(key=lambda r: r["bbox"][0])
        merged = " ".join(r["text"] for r in current_line if r.get("text"))
        lines.append(merged)

    return [line for line in lines if line.strip()]
