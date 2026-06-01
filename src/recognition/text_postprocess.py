"""
Text post-processing module.
Cleans and normalizes OCR output to improve extraction accuracy.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from src.recognition.ocr_engine import OcrResult, OcrPageResult
from src.utils.logger import get_logger

logger = get_logger("text_postprocess")


# Common OCR misread corrections (character-level)
OCR_CHAR_CORRECTIONS: Dict[str, str] = {
    "０": "0", "１": "1", "２": "2", "３": "3", "４": "4",
    "５": "5", "６": "6", "７": "7", "８": "8", "９": "9",
    "Ａ": "A", "Ｂ": "B", "Ｃ": "C", "Ｄ": "D", "Ｅ": "E",
    "Ｆ": "F", "Ｇ": "G", "Ｈ": "H", "Ｉ": "I", "Ｊ": "J",
    "Ｋ": "K", "Ｌ": "L", "Ｍ": "M", "Ｎ": "N", "Ｏ": "O",
    "Ｐ": "P", "Ｑ": "Q", "Ｒ": "R", "Ｓ": "S", "Ｔ": "T",
    "Ｕ": "U", "Ｖ": "V", "Ｗ": "W", "Ｘ": "X", "Ｙ": "Y", "Ｚ": "Z",
    "ａ": "a", "ｂ": "b", "ｃ": "c", "ｄ": "d", "ｅ": "e",
    "ｆ": "f", "ｇ": "g", "ｈ": "h", "ｉ": "i", "ｊ": "j",
    "ｋ": "k", "ｌ": "l", "ｍ": "m", "ｎ": "n", "ｏ": "o",
    "ｐ": "p", "ｑ": "q", "ｒ": "r", "ｓ": "s", "ｔ": "t",
    "ｕ": "u", "ｖ": "v", "ｗ": "w", "ｘ": "x", "ｙ": "y", "ｚ": "z",
}

# Common OCR word-level corrections
OCR_WORD_CORRECTIONS: Dict[str, str] = {
    "invioce": "invoice",
    "invoce": "invoice",
    "recipt": "receipt",
    "reciept": "receipt",
    "amout": "amount",
    "amounr": "amount",
    "totla": "total",
    "totol": "total",
    "adddress": "address",
    "comapny": "company",
    "compnay": "company",
    "numebr": "number",
    "nubmer": "number",
    "datc": "date",
    "dete": "date",
}


class TextPostProcessor:
    """
    Post-processes OCR output to improve quality.

    Operations:
    1. Full-width to half-width character conversion
    2. Common OCR misread correction
    3. Whitespace normalization
    4. Noise removal (isolated characters, stray marks)
    5. Context-aware corrections
    """

    def __init__(
        self,
        apply_char_corrections: bool = True,
        apply_word_corrections: bool = True,
        remove_noise: bool = True,
    ):
        self.apply_char_corrections = apply_char_corrections
        self.apply_word_corrections = apply_word_corrections
        self.remove_noise = remove_noise

    def process_page(self, page_result: OcrPageResult) -> OcrPageResult:
        """
        Post-process all OCR results from a page.

        Args:
            page_result: Raw OCR page result.

        Returns:
            Cleaned OCR page result.
        """
        cleaned_results = []
        for result in page_result.results:
            cleaned = self.process_text(result.text)
            cleaned_results.append(OcrResult(
                text=cleaned,
                confidence=result.confidence,
                bbox=result.bbox,
                region_type=result.region_type,
            ))

        full_text = self.process_text(page_result.full_text)

        return OcrPageResult(
            results=cleaned_results,
            full_text=full_text,
            total_time_ms=page_result.total_time_ms,
            avg_confidence=page_result.avg_confidence,
            engine_name=page_result.engine_name,
        )

    def process_text(self, text: str) -> str:
        """
        Apply all post-processing steps to a text string.

        Args:
            text: Raw OCR text.

        Returns:
            Cleaned text.
        """
        if not text:
            return ""

        # Step 1: Full-width to half-width
        if self.apply_char_corrections:
            text = self._correct_fullwidth(text)

        # Step 2: Word-level corrections
        if self.apply_word_corrections:
            text = self._correct_words(text)

        # Step 3: Noise removal
        if self.remove_noise:
            text = self._remove_noise(text)

        # Step 4: Whitespace normalization
        text = self._normalize_whitespace(text)

        return text.strip()

    def _correct_fullwidth(self, text: str) -> str:
        """Convert full-width characters to half-width."""
        result = []
        for ch in text:
            result.append(OCR_CHAR_CORRECTIONS.get(ch, ch))
        return "".join(result)

    def _correct_words(self, text: str) -> str:
        """Apply common OCR word-level corrections."""
        # Case-insensitive replacement
        lower_text = text.lower()
        for wrong, correct in OCR_WORD_CORRECTIONS.items():
            lower_text = lower_text.replace(wrong, correct)

        # Preserve original casing pattern for Chinese text
        # For mixed content, use the corrected lowercase version
        if any("\u4e00" <= c <= "\u9fff" for c in text):
            # Chinese text — only replace pure ASCII parts
            parts = re.split(r"([\u4e00-\u9fff]+)", text)
            corrected_parts = []
            for part in parts:
                if re.match(r"^[\u4e00-\u9fff]+$", part):
                    corrected_parts.append(part)
                else:
                    corrected_parts.append(
                        OCR_WORD_CORRECTIONS.get(part.lower(), part)
                    )
            return "".join(corrected_parts)

        # Pure ASCII — return corrected version
        return lower_text

    def _remove_noise(self, text: str) -> str:
        """Remove noise characters and isolated artifacts."""
        # Remove isolated non-alphanumeric characters (except common punctuation)
        lines = text.split("\n")
        cleaned_lines = []

        for line in lines:
            # Remove lines that are just noise (mostly special chars)
            alpha_count = sum(1 for c in line if c.isalnum() or "\u4e00" <= c <= "\u9fff")
            if alpha_count == 0 and len(line) < 3:
                continue

            # Remove isolated special characters at boundaries
            line = re.sub(r"^[^\w\u4e00-\u9fff]+", "", line)
            line = re.sub(r"[^\w\u4e00-\u9fff]+$", "", line)

            cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Normalize whitespace characters."""
        # Multiple spaces to single
        text = re.sub(r" {2,}", " ", text)
        # Multiple blank lines to double newline
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text


def merge_ocr_lines(
    results: List[OcrResult],
    vertical_threshold: int = 10,
) -> List[OcrResult]:
    """
    Merge OCR results that belong to the same text line.

    Groups results with similar y-coordinate (vertical position)
    and combines their text, sorted by x-position.

    Args:
        results: List of OCR results to merge.
        vertical_threshold: Max vertical distance to consider same line.

    Returns:
        Merged list of OCR results.
    """
    if not results:
        return []

    # Sort by y then x
    sorted_results = sorted(results, key=lambda r: (r.bbox[1], r.bbox[0]))

    lines = []
    current_line = [sorted_results[0]]
    current_y = sorted_results[0].bbox[1]

    for result in sorted_results[1:]:
        y = result.bbox[1]
        if abs(y - current_y) <= vertical_threshold:
            current_line.append(result)
        else:
            lines.append(_merge_line(current_line))
            current_line = [result]
            current_y = y

    if current_line:
        lines.append(_merge_line(current_line))

    return [l for l in lines if not l.is_empty]


def _merge_line(results: List[OcrResult]) -> OcrResult:
    """Merge a list of OCR results into a single line."""
    results.sort(key=lambda r: r.bbox[0])

    text = " ".join(r.text for r in results if r.text.strip())
    conf = float(sum(r.confidence for r in results)) / len(results) if results else 0.0

    # Combined bounding box
    min_x = min(r.bbox[0] for r in results)
    min_y = min(r.bbox[1] for r in results)
    max_x = max(r.bbox[0] + r.bbox[2] for r in results)
    max_y = max(r.bbox[1] + r.bbox[3] for r in results)

    return OcrResult(
        text=text,
        confidence=conf,
        bbox=(min_x, min_y, max_x - min_x, max_y - min_y),
    )
