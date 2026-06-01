"""
Image utility functions.
Loading, saving, format conversion, and common image operations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


def load_image(
    path: str | Path,
    color_mode: str = "rgb",
    target_dpi: Optional[int] = None,
    max_dimension: Optional[int] = None,
) -> np.ndarray:
    """
    Load an image from file with optional preprocessing.

    Args:
        path: Path to the image file.
        color_mode: 'rgb' (default) or 'gray'.
        target_dpi: Resample image to target DPI if specified.
        max_dimension: Downscale if image exceeds this dimension.

    Returns:
        Image as numpy array (H, W, C) for RGB or (H, W) for gray.

    Raises:
        FileNotFoundError: If image file does not exist.
        ValueError: If file is not a valid image.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    # Read image (OpenCV uses BGR)
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image (corrupt or unsupported): {path}")

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if color_mode == "gray":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Downscale if exceeds max dimension
    if max_dimension and max(img.shape[:2]) > max_dimension:
        scale = max_dimension / max(img.shape[:2])
        new_h, new_w = int(img.shape[0] * scale), int(img.shape[1] * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return img


def save_image(
    image: np.ndarray,
    path: str | Path,
    quality: int = 95,
) -> None:
    """
    Save an image to file.

    Args:
        image: Image array (RGB format expected).
        path: Output file path.
        quality: JPEG quality (1-100).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert RGB to BGR for OpenCV
    if len(image.shape) == 3:
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = image

    ext = path.suffix.lower()
    params = []
    if ext in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif ext == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 6]
    elif ext == ".webp":
        params = [cv2.IMWRITE_WEBP_QUALITY, quality]

    cv2.imwrite(str(path), img_bgr, params)


def rgb_to_gray(image: np.ndarray) -> np.ndarray:
    """Convert RGB image to grayscale."""
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def gray_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert grayscale image to RGB (3 channels)."""
    if len(image.shape) == 3:
        return image
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)


def estimate_dpi(image: np.ndarray) -> int:
    """
    Rough DPI estimation based on image dimensions.
    A4 at 72 DPI = 595x842, at 300 DPI = 2480x3508.
    """
    h, w = image.shape[:2]
    # Assume A4 aspect ratio
    area = h * w
    dpi_72_area = 595 * 842
    estimated = int(72 * (area / dpi_72_area) ** 0.5)
    return max(72, min(estimated, 600))


def get_image_hash(image: np.ndarray, hash_size: int = 8) -> str:
    """
    Compute a perceptual hash for image deduplication.
    Returns hex string of the hash.
    """
    gray = rgb_to_gray(image) if len(image.shape) == 3 else image
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    hash_val = 0
    for flat in diff.flatten():
        hash_val = (hash_val << 1) | int(flat)
    return f"{hash_val:0{hash_size * hash_size // 4}x}"
