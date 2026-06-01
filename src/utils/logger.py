"""
Logging utility for Smart Doc Extractor.
Provides consistent, configurable logging across all modules.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional


def setup_logging(
    name: str = "smart_doc",
    level: str = "INFO",
    log_format: Optional[str] = None,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Create and configure a logger instance.

    Args:
        name: Logger name (typically module name).
        level: Logging level string (DEBUG, INFO, WARNING, ERROR).
        log_format: Custom format string. Uses default if None.
        log_file: Optional file path for file logging. Console-only if None.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    fmt = log_format or "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    formatter = logging.Formatter(fmt)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a child logger under the smart_doc namespace."""
    return logging.getLogger(f"smart_doc.{name}")
