"""Unified logging helpers for the OTC Agent prototype.

The project uses the Python standard logging library only. Runtime modules can
use this helper to record Agent execution, Tool activity observed by the Agent,
Safety Guard decisions, and exception details without introducing a heavy
logging framework.
"""

from __future__ import annotations

import logging
import os


DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(level: str | None = None) -> None:
    """Configure project-level logging once."""
    log_level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format=DEFAULT_LOG_FORMAT,
        force=False,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a configured standard library logger."""
    configure_logging()
    return logging.getLogger(name)

