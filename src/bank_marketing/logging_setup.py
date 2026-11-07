"""Structured (JSON) logging setup.

A single configured logger tree under the ``bank_marketing`` namespace emits one JSON
object per line, which is convenient for log shipping and for feeding request/prediction
records into monitoring. Configuration is idempotent.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime

_ROOT_NAME = "bank_marketing"


class JsonFormatter(logging.Formatter):
    """Format log records as compact single-line JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Attach a JSON stdout handler to the package logger exactly once.

    Returns the package root logger. Calling this repeatedly updates the level but does
    not add duplicate handlers.
    """
    root = logging.getLogger(_ROOT_NAME)
    root.setLevel(level)
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        root.addHandler(handler)
        root.propagate = False
    return root


def get_logger(name: str) -> logging.Logger:
    """Return a child logger (``bank_marketing.<name>``) with logging configured."""
    configure_logging()
    return logging.getLogger(f"{_ROOT_NAME}.{name}")
