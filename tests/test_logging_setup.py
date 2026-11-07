"""Tests for :mod:`bank_marketing.logging_setup`."""

from __future__ import annotations

import json
import logging
import sys

from bank_marketing.logging_setup import JsonFormatter, configure_logging, get_logger

_ROOT_NAME = "bank_marketing"


def _reset_root_handlers() -> logging.Logger:
    root = logging.getLogger(_ROOT_NAME)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    return root


def test_configure_logging_is_idempotent() -> None:
    configure_logging()  # guarantee a handler exists so the reset loop body runs
    root = _reset_root_handlers()
    assert root.handlers == []
    first = configure_logging(level=logging.DEBUG)
    assert len(first.handlers) == 1
    assert first.level == logging.DEBUG
    # Second call must not add a duplicate handler but may change the level.
    second = configure_logging(level=logging.WARNING)
    assert second is first
    assert len(second.handlers) == 1
    assert second.level == logging.WARNING


def test_get_logger_namespacing() -> None:
    logger = get_logger("train")
    assert logger.name == "bank_marketing.train"
    assert logger.getEffectiveLevel() > 0


def test_json_formatter_basic_fields() -> None:
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="bank_marketing.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="hello %s",
        args=("world",),
        exc_info=None,
    )
    payload = json.loads(formatter.format(record))
    assert payload["message"] == "hello world"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "bank_marketing.test"
    assert "timestamp" in payload
    assert "exc_info" not in payload


def test_json_formatter_includes_exc_info() -> None:
    formatter = JsonFormatter()
    try:
        raise ValueError("boom")
    except ValueError:
        record = logging.LogRecord(
            name="bank_marketing.test",
            level=logging.ERROR,
            pathname=__file__,
            lineno=20,
            msg="failed",
            args=(),
            exc_info=sys.exc_info(),
        )
    payload = json.loads(formatter.format(record))
    assert "exc_info" in payload
    assert "ValueError" in payload["exc_info"]
