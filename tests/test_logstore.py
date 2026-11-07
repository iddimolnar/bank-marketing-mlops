"""Tests for :mod:`bank_marketing.logstore`."""

from __future__ import annotations

from pathlib import Path

from bank_marketing.logstore import JsonlFileLogStore


def test_append_then_read_roundtrip(tmp_path: Path) -> None:
    store = JsonlFileLogStore(tmp_path / "logs" / "requests.jsonl")
    store.append({"a": 1, "label": "yes"})
    store.append({"a": 2, "label": "no"})
    records = store.read_all()
    assert len(records) == 2
    assert records[0]["a"] == 1
    assert records[1]["label"] == "no"


def test_read_all_missing_file_returns_empty(tmp_path: Path) -> None:
    store = JsonlFileLogStore(tmp_path / "does-not-exist.jsonl")
    assert store.read_all() == []
