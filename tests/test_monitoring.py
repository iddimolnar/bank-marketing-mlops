"""Tests for :mod:`bank_marketing.monitoring`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from bank_marketing.config import Settings
from bank_marketing.logstore import JsonlFileLogStore
from bank_marketing.monitoring import (
    ReportNotReadyError,
    build_drift_report,
    generate_report,
    load_current,
    load_reference,
)


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        reference_path=tmp_path / "reference.parquet",
        request_log_path=tmp_path / "requests.jsonl",
        drift_report_path=tmp_path / "drift.html",
    )


def test_load_reference_none_when_missing(tmp_path: Path) -> None:
    assert load_reference(_settings(tmp_path)) is None


def test_load_reference_reads_parquet(tmp_path: Path, xy: tuple[pd.DataFrame, pd.Series]) -> None:
    settings = _settings(tmp_path)
    features, _ = xy
    features.to_parquet(settings.reference_path)
    loaded = load_reference(settings)
    assert loaded is not None
    assert len(loaded) == len(features)


def test_load_current_from_log(tmp_path: Path, valid_record: dict[str, Any]) -> None:
    settings = _settings(tmp_path)
    JsonlFileLogStore(settings.request_log_path).append({"inputs": valid_record, "prediction": 0})
    current = load_current(settings)
    assert len(current) == 1
    assert current.iloc[0]["age"] == valid_record["age"]


def test_build_drift_report_writes_html(tmp_path: Path, xy: tuple[pd.DataFrame, pd.Series]) -> None:
    settings = _settings(tmp_path)
    features, _ = xy
    output = build_drift_report(features, features.head(10), settings.drift_report_path)
    assert output.exists()
    assert output.read_text(encoding="utf-8").lstrip().startswith("<")


def test_generate_report_raises_without_reference(tmp_path: Path) -> None:
    with pytest.raises(ReportNotReadyError):
        generate_report(_settings(tmp_path))


def test_generate_report_raises_without_requests(tmp_path: Path, xy: tuple[pd.DataFrame, pd.Series]) -> None:
    settings = _settings(tmp_path)
    features, _ = xy
    features.to_parquet(settings.reference_path)  # reference present, but no requests logged
    with pytest.raises(ReportNotReadyError):
        generate_report(settings)


def test_generate_report_success(
    tmp_path: Path, xy: tuple[pd.DataFrame, pd.Series], valid_record: dict[str, Any]
) -> None:
    settings = _settings(tmp_path)
    features, _ = xy
    features.to_parquet(settings.reference_path)
    store = JsonlFileLogStore(settings.request_log_path)
    for _ in range(5):
        store.append({"inputs": valid_record})
    output = generate_report(settings)
    assert output.exists()
