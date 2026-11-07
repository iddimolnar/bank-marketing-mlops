"""Tests for :mod:`bank_marketing.config`."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from bank_marketing.config import Settings, get_settings

_MLFLOW_ENV = (
    "MLFLOW_TRACKING_URI",
    "MLFLOW_EXPERIMENT_NAME",
    "MLFLOW_REGISTERED_MODEL_NAME",
)
_BANK_ENV = (
    "BANK_DROP_DURATION",
    "BANK_RANDOM_STATE",
    "BANK_GATE_METRIC",
    "BANK_GATE_TOLERANCE",
    "BANK_TEST_SIZE",
)


def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (*_MLFLOW_ENV, *_BANK_ENV):
        monkeypatch.delenv(name, raising=False)


def test_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    settings = Settings(_env_file=None)
    assert settings.drop_duration is True
    assert settings.target_column == "y"
    assert settings.random_state == 42
    assert settings.gate_metric == "roc_auc"
    assert settings.mlflow_tracking_uri == "http://localhost:5000"
    assert settings.data_path == Path("data/fixtures/tiny_bank.csv")


def test_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("BANK_DROP_DURATION", "false")
    monkeypatch.setenv("BANK_RANDOM_STATE", "7")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    settings = Settings(_env_file=None)
    assert settings.drop_duration is False
    assert settings.random_state == 7
    assert settings.mlflow_tracking_uri == "http://mlflow:5000"


@pytest.mark.parametrize("bad_test_size", [0.0, 1.0, 1.5, -0.1])
def test_invalid_test_size_rejected(monkeypatch: pytest.MonkeyPatch, bad_test_size: float) -> None:
    _clear_env(monkeypatch)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, test_size=bad_test_size)


def test_negative_tolerance_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, gate_tolerance=-0.1)


def test_get_settings_is_cached() -> None:
    get_settings.cache_clear()
    first = get_settings()
    second = get_settings()
    assert first is second
    get_settings.cache_clear()
