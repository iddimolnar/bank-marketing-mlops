"""Tests for :mod:`bank_marketing.predict`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pytest_mock import MockerFixture
from sklearn.pipeline import Pipeline

from bank_marketing import predict as predict_module
from bank_marketing.config import Settings
from bank_marketing.logstore import JsonlFileLogStore
from bank_marketing.predict import ModelNotReadyError, Predictor, load_predictor
from bank_marketing.schemas import ClientRecord
from tests.conftest import InMemoryLogStore


def test_predict_returns_response_and_logs(
    fitted_pipeline: Pipeline, in_memory_store: InMemoryLogStore, valid_record: dict[str, Any]
) -> None:
    predictor = Predictor(pipeline=fitted_pipeline, log_store=in_memory_store, model_version="7")
    response = predictor.predict(ClientRecord(**valid_record))

    assert response.label in {"yes", "no"}
    assert response.prediction in {0, 1}
    assert 0.0 <= response.probability <= 1.0
    assert response.model_version == "7"

    assert len(in_memory_store.records) == 1
    logged = in_memory_store.read_all()[0]
    assert logged["prediction"] == response.prediction
    assert logged["inputs"]["age"] == valid_record["age"]
    assert "timestamp" in logged


def test_load_predictor_success(
    tmp_path: Path, fitted_pipeline: Pipeline, valid_record: dict[str, Any], mocker: MockerFixture
) -> None:
    mocker.patch.object(predict_module.registry, "setup_tracking")
    mocker.patch.object(predict_module.registry, "load_production_pipeline", return_value=fitted_pipeline)
    mocker.patch.object(predict_module.registry, "production_version", return_value="3")
    settings = Settings(_env_file=None, request_log_path=tmp_path / "requests.jsonl")

    predictor = load_predictor(settings)

    assert predictor.model_version == "3"
    assert isinstance(predictor.log_store, JsonlFileLogStore)
    predictor.predict(ClientRecord(**valid_record))
    assert (tmp_path / "requests.jsonl").exists()


def test_load_predictor_raises_without_production_model(mocker: MockerFixture) -> None:
    mocker.patch.object(predict_module.registry, "setup_tracking")
    mocker.patch.object(predict_module.registry, "load_production_pipeline", return_value=None)
    with pytest.raises(ModelNotReadyError, match="train"):
        load_predictor(Settings(_env_file=None))
