"""Tests for the FastAPI app (:mod:`bank_marketing.api`)."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient
from pytest_mock import MockerFixture
from sklearn.pipeline import Pipeline

from bank_marketing.api import deps, routes
from bank_marketing.api.app import create_app
from bank_marketing.api.deps import provide_predictor
from bank_marketing.predict import ModelNotReadyError, Predictor
from bank_marketing.train import TrainResult
from tests.conftest import InMemoryLogStore


def test_provide_predictor_delegates_to_load_predictor(mocker: MockerFixture) -> None:
    sentinel = object()
    mocker.patch.object(deps, "load_predictor", return_value=sentinel)
    assert deps.provide_predictor() is sentinel


def _client_with_predictor(predictor: Predictor) -> TestClient:
    app = create_app()
    app.dependency_overrides[provide_predictor] = lambda: predictor
    return TestClient(app)


def test_health() -> None:
    client = TestClient(create_app())
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_ok(fitted_pipeline: Pipeline, valid_record: dict[str, Any]) -> None:
    predictor = Predictor(pipeline=fitted_pipeline, log_store=InMemoryLogStore(), model_version="1")
    client = _client_with_predictor(predictor)
    response = client.post("/predict", json=valid_record)
    assert response.status_code == 200
    body = response.json()
    assert body["label"] in {"yes", "no"}
    assert body["prediction"] in {0, 1}
    assert 0.0 <= body["probability"] <= 1.0
    assert body["model_version"] == "1"


def test_predict_validation_error(valid_record: dict[str, Any]) -> None:
    predictor = Predictor(pipeline=object(), log_store=InMemoryLogStore())  # type: ignore[arg-type]
    client = _client_with_predictor(predictor)
    bad = {**valid_record, "age": -5}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_predict_model_not_ready(valid_record: dict[str, Any]) -> None:
    app = create_app()

    def _raise() -> Predictor:
        raise ModelNotReadyError("no production model")

    app.dependency_overrides[provide_predictor] = _raise
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post("/predict", json=valid_record)
    assert response.status_code == 503
    assert "production model" in response.json()["detail"]


def test_train_default(mocker: MockerFixture) -> None:
    result = TrainResult(
        run_id="r1",
        promoted=True,
        model_version="2",
        candidate_metrics={"roc_auc": 0.9},
        message="promoted",
    )
    run = mocker.patch.object(routes, "run_training", return_value=result)
    client = TestClient(create_app())
    response = client.post("/train", json={})
    assert response.status_code == 200
    assert response.json()["promoted"] is True
    assert response.json()["model_version"] == "2"
    run.assert_called_once()


def test_train_with_overrides(mocker: MockerFixture) -> None:
    result = TrainResult(
        run_id="r2",
        promoted=False,
        model_version="3",
        candidate_metrics={"roc_auc": 0.8},
        message="kept",
    )
    run = mocker.patch.object(routes, "run_training", return_value=result)
    client = TestClient(create_app())
    response = client.post("/train", json={"test_size": 0.25, "random_state": 7})
    assert response.status_code == 200
    assert response.json()["promoted"] is False
    passed_settings = run.call_args.args[0]
    assert passed_settings.test_size == 0.25
    assert passed_settings.random_state == 7
