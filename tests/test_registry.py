"""Integration tests for :mod:`bank_marketing.registry`.

These run MLflow against a local SQLite-backed tracking store in a temp directory, so no
MLflow server is required. The registry needs a database backend (not a file store).
"""

from __future__ import annotations

import pandas as pd
import pytest

from bank_marketing import registry
from bank_marketing.config import Settings
from bank_marketing.model import build_default_pipeline


@pytest.fixture
def local_mlflow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: object,
    xy: tuple[pd.DataFrame, pd.Series],
) -> Settings:
    """Configure MLflow to use a temp SQLite store and return matching Settings."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "test-exp")
    monkeypatch.setenv("MLFLOW_REGISTERED_MODEL_NAME", "test-model")
    monkeypatch.chdir(tmp_path)  # MLflow writes artifacts under the current directory
    settings = Settings(_env_file=None)
    registry.setup_tracking(settings)
    return settings


def _fitted_pipeline(xy: tuple[pd.DataFrame, pd.Series]):
    features, target = xy
    pipeline = build_default_pipeline(random_state=0)
    pipeline.fit(features, target)
    return pipeline


def test_load_production_returns_none_when_empty(local_mlflow: Settings) -> None:
    assert registry.load_production_pipeline(local_mlflow) is None


def test_log_register_promote_lifecycle(local_mlflow: Settings, xy: tuple[pd.DataFrame, pd.Series]) -> None:
    pipeline = _fitted_pipeline(xy)

    run_id, version = registry.log_training_run(local_mlflow, pipeline, {"model": "hgb"}, {"roc_auc": 0.9})
    assert run_id
    assert version == "1"
    # Registered but not yet promoted -> no Production model.
    assert registry.load_production_pipeline(local_mlflow) is None

    registry.promote_model(local_mlflow, version)
    loaded = registry.load_production_pipeline(local_mlflow)
    assert loaded is not None
    # The loaded pipeline can score the same data.
    features, _ = xy
    assert len(loaded.predict(features)) == len(features)
