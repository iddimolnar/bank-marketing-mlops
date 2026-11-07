"""Tests for :mod:`bank_marketing.train` (registry mocked)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pytest_mock import MockerFixture
from sklearn.pipeline import Pipeline

from bank_marketing import train
from bank_marketing.config import Settings
from bank_marketing.gate import PromotionDecision
from bank_marketing.model import build_default_pipeline
from bank_marketing.train import TrainResult, fit_candidate, main, run_training, write_reference


def _settings(tmp_path: Path, fixture_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        data_path=fixture_path,
        reference_path=tmp_path / "reference.parquet",
        test_size=0.3,
        random_state=42,
    )


def test_fit_candidate_returns_fitted_pipeline(tmp_path: Path, fixture_path: Path) -> None:
    pipeline, metrics, x_train, x_test, _y_test = fit_candidate(_settings(tmp_path, fixture_path))
    assert isinstance(pipeline, Pipeline)
    assert "roc_auc" in metrics
    assert len(x_train) + len(x_test) == 90


def test_write_reference_creates_parquet(tmp_path: Path, fixture_path: Path) -> None:
    settings = _settings(tmp_path, fixture_path)
    frame = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    write_reference(settings, frame)
    assert settings.reference_path.exists()
    assert pd.read_parquet(settings.reference_path).shape == (2, 2)


def test_run_training_promotes_when_no_production(tmp_path: Path, fixture_path: Path, mocker: MockerFixture) -> None:
    mocker.patch.object(train.registry, "setup_tracking")
    mocker.patch.object(train.registry, "load_production_pipeline", return_value=None)
    mocker.patch.object(train.registry, "log_training_run", return_value=("run123", "1"))
    promote = mocker.patch.object(train.registry, "promote_model")

    result = run_training(_settings(tmp_path, fixture_path))

    assert isinstance(result, TrainResult)
    assert result.promoted is True
    assert result.run_id == "run123"
    assert result.model_version == "1"
    promote.assert_called_once_with(mocker.ANY, "1")
    assert (tmp_path / "reference.parquet").exists()


def test_run_training_rejects_worse_candidate(
    tmp_path: Path, fixture_path: Path, xy: tuple[pd.DataFrame, pd.Series], mocker: MockerFixture
) -> None:
    features, target = xy
    current = build_default_pipeline(random_state=1)
    current.fit(features, target)

    mocker.patch.object(train.registry, "setup_tracking")
    mocker.patch.object(train.registry, "load_production_pipeline", return_value=current)
    mocker.patch.object(train.registry, "log_training_run", return_value=("run999", "2"))
    promote = mocker.patch.object(train.registry, "promote_model")
    mocker.patch.object(train, "decide_promotion", return_value=PromotionDecision(False, "worse"))

    result = run_training(_settings(tmp_path, fixture_path))

    assert result.promoted is False
    assert result.model_version == "2"
    promote.assert_not_called()
    assert not (tmp_path / "reference.parquet").exists()


def test_main_logs_result(mocker: MockerFixture) -> None:
    fake = TrainResult(
        run_id="r1",
        promoted=True,
        model_version="1",
        candidate_metrics={"roc_auc": 0.9},
        message="ok",
    )
    run = mocker.patch.object(train, "run_training", return_value=fake)
    main()
    run.assert_called_once_with()
