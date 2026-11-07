"""Training orchestration.

Ties the pieces together: load data, fit a candidate pipeline, evaluate it against the
current production model through the promotion gate, log everything to MLflow, and -- if
the gate approves -- promote the new version and refresh the Evidently reference data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pandas as pd
from sklearn.pipeline import Pipeline

from bank_marketing import registry
from bank_marketing.config import Settings, get_settings
from bank_marketing.data import load_data, prepare_xy, split
from bank_marketing.gate import decide_promotion
from bank_marketing.logging_setup import get_logger
from bank_marketing.metrics import SupportsPredict, evaluate
from bank_marketing.model import build_default_pipeline

logger = get_logger("train")


@dataclass(frozen=True)
class TrainResult:
    """Summary of a training run and the promotion decision."""

    run_id: str
    promoted: bool
    model_version: str
    candidate_metrics: dict[str, float]
    message: str


def fit_candidate(
    settings: Settings,
) -> tuple[Pipeline, dict[str, float], pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load data, fit a candidate pipeline and evaluate it on the held-out test split.

    Returns ``(pipeline, metrics, x_train, x_test, y_test)``.
    """
    frame = load_data(settings.data_path)
    features, target = prepare_xy(
        frame,
        target_column=settings.target_column,
        drop_duration=settings.drop_duration,
    )
    x_train, x_test, y_train, y_test = split(
        features,
        target,
        test_size=settings.test_size,
        random_state=settings.random_state,
    )
    pipeline = build_default_pipeline(
        drop_duration=settings.drop_duration,
        random_state=settings.random_state,
    )
    pipeline.fit(x_train, y_train)
    metrics = evaluate(cast("SupportsPredict", pipeline), x_test, y_test)
    return pipeline, metrics, x_train, x_test, y_test


def write_reference(settings: Settings, reference_features: pd.DataFrame) -> None:
    """Persist the training feature frame as the Evidently drift reference snapshot."""
    settings.reference_path.parent.mkdir(parents=True, exist_ok=True)
    reference_features.to_parquet(settings.reference_path)


def run_training(settings: Settings | None = None) -> TrainResult:
    """Run one full training + gated-promotion cycle and return its result."""
    settings = settings or get_settings()
    registry.setup_tracking(settings)

    pipeline, metrics, x_train, _x_test, y_test = fit_candidate(settings)

    current = registry.load_production_pipeline(settings)
    current_metrics = evaluate(cast("SupportsPredict", current), _x_test, y_test) if current is not None else None
    decision = decide_promotion(
        metrics,
        current_metrics,
        metric=settings.gate_metric,
        tolerance=settings.gate_tolerance,
    )

    params = {
        "model": "HistGradientBoostingClassifier",
        "drop_duration": settings.drop_duration,
        "random_state": settings.random_state,
        "test_size": settings.test_size,
    }
    run_id, version = registry.log_training_run(settings, pipeline, params, metrics)

    if decision.promote:
        registry.promote_model(settings, version)
        write_reference(settings, x_train)
        logger.info("Promoted model version %s (run %s)", version, run_id)
    else:
        logger.info("Candidate rejected (run %s): %s", run_id, decision.reason)

    return TrainResult(
        run_id=run_id,
        promoted=decision.promote,
        model_version=version,
        candidate_metrics=metrics,
        message=decision.reason,
    )


def main() -> None:
    """Console entry point (``bank-train``)."""
    result = run_training()
    logger.info(
        "Training complete: run=%s promoted=%s version=%s",
        result.run_id,
        result.promoted,
        result.model_version,
    )
