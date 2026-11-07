"""MLflow tracking and model-registry helpers.

All direct MLflow calls live here so the rest of the code depends on small, well-typed
functions. The model registry requires a database-backed tracking store (SQLite or
PostgreSQL); a plain file store is not sufficient for registered models.
"""

from __future__ import annotations

from typing import Any, cast

import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline

from bank_marketing.config import Settings

#: Registry stage used for the currently served model.
PRODUCTION_STAGE = "Production"
_ARTIFACT_PATH = "model"


def setup_tracking(settings: Settings) -> None:
    """Point MLflow at the configured tracking server and experiment."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)


def log_training_run(
    settings: Settings,
    pipeline: Pipeline,
    params: dict[str, Any],
    metrics: dict[str, float],
) -> tuple[str, str]:
    """Log one training run and register the resulting model.

    Returns the ``(run_id, model_version)`` pair. The registered version is *not*
    promoted here; promotion is a separate, gated step.
    """
    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        # Pass explicit pip requirements so MLflow does not spawn a module-inference
        # subprocess (which otherwise interferes with coverage measurement).
        info: Any = mlflow.sklearn.log_model(
            pipeline,
            artifact_path=_ARTIFACT_PATH,
            pip_requirements=mlflow.sklearn.get_default_pip_requirements(),
        )
        model_version = mlflow.register_model(info.model_uri, settings.registered_model_name)
        return cast("str", run.info.run_id), str(model_version.version)


def load_production_pipeline(settings: Settings) -> Pipeline | None:
    """Load the current ``Production`` pipeline, or ``None`` if none is registered."""
    uri = f"models:/{settings.registered_model_name}/{PRODUCTION_STAGE}"
    try:
        model = mlflow.sklearn.load_model(uri)
    except MlflowException:
        return None
    return cast("Pipeline", model)


def promote_model(settings: Settings, version: str) -> None:
    """Promote a model version to ``Production`` and archive the previous one."""
    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    client.transition_model_version_stage(
        name=settings.registered_model_name,
        version=version,
        stage=PRODUCTION_STAGE,
        archive_existing_versions=True,
    )
