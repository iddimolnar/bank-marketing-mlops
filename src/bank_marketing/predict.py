"""Prediction serving.

Loads the current production pipeline, validates incoming records, produces predictions
and logs every request/prediction pair so monitoring can detect drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import cast

import pandas as pd

from bank_marketing import registry
from bank_marketing.config import Settings, get_settings
from bank_marketing.logging_setup import get_logger
from bank_marketing.logstore import JsonlFileLogStore, RequestLogStore
from bank_marketing.metrics import SupportsPredict
from bank_marketing.schemas import ClientRecord, PredictionResponse

logger = get_logger("predict")


class ModelNotReadyError(RuntimeError):
    """Raised when a prediction is requested but no production model exists yet."""


@dataclass(frozen=True)
class Predictor:
    """Serves predictions from a fitted pipeline and logs each request."""

    pipeline: SupportsPredict
    log_store: RequestLogStore
    model_version: str | None = None

    def predict(self, record: ClientRecord) -> PredictionResponse:
        """Predict for a single validated client record and log the request."""
        frame = pd.DataFrame([record.model_dump()])
        probability = float(self.pipeline.predict_proba(frame)[0, 1])
        prediction = int(self.pipeline.predict(frame)[0])
        response = PredictionResponse(
            label="yes" if prediction == 1 else "no",
            prediction=prediction,
            probability=probability,
            model_version=self.model_version,
        )
        self._log(record, response)
        return response

    def _log(self, record: ClientRecord, response: PredictionResponse) -> None:
        self.log_store.append(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "inputs": record.model_dump(),
                "prediction": response.prediction,
                "label": response.label,
                "probability": response.probability,
                "model_version": response.model_version,
            }
        )


def load_predictor(settings: Settings | None = None) -> Predictor:
    """Build a :class:`Predictor` from the current production model.

    Raises :class:`ModelNotReadyError` if no production model is registered yet.
    """
    settings = settings or get_settings()
    registry.setup_tracking(settings)
    pipeline = registry.load_production_pipeline(settings)
    if pipeline is None:
        raise ModelNotReadyError("No production model registered; run training first.")
    store = JsonlFileLogStore(settings.request_log_path)
    version = registry.production_version(settings)
    logger.info("Loaded production model version %s", version)
    return Predictor(pipeline=cast("SupportsPredict", pipeline), log_store=store, model_version=version)
