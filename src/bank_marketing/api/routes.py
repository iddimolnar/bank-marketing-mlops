"""API routes: ``/health``, ``/predict`` and ``/train``."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from bank_marketing.api.deps import provide_predictor
from bank_marketing.config import Settings, get_settings
from bank_marketing.predict import Predictor
from bank_marketing.schemas import (
    ClientRecord,
    PredictionResponse,
    TrainRequest,
    TrainResponse,
)
from bank_marketing.train import run_training

router = APIRouter()


@router.get("/health", tags=["system"])
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@router.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict_endpoint(
    record: ClientRecord,
    predictor: Predictor = Depends(provide_predictor),
) -> PredictionResponse:
    """Predict term-deposit subscription for a single client."""
    return predictor.predict(record)


def _settings_for(request: TrainRequest) -> Settings:
    """Apply optional per-request overrides onto the base settings."""
    updates: dict[str, object] = {}
    if request.test_size is not None:
        updates["test_size"] = request.test_size
    if request.random_state is not None:
        updates["random_state"] = request.random_state
    settings = get_settings()
    return settings.model_copy(update=updates) if updates else settings


@router.post("/train", response_model=TrainResponse, tags=["training"])
def train_endpoint(request: TrainRequest) -> TrainResponse:
    """Trigger a training + gated-promotion cycle."""
    result = run_training(_settings_for(request))
    return TrainResponse(
        run_id=result.run_id,
        promoted=result.promoted,
        model_version=result.model_version,
        candidate_metrics=result.candidate_metrics,
        message=result.message,
    )
