"""Tests for :mod:`bank_marketing.schemas`."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from bank_marketing.schemas import (
    ClientRecord,
    PredictionResponse,
    TrainRequest,
    TrainResponse,
)

_VALID_RECORD = {
    "age": 30,
    "job": "unemployed",
    "marital": "married",
    "education": "primary",
    "default": "no",
    "balance": 1787,
    "housing": "no",
    "loan": "no",
    "contact": "cellular",
    "day": 19,
    "month": "oct",
    "campaign": 1,
    "pdays": -1,
    "previous": 0,
    "poutcome": "unknown",
}


def test_valid_record_parses() -> None:
    record = ClientRecord(**_VALID_RECORD)
    assert record.age == 30
    assert record.balance == 1787
    assert record.pdays == -1


def test_categorical_whitespace_is_stripped() -> None:
    record = ClientRecord(**{**_VALID_RECORD, "job": "  admin  "})
    assert record.job == "admin"


def test_extra_field_forbidden() -> None:
    with pytest.raises(ValidationError):
        ClientRecord(**{**_VALID_RECORD, "unexpected": 1})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("age", -1),
        ("age", 151),
        ("day", 0),
        ("day", 32),
        ("pdays", -2),
        ("previous", -1),
        ("campaign", -1),
        ("job", ""),
    ],
)
def test_out_of_range_values_rejected(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        ClientRecord(**{**_VALID_RECORD, field: value})


def test_negative_balance_allowed() -> None:
    record = ClientRecord(**{**_VALID_RECORD, "balance": -500})
    assert record.balance == -500


def test_prediction_response() -> None:
    resp = PredictionResponse(label="yes", prediction=1, probability=0.87, model_version="3")
    assert resp.label == "yes"
    assert resp.model_version == "3"


@pytest.mark.parametrize(
    ("prediction", "probability"),
    [(2, 0.5), (-1, 0.5), (1, 1.5), (0, -0.1)],
)
def test_prediction_response_bounds(prediction: int, probability: float) -> None:
    with pytest.raises(ValidationError):
        PredictionResponse(label="x", prediction=prediction, probability=probability)


def test_train_request_defaults_and_validation() -> None:
    empty = TrainRequest()
    assert empty.test_size is None
    assert empty.random_state is None
    with pytest.raises(ValidationError):
        TrainRequest(test_size=1.2)


def test_train_response_construction() -> None:
    resp = TrainResponse(
        run_id="abc",
        promoted=True,
        model_version="1",
        candidate_metrics={"roc_auc": 0.9},
        message="promoted",
    )
    assert resp.promoted is True
    assert resp.candidate_metrics["roc_auc"] == 0.9
