"""Pydantic v2 schemas for the serving boundary.

These models validate all data entering the system (secure-coding: validate at the
trust boundary). ``ClientRecord`` intentionally omits ``duration`` (dropped as target
leakage) and the target column ``y``.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

# A non-empty, whitespace-trimmed categorical string.
Category = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ClientRecord(BaseModel):
    """One bank client's features, as accepted by the ``/predict`` endpoint."""

    model_config = ConfigDict(extra="forbid")

    age: int = Field(ge=0, le=150)
    job: Category
    marital: Category
    education: Category
    default: Category
    balance: int
    housing: Category
    loan: Category
    contact: Category
    day: int = Field(ge=1, le=31)
    month: Category
    campaign: int = Field(ge=0)
    pdays: int = Field(ge=-1)
    previous: int = Field(ge=0)
    poutcome: Category


class PredictionResponse(BaseModel):
    """Result returned by ``/predict``."""

    # ``model_version`` would otherwise collide with pydantic's protected ``model_``
    # namespace; we opt out because it is a plain data field here.
    model_config = ConfigDict(protected_namespaces=())

    label: str
    prediction: int = Field(ge=0, le=1)
    probability: float = Field(ge=0.0, le=1.0)
    model_version: str | None = None


class TrainRequest(BaseModel):
    """Optional overrides for a ``/train`` run. Empty body uses configured defaults."""

    model_config = ConfigDict(extra="forbid")

    test_size: float | None = Field(default=None, gt=0.0, lt=1.0)
    random_state: int | None = Field(default=None, ge=0)


class TrainResponse(BaseModel):
    """Summary of a ``/train`` run and the promotion decision."""

    model_config = ConfigDict(protected_namespaces=())

    run_id: str
    promoted: bool
    model_version: str | None
    candidate_metrics: dict[str, float]
    message: str
