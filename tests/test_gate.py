"""Tests for :mod:`bank_marketing.gate`."""

from __future__ import annotations

import pytest

from bank_marketing.gate import PromotionDecision, decide_promotion


def test_promote_when_no_production_model() -> None:
    decision = decide_promotion({"roc_auc": 0.80}, None)
    assert isinstance(decision, PromotionDecision)
    assert decision.promote is True
    assert "No production model" in decision.reason


def test_promote_when_candidate_not_worse() -> None:
    decision = decide_promotion({"roc_auc": 0.90}, {"roc_auc": 0.90})
    assert decision.promote is True


def test_reject_when_candidate_worse() -> None:
    decision = decide_promotion({"roc_auc": 0.80}, {"roc_auc": 0.90})
    assert decision.promote is False
    assert "keeping production" in decision.reason


def test_tolerance_allows_small_regression() -> None:
    decision = decide_promotion({"roc_auc": 0.88}, {"roc_auc": 0.90}, tolerance=0.05)
    assert decision.promote is True


def test_missing_metric_raises_key_error() -> None:
    with pytest.raises(KeyError):
        decide_promotion({"f1": 0.5}, None, metric="roc_auc")
