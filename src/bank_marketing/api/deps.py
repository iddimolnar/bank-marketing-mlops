"""FastAPI dependency providers.

Kept tiny so tests can override them via ``app.dependency_overrides`` to inject fakes
instead of touching MLflow.
"""

from __future__ import annotations

from bank_marketing.predict import Predictor, load_predictor


def provide_predictor() -> Predictor:
    """Provide a predictor backed by the current production model."""
    return load_predictor()
