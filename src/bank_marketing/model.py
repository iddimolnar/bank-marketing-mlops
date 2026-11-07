"""Model construction: a single scikit-learn ``Pipeline``.

Wrapping preprocessing and the estimator in one Pipeline means the fitted preprocessing
travels with the model artifact, so training and serving apply identical transformations.
"""

from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

from bank_marketing.data import feature_columns
from bank_marketing.features import build_preprocessor


def build_pipeline(
    numeric: list[str],
    categorical: list[str],
    *,
    random_state: int = 42,
) -> Pipeline:
    """Build the full preprocessing + classifier pipeline."""
    preprocessor = build_preprocessor(numeric, categorical)
    classifier = HistGradientBoostingClassifier(random_state=random_state)
    return Pipeline([("pre", preprocessor), ("clf", classifier)])


def build_default_pipeline(*, drop_duration: bool = True, random_state: int = 42) -> Pipeline:
    """Convenience builder using the project's default feature columns."""
    numeric, categorical = feature_columns(drop_duration=drop_duration)
    return build_pipeline(numeric, categorical, random_state=random_state)
