"""Feature preprocessing: a scikit-learn ``ColumnTransformer``.

Numeric columns are standardised; categorical columns are one-hot encoded with
``handle_unknown="ignore"`` so categories unseen during training do not crash serving
(a guard against training-serving skew).
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """Build the preprocessing transformer for the given feature columns."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
        ],
        remainder="drop",
    )
