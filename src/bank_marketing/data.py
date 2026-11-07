"""Data loading, target preparation and train/test splitting.

The raw UCI Bank Marketing CSV is semicolon-delimited with quoted strings. ``duration``
is dropped by default because it is only known after a call completes and therefore leaks
the target (documented by UCI).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pandas as pd
from sklearn.model_selection import train_test_split

#: Numeric feature columns that are always available before a call is made.
NUMERIC_BASE: tuple[str, ...] = ("age", "balance", "day", "campaign", "pdays", "previous")
#: Leakage-prone numeric column (only known after the call).
DURATION_COLUMN = "duration"
#: Categorical feature columns.
CATEGORICAL_COLUMNS: tuple[str, ...] = (
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
)
#: Mapping from the raw yes/no target to a binary integer label.
TARGET_MAP = {"no": 0, "yes": 1}


def load_data(path: Path) -> pd.DataFrame:
    """Load the semicolon-delimited Bank Marketing CSV into a DataFrame."""
    return pd.read_csv(path, sep=";")


def feature_columns(*, drop_duration: bool = True) -> tuple[list[str], list[str]]:
    """Return the ``(numeric, categorical)`` feature column names.

    ``duration`` is appended to the numeric columns only when ``drop_duration`` is False.
    """
    numeric = list(NUMERIC_BASE)
    if not drop_duration:
        numeric.append(DURATION_COLUMN)
    return numeric, list(CATEGORICAL_COLUMNS)


def prepare_xy(
    df: pd.DataFrame,
    *,
    target_column: str = "y",
    drop_duration: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a raw DataFrame into the feature matrix ``X`` and binary target ``y``."""
    numeric, categorical = feature_columns(drop_duration=drop_duration)
    features = cast("pd.DataFrame", df[numeric + categorical].copy())

    # pandas' untyped __getitem__/map return ``Unknown`` under pyright; cast at this
    # boundary keeps the public signature precisely typed.
    raw_target = cast("Any", df[target_column])
    mapped = raw_target.map(TARGET_MAP)
    if bool(mapped.isna().any()):
        raise ValueError(f"Target column {target_column!r} contains values outside {set(TARGET_MAP)}.")
    target = cast("pd.Series", mapped.astype(int))
    target.name = target_column
    return features, target


def split(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split (preserves the class balance of ``target``)."""
    splits = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )
    # sklearn's train_test_split is untyped; cast the 4-tuple at this boundary.
    return cast(
        "tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]",
        tuple(splits),
    )
