"""Typed application configuration via pydantic-settings.

All configuration is environment-driven (12-factor). App settings use the ``BANK_``
prefix; MLflow / S3 settings read their conventional environment variable names so the
same ``.env`` works for the MLflow server and clients. No secrets are hard-coded.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, populated from environment variables / ``.env``."""

    model_config = SettingsConfigDict(
        env_prefix="BANK_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Data ---
    data_path: Path = Field(default=Path("data/fixtures/tiny_bank.csv"))
    target_column: str = Field(default="y")
    drop_duration: bool = Field(default=True)
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    random_state: int = Field(default=42, ge=0)

    # --- Model acceptance / promotion gate ---
    gate_metric: str = Field(default="roc_auc")
    gate_tolerance: float = Field(default=0.0, ge=0.0)

    # --- MLflow (conventional env var names, no BANK_ prefix) ---
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000",
        validation_alias="MLFLOW_TRACKING_URI",
    )
    mlflow_experiment_name: str = Field(
        default="bank-marketing",
        validation_alias="MLFLOW_EXPERIMENT_NAME",
    )
    registered_model_name: str = Field(
        default="bank-marketing-clf",
        validation_alias="MLFLOW_REGISTERED_MODEL_NAME",
    )

    # --- Runtime artifact locations ---
    reference_path: Path = Field(default=Path("data/reference/reference.parquet"))
    request_log_path: Path = Field(default=Path("data/logs/requests.jsonl"))
    drift_report_path: Path = Field(default=Path("data/reports/drift_report.html"))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-wide cached :class:`Settings` instance."""
    return Settings()
