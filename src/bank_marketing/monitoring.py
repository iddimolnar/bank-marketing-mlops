"""Data-drift monitoring with Evidently.

Compares the training feature distribution (the reference snapshot written at promotion
time) against the live traffic reconstructed from the request/prediction log, and renders
an HTML drift report that the Streamlit dashboard displays.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from bank_marketing.config import Settings, get_settings
from bank_marketing.logstore import JsonlFileLogStore


class ReportNotReadyError(RuntimeError):
    """Raised when there is not enough data to build a drift report yet."""


def load_reference(settings: Settings) -> pd.DataFrame | None:
    """Load the reference feature snapshot, or ``None`` if it does not exist yet."""
    if not settings.reference_path.exists():
        return None
    return pd.read_parquet(settings.reference_path)


def load_current(settings: Settings) -> pd.DataFrame:
    """Reconstruct the live feature frame from logged prediction requests."""
    store = JsonlFileLogStore(settings.request_log_path)
    inputs = [record["inputs"] for record in store.read_all() if "inputs" in record]
    return pd.DataFrame(inputs)


def build_drift_report(reference: pd.DataFrame, current: pd.DataFrame, output_path: Path) -> Path:
    """Build and save an Evidently data-drift HTML report."""
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(output_path))
    return output_path


def generate_report(settings: Settings | None = None) -> Path:
    """Generate the drift report from the reference snapshot and the request log.

    Raises :class:`ReportNotReadyError` if the reference snapshot is missing or no
    prediction requests have been logged yet.
    """
    settings = settings or get_settings()
    reference = load_reference(settings)
    current = load_current(settings)
    if reference is None or current.empty:
        raise ReportNotReadyError("A reference snapshot (train first) and at least one logged prediction are required.")
    return build_drift_report(reference, current, settings.drift_report_path)
