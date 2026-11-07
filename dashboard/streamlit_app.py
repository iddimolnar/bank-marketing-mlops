"""Streamlit dashboard for data-drift monitoring.

Thin UI layer: all logic lives in :mod:`bank_marketing.monitoring`. Run with::

    streamlit run dashboard/streamlit_app.py
"""

from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

from bank_marketing.config import get_settings
from bank_marketing.monitoring import ReportNotReadyError, generate_report


def main() -> None:  # pragma: no cover - thin UI layer; logic tested in monitoring
    st.set_page_config(page_title="Bank Marketing — Drift", layout="wide")
    st.title("Bank Marketing — Data Drift (Evidently)")
    settings = get_settings()

    if st.button("Generate / refresh drift report"):
        try:
            path = generate_report(settings)
            st.success(f"Report generated: {path}")
        except ReportNotReadyError as exc:
            st.warning(str(exc))

    if settings.drift_report_path.exists():
        html = settings.drift_report_path.read_text(encoding="utf-8")
        components.html(html, height=900, scrolling=True)
    else:
        st.info("No report yet. Train a model, send some /predict requests, then generate.")


if __name__ == "__main__":  # pragma: no cover
    main()
