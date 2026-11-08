"""Airflow DAG: scheduled model retraining.

Thin orchestration only. To avoid dependency conflicts (Airflow pins SQLAlchemy < 2.0,
MLflow needs >= 2.0), this DAG does **not** import the project package. Instead it triggers
retraining through the API's ``/train`` endpoint over HTTP, so all ML logic stays in the
``api`` service.
"""

from __future__ import annotations

import json
import urllib.request
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

API_TRAIN_URL = "http://api:8000/train"


def _trigger_training() -> None:
    """Call the API's /train endpoint and log the response."""
    request = urllib.request.Request(  # noqa: S310 - fixed internal compose URL, not user input
        API_TRAIN_URL,
        data=json.dumps({}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:  # noqa: S310 - internal URL
        print(f"/train response: {response.read().decode('utf-8')}")  # noqa: T201 - visible in Airflow logs


with DAG(
    dag_id="bank_marketing_retrain",
    description="Daily retraining of the bank-marketing model via the API.",
    schedule="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
    tags=["bank-marketing"],
) as dag:
    PythonOperator(task_id="retrain", python_callable=_trigger_training)