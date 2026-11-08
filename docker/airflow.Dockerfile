# syntax=docker/dockerfile:1

# Airflow runs in its own image built from an allowed official base (python:3.12-slim),
# because the namespaced `apache/airflow` image is blocked by this environment's registry
# proxy. Airflow is installed from PyPI with the official constraints file.
FROM python:3.12-slim

ENV AIRFLOW_HOME=/opt/airflow \
    PYTHONUNBUFFERED=1 \
    AIRFLOW__CORE__LOAD_EXAMPLES=False

ARG AIRFLOW_VERSION=2.10.3
ARG PYTHON_VERSION=3.12

RUN pip install --no-cache-dir \
    "apache-airflow==${AIRFLOW_VERSION}" \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"

EXPOSE 8080

# `airflow standalone` initialises the metadata DB, creates an admin user and runs the
# scheduler + webserver together (SQLite + SequentialExecutor) — enough for the demo.
CMD ["airflow", "standalone"]