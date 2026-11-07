# syntax=docker/dockerfile:1

# MLflow tracking server with a PostgreSQL backend store and proxied artifact
# access backed by a local volume (see docker-compose.yml). No S3/MinIO required.
FROM python:3.12-slim

RUN pip install --no-cache-dir \
    "mlflow==2.17.2" \
    "sqlalchemy==2.0.35" \
    "psycopg2-binary==2.9.10"

EXPOSE 5000

# BACKEND_STORE_URI is provided via the environment. Artifacts are stored under
# /mlartifacts (a mounted volume) and served to clients through the tracking server
# (clients use the ``mlflow-artifacts:`` scheme, so they need no storage credentials).
CMD ["sh", "-c", "mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri \"$BACKEND_STORE_URI\" --artifacts-destination /mlartifacts --default-artifact-root mlflow-artifacts:/ --serve-artifacts"]