# Screenshots

Evidence of the running system (captured from a live `docker compose up`).

| File | Shows |
|------|-------|
| `mlflow_run.png` | MLflow run detail: status, source, logged/registered model, logged **parameters** and **metrics** (roc_auc, f1, accuracy, precision, recall). |
| `mlflow_runs.png` | MLflow experiment run list with the registered model (`bank-marketing-clf v1`). |
| `mlflow_registry.png` | MLflow Model Registry: `bank-marketing-clf`, Version 1. |
| `api_swagger.png` | FastAPI Swagger UI with `/health`, `/predict`, `/train`. |
| `api_predict.png` | `/predict` executed via Swagger returning HTTP **200** with the prediction. |
| `docker.png` | Docker Desktop: `postgres`, `mlflow` (5001), `api` (8000) containers running. |
| `tests_coverage.png` | `pytest` run: **100%** coverage, all tests passing. |

Streamlit + Evidently and Airflow screenshots are added in later phases.