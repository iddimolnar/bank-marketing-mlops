# Bank Marketing MLOps

Production-ready MLOps system for the **UCI Bank Marketing** dataset, built for the
MLSecOps MSc semester project. Predicts whether a bank client will subscribe to a term
deposit (`y`, binary classification).

> **Status:** under active development. This README grows with the project; see
> `docs/` and the per-phase sections below as they land.

## What this project demonstrates

- Reproducible preprocessing + model in a single scikit-learn `Pipeline`
- Experiment tracking, model registry and artifact storage with **MLflow**
- A **FastAPI** REST service with `/train` and `/predict` endpoints
- Request/prediction logging feeding **Evidently** drift reports
- A **Streamlit** dashboard rendering the drift report
- One-command startup via **Docker Compose** (MLflow, MinIO, API, Streamlit, Airflow)
- CI/CD with **GitHub Actions** + **pre-commit** (tests, coverage, docker build)
- Data versioning with **DVC** (MinIO S3 backend)

## Dataset

UCI Bank Marketing (`bank.csv`, 4521 rows, 17 columns, `;`-delimited, CC BY 4.0, no
personal data). Target `y` (yes/no). The `duration` feature is dropped by default because
it is only known *after* a call completes and therefore leaks the target (documented by
UCI). See `data/` and the dataset report.

## Run the whole system

```bash
docker compose up -d --build
```

This starts PostgreSQL, the MLflow tracking server and the FastAPI service. Then:

| Service | URL |
|---------|-----|
| REST API (Swagger) | http://localhost:8000/docs |
| MLflow UI | http://localhost:5001 |

Smoke test:

```bash
curl -X POST http://localhost:8000/train          # train + gated promotion
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":30,"job":"unemployed","marital":"married","education":"primary","default":"no","balance":1787,"housing":"no","loan":"no","contact":"cellular","day":19,"month":"oct","campaign":1,"pdays":-1,"previous":0,"poutcome":"unknown"}'
docker compose down          # stop (add -v to also drop volumes)
```

> **Artifact store note.** The course used MinIO (S3) as the MLflow artifact store, but
> this environment's Docker registry proxy only allows official library images, so MinIO
> cannot be pulled. MLflow therefore stores artifacts on a shared volume and serves them
> through the tracking server (proxied artifact access) — same functionality, no S3.
> The MLflow UI is mapped to host port **5001** because macOS AirPlay often occupies 5000.

## Local development


```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

flake8 src tests          # PEP 8 (line length 120)
pyright                   # type checking (0 errors in src/)
pytest --cov=src --cov=tests --cov-report=term-missing --cov-fail-under=95
```

## License

Code: MIT. Dataset: CC BY 4.0 (UCI Machine Learning Repository).