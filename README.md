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