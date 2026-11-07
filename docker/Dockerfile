# syntax=docker/dockerfile:1

# --- Build stage: install the package and its dependencies ---
FROM python:3.12-slim AS builder

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

# --- Runtime stage: slim image, non-root user ---
FROM python:3.12-slim

WORKDIR /app

# Security: run as a non-root user.
RUN useradd --create-home appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copy the installed packages and console scripts from the builder.
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Bundle the committed fixture so training works even without a mounted dataset.
COPY --chown=appuser:appuser data/fixtures ./data/fixtures

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "bank_marketing.api.app:app", "--host", "0.0.0.0", "--port", "8000"]