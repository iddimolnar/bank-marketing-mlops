"""FastAPI application factory.

Run locally with::

    uvicorn bank_marketing.api.app:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from bank_marketing.api.routes import router
from bank_marketing.predict import ModelNotReadyError


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    app = FastAPI(
        title="Bank Marketing MLOps API",
        version="0.1.0",
        description="Train and serve a term-deposit subscription classifier.",
    )
    app.include_router(router)

    @app.exception_handler(ModelNotReadyError)
    async def _model_not_ready(_request: Request, exc: ModelNotReadyError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    return app


app = create_app()
