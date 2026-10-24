"""FastAPI application factory and global error handlers."""

import logging

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from langchain_core.exceptions import ModelRateLimitError

from app.api.routes import router
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.rag.embeddings import is_retryable
from app.rag.providers import MissingAPIKeyError

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Build the app; /docs (Swagger) and /redoc are generated from the Pydantic schemas."""
    settings = get_settings()
    secret = settings.google_api_key.get_secret_value() if settings.google_api_key else None
    setup_logging(settings.log_level, secret)

    app = FastAPI(
        title="Tax & Legal Document Q&A",
        description="Bilingual (Arabic/English) RAG API with grounded, cited answers.",
        version="0.1.0",
    )
    app.include_router(router)

    @app.exception_handler(MissingAPIKeyError)
    async def missing_key_handler(_: Request, exc: MissingAPIKeyError) -> JSONResponse:
        logger.error("Request needs Gemini but GOOGLE_API_KEY is missing")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Server misconfigured: GOOGLE_API_KEY is not set."},
        )

    @app.exception_handler(Exception)
    async def unhandled_handler(_: Request, exc: Exception) -> JSONResponse:
        # Rate limits that survived all retries -> 429 so clients know to back off.
        if isinstance(exc, ModelRateLimitError) or is_retryable(exc):
            logger.warning(
                "Upstream rate limit / unavailable after retries: %s", type(exc).__name__
            )
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Model provider is rate-limiting. Try again shortly."},
            )
        # Log the full traceback server-side, return a generic message (no internals leaked).
        logger.exception("Unhandled error")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error."},
        )

    return app


app = create_app()
