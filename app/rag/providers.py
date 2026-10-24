"""Factories for the LLM, embeddings and vector store.

Everything external is created here, behind small functions, so tests can swap
them for fakes (see tests/conftest.py) and the API runs without network calls.
"""

import logging
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from app.core.config import Settings, get_settings
from app.rag.embeddings import BatchedRetryingEmbeddings

logger = logging.getLogger(__name__)


class MissingAPIKeyError(RuntimeError):
    """Raised when a Gemini call is needed but GOOGLE_API_KEY is not set."""


def _require_key(settings: Settings):
    if settings.google_api_key is None or not settings.google_api_key.get_secret_value():
        raise MissingAPIKeyError("GOOGLE_API_KEY is not set. Copy .env.example to .env.")
    return settings.google_api_key


def build_embeddings(settings: Settings) -> Embeddings:
    """Gemini embeddings wrapped with batching + retry."""
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    inner = GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=_require_key(settings),
    )
    return BatchedRetryingEmbeddings(
        inner, batch_size=settings.embed_batch_size, max_retries=settings.embed_max_retries
    )


def build_llm(settings: Settings) -> BaseChatModel:
    """Gemini chat model; built-in retries cover transient errors and rate limits."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    kwargs = {}
    if settings.llm_temperature is not None:
        kwargs["temperature"] = settings.llm_temperature
    return ChatGoogleGenerativeAI(
        model=settings.llm_model,
        google_api_key=_require_key(settings),
        max_retries=settings.embed_max_retries,
        **kwargs,
    )


@lru_cache
def get_embeddings() -> Embeddings:
    return build_embeddings(get_settings())


@lru_cache
def get_llm() -> BaseChatModel:
    return build_llm(get_settings())


@lru_cache
def get_vector_store() -> Chroma:
    """Persistent local Chroma collection using cosine distance."""
    settings = get_settings()
    logger.info(
        "Opening Chroma at %s (collection=%s)", settings.chroma_dir, settings.collection_name
    )
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(settings.chroma_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )
