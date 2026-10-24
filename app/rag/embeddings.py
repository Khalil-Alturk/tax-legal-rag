"""Embeddings wrapper that batches requests and retries on rate limits (free-tier friendly)."""

import logging

from langchain_core.embeddings import Embeddings
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

logger = logging.getLogger(__name__)

_RETRYABLE_CODES = {429, 500, 502, 503, 504}
_RETRYABLE_TEXT = ("429", "RESOURCE_EXHAUSTED", "rate limit", "UNAVAILABLE", "503")


def is_retryable(exc: BaseException) -> bool:
    """True for rate-limit / transient server errors.

    langchain-google-genai wraps google.genai errors, so walk the __cause__ chain
    looking for an HTTP status code, then fall back to the message text.
    """
    current: BaseException | None = exc
    while current is not None:
        code = getattr(current, "code", None)
        if isinstance(code, int) and code in _RETRYABLE_CODES:
            return True
        current = current.__cause__
    message = str(exc)
    return any(marker.lower() in message.lower() for marker in _RETRYABLE_TEXT)


class BatchedRetryingEmbeddings(Embeddings):
    """Wraps any LangChain Embeddings: splits into batches, retries each batch with backoff.

    Retrying per batch means a 429 halfway through ingestion only repeats that batch.
    """

    def __init__(
        self,
        inner: Embeddings,
        batch_size: int = 20,
        max_retries: int = 5,
        max_wait_seconds: float = 60.0,
    ) -> None:
        self.inner = inner
        self.batch_size = batch_size
        self._retry = retry(
            retry=retry_if_exception(is_retryable),
            # Exponential backoff with jitter: ~1s, 2s, 4s ... capped, randomized.
            wait=wait_random_exponential(multiplier=1, max=max_wait_seconds),
            stop=stop_after_attempt(max_retries),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        total = len(texts)
        for start in range(0, total, self.batch_size):
            batch = texts[start : start + self.batch_size]
            vectors.extend(self._retry(self.inner.embed_documents)(batch))
            logger.info("Embedded %d/%d chunks", min(start + self.batch_size, total), total)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._retry(self.inner.embed_query)(text)
