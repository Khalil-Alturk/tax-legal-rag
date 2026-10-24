"""Batching and retry behaviour of the embeddings wrapper."""

import pytest
from langchain_core.embeddings import Embeddings

from app.rag.embeddings import BatchedRetryingEmbeddings, is_retryable


class _RateLimited(Exception):
    code = 429


class _FlakyEmbeddings(Embeddings):
    """Fails with 429 the first `failures` calls, then succeeds."""

    def __init__(self, failures: int = 0, error: Exception | None = None) -> None:
        self.failures = failures
        self.error = error or _RateLimited("429 RESOURCE_EXHAUSTED")
        self.batches: list[int] = []

    def embed_documents(self, texts):
        if self.failures > 0:
            self.failures -= 1
            raise self.error
        self.batches.append(len(texts))
        return [[float(len(t))] for t in texts]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def _wrap(inner, **kw):
    return BatchedRetryingEmbeddings(inner, max_wait_seconds=0.01, **kw)


def test_texts_are_sent_in_batches():
    inner = _FlakyEmbeddings()
    vectors = _wrap(inner, batch_size=3).embed_documents(["a"] * 8)
    assert inner.batches == [3, 3, 2]
    assert len(vectors) == 8


def test_rate_limit_is_retried_then_succeeds():
    inner = _FlakyEmbeddings(failures=2)
    assert _wrap(inner, max_retries=5).embed_documents(["abc"]) == [[3.0]]


def test_gives_up_after_max_retries():
    inner = _FlakyEmbeddings(failures=10)
    with pytest.raises(_RateLimited):
        _wrap(inner, max_retries=3).embed_documents(["abc"])


def test_non_retryable_error_fails_fast():
    inner = _FlakyEmbeddings(failures=1, error=ValueError("invalid model name"))
    with pytest.raises(ValueError):
        _wrap(inner, max_retries=5).embed_documents(["abc"])
    assert inner.failures == 0  # called exactly once


def test_is_retryable_follows_wrapped_cause():
    wrapper = RuntimeError("Error embedding content")
    wrapper.__cause__ = _RateLimited()
    assert is_retryable(wrapper)
    assert not is_retryable(ValueError("bad input"))
