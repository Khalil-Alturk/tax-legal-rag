"""Read side of the vector store: similarity search with optional metadata filters."""

import logging
from typing import Any

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

logger = logging.getLogger(__name__)


def build_filter(file_name: str | None = None, language: str | None = None) -> dict[str, Any] | None:
    """Translate API filters into a Chroma `where` clause.

    Chroma allows one condition per dict, so two conditions must be wrapped in $and.
    """
    conditions = []
    if file_name:
        conditions.append({"file_name": file_name})
    if language:
        conditions.append({"language": language})
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def retrieve(
    store: VectorStore,
    question: str,
    top_k: int,
    file_name: str | None = None,
    language: str | None = None,
) -> list[Document]:
    """Return the top_k most similar chunks, each with its relevance score in metadata."""
    where = build_filter(file_name, language)
    results = store.similarity_search_with_relevance_scores(question, k=top_k, filter=where)
    docs = []
    for doc, score in results:
        doc.metadata["score"] = round(float(score), 4)
        docs.append(doc)
    logger.info(
        "Retrieved %d chunks (top_k=%d, filter=%s, best=%s)",
        len(docs),
        top_k,
        where,
        docs[0].metadata["score"] if docs else None,
    )
    return docs
