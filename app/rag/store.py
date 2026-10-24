"""Write side of the vector store: index, replace and count chunks."""

import logging

from langchain_chroma import Chroma
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def index_documents(store: Chroma, docs: list[Document], replace_all: bool = False) -> int:
    """Embed and store chunks; returns how many were written.

    - replace_all=True: wipe the collection first (full re-index of data/).
    - otherwise: delete old chunks of the same files first, so a shorter new version
      of a file doesn't leave stale chunks behind.
    """
    if replace_all:
        store.reset_collection()
    else:
        for file_name in sorted({d.metadata["file_name"] for d in docs}):
            store.delete(where={"file_name": file_name})
    if not docs:
        return 0
    ids = [d.metadata["chunk_id"] for d in docs]
    store.add_documents(docs, ids=ids)
    logger.info("Indexed %d chunks", len(docs))
    return len(docs)


def count_chunks(store: Chroma) -> int:
    """Number of chunks currently in the collection."""
    return store._collection.count()
