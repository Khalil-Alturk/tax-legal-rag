"""HTTP endpoints: /health, /ingest, /query."""

import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import pymupdf
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from langchain_chroma import Chroma
from langchain_core.language_models import BaseChatModel

from app.core.config import Settings, get_settings
from app.rag import providers
from app.rag.chain import answer_question
from app.rag.ingest import load_and_chunk
from app.rag.retriever import retrieve
from app.rag.store import count_chunks, index_documents
from app.schemas import HealthResponse, IngestResponse, QueryRequest, QueryResponse

logger = logging.getLogger(__name__)
router = APIRouter()


def get_store_factory() -> Callable[[], Chroma]:
    """Dependency returning a *factory*, not the store itself.

    FastAPI resolves dependencies before validating the body, so building the store
    here would turn a bad request (422) into a missing-key error (500). Endpoints call
    the factory only after validation. Tests override this to return a fake store.
    """
    return providers.get_vector_store


def get_llm_factory() -> Callable[[], BaseChatModel]:
    """Dependency returning the chat-model factory (tests override with a fake LLM)."""
    return providers.get_llm


SettingsDep = Annotated[Settings, Depends(get_settings)]
StoreFactory = Annotated[Callable[[], Chroma], Depends(get_store_factory)]
LLMFactory = Annotated[Callable[[], BaseChatModel], Depends(get_llm_factory)]

_SAFE_NAME = re.compile(r"[^\w.\-؀-ۿ]+")


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health(settings: SettingsDep) -> HealthResponse:
    """Liveness check. Works even without an API key so containers can report health."""
    key_ok = bool(settings.google_api_key and settings.google_api_key.get_secret_value())
    indexed = None
    if key_ok:
        try:
            indexed = count_chunks(providers.get_vector_store())
        except Exception:  # health must never crash
            logger.exception("Could not open vector store for health check")
    return HealthResponse(
        llm_model=settings.llm_model,
        embedding_model=settings.embedding_model,
        api_key_configured=key_ok,
        indexed_chunks=indexed,
    )


def _safe_filename(name: str) -> str:
    """Drop any path components and odd characters from a client-supplied file name."""
    base = Path(name).name
    cleaned = _SAFE_NAME.sub("_", base).strip("._") or "upload"
    return cleaned if cleaned.lower().endswith(".pdf") else f"{cleaned}.pdf"


def _read_validated_pdf(file: UploadFile, max_bytes: int) -> bytes:
    """Check extension, size and PDF magic bytes; raise 400/413 with a clear message."""
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Only .pdf files are accepted.")
    # Read at most max_bytes+1 so an oversized upload is never fully loaded into memory.
    content = file.file.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            f"File too large. Limit is {max_bytes // (1024 * 1024)} MB.",
        )
    if not content.startswith(b"%PDF-"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "File is not a valid PDF.")
    try:
        with pymupdf.open(stream=content, filetype="pdf"):
            pass
    except Exception as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "PDF is corrupted or encrypted.") from exc
    return content


@router.post("/ingest", response_model=IngestResponse, tags=["ingestion"])
def ingest(
    settings: SettingsDep,
    store_factory: StoreFactory,
    file: Annotated[UploadFile | None, File(description="Optional single PDF to add")] = None,
) -> IngestResponse:
    """Upload one PDF to add it to the index, or send no file to re-index the whole data folder."""
    if file is not None:
        content = _read_validated_pdf(file, settings.max_upload_bytes)
        upload_dir = settings.data_dir / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        target = upload_dir / _safe_filename(file.filename or "upload.pdf")
        target.write_bytes(content)
        paths, replace_all = [target], False
    else:
        paths = sorted(settings.data_dir.rglob("*.pdf"))
        replace_all = True
        if not paths:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST, f"No PDF files found in '{settings.data_dir}'."
            )

    docs = load_and_chunk(paths, settings.chunk_size, settings.chunk_overlap)
    if not docs:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "No extractable text found. Scanned PDFs need OCR (see README: next steps).",
        )
    count = index_documents(store_factory(), docs, replace_all=replace_all)
    files = sorted({d.metadata["file_name"] for d in docs})
    return IngestResponse(
        files_processed=files,
        chunks_indexed=count,
        message="Re-indexed data folder." if replace_all else f"Added {files[0]}.",
    )


@router.post("/query", response_model=QueryResponse, tags=["qa"])
def query(
    body: QueryRequest,
    settings: SettingsDep,
    store_factory: StoreFactory,
    llm_factory: LLMFactory,
) -> QueryResponse:
    """Answer a question from the indexed documents, with [file, page] citations."""
    top_k = min(body.top_k or settings.default_top_k, settings.max_top_k)
    docs = retrieve(store_factory(), body.question, top_k, body.file_filter, body.language_filter)
    return answer_question(llm_factory(), body.question, docs)
