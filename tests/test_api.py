"""Endpoint tests with mocked LLM + embeddings."""

from app.api.routes import get_llm_factory
from app.main import app
from app.rag.providers import MissingAPIKeyError
from tests.conftest import ENGLISH_PAGES, make_pdf


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["api_key_configured"] is False
    assert body["llm_model"]


def test_docs_are_generated(client):
    assert client.get("/docs").status_code == 200
    assert {"/health", "/ingest", "/query"} <= set(client.get("/openapi.json").json()["paths"])


def test_ingest_reindexes_data_folder(client, settings):
    make_pdf(settings.data_dir / "a.pdf", ENGLISH_PAGES)
    body = client.post("/ingest").json()
    assert body["files_processed"] == ["a.pdf"]
    assert body["chunks_indexed"] >= 2


def test_ingest_empty_folder_is_400(client, settings):
    settings.data_dir.mkdir(parents=True)
    response = client.post("/ingest")
    assert response.status_code == 400
    assert "No PDF files" in response.json()["detail"]


def test_upload_valid_pdf(client, tmp_path):
    pdf = make_pdf(tmp_path / "upload.pdf", ENGLISH_PAGES).read_bytes()
    response = client.post(
        "/ingest", files={"file": ("../../evil name.pdf", pdf, "application/pdf")}
    )
    assert response.status_code == 200
    # Path traversal stripped, unsafe characters replaced.
    assert response.json()["files_processed"] == ["evil_name.pdf"]


def test_upload_rejects_non_pdf_extension(client):
    response = client.post("/ingest", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 400
    assert "Only .pdf" in response.json()["detail"]


def test_upload_rejects_fake_pdf(client):
    response = client.post("/ingest", files={"file": ("fake.pdf", b"not a pdf", "application/pdf")})
    assert response.status_code == 400
    assert "not a valid PDF" in response.json()["detail"]


def test_upload_rejects_too_large(client):
    big = b"%PDF-1.7\n" + b"0" * (1024 * 1024 + 10)  # settings.max_upload_mb = 1
    response = client.post("/ingest", files={"file": ("big.pdf", big, "application/pdf")})
    assert response.status_code == 413


def test_query_returns_structured_answer_with_citations(indexed_client):
    response = indexed_client.post("/query", json={"question": "What is the standard VAT rate?"})
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"answer", "citations", "confidence", "retrieved_chunks_count"}
    assert "15%" in body["answer"]
    assert body["citations"][0] == {
        "file": "vat_test_en.pdf",
        "page": 1,
        "snippet": body["citations"][0]["snippet"],
    }
    assert body["confidence"] == "high"
    assert body["retrieved_chunks_count"] >= 1


def test_query_top_k_limits_retrieval(indexed_client):
    body = indexed_client.post("/query", json={"question": "VAT rate?", "top_k": 1}).json()
    assert body["retrieved_chunks_count"] == 1


def test_query_file_filter_with_unknown_file(indexed_client, fake_llm):
    body = indexed_client.post(
        "/query", json={"question": "What is the VAT rate?", "file_filter": "missing.pdf"}
    ).json()
    assert body["retrieved_chunks_count"] == 0
    assert body["citations"] == []
    assert fake_llm.calls == 0  # nothing retrieved -> LLM never called


def test_query_without_api_key_is_500_with_clear_message(indexed_client):
    def raise_missing():
        raise MissingAPIKeyError("GOOGLE_API_KEY is not set.")

    app.dependency_overrides[get_llm_factory] = lambda: raise_missing
    response = indexed_client.post("/query", json={"question": "What is the VAT rate?"})
    assert response.status_code == 500
    assert "GOOGLE_API_KEY" in response.json()["detail"]
