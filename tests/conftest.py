"""Shared fixtures: fake embeddings, fake LLM, temp Chroma store and a wired TestClient.

Nothing here calls Google, so the suite runs in CI without GOOGLE_API_KEY.
"""

import hashlib
import math
import re
import uuid
from pathlib import Path

import pymupdf
import pytest
from fastapi.testclient import TestClient
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import RunnableLambda

from app.api.routes import get_llm_factory, get_store_factory
from app.core.config import Settings, get_settings
from app.main import app
from app.rag.chain import IDK_AR, IDK_EN
from app.rag.ingest import detect_language
from app.schemas import AnswerOutput, Citation

_WORD = re.compile(r"\w+")
_STOPWORDS = {"what", "which", "does", "with", "from", "that", "this", "about", "under", "there"}


def keywords(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if len(w) > 3 and w not in _STOPWORDS]


class HashEmbeddings(Embeddings):
    """Deterministic bag-of-words embeddings: texts sharing words get similar vectors."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for word in _WORD.findall(text.lower()):
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


_SOURCE = re.compile(r'<source file="(.*?)" page="(\d+)"[^>]*>\n(.*?)\n</source>', re.DOTALL)


class FakeGroundedLLM:
    """Stands in for ChatGoogleGenerativeAI.with_structured_output(AnswerOutput).

    Behaves like a well-grounded model: answers only if a source contains most of the
    question's keywords, otherwise returns the exact "I don't know" sentence.
    """

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.calls = 0

    def with_structured_output(self, schema):
        assert schema is AnswerOutput
        return RunnableLambda(self._respond)

    def _respond(self, prompt: PromptValue) -> AnswerOutput:
        self.calls += 1
        messages = prompt.to_messages()
        system, question = messages[0].content, messages[1].content.removeprefix("Question: ")
        self.prompts.append(system)
        q_words = keywords(question)
        best, best_score = None, 0.0
        for file, page, text in _SOURCE.findall(system):
            score = sum(w in text.lower() for w in q_words) / max(len(q_words), 1)
            if score > best_score:
                best, best_score = (file, int(page), text), score
        if best is None or best_score < 0.6:
            idk = IDK_AR if detect_language(question) == "ar" else IDK_EN
            return AnswerOutput(answer=idk, citations=[], confidence="low")
        file, page, text = best
        return AnswerOutput(
            answer=f"{text[:120]} [{file}, {page}]",
            citations=[Citation(file=file, page=page, snippet=text[:150])],
            confidence="high",
        )


def make_pdf(path: Path, pages: list[str]) -> Path:
    """Write a simple multi-page PDF (Latin text) for tests."""
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(50, 50, 550, 800), text, fontsize=11)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path)
    doc.close()
    return path


ENGLISH_PAGES = [
    "Article 1. The standard Value Added Tax rate is fifteen percent (15%) "
    "on all taxable supplies of goods and services.",
    "Article 2. A person must register for VAT when annual taxable turnover "
    "exceeds 375,000 riyals. Exports of goods are zero-rated.",
]


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        google_api_key=None,
        data_dir=tmp_path / "data",
        chroma_dir=tmp_path / "chroma",
        chunk_size=300,
        chunk_overlap=50,
        max_upload_mb=1,
    )


@pytest.fixture
def store(tmp_path: Path) -> Chroma:
    return Chroma(
        collection_name=f"test_{uuid.uuid4().hex[:8]}",
        embedding_function=HashEmbeddings(),
        persist_directory=str(tmp_path / "chroma"),
        collection_metadata={"hnsw:space": "cosine"},
    )


@pytest.fixture
def fake_llm() -> FakeGroundedLLM:
    return FakeGroundedLLM()


@pytest.fixture
def client(settings: Settings, store: Chroma, fake_llm: FakeGroundedLLM):
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_store_factory] = lambda: lambda: store
    app.dependency_overrides[get_llm_factory] = lambda: lambda: fake_llm
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def indexed_client(client: TestClient, settings: Settings) -> TestClient:
    """Client with the English sample PDF already ingested."""
    make_pdf(settings.data_dir / "vat_test_en.pdf", ENGLISH_PAGES)
    assert client.post("/ingest").status_code == 200
    return client
