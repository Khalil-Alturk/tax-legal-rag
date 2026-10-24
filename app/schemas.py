"""Pydantic models for the API contract and the LLM's structured output."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Language = Literal["ar", "en", "unknown"]
Confidence = Literal["low", "medium", "high"]


class QueryRequest(BaseModel):
    """Body of POST /query."""

    question: str = Field(..., min_length=3, max_length=2000, examples=["What is the VAT rate?"])
    top_k: int | None = Field(None, ge=1, le=20, description="Chunks to retrieve (default 4).")
    file_filter: str | None = Field(None, description="Only search this file name.")
    language_filter: Language | None = Field(None, description="Only search chunks in 'ar'/'en'.")

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 3:
            raise ValueError("question must contain at least 3 non-space characters")
        return v


class Citation(BaseModel):
    """One source the answer relies on."""

    file: str = Field(..., description="Source file name, e.g. vat_regulation_en.pdf")
    page: int = Field(..., ge=1, description="1-based page number")
    snippet: str = Field(..., description="Short quote from the chunk supporting the answer")


class AnswerOutput(BaseModel):
    """What the LLM must return (enforced via with_structured_output)."""

    answer: str = Field(
        ...,
        description="Answer in the question's language, with inline [file, page] citations.",
    )
    citations: list[Citation] = Field(default_factory=list)
    confidence: Confidence = Field(..., description="How well the context supports the answer.")


class QueryResponse(AnswerOutput):
    """API response: the LLM output plus retrieval metadata added by our code."""

    retrieved_chunks_count: int = Field(..., ge=0)


class IngestResponse(BaseModel):
    """Result of POST /ingest."""

    files_processed: list[str]
    chunks_indexed: int
    message: str


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    llm_model: str
    embedding_model: str
    indexed_chunks: int
