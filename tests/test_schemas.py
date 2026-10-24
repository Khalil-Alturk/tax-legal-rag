"""Request validation, both at the Pydantic level and through the API (422)."""

import pytest
from pydantic import ValidationError

from app.rag.retriever import build_filter
from app.schemas import QueryRequest


def test_valid_request_strips_whitespace():
    req = QueryRequest(question="  What is the VAT rate?  ", top_k=3)
    assert req.question == "What is the VAT rate?"
    assert req.top_k == 3


@pytest.mark.parametrize(
    "payload",
    [
        {},  # missing question
        {"question": "   "},  # blank
        {"question": "ok?", "top_k": 0},  # top_k too small
        {"question": "ok?", "top_k": 21},  # top_k too large
        {"question": "ok?", "language_filter": "fr"},  # unsupported language
        {"question": "x" * 2001},  # too long
    ],
)
def test_invalid_requests_raise(payload):
    with pytest.raises(ValidationError):
        QueryRequest(**payload)


@pytest.mark.parametrize("payload", [{}, {"question": " "}, {"question": "abc", "top_k": 50}])
def test_api_returns_422_for_invalid_body(client, payload):
    response = client.post("/query", json=payload)
    assert response.status_code == 422
    assert "detail" in response.json()


def test_build_filter_combinations():
    assert build_filter() is None
    assert build_filter(file_name="a.pdf") == {"file_name": "a.pdf"}
    assert build_filter(file_name="a.pdf", language="ar") == {
        "$and": [{"file_name": "a.pdf"}, {"language": "ar"}]
    }
