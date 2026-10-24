"""Text cleaning, language detection and metadata extracted from real PDF files."""

import logging

import pytest

from app.core.logging import RedactSecretsFilter
from app.rag.ingest import chunk_pages, clean_text, detect_language, load_and_chunk, load_pdf_pages
from tests.conftest import ENGLISH_PAGES, make_pdf


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("المادة الخامسة: نسبة الضريبة", "ar"),
        ("Article 5: the VAT rate", "en"),
        ("المادة 5 of the VAT law نسبة الضريبة", "ar"),  # mostly Arabic letters
        ("12345 %%%", "unknown"),
    ],
)
def test_detect_language(text, expected):
    assert detect_language(text) == expected


def test_clean_text_normalizes_arabic():
    raw = "ﻻ   ضريبةـــ على الدَّخْلِ"  # presentation form, tatweel, diacritics, spaces
    assert clean_text(raw) == "لا ضريبة على الدخل"


def test_clean_text_joins_hyphenated_words_and_collapses_blank_lines():
    assert clean_text("compu-\nted\n\n\n\nnext") == "computed\n\nnext"


def test_load_pdf_pages_returns_one_based_pages(tmp_path):
    pdf = make_pdf(tmp_path / "law.pdf", ENGLISH_PAGES)
    pages = load_pdf_pages(pdf)
    assert [p.page for p in pages] == [1, 2]
    assert all(p.file_name == "law.pdf" for p in pages)
    assert "fifteen percent" in pages[0].text


def test_blank_pages_are_skipped(tmp_path):
    pdf = make_pdf(tmp_path / "scan.pdf", ["", "Some text"])
    assert [p.page for p in load_pdf_pages(pdf)] == [2]


def test_every_chunk_has_required_metadata(tmp_path):
    pdf = make_pdf(tmp_path / "law.pdf", ENGLISH_PAGES)
    docs = load_and_chunk([pdf], chunk_size=300, chunk_overlap=50)
    assert docs
    for d in docs:
        assert set(d.metadata) == {"file_name", "page", "language", "chunk_id"}
        assert d.metadata["file_name"] == "law.pdf"
        assert d.metadata["language"] == "en"


def test_chunk_language_is_detected_per_chunk():
    from app.rag.ingest import PageText

    docs = chunk_pages([PageText("ar.pdf", 1, "تفرض الضريبة على التوريدات")], 300, 0)
    assert docs[0].metadata["language"] == "ar"


def test_api_key_is_redacted_from_logs():
    fake_key = "AIza" + "x" * 35
    record = logging.LogRecord("t", logging.INFO, "", 0, "key=%s", (fake_key,), None)
    RedactSecretsFilter(secret=fake_key).filter(record)
    assert fake_key not in record.getMessage()
    assert "***" in record.getMessage()


def test_committed_arabic_sample_extracts_in_logical_order():
    """Regression: sort=True reversed Arabic word order; words must read right-to-left."""
    from pathlib import Path

    pages = load_pdf_pages(Path("data/income_tax_regulation_ar.pdf"))
    text = " ".join(pages[0].text.split())
    assert "تفرض ضريبة الدخل على الشركات المقيمة بنسبة 20 بالمئة" in text
    assert all(detect_language(p.text) == "ar" for p in pages)
