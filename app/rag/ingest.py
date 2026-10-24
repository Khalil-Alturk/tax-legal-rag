"""Ingestion: PDF -> cleaned page text -> chunks with metadata.

Pure functions only; embedding and storage live in app.rag.store so this module
can be unit-tested without any API key.
"""

import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

_ARABIC_LETTER = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿ]")
_LATIN_LETTER = re.compile(r"[A-Za-z]")
_TASHKEEL = re.compile(r"[ً-ْٰ]")  # Arabic diacritics
_TATWEEL = "ـ"  # Arabic elongation character, purely visual

# Try paragraph -> line -> sentence (Latin and Arabic punctuation) -> clause -> word.
SEPARATORS = ["\n\n", "\n", ". ", "۔ ", "؟ ", "? ", "! ", "؛ ", "; ", "، ", ", ", " ", ""]


@dataclass(frozen=True)
class PageText:
    """Raw text of one PDF page (page is 1-based)."""

    file_name: str
    page: int
    text: str


def clean_text(text: str) -> str:
    """Normalize PDF text for both Arabic and English.

    - NFKC turns Arabic presentation forms (e.g. U+FEFB) back into normal letters.
    - Tatweel and diacritics are removed so the same word always embeds the same.
    - Words hyphenated across line breaks are re-joined; whitespace is collapsed.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace(_TATWEEL, "")
    text = _TASHKEEL.sub("", text)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def detect_language(text: str) -> str:
    """Return 'ar', 'en' or 'unknown' from the share of Arabic vs Latin letters."""
    arabic = len(_ARABIC_LETTER.findall(text))
    latin = len(_LATIN_LETTER.findall(text))
    total = arabic + latin
    if total == 0:
        return "unknown"
    return "ar" if arabic / total >= 0.3 else "en"


def load_pdf_pages(path: Path) -> list[PageText]:
    """Extract and clean text page by page. Pages with no text (e.g. scans) are skipped."""
    pages: list[PageText] = []
    with pymupdf.open(path) as doc:
        for index, page in enumerate(doc, start=1):
            text = clean_text(page.get_text("text", sort=True))
            if text:
                pages.append(PageText(file_name=path.name, page=index, text=text))
            else:
                logger.warning("No extractable text on %s page %d (scanned?)", path.name, index)
    return pages


def make_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    """Character-based splitter with Arabic-aware separators."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS,
        keep_separator="end",
        strip_whitespace=True,
    )


def chunk_pages(pages: list[PageText], chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Split each page separately so every chunk maps to exactly one page."""
    splitter = make_splitter(chunk_size, chunk_overlap)
    docs: list[Document] = []
    for page in pages:
        for i, chunk in enumerate(splitter.split_text(page.text)):
            stem = Path(page.file_name).stem
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "file_name": page.file_name,
                        "page": page.page,
                        "language": detect_language(chunk),
                        # Deterministic id: re-ingesting the same file overwrites, not duplicates.
                        "chunk_id": f"{stem}-p{page.page}-c{i}",
                    },
                )
            )
    return docs


def load_and_chunk(paths: list[Path], chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Full offline pipeline for several PDFs."""
    docs: list[Document] = []
    for path in paths:
        pages = load_pdf_pages(path)
        file_docs = chunk_pages(pages, chunk_size, chunk_overlap)
        logger.info(
            "Parsed %s: %d pages with text, %d chunks", path.name, len(pages), len(file_docs)
        )
        docs.extend(file_docs)
    return docs
