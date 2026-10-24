"""Chunking: size limits, overlap, Arabic separators, page mapping, stable ids."""

from app.rag.ingest import PageText, chunk_pages, make_splitter


def test_chunks_respect_size_limit():
    pages = [PageText("doc.pdf", 1, "word " * 1000)]
    docs = chunk_pages(pages, chunk_size=300, chunk_overlap=50)
    assert len(docs) > 1
    assert all(len(d.page_content) <= 300 for d in docs)


def test_consecutive_chunks_overlap():
    text = " ".join(f"token{i}" for i in range(300))
    docs = chunk_pages([PageText("doc.pdf", 1, text)], chunk_size=200, chunk_overlap=60)
    first_tail = docs[0].page_content.split()[-1]
    assert first_tail in docs[1].page_content


def test_arabic_text_splits_on_arabic_punctuation():
    sentence = "تفرض ضريبة القيمة المضافة بنسبة خمسة عشر بالمئة على التوريدات، "
    chunks = make_splitter(chunk_size=150, chunk_overlap=0).split_text(sentence * 10)
    assert len(chunks) > 1
    # Splits happen after the Arabic comma, never mid-word.
    assert all(c.endswith("،") or c.endswith("التوريدات") for c in chunks[:-1])


def test_each_chunk_maps_to_its_page():
    pages = [PageText("a.pdf", 1, "first page " * 50), PageText("a.pdf", 2, "second page " * 50)]
    docs = chunk_pages(pages, chunk_size=200, chunk_overlap=20)
    assert {d.metadata["page"] for d in docs} == {1, 2}
    assert all(("first" in d.page_content) == (d.metadata["page"] == 1) for d in docs)


def test_chunk_ids_are_deterministic_and_unique():
    pages = [PageText("reg.pdf", 3, "alpha beta gamma " * 100)]
    first = [d.metadata["chunk_id"] for d in chunk_pages(pages, 200, 20)]
    second = [d.metadata["chunk_id"] for d in chunk_pages(pages, 200, 20)]
    assert first == second
    assert len(set(first)) == len(first)
    assert first[0] == "reg-p3-c0"
