import pytest

from src.chunking import (
    build_chunks,
    extract_source_span,
    split_text_with_offsets,
    split_text_with_overlap,
)
from src.corpus import Document


def test_split_text_with_overlap_returns_multiple_chunks() -> None:
    text = "abcdefghijklmnopqrstuvwxyz " * 4
    chunks = split_text_with_overlap(text, chunk_size=30, overlap=10)
    assert len(chunks) >= 3


def test_split_text_with_overlap_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError):
        split_text_with_overlap("abc", chunk_size=10, overlap=10)


def test_split_prefers_sentence_endings_when_they_fit() -> None:
    text = "First sentence has useful context. Second sentence continues the topic."
    chunks = split_text_with_overlap(text, chunk_size=42, overlap=0)
    assert chunks[0] == "First sentence has useful context."


def test_split_never_breaks_ordinary_words() -> None:
    text = "alpha extraordinaryword omega"
    chunks = split_text_with_overlap(text, chunk_size=12, overlap=0)
    assert "".join(chunks).replace(" ", "") == text.replace(" ", "")
    assert all(not chunk.endswith("extraordinary") for chunk in chunks)
    assert all(not chunk.startswith("word") for chunk in chunks)


def test_split_keeps_a_word_longer_than_the_target_intact() -> None:
    word = "extraordinaryword"
    chunks = split_text_with_overlap(word, chunk_size=5, overlap=0)
    assert chunks == [word]


def test_split_empty_text_returns_no_chunks() -> None:
    assert split_text_with_overlap("  \n\t ") == []


def test_split_with_zero_overlap_has_no_repeated_words() -> None:
    text = "one two three four five six seven eight"
    chunks = split_text_with_overlap(text, chunk_size=12, overlap=0)
    assert " ".join(chunks) == text


def test_split_handles_overlap_close_to_chunk_size() -> None:
    text = "one two three four five six seven eight nine ten"
    chunks = split_text_with_overlap(text, chunk_size=12, overlap=11)
    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)
    assert " ".join(chunks).replace("  ", " ").startswith("one two")


def test_offsets_extract_exact_original_unicode_text() -> None:
    text = "\nHeader.\n\nCafé has useful context. More text follows.  "
    spans = split_text_with_offsets(text, chunk_size=24, overlap=4)
    assert spans
    for chunk_text, start, end in spans:
        assert text[start:end] == chunk_text
        assert 0 <= start < end <= len(text)


def test_chunk_ids_do_not_depend_on_unrelated_documents() -> None:
    target = Document(
        doc_id="book-1",
        source_id="book-source",
        title="Sample Book",
        section="Chapter 1",
        text="First sentence. Second sentence contains detail.",
    )
    unrelated = Document("extra", "Extra", "Unrelated document text.")
    before = [chunk.chunk_id for chunk in build_chunks([target], chunk_size=24, overlap=0)]
    after = [
        chunk.chunk_id
        for chunk in build_chunks([unrelated, target], chunk_size=24, overlap=0)
        if chunk.doc_id == target.doc_id
    ]
    assert before == after


def test_repeated_passages_get_distinct_offset_ids() -> None:
    repeated_text = "Repeated sentence with details."
    document = Document("repeat", "Repeat", f"{repeated_text} {repeated_text}")
    chunks = build_chunks([document], chunk_size=100, overlap=0)
    assert len(chunks) == 1
    document = Document("repeat", "Repeat", f"{repeated_text} {repeated_text}")
    chunks = build_chunks([document], chunk_size=36, overlap=0)
    repeated = [chunk for chunk in chunks if chunk.text == repeated_text]
    assert len(repeated) == 2
    assert repeated[0].chunk_id != repeated[1].chunk_id


def test_source_span_helper_rejects_changed_source_text() -> None:
    document = Document("source", "Title", "A source sentence. Another sentence.")
    chunk = build_chunks([document], chunk_size=30, overlap=0)[0]
    assert extract_source_span(chunk, document.text) == chunk.text
    with pytest.raises(ValueError):
        extract_source_span(chunk, "changed source")
