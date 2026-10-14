import pytest

from src.chunking import split_text_with_overlap


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
