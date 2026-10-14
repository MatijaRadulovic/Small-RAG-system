import pytest

from src.chunking import split_text_with_overlap


def test_split_text_with_overlap_returns_multiple_chunks() -> None:
    text = "abcdefghijklmnopqrstuvwxyz" * 4
    chunks = split_text_with_overlap(text, chunk_size=30, overlap=10)
    assert len(chunks) >= 3


def test_split_text_with_overlap_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError):
        split_text_with_overlap("abc", chunk_size=10, overlap=10)
