from dataclasses import dataclass
import re

from .corpus import Document


DEFAULT_CHUNK_SIZE = 180
DEFAULT_OVERLAP = 0
SENTENCE_END_RE = re.compile(r"[.!?](?:[\"')\]]+)?(?=\s|$)")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str


def _choose_chunk_end(
    text: str,
    start: int,
    hard_end: int,
    chunk_size: int,
    overlap: int,
) -> int:
    """Prefer a sentence boundary, then a word boundary, before hard_end."""
    if hard_end >= len(text):
        return len(text)

    minimum_length = max(chunk_size // 2, overlap + 1)
    sentence_ends = [
        start + match.end()
        for match in SENTENCE_END_RE.finditer(text[start:hard_end])
        if match.end() >= minimum_length
    ]
    if sentence_ends:
        return sentence_ends[-1]

    word_end = text.rfind(" ", start + 1, hard_end + 1)
    if word_end > start:
        return word_end

    # A single word may be longer than chunk_size. Keeping the word intact is
    # more useful for retrieval than cutting it into meaningless fragments.
    next_space = text.find(" ", hard_end)
    return len(text) if next_space == -1 else next_space


def _choose_next_start(text: str, previous_start: int, end: int, overlap: int) -> int:
    """Move back by about overlap characters and start at a word boundary."""
    if overlap == 0:
        next_start = end
    else:
        target = max(previous_start + 1, end - overlap)
        word_start = text.rfind(" ", previous_start + 1, target + 1)
        if word_start != -1:
            next_start = word_start + 1
        else:
            next_space = text.find(" ", target, end)
            next_start = next_space + 1 if next_space != -1 else end

    while next_start < len(text) and text[next_start] == " ":
        next_start += 1
    return next_start


def split_text_with_overlap(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    cleaned = " ".join(text.split())
    if not cleaned:
        return []

    chunks: list[str] = []
    start = 0

    while start < len(cleaned):
        hard_end = min(start + chunk_size, len(cleaned))
        end = _choose_chunk_end(
            cleaned,
            start=start,
            hard_end=hard_end,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(cleaned):
            break
        next_start = _choose_next_start(cleaned, start, end, overlap)
        start = next_start if next_start > start else end

    return chunks


def build_chunks(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in documents:
        parts = split_text_with_overlap(doc.text, chunk_size=chunk_size, overlap=overlap)
        for i, part in enumerate(parts):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}-chunk-{i}",
                    doc_id=doc.doc_id,
                    title=doc.title,
                    text=part,
                )
            )
    return chunks
