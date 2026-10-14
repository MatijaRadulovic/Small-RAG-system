from dataclasses import dataclass
import re

from .corpus import Document


DEFAULT_CHUNK_SIZE = 220
DEFAULT_OVERLAP = 0
SENTENCE_END_RE = re.compile(r"[.!?](?:[\"')\]]+)?(?=\s|$)")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    source_id: str = ""
    section: str | None = None
    start_offset: int = 0
    end_offset: int = 0


def _last_whitespace(text: str, start: int, end: int) -> int:
    return max((text.rfind(char, start, end) for char in " \t\n\r\f\v"), default=-1)


def _next_whitespace(text: str, start: int, end: int) -> int:
    positions = [
        position
        for char in " \t\n\r\f\v"
        if (position := text.find(char, start, end)) != -1
    ]
    return min(positions, default=-1)


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

    word_end = _last_whitespace(text, start + 1, hard_end + 1)
    if word_end > start:
        return word_end

    # A single word may be longer than chunk_size. Keeping the word intact is
    # more useful for retrieval than cutting it into meaningless fragments.
    next_space = _next_whitespace(text, hard_end, len(text))
    return len(text) if next_space == -1 else next_space


def _choose_next_start(text: str, previous_start: int, end: int, overlap: int) -> int:
    """Move back by about overlap characters and start at a word boundary."""
    if overlap == 0:
        next_start = end
    else:
        target = max(previous_start + 1, end - overlap)
        word_start = _last_whitespace(text, previous_start + 1, target + 1)
        if word_start != -1:
            next_start = word_start + 1
        else:
            next_space = _next_whitespace(text, target, end)
            next_start = next_space + 1 if next_space != -1 else end

    while next_start < len(text) and text[next_start].isspace():
        next_start += 1
    return next_start


def split_text_with_offsets(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[tuple[str, int, int]]:
    """Return chunks with exact start and end offsets into the original text."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    if not text.strip():
        return []

    spans: list[tuple[str, int, int]] = []
    start = 0
    while start < len(text):
        while start < len(text) and text[start].isspace():
            start += 1
        if start >= len(text):
            break

        hard_end = min(start + chunk_size, len(text))
        end = _choose_chunk_end(
            text,
            start=start,
            hard_end=hard_end,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        segment = text[start:end]
        leading = len(segment) - len(segment.lstrip())
        trailing_end = len(segment.rstrip())
        source_start = start + leading
        source_end = start + trailing_end
        if source_end > source_start:
            spans.append((text[source_start:source_end], source_start, source_end))
        if end >= len(text):
            break

        next_start = _choose_next_start(text, start, end, overlap)
        start = next_start if next_start > start else end
    return spans


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

    return [
        part
        for part, _start, _end in split_text_with_offsets(
            text,
            chunk_size=chunk_size,
            overlap=overlap,
        )
    ]


def build_chunks(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in documents:
        parts = split_text_with_offsets(doc.text, chunk_size=chunk_size, overlap=overlap)
        source_id = doc.source_id or doc.doc_id
        section = doc.section or doc.title
        for part, start_offset, end_offset in parts:
            chunk_id = f"{source_id}@{start_offset}:{end_offset}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc.doc_id,
                    title=doc.title,
                    text=part,
                    source_id=source_id,
                    section=section,
                    start_offset=start_offset,
                    end_offset=end_offset,
                )
            )
    return chunks


def extract_source_span(chunk: Chunk, source_text: str) -> str:
    """Return the exact original text cited by a chunk, validating its ID span."""
    if not 0 <= chunk.start_offset <= chunk.end_offset <= len(source_text):
        raise ValueError("chunk offsets are outside the source text")
    source_span = source_text[chunk.start_offset:chunk.end_offset]
    if source_span != chunk.text:
        raise ValueError("chunk text does not match the cited source span")
    return source_span
