from collections import Counter
from dataclasses import dataclass
from math import log, sqrt
import re

from .chunking import Chunk


TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")
BM25_K1 = 1.5
BM25_B = 0.75
TITLE_BOOST = 0.25
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "does",
    "for",
    "from",
    "how",
    "if",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "should",
    "that",
    "the",
    "to",
    "was",
    "what",
    "when",
    "who",
    "why",
    "with",
}


def tokenize(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if token.lower() not in STOP_WORDS
    ]


@dataclass(frozen=True)
class RetrievalResult:
    chunk: Chunk
    score: float


def lexical_score(query: str, chunk_text: str) -> float:
    """Return binary cosine similarity for small secondary signals."""
    q_tokens = set(tokenize(query))
    c_tokens = set(tokenize(chunk_text))
    if not q_tokens or not c_tokens:
        return 0.0

    overlap = len(q_tokens & c_tokens)
    return overlap / sqrt(len(q_tokens) * len(c_tokens))


def bm25_scores(
    query: str,
    chunks: list[Chunk],
    k1: float = BM25_K1,
    b: float = BM25_B,
) -> list[float]:
    """Score every chunk with BM25 using the supplied chunks as the index."""
    query_terms = set(tokenize(query))
    tokenized_chunks = [tokenize(chunk.text) for chunk in chunks]
    if not query_terms or not tokenized_chunks:
        return [0.0] * len(chunks)

    document_frequency: Counter[str] = Counter()
    for tokens in tokenized_chunks:
        document_frequency.update(set(tokens))

    chunk_count = len(tokenized_chunks)
    average_length = sum(len(tokens) for tokens in tokenized_chunks) / chunk_count
    if average_length == 0:
        return [0.0] * len(chunks)

    scores: list[float] = []
    for tokens in tokenized_chunks:
        term_frequency = Counter(tokens)
        chunk_length = len(tokens)
        score = 0.0
        for term in query_terms:
            frequency = term_frequency[term]
            if frequency == 0:
                continue

            frequency_in_chunks = document_frequency[term]
            inverse_document_frequency = log(
                1 + (chunk_count - frequency_in_chunks + 0.5) / (frequency_in_chunks + 0.5)
            )
            length_normalization = k1 * (
                1 - b + b * chunk_length / average_length
            )
            score += inverse_document_frequency * (
                frequency * (k1 + 1) / (frequency + length_normalization)
            )
        scores.append(score)

    return scores


def retrieve_top_k(
    query: str,
    chunks: list[Chunk],
    k: int = 3,
    min_score: float = 0.01,
) -> list[RetrievalResult]:
    if k <= 0:
        raise ValueError("k must be > 0")

    scored = []
    content_scores = bm25_scores(query, chunks)
    for chunk, content_score in zip(chunks, content_scores):
        title_score = lexical_score(query, chunk.title)
        score = content_score + (TITLE_BOOST * title_score)
        scored.append(RetrievalResult(chunk=chunk, score=score))
    filtered = [item for item in scored if item.score >= min_score]
    filtered.sort(key=lambda item: (-item.score, item.chunk.chunk_id))
    return filtered[:k]
