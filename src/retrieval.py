from dataclasses import dataclass
from math import sqrt
import re

from .chunking import Chunk


TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")
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
    q_tokens = set(tokenize(query))
    c_tokens = set(tokenize(chunk_text))
    if not q_tokens or not c_tokens:
        return 0.0

    overlap = len(q_tokens & c_tokens)
    return overlap / sqrt(len(q_tokens) * len(c_tokens))


def retrieve_top_k(
    query: str,
    chunks: list[Chunk],
    k: int = 3,
    min_score: float = 0.01,
) -> list[RetrievalResult]:
    if k <= 0:
        raise ValueError("k must be > 0")

    scored = []
    for chunk in chunks:
        text_score = lexical_score(query, chunk.text)
        title_score = lexical_score(query, chunk.title)
        score = (0.85 * text_score) + (0.15 * title_score)
        scored.append(RetrievalResult(chunk=chunk, score=score))
    filtered = [item for item in scored if item.score >= min_score]
    filtered.sort(key=lambda item: (-item.score, item.chunk.chunk_id))
    return filtered[:k]
