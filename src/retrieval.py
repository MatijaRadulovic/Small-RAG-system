from collections import Counter
from dataclasses import dataclass
from math import log, sqrt
import re
import unicodedata

from .chunking import Chunk


TOKEN_RE = re.compile(r"[^\W_]+(?:[-./][^\W_]+)*", re.UNICODE)
BM25_K1 = 1.5
BM25_B = 0.75
TITLE_BOOST = 0.25
ENGLISH_STOP_WORDS = {
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
SERBIAN_STOP_WORDS = {
    "a",
    "ali",
    "ako",
    "bi",
    "bila",
    "bile",
    "bili",
    "bio",
    "bismo",
    "biste",
    "biti",
    "da",
    "do",
    "dok",
    "ga",
    "ih",
    "i",
    "iz",
    "ja",
    "je",
    "još",
    "kao",
    "kada",
    "koja",
    "koje",
    "koji",
    "kroz",
    "li",
    "me",
    "mi",
    "moj",
    "na",
    "nam",
    "nas",
    "ne",
    "nego",
    "njega",
    "njih",
    "njihov",
    "nju",
    "o",
    "od",
    "on",
    "ona",
    "oni",
    "ono",
    "pa",
    "po",
    "pod",
    "pri",
    "sa",
    "sam",
    "se",
    "si",
    "su",
    "sve",
    "sta",
    "šta",
    "ta",
    "taj",
    "te",
    "ti",
    "to",
    "u",
    "uz",
    "već",
    "vi",
    "za",
    "što",
}
SERBIAN_CYRILLIC_STOP_WORDS = {
    "а",
    "али",
    "ако",
    "би",
    "била",
    "биле",
    "били",
    "био",
    "бити",
    "да",
    "до",
    "док",
    "и",
    "из",
    "је",
    "како",
    "као",
    "када",
    "која",
    "које",
    "који",
    "кроз",
    "ли",
    "ми",
    "на",
    "не",
    "него",
    "њега",
    "њих",
    "њихов",
    "њу",
    "о",
    "од",
    "он",
    "она",
    "они",
    "оно",
    "па",
    "по",
    "под",
    "при",
    "са",
    "сам",
    "се",
    "су",
    "све",
    "та",
    "тај",
    "те",
    "ти",
    "то",
    "у",
    "уз",
    "већ",
    "ви",
    "за",
    "што",
    "шта",
}
STOP_WORDS_BY_LANGUAGE = {
    "en": ENGLISH_STOP_WORDS,
    "sr": SERBIAN_STOP_WORDS | SERBIAN_CYRILLIC_STOP_WORDS,
    "mixed": ENGLISH_STOP_WORDS | SERBIAN_STOP_WORDS | SERBIAN_CYRILLIC_STOP_WORDS,
}


def tokenize(text: str, language: str = "en") -> list[str]:
    """Normalize text and return Unicode word tokens for the selected language."""
    if language not in STOP_WORDS_BY_LANGUAGE:
        raise ValueError("language must be 'en', 'sr', or 'mixed'")

    normalized = unicodedata.normalize("NFKC", text).casefold()
    stop_words = STOP_WORDS_BY_LANGUAGE[language]
    return [
        token
        for token in TOKEN_RE.findall(normalized)
        if token not in stop_words
    ]


@dataclass(frozen=True)
class RetrievalResult:
    chunk: Chunk
    score: float


def lexical_score(query: str, chunk_text: str, language: str = "en") -> float:
    """Return binary cosine similarity for small secondary signals."""
    q_tokens = set(tokenize(query, language=language))
    c_tokens = set(tokenize(chunk_text, language=language))
    if not q_tokens or not c_tokens:
        return 0.0

    overlap = len(q_tokens & c_tokens)
    return overlap / sqrt(len(q_tokens) * len(c_tokens))


def bm25_scores(
    query: str,
    chunks: list[Chunk],
    k1: float = BM25_K1,
    b: float = BM25_B,
    language: str = "en",
) -> list[float]:
    """Score every chunk with BM25 using the supplied chunks as the index."""
    query_terms = set(tokenize(query, language=language))
    tokenized_chunks = [tokenize(chunk.text, language=language) for chunk in chunks]
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
    language: str = "en",
) -> list[RetrievalResult]:
    if k <= 0:
        raise ValueError("k must be > 0")

    scored = []
    content_scores = bm25_scores(query, chunks, language=language)
    for chunk, content_score in zip(chunks, content_scores):
        title_score = lexical_score(query, chunk.title, language=language)
        score = content_score + (TITLE_BOOST * title_score)
        scored.append(RetrievalResult(chunk=chunk, score=score))
    filtered = [item for item in scored if item.score >= min_score]
    filtered.sort(key=lambda item: (-item.score, item.chunk.chunk_id))
    return filtered[:k]
