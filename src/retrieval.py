from collections import Counter
from dataclasses import dataclass
from math import log, sqrt
import re
import unicodedata
from typing import Sequence

from .chunking import Chunk


TOKEN_RE = re.compile(r"[^\W_]+(?:[-./][^\W_]+)*", re.UNICODE)
BM25_K1 = 1.5
BM25_B = 0.75
TITLE_RRF_WEIGHT = 0.01
RRF_CONSTANT = 60
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
    content_score: float = 0.0
    title_score: float = 0.0
    matched_terms: tuple[str, ...] = ()
    redundancy: float = 0.0
    diversity_score: float = 1.0
    semantic_score: float | None = None
    lexical_rank: int | None = None
    semantic_rank: int | None = None


class _BM25Field:
    """BM25 statistics and scoring for one text field across all chunks."""

    def __init__(
        self,
        values: Sequence[str],
        *,
        language: str,
        k1: float,
        b: float,
    ) -> None:
        self.tokens = [tokenize(value, language=language) for value in values]
        self.term_frequencies = [Counter(tokens) for tokens in self.tokens]
        self.document_frequency: Counter[str] = Counter()
        for tokens in self.tokens:
            self.document_frequency.update(set(tokens))
        self.document_count = len(self.tokens)
        self.average_length = (
            sum(map(len, self.tokens)) / self.document_count
            if self.document_count
            else 0.0
        )
        self.k1 = k1
        self.b = b

    def score(self, query_terms: set[str]) -> list[float]:
        if not query_terms or not self.document_count or self.average_length == 0:
            return [0.0] * self.document_count

        scores: list[float] = []
        for tokens, term_frequency in zip(self.tokens, self.term_frequencies):
            length_normalization = self.k1 * (
                1 - self.b + self.b * len(tokens) / self.average_length
            )
            score = 0.0
            for term in query_terms:
                frequency = term_frequency[term]
                if frequency == 0:
                    continue
                frequency_in_chunks = self.document_frequency[term]
                inverse_document_frequency = log(
                    1
                    + (self.document_count - frequency_in_chunks + 0.5)
                    / (frequency_in_chunks + 0.5)
                )
                score += inverse_document_frequency * (
                    frequency * (self.k1 + 1) / (frequency + length_normalization)
                )
            scores.append(score)
        return scores


class BM25Index:
    """Reusable BM25 content/title statistics for repeated queries."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        *,
        language: str = "en",
        k1: float = BM25_K1,
        b: float = BM25_B,
    ) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be > 0")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")
        # Validate the language even when the chunk list is empty.
        tokenize("", language=language)

        self.chunks = tuple(chunks)
        self.language = language
        self.k1 = k1
        self.b = b
        self._content = _BM25Field(
            [chunk.text for chunk in self.chunks],
            language=language,
            k1=k1,
            b=b,
        )
        self._titles = _BM25Field(
            [chunk.title for chunk in self.chunks],
            language=language,
            k1=k1,
            b=b,
        )

    @property
    def vocabulary_size(self) -> int:
        return len(self._content.document_frequency)

    @property
    def indexed_token_count(self) -> int:
        return sum(map(len, self._content.tokens))

    @property
    def average_chunk_length(self) -> float:
        return self._content.average_length

    def score(self, query: str) -> tuple[list[float], list[float]]:
        query_terms = set(tokenize(query, language=self.language))
        return self._content.score(query_terms), self._titles.score(query_terms)


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
    """Compatibility helper for one-off content-only BM25 scoring."""
    index = BM25Index(chunks, language=language, k1=k1, b=b)
    return index.score(query)[0]


def _rank_scores(scores: list[float], chunks: Sequence[Chunk]) -> dict[int, int]:
    ranked = sorted(
        (index for index, score in enumerate(scores) if score > 0),
        key=lambda index: (-scores[index], chunks[index].chunk_id),
    )
    return {index: rank for rank, index in enumerate(ranked, start=1)}


def _token_overlap(left: set[str], right: set[str]) -> float:
    """Return Jaccard overlap, a simple lexical near-duplicate signal."""
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _validate_diversity_options(
    max_per_source: int | None,
    duplicate_threshold: float,
) -> None:
    if max_per_source is not None and max_per_source <= 0:
        raise ValueError("max_per_source must be > 0 or None")
    if not 0 <= duplicate_threshold <= 1:
        raise ValueError("duplicate_threshold must be between 0 and 1")


def _select_diverse_results(
    candidates: Sequence[RetrievalResult],
    *,
    k: int,
    language: str,
    max_per_source: int | None,
    duplicate_threshold: float,
) -> list[RetrievalResult]:
    _validate_diversity_options(max_per_source, duplicate_threshold)
    selected: list[RetrievalResult] = []
    source_counts: Counter[str] = Counter()
    selected_tokens: list[tuple[RetrievalResult, set[str]]] = []
    for item in candidates:
        source = item.chunk.source_id or item.chunk.doc_id
        if max_per_source is not None and source_counts[source] >= max_per_source:
            continue

        terms = set(tokenize(item.chunk.text, language=language))
        redundancy = 0.0
        duplicate = False
        normalized_text = " ".join(item.chunk.text.casefold().split())
        for existing, existing_terms in selected_tokens:
            existing_text = " ".join(existing.chunk.text.casefold().split())
            overlap = _token_overlap(terms, existing_terms)
            redundancy = max(redundancy, overlap)
            same_source = (existing.chunk.source_id or existing.chunk.doc_id) == source
            if normalized_text == existing_text or (
                same_source and overlap >= duplicate_threshold
            ):
                duplicate = True
                break
        if duplicate:
            continue

        selected_item = RetrievalResult(
            chunk=item.chunk,
            score=item.score,
            content_score=item.content_score,
            title_score=item.title_score,
            matched_terms=item.matched_terms,
            semantic_score=item.semantic_score,
            lexical_rank=item.lexical_rank,
            semantic_rank=item.semantic_rank,
            redundancy=redundancy,
            diversity_score=1.0 - redundancy,
        )
        selected.append(selected_item)
        selected_tokens.append((selected_item, terms))
        source_counts[source] += 1
        if len(selected) == k:
            break
    return selected


def retrieve_top_k(
    query: str,
    chunks: Sequence[Chunk] | BM25Index,
    k: int = 3,
    min_score: float = 0.0,
    language: str | None = None,
    title_weight: float = TITLE_RRF_WEIGHT,
    max_per_source: int | None = 2,
    duplicate_threshold: float = 0.85,
) -> list[RetrievalResult]:
    if k <= 0:
        raise ValueError("k must be > 0")
    if min_score < 0:
        raise ValueError("min_score must be >= 0")
    if not 0 <= title_weight <= 1:
        raise ValueError("title_weight must be between 0 and 1")
    _validate_diversity_options(max_per_source, duplicate_threshold)

    if isinstance(chunks, BM25Index):
        index = chunks
        if language is not None and language != index.language:
            raise ValueError("language must match the BM25Index language")
    else:
        index = BM25Index(chunks, language=language or "en")

    content_scores, title_scores = index.score(query)
    content_ranks = _rank_scores(content_scores, index.chunks)
    title_ranks = _rank_scores(title_scores, index.chunks)
    query_terms = set(tokenize(query, language=index.language))

    scored: list[RetrievalResult] = []
    for position, chunk in enumerate(index.chunks):
        content_rank = content_ranks.get(position)
        title_rank = title_ranks.get(position)
        if content_rank is None and title_rank is None:
            continue
        score = (1 / (RRF_CONSTANT + content_rank)) if content_rank else 0.0
        if title_rank:
            score += title_weight / (RRF_CONSTANT + title_rank)
        chunk_terms = set(tokenize(chunk.text, language=index.language))
        scored.append(
            RetrievalResult(
                chunk=chunk,
                score=score,
                content_score=content_scores[position],
                title_score=title_scores[position],
                matched_terms=tuple(sorted(query_terms & chunk_terms)),
            )
        )
    candidates = [item for item in scored if item.score >= min_score]
    candidates.sort(key=lambda item: (-item.score, item.chunk.chunk_id))

    return _select_diverse_results(
        candidates,
        k=k,
        language=index.language,
        max_per_source=max_per_source,
        duplicate_threshold=duplicate_threshold,
    )
