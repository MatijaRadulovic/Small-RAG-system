"""Dense cosine search and lexical/semantic rank fusion."""

from collections.abc import Sequence
from math import isfinite, sqrt
from pathlib import Path

from .chunking import Chunk
from .embeddings import Embedding, EmbeddingProvider
from .index_cache import load_vectors, store_vectors
from .retrieval import (
    BM25Index,
    RRF_CONSTANT,
    RetrievalResult,
    _select_diverse_results,
    retrieve_top_k as retrieve_lexical_top_k,
    tokenize,
)


def _cosine(left: Embedding, right: Embedding) -> float:
    if len(left) != len(right):
        raise ValueError("embedding dimensions do not match")
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        raise ValueError("cannot compare a zero embedding")
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


class SemanticIndex:
    """Embeddings and cosine-ranking for an immutable chunk collection."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        provider: EmbeddingProvider,
        *,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.chunks = tuple(chunks)
        self.provider = provider
        cached_vectors: list[Embedding] | None = None
        self.cache_status = "disabled"
        if cache_dir is not None:
            cached_vectors, self.cache_status = load_vectors(
                cache_dir, provider.model_name, self.chunks
            )
        vectors = cached_vectors or provider.embed_passages(
            [chunk.text for chunk in self.chunks]
        )
        if len(vectors) != len(self.chunks):
            raise ValueError("embedding provider returned the wrong number of passages")
        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) > 1:
            raise ValueError("passage embeddings have inconsistent dimensions")
        if any(
            not vector
            or not all(isfinite(value) for value in vector)
            or not any(value != 0 for value in vector)
            for vector in vectors
        ):
            raise ValueError("passage embeddings must be finite, non-empty, and non-zero")
        self._vectors = tuple(vectors)
        self.dimension = next(iter(dimensions), 0)
        if cache_dir is not None and cached_vectors is None:
            store_vectors(cache_dir, provider.model_name, self.chunks, self._vectors)
            self.cache_status = "rebuilt" if self.cache_status == "corrupt" else "stored"

    def rank(self, query: str) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            return []
        query_vector = self.provider.embed_query(query)
        scores = [
            (chunk, _cosine(query_vector, vector))
            for chunk, vector in zip(self.chunks, self._vectors)
        ]
        return sorted(scores, key=lambda item: (-item[1], item[0].chunk_id))

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        if k <= 0:
            raise ValueError("k must be > 0")
        return self.rank(query)[:k]


class HybridRetriever:
    """Reuse BM25 and semantic indexes, then fuse their rankings with RRF."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        provider: EmbeddingProvider,
        *,
        language: str = "en",
        semantic_weight: float = 1.0,
        rrf_constant: int = RRF_CONSTANT,
        embedding_cache_dir: str | Path | None = None,
    ) -> None:
        if not isfinite(semantic_weight) or semantic_weight < 0:
            raise ValueError("semantic_weight must be >= 0")
        if rrf_constant <= 0:
            raise ValueError("rrf_constant must be > 0")
        self.chunks = tuple(chunks)
        self.language = language
        self.semantic_weight = semantic_weight
        self.rrf_constant = rrf_constant
        self.lexical_index = BM25Index(self.chunks, language=language)
        self.semantic_index = SemanticIndex(
            self.chunks, provider, cache_dir=embedding_cache_dir
        )

    def retrieve_top_k(
        self,
        query: str,
        k: int = 3,
        *,
        max_per_source: int | None = 2,
        duplicate_threshold: float = 0.85,
        min_score: float = 0.0,
    ) -> list[RetrievalResult]:
        if k <= 0:
            raise ValueError("k must be > 0")
        if min_score < 0:
            raise ValueError("min_score must be >= 0")

        lexical_results = retrieve_lexical_top_k(
            query,
            self.lexical_index,
            k=max(len(self.chunks), 1),
            language=self.language,
            max_per_source=None,
            duplicate_threshold=1.0,
        )
        lexical_by_id = {item.chunk.chunk_id: item for item in lexical_results}
        lexical_ranks = {
            item.chunk.chunk_id: rank
            for rank, item in enumerate(lexical_results, start=1)
        }
        semantic_results = self.semantic_index.rank(query)
        semantic_ranks = {
            chunk.chunk_id: rank
            for rank, (chunk, _) in enumerate(semantic_results, start=1)
        }
        semantic_scores = {
            chunk.chunk_id: score for chunk, score in semantic_results
        }
        query_terms = set(tokenize(query, language=self.language))

        fused: list[RetrievalResult] = []
        for chunk in self.chunks:
            lexical_rank = lexical_ranks.get(chunk.chunk_id)
            semantic_rank = semantic_ranks.get(chunk.chunk_id)
            lexical_part = (
                1 / (self.rrf_constant + lexical_rank) if lexical_rank else 0.0
            )
            semantic_part = (
                self.semantic_weight / (self.rrf_constant + semantic_rank)
                if semantic_rank
                else 0.0
            )
            score = lexical_part + semantic_part
            if score < min_score or score == 0:
                continue
            lexical_result = lexical_by_id.get(chunk.chunk_id)
            chunk_terms = set(tokenize(chunk.text, language=self.language))
            fused.append(
                RetrievalResult(
                    chunk=chunk,
                    score=score,
                    content_score=lexical_result.content_score if lexical_result else 0.0,
                    title_score=lexical_result.title_score if lexical_result else 0.0,
                    matched_terms=tuple(sorted(query_terms & chunk_terms)),
                    semantic_score=semantic_scores.get(chunk.chunk_id),
                    lexical_rank=lexical_rank,
                    semantic_rank=semantic_rank,
                )
            )
        fused.sort(key=lambda item: (-item.score, item.chunk.chunk_id))
        return _select_diverse_results(
            fused,
            k=k,
            language=self.language,
            max_per_source=max_per_source,
            duplicate_threshold=duplicate_threshold,
        )
