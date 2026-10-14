"""Validated settings shared by the local RAG pipeline."""

from dataclasses import dataclass
from math import isfinite

from .chunking import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP, Chunk
from .embeddings import DEFAULT_EMBEDDING_MODEL
from .reranking import DEFAULT_RERANKER_MODEL


@dataclass(frozen=True)
class RagConfig:
    chunk_size: int = DEFAULT_CHUNK_SIZE
    overlap: int = DEFAULT_OVERLAP
    top_k: int = 2
    retrieval_mode: str = "lexical"
    language: str = "en"
    max_per_source: int | None = 2
    min_evidence_matches: int = 2
    min_evidence_coverage: float = 0.5
    semantic_weight: float = 1.0
    rrf_constant: int = 60
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    rerank: bool = False
    candidate_k: int = 20
    reranker_model: str = DEFAULT_RERANKER_MODEL
    model_cache_dir: str | None = None
    index_cache_dir: str | None = None
    offline_models: bool = False

    def validate(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if self.overlap < 0 or self.overlap >= self.chunk_size:
            raise ValueError("overlap must be >= 0 and smaller than chunk_size")
        if self.top_k <= 0:
            raise ValueError("top_k must be > 0")
        if self.retrieval_mode not in {"lexical", "hybrid"}:
            raise ValueError("retrieval_mode must be 'lexical' or 'hybrid'")
        if self.language not in {"en", "sr", "mixed"}:
            raise ValueError("language must be 'en', 'sr', or 'mixed'")
        if self.max_per_source is not None and self.max_per_source <= 0:
            raise ValueError("max_per_source must be > 0 or None")
        if self.min_evidence_matches <= 0:
            raise ValueError("min_evidence_matches must be > 0")
        if not isfinite(self.min_evidence_coverage) or not 0 <= self.min_evidence_coverage <= 1:
            raise ValueError("min_evidence_coverage must be between 0 and 1")
        if not isfinite(self.semantic_weight) or self.semantic_weight < 0:
            raise ValueError("semantic_weight must be finite and >= 0")
        if self.rrf_constant <= 0:
            raise ValueError("rrf_constant must be > 0")
        if self.candidate_k <= 0:
            raise ValueError("candidate_k must be > 0")
        if not self.embedding_model.strip() or not self.reranker_model.strip():
            raise ValueError("model names must not be empty")

    def build_retriever(self, chunks: list[Chunk]):
        """Build the configured retriever, loading optional runtimes only if needed."""
        self.validate()
        from .hybrid_retrieval import HybridRetriever
        from .retrieval import BM25Index

        if self.retrieval_mode == "hybrid":
            from .embeddings import FastEmbedProvider

            provider = FastEmbedProvider(
                self.embedding_model,
                cache_dir=self.model_cache_dir,
                local_files_only=self.offline_models,
            )
            index = HybridRetriever(
                chunks,
                provider,
                language=self.language,
                semantic_weight=self.semantic_weight,
                rrf_constant=self.rrf_constant,
                embedding_cache_dir=self.index_cache_dir,
            )
        else:
            index = BM25Index(chunks, language=self.language)

        if self.rerank:
            from .reranking import FastEmbedReranker, RerankingRetriever

            reranker = FastEmbedReranker(
                self.reranker_model,
                cache_dir=self.model_cache_dir,
                local_files_only=self.offline_models,
            )
            index = RerankingRetriever(index, reranker, candidate_k=self.candidate_k)
        return index
