"""Optional cross-encoder reranking of a larger first-stage candidate set."""

from collections.abc import Sequence
from dataclasses import replace
from math import isfinite
from typing import Protocol

from .chunking import Chunk
from .hybrid_retrieval import HybridRetriever
from .retrieval import (
    BM25Index,
    RetrievalResult,
    _select_diverse_results,
    retrieve_top_k,
)


DEFAULT_RERANKER_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


class Reranker(Protocol):
    """Scores each query/passage pair jointly; scores are rank values, not probabilities."""

    model_name: str

    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


class FastEmbedReranker:
    """Local ONNX cross-encoder supplied by the optional FastEmbed runtime."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        *,
        cache_dir: str | None = None,
        local_files_only: bool = False,
        threads: int | None = None,
    ) -> None:
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
        except ImportError as error:
            raise RuntimeError(
                "Reranking needs the optional local runtime. "
                "Install it with `python3 -m pip install -r requirements-embeddings.txt`."
            ) from error

        self.model_name = model_name
        try:
            self._model = TextCrossEncoder(
                model_name=model_name,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                threads=threads,
            )
        except Exception as error:
            mode = "offline cache" if local_files_only else "local cache or model download"
            raise RuntimeError(
                f"Could not load reranker {model_name!r} from the {mode}. "
                "Check the model name, network access, and cache path."
            ) from error

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        return [float(score) for score in self._model.rerank(query, list(passages))]


class RerankingRetriever:
    """Rerank first-stage candidates, then apply duplicate/source diversity."""

    def __init__(
        self,
        first_stage: BM25Index | HybridRetriever,
        reranker: Reranker,
        *,
        candidate_k: int = 20,
    ) -> None:
        if candidate_k <= 0:
            raise ValueError("candidate_k must be > 0")
        self.first_stage = first_stage
        self.reranker = reranker
        self.candidate_k = candidate_k
        self.language = first_stage.language
        self.chunks: Sequence[Chunk] = first_stage.chunks

    def retrieve_top_k(
        self,
        query: str,
        k: int = 3,
        *,
        max_per_source: int | None = 2,
        duplicate_threshold: float = 0.85,
    ) -> list[RetrievalResult]:
        if k <= 0:
            raise ValueError("k must be > 0")
        if isinstance(self.first_stage, HybridRetriever):
            candidates = self.first_stage.retrieve_top_k(
                query,
                k=self.candidate_k,
                max_per_source=None,
                duplicate_threshold=1.0,
            )
        else:
            candidates = retrieve_top_k(
                query,
                self.first_stage,
                k=self.candidate_k,
                max_per_source=None,
                duplicate_threshold=1.0,
            )
        if not candidates:
            return []

        scores = self.reranker.score(query, [item.chunk.text for item in candidates])
        if len(scores) != len(candidates):
            raise ValueError("reranker returned the wrong number of scores")
        if not all(isfinite(score) for score in scores):
            raise ValueError("reranker scores must be finite")

        rescored = [
            replace(
                item,
                score=score,
                rerank_score=score,
                first_stage_score=item.score,
            )
            for item, score in zip(candidates, scores)
        ]
        rescored.sort(
            key=lambda item: (
                -item.score,
                -(item.first_stage_score or 0.0),
                item.chunk.chunk_id,
            )
        )
        return _select_diverse_results(
            rescored,
            k=k,
            language=self.language,
            max_per_source=max_per_source,
            duplicate_threshold=duplicate_threshold,
        )
