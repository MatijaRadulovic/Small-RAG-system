"""Structured query traces that omit question and passage text."""

from dataclasses import asdict, dataclass, field
from uuid import uuid4

from .retrieval import RetrievalResult


@dataclass(frozen=True)
class ChunkTrace:
    chunk_id: str
    score: float
    content_score: float
    title_score: float
    semantic_score: float | None
    rerank_score: float | None
    first_stage_score: float | None
    matched_term_count: int
    redundancy: float
    diversity_score: float

    @classmethod
    def from_result(cls, result: RetrievalResult) -> "ChunkTrace":
        return cls(
            chunk_id=result.chunk.chunk_id,
            score=result.score,
            content_score=result.content_score,
            title_score=result.title_score,
            semantic_score=result.semantic_score,
            rerank_score=result.rerank_score,
            first_stage_score=result.first_stage_score,
            matched_term_count=len(result.matched_terms),
            redundancy=result.redundancy,
            diversity_score=result.diversity_score,
        )


@dataclass
class RequestTrace:
    query_id: str = field(default_factory=lambda: str(uuid4()))
    retrieval_mode: str = "lexical"
    index_cache_status: str | None = None
    selected_chunks: list[ChunkTrace] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    refusal_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialize a trace without including user query or source passage text."""
        return asdict(self)
