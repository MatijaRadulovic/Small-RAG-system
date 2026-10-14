import pytest

from src.chunking import Chunk
from src.retrieval import BM25Index, retrieve_top_k
from src.reranking import RerankingRetriever


class FakeReranker:
    model_name = "fake-cross-encoder"

    def score(self, query: str, passages: list[str]) -> list[float]:
        return [0.9 if "citation" in passage else 0.1 for passage in passages]


def test_reranker_changes_order_and_preserves_first_stage_scores() -> None:
    chunks = [
        Chunk(chunk_id="a", doc_id="a", title="A", text="retrieval finds general passages"),
        Chunk(chunk_id="b", doc_id="b", title="B", text="retrieval citation evidence supports claims"),
        Chunk(chunk_id="c", doc_id="c", title="C", text="retrieval searches a corpus"),
    ]
    first_stage = BM25Index(chunks)
    before = {item.chunk.chunk_id: item.score for item in retrieve_top_k("retrieval", first_stage, k=3)}
    retriever = RerankingRetriever(first_stage, FakeReranker(), candidate_k=3)

    results = retriever.retrieve_top_k("retrieval", k=3)

    assert results[0].chunk.chunk_id == "b"
    assert results[0].score == 0.9
    assert results[0].rerank_score == 0.9
    assert results[0].first_stage_score is not None
    assert results[0].first_stage_score == before["b"]
    assert all(item.rerank_score is not None for item in results)


def test_reranker_requires_candidate_count_to_match() -> None:
    class ShortReranker(FakeReranker):
        def score(self, query: str, passages: list[str]) -> list[float]:
            return []

    chunks = [Chunk(chunk_id="a", doc_id="a", title="A", text="retrieval")]
    retriever = RerankingRetriever(BM25Index(chunks), ShortReranker())
    with pytest.raises(ValueError, match="wrong number"):
        retriever.retrieve_top_k("retrieval", k=1)


def test_reranker_rejects_non_finite_scores() -> None:
    class InvalidReranker(FakeReranker):
        def score(self, query: str, passages: list[str]) -> list[float]:
            return [float("nan")] * len(passages)

    chunks = [Chunk(chunk_id="a", doc_id="a", title="A", text="retrieval")]
    retriever = RerankingRetriever(BM25Index(chunks), InvalidReranker())
    with pytest.raises(ValueError, match="finite"):
        retriever.retrieve_top_k("retrieval", k=1)


def test_reranker_validates_candidate_limit() -> None:
    chunks = [Chunk(chunk_id="a", doc_id="a", title="A", text="retrieval")]
    with pytest.raises(ValueError, match="candidate_k"):
        RerankingRetriever(BM25Index(chunks), FakeReranker(), candidate_k=0)
