from src.chunking import Chunk
from src.retrieval import lexical_score, retrieve_top_k


def test_retrieve_top_k_orders_by_score_desc() -> None:
    chunks = [
        Chunk(chunk_id="a", doc_id="1", title="A", text="rag retrieval citations evidence"),
        Chunk(chunk_id="b", doc_id="1", title="B", text="weather forecast rain tomorrow"),
    ]
    results = retrieve_top_k("rag citations", chunks, k=2)
    assert results
    assert results[0].chunk.chunk_id == "a"


def test_retrieve_top_k_returns_empty_for_unrelated_query() -> None:
    chunks = [
        Chunk(chunk_id="a", doc_id="1", title="A", text="vector database retrieval"),
    ]
    results = retrieve_top_k("basketball playoffs", chunks, k=2)
    assert results == []


def test_lexical_score_rewards_more_shared_meaningful_terms() -> None:
    strong = lexical_score("chunk overlap retrieval", "chunk overlap improves retrieval")
    weak = lexical_score("chunk overlap retrieval", "chunk storage format")
    assert strong > weak > 0


def test_retrieve_top_k_respects_limit_and_descending_scores() -> None:
    chunks = [
        Chunk(chunk_id="a", doc_id="1", title="A", text="rag retrieval citations evidence"),
        Chunk(chunk_id="b", doc_id="2", title="B", text="retrieval evidence"),
        Chunk(chunk_id="c", doc_id="3", title="C", text="retrieval"),
    ]
    results = retrieve_top_k("rag retrieval citations", chunks, k=2)
    assert len(results) == 2
    assert results[0].score >= results[1].score


def test_retrieve_top_k_uses_title_as_a_small_relevance_signal() -> None:
    chunks = [
        Chunk(
            chunk_id="a",
            doc_id="1",
            title="Chunking Guide",
            text="Passages preserve surrounding context.",
        ),
        Chunk(
            chunk_id="b",
            doc_id="2",
            title="Unrelated Guide",
            text="Passages preserve surrounding context.",
        ),
    ]
    results = retrieve_top_k("chunking context", chunks, k=2)
    assert results[0].chunk.chunk_id == "a"
