from src.chunking import Chunk
from src.retrieval import bm25_scores, lexical_score, retrieve_top_k


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


def test_bm25_gives_rare_terms_more_weight() -> None:
    chunks = [
        Chunk(chunk_id="a", doc_id="1", title="A", text="common common raretoken"),
        Chunk(chunk_id="b", doc_id="2", title="B", text="common common common"),
        Chunk(chunk_id="c", doc_id="3", title="C", text="raretoken appears"),
    ]
    scores = bm25_scores("raretoken", chunks)
    assert scores[0] > scores[1]
    assert scores[2] > 0


def test_bm25_repeated_term_gain_saturates() -> None:
    chunks = [
        Chunk(chunk_id="a", doc_id="1", title="A", text="signal"),
        Chunk(chunk_id="b", doc_id="2", title="B", text="signal signal"),
        Chunk(chunk_id="c", doc_id="3", title="C", text="other words"),
    ]
    scores = bm25_scores("signal", chunks)
    assert scores[1] > scores[0]
    assert scores[1] < scores[0] * 2


def test_bm25_normalizes_for_chunk_length() -> None:
    chunks = [
        Chunk(chunk_id="a", doc_id="1", title="A", text="rareterm"),
        Chunk(
            chunk_id="b",
            doc_id="2",
            title="B",
            text="rareterm " + "filler " * 40,
        ),
        Chunk(chunk_id="c", doc_id="3", title="C", text="other content"),
    ]
    scores = bm25_scores("rareterm", chunks)
    assert scores[0] > scores[1]


def test_bm25_handles_empty_queries_and_corpora() -> None:
    chunk = Chunk(chunk_id="a", doc_id="1", title="A", text="some content")
    assert bm25_scores("the and", [chunk]) == [0.0]
    assert bm25_scores("query", []) == []


def test_bm25_returns_zero_for_unrelated_query() -> None:
    chunks = [
        Chunk(chunk_id="a", doc_id="1", title="A", text="retrieval evidence"),
        Chunk(chunk_id="b", doc_id="2", title="B", text="citation source"),
    ]
    assert bm25_scores("basketball playoffs", chunks) == [0.0, 0.0]
