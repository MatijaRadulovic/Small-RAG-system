from src.chunking import Chunk
from src.retrieval import BM25Index, bm25_scores, lexical_score, retrieve_top_k, tokenize


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


def test_tokenize_normalizes_unicode_and_casefolds() -> None:
    assert tokenize("Cafe\u0301 STRASSE") == ["café", "strasse"]


def test_tokenize_supports_serbian_latin_and_cyrillic_stop_words() -> None:
    assert tokenize("Šta su dokumenti?", language="sr") == ["dokumenti"]
    assert tokenize("Како су документи?", language="sr") == ["документи"]


def test_tokenize_preserves_versioned_identifiers() -> None:
    assert tokenize("Use RAG-v2.1 now", language="en") == ["use", "rag-v2.1", "now"]


def test_tokenize_rejects_unknown_language() -> None:
    import pytest

    with pytest.raises(ValueError):
        tokenize("hello", language="xx")


def test_retrieval_supports_serbian_tokens() -> None:
    chunks = [
        Chunk(
            chunk_id="sr-1",
            doc_id="sr",
            title="Документација",
            text="Документ садржи релевантне податке.",
        ),
        Chunk(
            chunk_id="en-1",
            doc_id="en",
            title="Weather",
            text="Rain and wind are expected tomorrow.",
        ),
    ]
    results = retrieve_top_k(
        "Који документ садржи релевантне податке?",
        chunks,
        k=1,
        language="sr",
    )
    assert results[0].chunk.chunk_id == "sr-1"


def test_bm25_index_can_be_reused_for_multiple_queries() -> None:
    chunks = [
        Chunk(chunk_id="a", doc_id="1", title="RAG", text="retrieval evidence"),
        Chunk(chunk_id="b", doc_id="2", title="Other", text="citation quality"),
    ]
    index = BM25Index(chunks)
    first = retrieve_top_k("retrieval", index, k=1)
    second = retrieve_top_k("citation", index, k=1)
    assert first[0].chunk.chunk_id == "a"
    assert second[0].chunk.chunk_id == "b"
    assert index.vocabulary_size > 0
    assert index.indexed_token_count == 4
    assert index.average_chunk_length == 2


def test_bm25_index_validates_parameters() -> None:
    import pytest

    chunks = [Chunk(chunk_id="a", doc_id="1", title="A", text="text")]
    with pytest.raises(ValueError):
        BM25Index(chunks, k1=0)
    with pytest.raises(ValueError):
        BM25Index(chunks, b=1.1)
    with pytest.raises(ValueError):
        retrieve_top_k("query", chunks, title_weight=1.1)
    with pytest.raises(ValueError):
        retrieve_top_k("query", chunks, min_score=-1)
    with pytest.raises(ValueError):
        retrieve_top_k("query", chunks, k=0)


def test_rank_fusion_keeps_component_scores_separate() -> None:
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
            title="Other Guide",
            text="Passages preserve surrounding context.",
        ),
    ]
    results = retrieve_top_k("chunking context", chunks, k=2)
    assert results[0].chunk.chunk_id == "a"
    assert results[0].content_score > 0
    assert results[0].title_score > 0
    assert results[0].score < 0.03


def test_retrieval_collapses_exact_duplicate_text() -> None:
    chunks = [
        Chunk(chunk_id="a1", doc_id="a", title="A", text="retrieval citation evidence"),
        Chunk(chunk_id="a2", doc_id="a", title="A", text="retrieval citation evidence"),
        Chunk(chunk_id="b1", doc_id="b", title="B", text="retrieval citation source"),
    ]

    results = retrieve_top_k("retrieval citation evidence", chunks, k=3)

    assert len(results) == 2
    assert len({item.chunk.text for item in results}) == 2


def test_retrieval_skips_near_duplicate_adjacent_chunks() -> None:
    chunks = [
        Chunk(
            chunk_id="a1",
            doc_id="a",
            title="A",
            text="retrieval citation evidence source traceability metadata",
        ),
        Chunk(
            chunk_id="a2",
            doc_id="a",
            title="A",
            text="retrieval citation evidence source traceability metadata notes",
        ),
        Chunk(chunk_id="b1", doc_id="b", title="B", text="retrieval citation evidence quality"),
    ]

    results = retrieve_top_k("retrieval citation evidence", chunks, k=3)

    assert "a1" in [item.chunk.chunk_id for item in results]
    assert "a2" not in [item.chunk.chunk_id for item in results]
    assert all(item.diversity_score == 1 - item.redundancy for item in results)


def test_retrieval_limits_source_repetition_but_keeps_distinct_facts() -> None:
    chunks = [
        Chunk(chunk_id="a1", doc_id="a", title="A", text="retrieval ranks relevant passages"),
        Chunk(chunk_id="a2", doc_id="a", title="A", text="retrieval cites source passages"),
        Chunk(chunk_id="a3", doc_id="a", title="A", text="retrieval filters weak passages"),
    ]

    results = retrieve_top_k("retrieval passages", chunks, k=3, max_per_source=2)

    assert len(results) == 2
    assert [item.chunk.chunk_id for item in results] == ["a1", "a2"]


def test_retrieval_diversity_options_are_validated() -> None:
    import pytest

    chunk = Chunk(chunk_id="a", doc_id="a", title="A", text="retrieval evidence")
    with pytest.raises(ValueError):
        retrieve_top_k("retrieval", [chunk], max_per_source=0)
    with pytest.raises(ValueError):
        retrieve_top_k("retrieval", [chunk], duplicate_threshold=1.1)
