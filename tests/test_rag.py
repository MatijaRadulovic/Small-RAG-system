from src.chunking import build_chunks
from src.corpus import load_default_corpus
from src.rag import build_grounded_answer
from src.retrieval import retrieve_top_k


def test_build_grounded_answer_returns_citations_for_supported_query() -> None:
    chunks = build_chunks(load_default_corpus(), chunk_size=120, overlap=20)
    question = "What is retrieval augmented generation?"
    retrieved = retrieve_top_k(question, chunks, k=2)
    result = build_grounded_answer(question, chunks, top_k=2)
    assert result.refused is False
    assert result.citations == [item.chunk.chunk_id for item in retrieved]


def test_supported_answer_separates_body_and_references() -> None:
    chunks = build_chunks(load_default_corpus())
    result = build_grounded_answer(
        "How do citations connect claims to retrieved chunks?",
        chunks,
        top_k=2,
    )
    assert result.answer.startswith("Answer:\n")
    assert "\n\nReferences:\n" in result.answer
    for citation in result.citations:
        assert result.answer.count(f"[{citation}]") == 2


def test_build_grounded_answer_refuses_unsupported_query() -> None:
    chunks = build_chunks(load_default_corpus(), chunk_size=120, overlap=20)
    result = build_grounded_answer("Who was emperor of Mars?", chunks, top_k=2)
    assert result.refused is True
    assert result.citations == []


def test_build_grounded_answer_refuses_weak_one_term_overlap() -> None:
    chunks = build_chunks(load_default_corpus())
    question = "What color is retrieval on Mars?"
    retrieved = retrieve_top_k(question, chunks, k=2)
    result = build_grounded_answer(question, chunks, top_k=2)

    assert retrieved
    assert result.refused is True
    assert result.citations == []
    assert "not have enough grounded evidence" in result.answer


def test_grounding_uses_query_term_coverage_not_raw_bm25_scale() -> None:
    chunks = build_chunks(load_default_corpus())
    result = build_grounded_answer(
        "What color is retrieval on Mars?", chunks, top_k=2
    )
    assert result.refused is True
    assert result.citations == []
