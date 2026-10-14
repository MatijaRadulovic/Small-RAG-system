from src.chunking import Chunk, build_chunks
from src.corpus import load_default_corpus
from src.rag import assess_evidence, build_grounded_answer
from src.retrieval import RetrievalResult, retrieve_top_k


def test_build_grounded_answer_returns_citations_for_supported_query() -> None:
    chunks = build_chunks(load_default_corpus(), chunk_size=120, overlap=20)
    question = "What is retrieval augmented generation?"
    retrieved = retrieve_top_k(question, chunks, k=2)
    result = build_grounded_answer(question, chunks, top_k=2)
    assert result.refused is False
    supported_ids = [
        item.chunk_id for item in result.evidence_assessments if item.sufficient
    ]
    assert result.citations == supported_ids
    assert set(result.citations) <= {item.chunk.chunk_id for item in retrieved}


def test_supported_answer_separates_body_and_references() -> None:
    chunks = build_chunks(load_default_corpus())
    result = build_grounded_answer(
        "How do citations connect claims to retrieved chunks?",
        chunks,
        top_k=2,
    )
    assert result.answer.startswith(
        "Answer (based only on the individually supported passages):\n"
    )
    assert "\n\nReferences:\n" in result.answer
    assert len(result.evidence_assessments) == 2
    assert sum(item.sufficient for item in result.evidence_assessments) == 1
    for citation in result.citations:
        assert result.answer.count(f"[{citation}]") == 2
    assert "source characters" in result.answer


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


def test_only_individually_supported_results_are_answered_and_cited() -> None:
    chunks = build_chunks(load_default_corpus())
    question = "How do citations connect claims to retrieved chunks?"
    result = build_grounded_answer(question, chunks, top_k=3)

    assert result.refused is False
    assert [item.sufficient for item in result.evidence_assessments] == [True, False, False]
    assert result.citations == [result.evidence_assessments[0].chunk_id]
    assert result.evidence_assessments[1].chunk_id not in result.answer
    assert "individually supported passages" in result.answer


def test_title_match_alone_is_not_sufficient_evidence() -> None:
    chunk = Chunk(
        chunk_id="title-only@0:20",
        doc_id="title-only",
        title="retrieval Mars color",
        text="This passage discusses unrelated gardening advice.",
    )
    result = RetrievalResult(chunk=chunk, score=0.9)

    assessment = assess_evidence("What color is retrieval on Mars?", result)

    assert assessment.sufficient is False
    assert assessment.matched_terms == ()
    assert assessment.term_coverage == 0.0
    assert "requires at least" in assessment.reason


def test_assessment_records_score_matches_coverage_and_reason() -> None:
    chunk = Chunk(
        chunk_id="evidence@0:40",
        doc_id="evidence",
        title="Evidence",
        text="Citations connect claims to retrieved chunks.",
    )
    result = RetrievalResult(chunk=chunk, score=0.42)

    assessment = assess_evidence("How do citations connect claims?", result)

    assert assessment.score == 0.42
    assert assessment.matched_terms == ("citations", "claims", "connect")
    assert assessment.term_coverage == 1.0
    assert assessment.sufficient is True
    assert "matched 3 of 3" in assessment.reason
