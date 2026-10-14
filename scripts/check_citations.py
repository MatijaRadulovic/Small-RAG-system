"""Print and validate Outcome 3 citation evidence."""

from src.chunking import build_chunks
from src.corpus import load_default_corpus
from src.rag import build_grounded_answer
from src.retrieval import BM25Index, retrieve_top_k


QUERY = "How do citations connect claims to retrieved chunks?"


def main() -> None:
    chunks = build_chunks(load_default_corpus())
    index = BM25Index(chunks)
    retrieved = retrieve_top_k(QUERY, index, k=2)
    result = build_grounded_answer(QUERY, index, top_k=2)
    retrieved_ids = [item.chunk.chunk_id for item in retrieved]

    assert result.refused is False
    expected_citations = [
        assessment.chunk_id
        for assessment in result.evidence_assessments
        if assessment.sufficient
    ]
    assert result.citations == expected_citations
    assert "\n\nReferences:\n" in result.answer
    assert all(result.answer.count(f"[{chunk_id}]") == 2 for chunk_id in result.citations)

    print("OUTCOME 3: CITATION CHECK")
    print(f"Query: {QUERY}")
    print("Retrieved chunk IDs:", ", ".join(retrieved_ids))
    print(
        "Individually supported IDs:",
        ", ".join(result.citations),
    )
    for assessment in result.evidence_assessments:
        print(
            f"  {assessment.chunk_id}: "
            f"{len(assessment.matched_terms)} terms, "
            f"{assessment.term_coverage:.0%} coverage, "
            f"sufficient={assessment.sufficient} ({assessment.reason})"
        )
    print("\nAnswer with citations:")
    print(result.answer)
    print("\nPASS: answer citations exactly match individually supported retrieved chunks.")


if __name__ == "__main__":
    main()
