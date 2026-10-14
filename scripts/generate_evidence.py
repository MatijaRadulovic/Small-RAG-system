"""Generate and validate the final supported/unsupported evidence pair."""

from src.chunking import Chunk, build_chunks
from src.corpus import load_default_corpus
from src.rag import RagAnswer, build_grounded_answer
from src.retrieval import BM25Index, RetrievalResult, retrieve_top_k


SUPPORTED_QUERY = "How do citations connect claims to retrieved chunks?"
UNSUPPORTED_QUERY = "What color is retrieval on Mars?"


def print_case(
    label: str,
    question: str,
    retrieved: list[RetrievalResult],
    result: RagAnswer,
) -> None:
    print(f"{label} CASE")
    print(f"Question: {question}")
    print("Retrieved chunks:")
    assessments = {item.chunk_id: item for item in result.evidence_assessments}
    for retrieval in retrieved:
        chunk: Chunk = retrieval.chunk
        assessment = assessments.get(chunk.chunk_id)
        support = (
            f"supported={assessment.sufficient} "
            f"| coverage={assessment.term_coverage:.0%}"
            if assessment is not None
            else "evidence assessment unavailable"
        )
        print(f"- {chunk.chunk_id} | {chunk.title} | score={retrieval.score:.3f}")
        print(f"  Evidence: {support}")
        print(f"  {chunk.text}")
    print("Answer:")
    print(result.answer)
    print("Citations:", ", ".join(result.citations) if result.citations else "None")
    print("Refused:", result.refused)


def main() -> None:
    chunks = build_chunks(load_default_corpus())
    index = BM25Index(chunks)

    supported_retrieval = retrieve_top_k(SUPPORTED_QUERY, index, k=2)
    supported_result = build_grounded_answer(SUPPORTED_QUERY, index, top_k=2)
    supported_ids = [
        item.chunk_id for item in supported_result.evidence_assessments if item.sufficient
    ]
    assert supported_result.refused is False
    assert supported_result.citations == supported_ids
    assert set(supported_ids) <= {item.chunk.chunk_id for item in supported_retrieval}

    unsupported_retrieval = retrieve_top_k(UNSUPPORTED_QUERY, index, k=2)
    unsupported_result = build_grounded_answer(UNSUPPORTED_QUERY, index, top_k=2)
    assert unsupported_result.refused is True
    assert unsupported_result.citations == []

    print("OUTCOME 5: FINAL EVIDENCE PAIR\n")
    print_case(
        "SUPPORTED",
        SUPPORTED_QUERY,
        supported_retrieval,
        supported_result,
    )
    print("\n" + "=" * 72 + "\n")
    print_case(
        "UNSUPPORTED",
        UNSUPPORTED_QUERY,
        unsupported_retrieval,
        unsupported_result,
    )
    print("\nPASS: supported and unsupported evidence satisfy the homework checks.")


if __name__ == "__main__":
    main()
