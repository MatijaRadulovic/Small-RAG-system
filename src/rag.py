from dataclasses import dataclass

from .chunking import Chunk
from .retrieval import BM25Index, RetrievalResult, retrieve_top_k, tokenize


REFUSAL_MESSAGE = "I do not have enough grounded evidence in the provided corpus to answer that question."
MIN_MATCHED_TERMS = 2
MIN_QUERY_TERM_COVERAGE = 0.5


@dataclass(frozen=True)
class RagAnswer:
    answer: str
    citations: list[str]
    refused: bool


def has_sufficient_evidence(
    question: str,
    result: RetrievalResult,
    language: str = "en",
) -> bool:
    query_terms = set(tokenize(question, language=language))
    evidence_terms = set(
        tokenize(f"{result.chunk.title} {result.chunk.text}", language=language)
    )
    matched_terms = len(query_terms & evidence_terms)
    required_matches = 1 if len(query_terms) == 1 else MIN_MATCHED_TERMS
    term_coverage = matched_terms / len(query_terms) if query_terms else 0.0
    return (
        matched_terms >= required_matches
        and term_coverage >= MIN_QUERY_TERM_COVERAGE
    )


def build_grounded_answer(
    question: str,
    chunks: list[Chunk] | BM25Index,
    top_k: int = 2,
    language: str | None = None,
) -> RagAnswer:
    effective_language = language or (chunks.language if isinstance(chunks, BM25Index) else "en")
    results = retrieve_top_k(
        question,
        chunks,
        k=top_k,
        language=language,
    )
    if not results:
        return RagAnswer(answer=REFUSAL_MESSAGE, citations=[], refused=True)

    best = results[0]
    if not has_sufficient_evidence(question, best, language=effective_language):
        return RagAnswer(answer=REFUSAL_MESSAGE, citations=[], refused=True)

    supporting_points: list[str] = []
    references: list[str] = []
    citations: list[str] = []
    for item in results:
        chunk_id = item.chunk.chunk_id
        citations.append(chunk_id)
        supporting_points.append(f"- {item.chunk.text} [{chunk_id}]")
        references.append(f"- [{chunk_id}] {item.chunk.title}")

    answer = (
        "Answer:\n"
        + "\n".join(supporting_points)
        + "\n\nReferences:\n"
        + "\n".join(references)
    )
    return RagAnswer(answer=answer, citations=citations, refused=False)
