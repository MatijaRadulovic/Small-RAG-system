from dataclasses import dataclass, field

from .chunking import Chunk
from .hybrid_retrieval import HybridRetriever
from .retrieval import BM25Index, RetrievalResult, retrieve_top_k, tokenize


REFUSAL_MESSAGE = "I do not have enough grounded evidence in the provided corpus to answer that question."
MIN_MATCHED_TERMS = 2
MIN_QUERY_TERM_COVERAGE = 0.5


@dataclass(frozen=True)
class RagAnswer:
    answer: str
    citations: list[str]
    refused: bool
    evidence_assessments: list["EvidenceAssessment"] = field(default_factory=list)


@dataclass(frozen=True)
class EvidenceAssessment:
    chunk_id: str
    score: float
    matched_terms: tuple[str, ...]
    term_coverage: float
    sufficient: bool
    reason: str


def assess_evidence(
    question: str,
    result: RetrievalResult,
    language: str = "en",
) -> EvidenceAssessment:
    query_terms = set(tokenize(question, language=language))
    passage_terms = set(tokenize(result.chunk.text, language=language))
    matched_terms = tuple(sorted(query_terms & passage_terms))
    required_matches = 1 if len(query_terms) == 1 else MIN_MATCHED_TERMS
    term_coverage = len(matched_terms) / len(query_terms) if query_terms else 0.0
    sufficient = bool(query_terms) and (
        len(matched_terms) >= required_matches
        and term_coverage >= MIN_QUERY_TERM_COVERAGE
    )
    if sufficient:
        reason = f"matched {len(matched_terms)} of {len(query_terms)} meaningful query terms"
    elif not query_terms:
        reason = "question has no meaningful terms after tokenization"
    else:
        reason = (
            f"matched {len(matched_terms)} of {len(query_terms)} meaningful terms; "
            f"requires at least {required_matches} matches and "
            f"{MIN_QUERY_TERM_COVERAGE:.0%} coverage"
        )
    return EvidenceAssessment(
        chunk_id=result.chunk.chunk_id,
        score=result.score,
        matched_terms=matched_terms,
        term_coverage=term_coverage,
        sufficient=sufficient,
        reason=reason,
    )


def has_sufficient_evidence(
    question: str,
    result: RetrievalResult,
    language: str = "en",
) -> bool:
    """Compatibility wrapper for callers that only need the pass/fail decision."""
    return assess_evidence(question, result, language=language).sufficient


def build_grounded_answer(
    question: str,
    chunks: list[Chunk] | BM25Index | HybridRetriever,
    top_k: int = 2,
    language: str | None = None,
    max_per_source: int | None = 2,
) -> RagAnswer:
    effective_language = (
        language
        or (chunks.language if isinstance(chunks, (BM25Index, HybridRetriever)) else "en")
    )
    if isinstance(chunks, HybridRetriever):
        results = chunks.retrieve_top_k(
            question,
            k=top_k,
            max_per_source=max_per_source,
        )
    else:
        results = retrieve_top_k(
            question,
            chunks,
            k=top_k,
            language=language,
            max_per_source=max_per_source,
        )
    if not results:
        return RagAnswer(answer=REFUSAL_MESSAGE, citations=[], refused=True)

    assessments = [
        assess_evidence(question, item, language=effective_language)
        for item in results
    ]
    accepted_ids = {item.chunk_id for item in assessments if item.sufficient}
    accepted_results = [item for item in results if item.chunk.chunk_id in accepted_ids]
    if not accepted_results:
        return RagAnswer(
            answer=REFUSAL_MESSAGE,
            citations=[],
            refused=True,
            evidence_assessments=assessments,
        )

    supporting_points: list[str] = []
    references: list[str] = []
    citations: list[str] = []
    for item in accepted_results:
        chunk_id = item.chunk.chunk_id
        citations.append(chunk_id)
        supporting_points.append(f"- {item.chunk.text} [{chunk_id}]")
        references.append(
            f"- [{chunk_id}] {item.chunk.title}; section {item.chunk.section}; "
            f"source characters {item.chunk.start_offset}-{item.chunk.end_offset}"
        )

    answer_heading = (
        "Answer (based only on the individually supported passages):\n"
        if len(accepted_results) < len(results)
        else "Answer:\n"
    )
    answer = (
        answer_heading
        + "\n".join(supporting_points)
        + "\n\nReferences:\n"
        + "\n".join(references)
    )
    return RagAnswer(
        answer=answer,
        citations=citations,
        refused=False,
        evidence_assessments=assessments,
    )
