from dataclasses import dataclass, field
from collections.abc import Iterable

from .chunking import Chunk
from .generation import AnswerGenerator, build_generation_prompts, validate_generated_answer
from .hybrid_retrieval import HybridRetriever
from .retrieval import BM25Index, RetrievalResult, retrieve_top_k, tokenize
from .reranking import RerankingRetriever


REFUSAL_MESSAGE = "I do not have enough grounded evidence in the provided corpus to answer that question."
MIN_MATCHED_TERMS = 2
MIN_QUERY_TERM_COVERAGE = 0.5


@dataclass(frozen=True)
class RagAnswer:
    answer: str
    citations: list[str]
    refused: bool
    evidence_assessments: list["EvidenceAssessment"] = field(default_factory=list)
    generation_status: str = "extractive"
    generation_reason: str | None = None


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
    chunks: list[Chunk] | BM25Index | HybridRetriever | RerankingRetriever,
    top_k: int = 2,
    language: str | None = None,
    max_per_source: int | None = 2,
    generator: AnswerGenerator | None = None,
) -> RagAnswer:
    effective_language = (
        language
        or (
            chunks.language
            if isinstance(chunks, (BM25Index, HybridRetriever, RerankingRetriever))
            else "en"
        )
    )
    if isinstance(chunks, (HybridRetriever, RerankingRetriever)):
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

    if generator is not None:
        system_prompt, user_prompt = build_generation_prompts(
            question, [item.chunk for item in accepted_results]
        )
        try:
            generated = generator.generate(system_prompt, user_prompt)
            validation = validate_generated_answer(
                generated,
                [item.chunk for item in accepted_results],
                language=effective_language,
            )
            if validation.valid:
                references = _format_references(
                    item.chunk
                    for item in accepted_results
                    if item.chunk.chunk_id in validation.citations
                )
                return RagAnswer(
                    answer=generated.strip() + "\n\nReferences:\n" + "\n".join(references),
                    citations=list(validation.citations),
                    refused=False,
                    evidence_assessments=assessments,
                    generation_status="generated",
                    generation_reason=validation.reason,
                )
            fallback_reason = validation.reason
        except Exception as error:
            fallback_reason = f"generator failed ({type(error).__name__}); used extractive answer"
    else:
        fallback_reason = None

    supporting_points = [
        f"- {item.chunk.text} [{item.chunk.chunk_id}]" for item in accepted_results
    ]
    citations = [item.chunk.chunk_id for item in accepted_results]
    references = _format_references(item.chunk for item in accepted_results)

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
        generation_status="fallback" if generator is not None else "extractive",
        generation_reason=fallback_reason,
    )


def _format_references(chunks: Iterable[Chunk]) -> list[str]:
    """Format references without trusting model-generated citation metadata."""
    return [
        f"- [{chunk.chunk_id}] {chunk.title}; section {chunk.section}; "
        f"source characters {chunk.start_offset}-{chunk.end_offset}"
        for chunk in chunks
    ]
