"""Evaluate retrieval, citation coverage, and refusal decisions on labeled cases."""

import argparse
from collections import Counter

from src.chunking import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP, build_chunks
from src.corpus import load_default_corpus
from src.evaluation import (
    DEFAULT_DATASET,
    EvaluationCase,
    load_evaluation_cases,
    ndcg_at_k,
    precision_recall,
    recall_at_k,
    reciprocal_rank,
)
from src.rag import build_grounded_answer
from src.retrieval import BM25Index, RetrievalResult, retrieve_top_k


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--split", choices=("all", "tuning", "heldout"), default="heldout")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    parser.add_argument("--details", action="store_true", help="Print per-question outcomes")
    return parser.parse_args()


def _span_offsets(case: EvaluationCase, documents: dict[str, str]) -> list[tuple[str, int, int]]:
    spans: list[tuple[str, int, int]] = []
    for expected in case.expected_source_spans:
        text = documents.get(expected.doc_id)
        if text is None:
            raise ValueError(f"{case.case_id}: unknown span document {expected.doc_id!r}")
        start = text.find(expected.text)
        if start < 0:
            raise ValueError(f"{case.case_id}: expected span not found in {expected.doc_id}")
        spans.append((expected.doc_id, start, start + len(expected.text)))
    return spans


def evaluate_cases(
    cases: list[EvaluationCase],
    *,
    top_k: int,
    chunk_size: int,
    overlap: int,
    details: bool = False,
) -> dict[str, float | int]:
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    documents = load_default_corpus()
    source_texts = {document.doc_id: document.text for document in documents}
    chunks = build_chunks(documents, chunk_size=chunk_size, overlap=overlap)
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    index = BM25Index(chunks)

    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    span_recalls: list[float] = []
    span_reciprocal_ranks: list[float] = []
    citation_counts: Counter[str] = Counter()
    refusal_counts: Counter[str] = Counter()
    expected_span_count = 0
    cited_span_count = 0

    for case in cases:
        results = retrieve_top_k(case.question, index, k=top_k)
        retrieved_doc_ids = [item.chunk.doc_id for item in results]
        answer = build_grounded_answer(case.question, index, top_k=top_k)
        relevant = set(case.expected_documents)
        if case.answerable:
            recalls.append(recall_at_k(relevant, retrieved_doc_ids, top_k))
            reciprocal_ranks.append(reciprocal_rank(relevant, retrieved_doc_ids))
            ndcgs.append(ndcg_at_k(case.expected_documents, retrieved_doc_ids, top_k))

        expected_spans = _span_offsets(case, source_texts)
        expected_span_count += len(expected_spans)
        def covers_span(result: RetrievalResult, span: tuple[str, int, int]) -> bool:
            doc_id, start, end = span
            return (
                result.chunk.doc_id == doc_id
                and result.chunk.start_offset <= start
                and result.chunk.end_offset >= end
            )

        covered_at_k = {
            span_index
            for span_index, span in enumerate(expected_spans)
            if any(covers_span(result, span) for result in results)
        }
        if case.answerable:
            span_recalls.append(
                len(covered_at_k) / len(expected_spans) if expected_spans else 0.0
            )
            first_span_rank = next(
                (
                    rank
                    for rank, result in enumerate(results, start=1)
                    if any(covers_span(result, span) for span in expected_spans)
                ),
                0,
            )
            span_reciprocal_ranks.append(1 / first_span_rank if first_span_rank else 0.0)
        cited_chunks = []
        for citation in answer.citations:
            chunk = chunks_by_id.get(citation)
            if chunk is None:
                citation_counts["incorrect"] += 1
                continue
            cited_chunks.append(chunk)
            covers_expected_span = any(
                chunk.doc_id == doc_id and chunk.start_offset <= start and chunk.end_offset >= end
                for doc_id, start, end in expected_spans
            )
            citation_counts["correct" if covers_expected_span else "incorrect"] += 1

        cited_span_count += sum(
            any(
                chunk.doc_id == doc_id
                and chunk.start_offset <= start
                and chunk.end_offset >= end
                for chunk in cited_chunks
            )
            for doc_id, start, end in expected_spans
        )

        if not case.answerable and answer.refused:
            refusal_counts["true_positive"] += 1
        elif not case.answerable:
            refusal_counts["false_negative"] += 1
        elif answer.refused:
            refusal_counts["false_positive"] += 1
        else:
            refusal_counts["true_negative"] += 1

        if details:
            span_recall = len(covered_at_k) / len(expected_spans) if expected_spans else 0.0
            print(
                f"{case.case_id} [{case.category}] answerable={case.answerable} "
                f"retrieved={','.join(retrieved_doc_ids) or '-'} "
                f"span_recall@{top_k}={span_recall:.2f} "
                f"citations={','.join(answer.citations) or '-'} refused={answer.refused}"
            )

    citation_precision, _ = precision_recall(
        citation_counts["correct"], citation_counts["incorrect"], 0
    )
    refusal_precision, refusal_recall = precision_recall(
        refusal_counts["true_positive"],
        refusal_counts["false_positive"],
        refusal_counts["false_negative"],
    )
    return {
        "case_count": len(cases),
        "answerable_count": len(recalls),
        "recall_at_k": sum(recalls) / len(recalls) if recalls else 0.0,
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0,
        "ndcg_at_k": sum(ndcgs) / len(ndcgs) if ndcgs else 0.0,
        "span_recall_at_k": sum(span_recalls) / len(span_recalls) if span_recalls else 0.0,
        "span_mrr": sum(span_reciprocal_ranks) / len(span_reciprocal_ranks) if span_reciprocal_ranks else 0.0,
        "citation_precision": citation_precision,
        "citation_recall": cited_span_count / expected_span_count if expected_span_count else 0.0,
        "refusal_precision": refusal_precision,
        "refusal_recall": refusal_recall,
        "citations": citation_counts["correct"] + citation_counts["incorrect"],
        "expected_spans": expected_span_count,
    }


def main() -> None:
    args = parse_args()
    cases = load_evaluation_cases(
        args.dataset,
        split=None if args.split == "all" else args.split,
    )
    if not cases:
        raise SystemExit("The selected split has no evaluation cases.")
    metrics = evaluate_cases(
        cases,
        top_k=args.top_k,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        details=args.details,
    )
    print(
        f"RAG evaluation: {args.split}; {metrics['case_count']} cases; "
        f"chunk={args.chunk_size}/{args.overlap}; k={args.top_k}"
    )
    if metrics["answerable_count"]:
        print(
            f"Retrieval — Recall@{args.top_k}: {metrics['recall_at_k']:.3f}; "
            f"MRR: {metrics['mrr']:.3f}; nDCG@{args.top_k}: {metrics['ndcg_at_k']:.3f}; "
            f"source-span Recall@{args.top_k}: {metrics['span_recall_at_k']:.3f}; "
            f"span MRR: {metrics['span_mrr']:.3f}"
        )
    print(
        f"Citations — precision: {metrics['citation_precision']:.3f}; "
        f"source-span recall: {metrics['citation_recall']:.3f} "
        f"({metrics['citations']} citations; {metrics['expected_spans']} expected spans)"
    )
    print(
        f"Refusals — precision: {metrics['refusal_precision']:.3f}; "
        f"recall: {metrics['refusal_recall']:.3f}"
    )


if __name__ == "__main__":
    main()
