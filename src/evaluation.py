"""Load labeled RAG evaluation cases and calculate common retrieval metrics."""

from dataclasses import dataclass
import json
from math import log2
from pathlib import Path
from typing import Any, Iterable


DEFAULT_DATASET = Path(__file__).resolve().parents[1] / "data" / "rag_eval_v1.jsonl"


@dataclass(frozen=True)
class ExpectedSpan:
    doc_id: str
    text: str


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    split: str
    category: str
    question: str
    expected_documents: dict[str, int]
    expected_source_spans: tuple[ExpectedSpan, ...]
    answerable: bool
    expected_facts: tuple[str, ...]
    notes: str


def load_evaluation_cases(
    path: str | Path = DEFAULT_DATASET,
    split: str | None = None,
) -> list[EvaluationCase]:
    cases: list[EvaluationCase] = []
    seen_ids: set[str] = set()
    with Path(path).open(encoding="utf-8") as dataset_file:
        for line_number, line in enumerate(dataset_file, start=1):
            if not line.strip():
                continue
            try:
                row: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number}: {error}") from error
            case = EvaluationCase(
                case_id=row["id"],
                split=row["split"],
                category=row["category"],
                question=row["question"],
                expected_documents={
                    str(doc_id): int(grade)
                    for doc_id, grade in row["expected_documents"].items()
                },
                expected_source_spans=tuple(
                    ExpectedSpan(doc_id=span["doc_id"], text=span["text"])
                    for span in row["expected_source_spans"]
                ),
                answerable=bool(row["answerable"]),
                expected_facts=tuple(row["expected_facts"]),
                notes=row["notes"],
            )
            if case.case_id in seen_ids:
                raise ValueError(f"Duplicate evaluation ID {case.case_id!r}")
            seen_ids.add(case.case_id)
            if case.split not in {"tuning", "heldout"}:
                raise ValueError(f"Unknown split {case.split!r} on line {line_number}")
            if case.answerable != bool(case.expected_documents):
                raise ValueError(f"Answerability labels disagree on line {line_number}")
            span_documents = {span.doc_id for span in case.expected_source_spans}
            if case.answerable != bool(case.expected_source_spans):
                raise ValueError(f"Expected source spans disagree with answerability on line {line_number}")
            if not set(case.expected_documents) <= span_documents:
                raise ValueError(f"Relevant documents lack source spans on line {line_number}")
            if split is None or case.split == split:
                cases.append(case)
    if split not in {None, "tuning", "heldout"}:
        raise ValueError("split must be 'tuning', 'heldout', or None")
    return cases


def reciprocal_rank(relevant_documents: set[str], ranked_documents: Iterable[str]) -> float:
    for rank, doc_id in enumerate(unique_ranked_documents(ranked_documents), start=1):
        if doc_id in relevant_documents:
            return 1.0 / rank
    return 0.0


def unique_ranked_documents(ranked_documents: Iterable[str]) -> list[str]:
    """Keep the first occurrence of each source document in chunk ranking order."""
    return list(dict.fromkeys(ranked_documents))


def recall_at_k(relevant_documents: set[str], ranked_documents: list[str], k: int) -> float:
    if not relevant_documents:
        return 0.0
    unique_ranked = unique_ranked_documents(ranked_documents)
    return len(relevant_documents & set(unique_ranked[:k])) / len(relevant_documents)


def ndcg_at_k(relevance_by_document: dict[str, int], ranked_documents: list[str], k: int) -> float:
    def gain(relevance: int, rank: int) -> float:
        return (2**relevance - 1) / log2(rank + 1)

    unique_ranked = unique_ranked_documents(ranked_documents)
    dcg = sum(
        gain(relevance_by_document.get(doc_id, 0), rank)
        for rank, doc_id in enumerate(unique_ranked[:k], start=1)
    )
    ideal_grades = sorted(relevance_by_document.values(), reverse=True)[:k]
    ideal = sum(gain(grade, rank) for rank, grade in enumerate(ideal_grades, start=1))
    return dcg / ideal if ideal else 0.0


def precision_recall(true_positive: int, false_positive: int, false_negative: int) -> tuple[float, float]:
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    return precision, recall
