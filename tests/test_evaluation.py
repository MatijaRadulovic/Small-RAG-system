import pytest

from src.evaluation import (
    load_evaluation_cases,
    ndcg_at_k,
    precision_recall,
    recall_at_k,
    reciprocal_rank,
)


def test_evaluation_dataset_is_labeled_and_split() -> None:
    tuning = load_evaluation_cases(split="tuning")
    heldout = load_evaluation_cases(split="heldout")
    all_cases = load_evaluation_cases()

    assert len(tuning) == 10
    assert len(heldout) == 10
    assert len(all_cases) == len(tuning) + len(heldout)
    assert {case.category for case in all_cases} >= {
        "exact",
        "paraphrase",
        "multi_document",
        "weak_overlap",
        "unsupported",
        "adversarial",
    }
    assert all(case.expected_source_spans for case in all_cases if case.answerable)
    assert all(not case.expected_documents for case in all_cases if not case.answerable)


def test_retrieval_metrics_compute_recall_mrr_and_ndcg() -> None:
    relevant = {"doc-a", "doc-b"}
    ranked = ["doc-x", "doc-a", "doc-b", "doc-a"]

    assert recall_at_k(relevant, ranked, 3) == 1.0
    assert reciprocal_rank(relevant, ranked) == 0.5
    assert ndcg_at_k({"doc-a": 3, "doc-b": 2}, ranked, 3) <= 1.0
    assert ndcg_at_k({"doc-a": 3, "doc-b": 2}, ["doc-a", "doc-b"], 2) == 1.0


def test_document_metrics_do_not_count_repeated_chunks_as_new_ranks() -> None:
    assert reciprocal_rank({"doc-b"}, ["doc-a", "doc-a", "doc-b"]) == 0.5
    assert recall_at_k({"doc-a", "doc-b"}, ["doc-a", "doc-a", "doc-b"], 2) == 1.0


def test_precision_recall_handles_empty_and_mixed_decisions() -> None:
    assert precision_recall(0, 0, 0) == (0.0, 0.0)
    assert precision_recall(3, 1, 2) == (0.75, 0.6)


def test_loader_rejects_unknown_split() -> None:
    with pytest.raises(ValueError):
        load_evaluation_cases(split="production")
