"""Compare chunking configurations on representative retrieval queries."""

from dataclasses import dataclass

from src.chunking import build_chunks
from src.corpus import load_default_corpus
from src.retrieval import retrieve_top_k


@dataclass(frozen=True)
class EvaluationQuery:
    question: str
    expected_doc_id: str


QUERIES = [
    EvaluationQuery(
        question="What does retrieval augmented generation combine?",
        expected_doc_id="doc-1",
    ),
    EvaluationQuery(
        question="How do chunk size and overlap affect retrieval quality?",
        expected_doc_id="doc-2",
    ),
    EvaluationQuery(
        question="What should happen when context does not support a claim?",
        expected_doc_id="doc-3",
    ),
]

CONFIGURATIONS = [
    (80, 0),
    (100, 20),
    (140, 30),
    (180, 40),
    (220, 50),
]


def reciprocal_rank(expected_doc_id: str, doc_ids: list[str]) -> float:
    for rank, doc_id in enumerate(doc_ids, start=1):
        if doc_id == expected_doc_id:
            return 1.0 / rank
    return 0.0


def main() -> None:
    documents = load_default_corpus()
    summaries: list[tuple[int, int, int, float, float, float]] = []

    print("CHUNKING ANALYSIS")
    print("Each query has one expected source document; retrieval uses top_k=3.\n")

    for chunk_size, overlap in CONFIGURATIONS:
        chunks = build_chunks(documents, chunk_size=chunk_size, overlap=overlap)
        reciprocal_ranks: list[float] = []
        top_scores: list[float] = []

        print(f"Configuration: chunk_size={chunk_size}, overlap={overlap}")
        print(f"Chunks produced: {len(chunks)}")
        for query in QUERIES:
            results = retrieve_top_k(query.question, chunks, k=3)
            doc_ids = [result.chunk.doc_id for result in results]
            rank_score = reciprocal_rank(query.expected_doc_id, doc_ids)
            reciprocal_ranks.append(rank_score)
            top_scores.append(results[0].score if results else 0.0)
            ranking = ", ".join(
                f"{result.chunk.chunk_id} ({result.score:.3f})" for result in results
            ) or "no results"
            expected_rank = int(1 / rank_score) if rank_score else "missing"
            print(f"  Query: {query.question}")
            print(f"  Expected: {query.expected_doc_id}; rank: {expected_rank}")
            print(f"  Results: {ranking}")

        hit_rate = sum(score > 0 for score in reciprocal_ranks) / len(QUERIES)
        mean_reciprocal_rank = sum(reciprocal_ranks) / len(QUERIES)
        mean_top_score = sum(top_scores) / len(QUERIES)
        summaries.append(
            (
                chunk_size,
                overlap,
                len(chunks),
                hit_rate,
                mean_reciprocal_rank,
                mean_top_score,
            )
        )
        print()

    print("ANALYSIS SUMMARY")
    print("size overlap chunks hit_rate MRR mean_top_score")
    for size, overlap, count, hit_rate, mrr, mean_score in summaries:
        print(
            f"{size:>4} {overlap:>7} {count:>6} "
            f"{hit_rate:>8.0%} {mrr:>4.3f} {mean_score:>14.3f}"
        )

    best = max(
        summaries,
        key=lambda item: (item[4], item[1] > 0, item[5], -item[2]),
    )
    print(
        "\nRecommended measured configuration: "
        f"chunk_size={best[0]}, overlap={best[1]} "
        "(highest MRR, then boundary-preserving overlap, mean top score, and fewer chunks)."
    )


if __name__ == "__main__":
    main()
