"""Evaluate a size-by-overlap grid on known relevant-source queries."""

from dataclasses import dataclass
from itertools import product

from src.chunking import build_chunks
from src.corpus import load_default_corpus
from src.evaluation import load_evaluation_cases, reciprocal_rank
from src.retrieval import BM25Index, retrieve_top_k

CHUNK_SIZES = (60, 80, 100, 120, 140, 180, 220)
OVERLAPS = (0, 10, 20, 30, 40, 50)
TOP_K = 3


@dataclass(frozen=True)
class Summary:
    chunk_size: int
    overlap: int
    chunk_count: int
    hit_at_1: float
    hit_at_3: float
    mrr: float


def evaluate(chunk_size: int, overlap: int, documents: list) -> Summary:
    chunks = build_chunks(documents, chunk_size=chunk_size, overlap=overlap)
    index = BM25Index(chunks)
    ranks: list[float] = []
    hit_at_1: list[bool] = []
    hit_at_3: list[bool] = []

    queries = load_evaluation_cases(split="tuning")
    for query in queries:
        results = retrieve_top_k(query.question, index, k=TOP_K)
        doc_ids = [result.chunk.doc_id for result in results]
        relevant_docs = set(query.expected_documents)
        rank = reciprocal_rank(relevant_docs, doc_ids)
        ranks.append(rank)
        hit_at_1.append(bool(doc_ids and doc_ids[0] in relevant_docs))
        hit_at_3.append(bool(relevant_docs & set(doc_ids)))

    query_count = len(queries)
    return Summary(
        chunk_size=chunk_size,
        overlap=overlap,
        chunk_count=len(chunks),
        hit_at_1=sum(hit_at_1) / query_count,
        hit_at_3=sum(hit_at_3) / query_count,
        mrr=sum(ranks) / query_count,
    )


def main() -> None:
    documents = load_default_corpus()
    configurations = [
        (size, overlap)
        for size, overlap in product(CHUNK_SIZES, OVERLAPS)
        if overlap < size
    ]
    summaries = [evaluate(size, overlap, documents) for size, overlap in configurations]

    print("CHUNK SIZE × OVERLAP GRID")
    print(
        f"{len(summaries)} configurations; "
        f"{len(load_evaluation_cases(split='tuning'))} versioned tuning queries; "
        f"top_k={TOP_K}; relevance judged by expected source document."
    )
    print("size overlap chunks hit@1 hit@3 MRR")
    for item in summaries:
        print(
            f"{item.chunk_size:>4} {item.overlap:>7} {item.chunk_count:>6} "
            f"{item.hit_at_1:>5.0%} {item.hit_at_3:>5.0%} {item.mrr:>5.3f}"
        )

    # Rank retrieval quality first. When quality ties, prefer less overlap and
    # then fewer chunks, since these reduce duplicated text and index size.
    best = max(
        summaries,
        key=lambda item: (
            item.hit_at_1,
            item.mrr,
            item.hit_at_3,
            -item.overlap,
            -item.chunk_count,
        ),
    )
    print(
        "\nRecommended measured configuration: "
        f"chunk_size={best.chunk_size}, overlap={best.overlap} "
        f"(hit@1={best.hit_at_1:.1%}, hit@3={best.hit_at_3:.1%}, "
        f"MRR={best.mrr:.3f}, chunks={best.chunk_count})."
    )


if __name__ == "__main__":
    main()
