"""Evaluate a size-by-overlap grid on known relevant-source queries."""

from dataclasses import dataclass
from itertools import product

from src.chunking import build_chunks
from src.corpus import load_default_corpus
from src.retrieval import retrieve_top_k


@dataclass(frozen=True)
class EvaluationQuery:
    question: str
    expected_doc_id: str


# Three questions per source document reduce the risk of choosing settings for
# one convenient wording. Expected sources are fixed before evaluating settings.
QUERIES = [
    EvaluationQuery("What does retrieval augmented generation combine?", "doc-1"),
    EvaluationQuery("How does retrieval provide context to the generator?", "doc-1"),
    EvaluationQuery("Why can RAG use domain-specific information without retraining?", "doc-1"),
    EvaluationQuery("How can chunks that are too small or large affect retrieval?", "doc-2"),
    EvaluationQuery("Why is overlap added between adjacent text chunks?", "doc-2"),
    EvaluationQuery("What metadata helps trace a retrieved source?", "doc-2"),
    EvaluationQuery("When should a grounded assistant refuse to answer?", "doc-3"),
    EvaluationQuery("Why should a system avoid inventing unsupported answers?", "doc-3"),
    EvaluationQuery("How does refusal behavior affect user trust?", "doc-3"),
    EvaluationQuery("What do hit rate and mean reciprocal rank measure?", "doc-4"),
    EvaluationQuery("How should top-k retrieval results be ordered?", "doc-4"),
    EvaluationQuery("Why compare several representative retrieval queries?", "doc-4"),
    EvaluationQuery("What should citation identifiers match?", "doc-5"),
    EvaluationQuery("How do references help readers inspect evidence?", "doc-5"),
    EvaluationQuery("Why is a citation alone insufficient?", "doc-5"),
]

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


def reciprocal_rank(expected_doc_id: str, doc_ids: list[str]) -> float:
    for rank, doc_id in enumerate(doc_ids, start=1):
        if doc_id == expected_doc_id:
            return 1.0 / rank
    return 0.0


def evaluate(chunk_size: int, overlap: int, documents: list) -> Summary:
    chunks = build_chunks(documents, chunk_size=chunk_size, overlap=overlap)
    ranks: list[float] = []
    hit_at_1: list[bool] = []
    hit_at_3: list[bool] = []

    for query in QUERIES:
        results = retrieve_top_k(query.question, chunks, k=TOP_K)
        doc_ids = [result.chunk.doc_id for result in results]
        rank = reciprocal_rank(query.expected_doc_id, doc_ids)
        ranks.append(rank)
        hit_at_1.append(bool(doc_ids and doc_ids[0] == query.expected_doc_id))
        hit_at_3.append(query.expected_doc_id in doc_ids)

    query_count = len(QUERIES)
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
        f"{len(summaries)} configurations; {len(QUERIES)} fixed queries; "
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
