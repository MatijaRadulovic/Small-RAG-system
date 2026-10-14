"""Print and validate Outcome 2 top-k retrieval evidence."""

from src.chunking import build_chunks
from src.corpus import load_default_corpus
from src.retrieval import BM25Index, RetrievalResult, retrieve_top_k


QUERY = "How do chunk size and overlap improve retrieval quality?"


def print_results(k: int, results: list[RetrievalResult]) -> None:
    print(f"top_k={k}")
    for rank, result in enumerate(results, start=1):
        preview = result.chunk.text.replace("\n", " ")[:90]
        semantic_details = (
            f" | semantic cosine={result.semantic_score:.3f} "
            f"| lexical rank={result.lexical_rank} | semantic rank={result.semantic_rank}"
            if result.semantic_score is not None
            else ""
        )
        print(
            f"  {rank}. {result.chunk.chunk_id} | {result.chunk.title} "
            f"| fused={result.score:.5f} | BM25={result.content_score:.3f} "
            f"| title BM25={result.title_score:.3f} "
            f"{semantic_details} "
            f"| diversity={result.diversity_score:.2f} "
            f"(redundancy={result.redundancy:.2f}) "
            f"| terms={','.join(result.matched_terms)} | {preview}..."
        )


def main() -> None:
    chunks = build_chunks(load_default_corpus())
    index = BM25Index(chunks)
    print("OUTCOME 2: TOP-K RETRIEVAL CHECK")
    print(f"Query: {QUERY}")
    print(f"Chunks searched: {len(chunks)}\n")

    for k in (2, 3):
        results = retrieve_top_k(QUERY, index, k=k)
        assert len(results) == k
        assert results == sorted(results, key=lambda item: item.score, reverse=True)
        assert results[0].chunk.doc_id == "doc-2"
        print_results(k, results)
        print()

    print("PASS: limits are respected, scores descend, and the expected source ranks first.")


if __name__ == "__main__":
    main()
