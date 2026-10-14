"""Print and validate Outcome 2 top-k retrieval evidence."""

from src.chunking import build_chunks
from src.corpus import load_default_corpus
from src.retrieval import RetrievalResult, retrieve_top_k


QUERY = "How do chunk size and overlap improve retrieval quality?"


def print_results(k: int, results: list[RetrievalResult]) -> None:
    print(f"top_k={k}")
    for rank, result in enumerate(results, start=1):
        preview = result.chunk.text.replace("\n", " ")[:90]
        print(
            f"  {rank}. {result.chunk.chunk_id} | {result.chunk.title} "
            f"| score={result.score:.3f} | {preview}..."
        )


def main() -> None:
    chunks = build_chunks(load_default_corpus())
    print("OUTCOME 2: TOP-K RETRIEVAL CHECK")
    print(f"Query: {QUERY}")
    print(f"Chunks searched: {len(chunks)}\n")

    for k in (2, 3):
        results = retrieve_top_k(QUERY, chunks, k=k)
        assert len(results) == k
        assert results == sorted(results, key=lambda item: item.score, reverse=True)
        assert results[0].chunk.doc_id == "doc-2"
        print_results(k, results)
        print()

    print("PASS: limits are respected, scores descend, and the expected source ranks first.")


if __name__ == "__main__":
    main()
