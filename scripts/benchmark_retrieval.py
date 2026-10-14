"""Measure one-time BM25 index construction and repeated query latency."""

from statistics import mean
from time import perf_counter
import tracemalloc

from scripts.analyze_chunking import QUERIES
from src.chunking import build_chunks
from src.corpus import load_default_corpus
from src.retrieval import BM25Index, retrieve_top_k


def main() -> None:
    chunks = build_chunks(load_default_corpus())
    tracemalloc.start()
    build_started = perf_counter()
    index = BM25Index(chunks)
    build_ms = (perf_counter() - build_started) * 1000
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    query_times: list[float] = []
    for query in QUERIES:
        started = perf_counter()
        retrieve_top_k(query.question, index, k=3)
        query_times.append((perf_counter() - started) * 1000)

    print("BM25 INDEX BENCHMARK")
    print(f"Chunks: {len(index.chunks)}")
    print(f"Content vocabulary size: {index.vocabulary_size}")
    print(f"Indexed content tokens: {index.indexed_token_count}")
    print(f"Average chunk length: {index.average_chunk_length:.2f} tokens")
    print(f"Index build: {build_ms:.3f} ms")
    print(f"Peak traced build allocation: {peak_bytes / 1024:.1f} KiB")
    print(f"Repeated query mean: {mean(query_times):.3f} ms")
    print(f"Repeated query max: {max(query_times):.3f} ms")
    print("Timing values are machine-dependent; token counts are corpus-dependent.")


if __name__ == "__main__":
    main()
