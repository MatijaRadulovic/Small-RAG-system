"""Measure local embedding load, indexing, query latency, and memory."""

import argparse
from pathlib import Path
import resource
import statistics
import time
import tracemalloc

from src.chunking import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP, build_chunks
from src.corpus import load_default_corpus
from src.embeddings import DEFAULT_EMBEDDING_MODEL, FastEmbedProvider
from src.evaluation import load_evaluation_cases
from src.hybrid_retrieval import HybridRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-cache", default=".cache/fastembed")
    parser.add_argument("--offline-embeddings", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    parser.add_argument("--repeats", type=int, default=5)
    return parser.parse_args()


def _directory_size(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(
        path.stat().st_size
        for path in directory.rglob("*")
        if path.is_file() and not path.is_symlink()
    )


def _peak_rss_mib() -> float | None:
    try:
        # Linux reports KiB; macOS reports bytes.
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (ImportError, AttributeError):
        return None
    return peak / (1024 * 1024) if peak > 10**8 else peak / 1024


def main() -> None:
    args = parse_args()
    if args.repeats <= 0:
        raise SystemExit("--repeats must be > 0")
    chunks = build_chunks(
        load_default_corpus(),
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    tracemalloc.start()
    load_started = time.perf_counter()
    provider = FastEmbedProvider(
        args.embedding_model,
        cache_dir=args.embedding_cache,
        local_files_only=args.offline_embeddings,
    )
    model_load_seconds = time.perf_counter() - load_started
    build_started = time.perf_counter()
    retriever = HybridRetriever(chunks, provider)
    index_build_seconds = time.perf_counter() - build_started

    questions = [
        case.question
        for case in load_evaluation_cases(split="heldout")
        if case.answerable
    ]
    latencies: list[float] = []
    for _ in range(args.repeats):
        for question in questions:
            started = time.perf_counter()
            retriever.retrieve_top_k(question, k=3)
            latencies.append(time.perf_counter() - started)
    latencies.sort()
    p95_index = min(int(0.95 * len(latencies)), len(latencies) - 1)
    _, peak_traced_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    vector_bytes = len(chunks) * retriever.semantic_index.dimension * 4
    cache_bytes = _directory_size(Path(args.embedding_cache))
    print(f"Model: {args.embedding_model}")
    print(f"Chunks: {len(chunks)}; dimensions: {retriever.semantic_index.dimension}")
    print(f"Model load/init: {model_load_seconds * 1000:.1f} ms")
    print(f"Chunk embedding/index build: {index_build_seconds * 1000:.1f} ms")
    print(
        f"Warm query mean: {statistics.mean(latencies) * 1000:.2f} ms; "
        f"p95: {latencies[p95_index] * 1000:.2f} ms ({len(latencies)} runs)"
    )
    print(f"Embedding matrix lower bound (float32): {vector_bytes / 1024:.1f} KiB")
    print(f"Cached model files: {cache_bytes / (1024 * 1024):.1f} MiB")
    print(f"Python traced peak allocation: {peak_traced_bytes / (1024 * 1024):.1f} MiB")
    peak_rss = _peak_rss_mib()
    print(f"Process peak RSS: {peak_rss:.1f} MiB" if peak_rss is not None else "Process peak RSS: unavailable")


if __name__ == "__main__":
    main()
