"""Measure first-stage and cross-encoder latency on the held-out questions."""

import argparse
from math import ceil
from statistics import mean
from time import perf_counter

from src.chunking import build_chunks
from src.corpus import load_default_corpus
from src.embeddings import DEFAULT_EMBEDDING_MODEL, FastEmbedProvider
from src.evaluation import load_evaluation_cases
from src.hybrid_retrieval import HybridRetriever
from src.reranking import DEFAULT_RERANKER_MODEL, FastEmbedReranker


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, ceil(len(ordered) * fraction) - 1)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-k", type=int, default=14)
    parser.add_argument("--chunk-size", type=int, default=220)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--cache-dir")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if args.candidate_k <= 0:
        parser.error("--candidate-k must be > 0")

    documents = load_default_corpus()
    chunks = build_chunks(documents, chunk_size=args.chunk_size, overlap=args.overlap)
    embeddings = FastEmbedProvider(
        args.embedding_model, cache_dir=args.cache_dir, local_files_only=args.offline
    )
    index_start = perf_counter()
    first_stage = HybridRetriever(chunks, embeddings)
    index_seconds = perf_counter() - index_start
    model_start = perf_counter()
    reranker = FastEmbedReranker(
        args.reranker_model, cache_dir=args.cache_dir, local_files_only=args.offline
    )
    model_seconds = perf_counter() - model_start

    first_stage_times: list[float] = []
    rerank_times: list[float] = []
    counts: list[int] = []
    for case in load_evaluation_cases(split="heldout"):
        start = perf_counter()
        candidates = first_stage.retrieve_top_k(
            case.question,
            k=args.candidate_k,
            max_per_source=None,
            duplicate_threshold=1.0,
        )
        first_stage_times.append(perf_counter() - start)
        start = perf_counter()
        reranker.score(case.question, [item.chunk.text for item in candidates])
        rerank_times.append(perf_counter() - start)
        counts.append(len(candidates))

    print(f"heldout_queries={len(first_stage_times)} candidates_mean={mean(counts):.1f}")
    print(f"index_build_ms={index_seconds * 1000:.2f}")
    print(f"reranker_load_ms={model_seconds * 1000:.2f}")
    for label, timings in (
        ("first_stage_retrieval", first_stage_times),
        ("cross_encoder_reranking", rerank_times),
    ):
        print(
            f"{label}_ms mean={mean(timings) * 1000:.2f} "
            f"p95={percentile(timings, 0.95) * 1000:.2f}"
        )


if __name__ == "__main__":
    main()
