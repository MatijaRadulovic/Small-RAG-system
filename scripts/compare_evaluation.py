"""Compare two chunking configurations on the same labeled evaluation split."""

import argparse

from src.evaluation import load_evaluation_cases
from src.embeddings import DEFAULT_EMBEDDING_MODEL, FastEmbedProvider
from scripts.evaluate_rag import evaluate_cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("heldout", "tuning", "all"), default="heldout")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--baseline-size", type=int, default=220)
    parser.add_argument("--baseline-overlap", type=int, default=0)
    parser.add_argument("--candidate-size", type=int, default=100)
    parser.add_argument("--candidate-overlap", type=int, default=20)
    parser.add_argument("--baseline-mode", choices=("lexical", "hybrid"), default="lexical")
    parser.add_argument("--candidate-mode", choices=("lexical", "hybrid"), default="lexical")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-cache")
    parser.add_argument("--offline-embeddings", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = load_evaluation_cases(split=None if args.split == "all" else args.split)
    embedding_provider = (
        FastEmbedProvider(
            args.embedding_model,
            cache_dir=args.embedding_cache,
            local_files_only=args.offline_embeddings,
        )
        if "hybrid" in {args.baseline_mode, args.candidate_mode}
        else None
    )
    configurations = {
        "baseline": (args.baseline_size, args.baseline_overlap, args.baseline_mode),
        "candidate": (args.candidate_size, args.candidate_overlap, args.candidate_mode),
    }
    metrics = {
        label: evaluate_cases(
            cases,
            top_k=args.top_k,
            chunk_size=size,
            overlap=overlap,
            retrieval_mode=mode,
            embedding_provider=embedding_provider,
        )
        for label, (size, overlap, mode) in configurations.items()
    }
    metric_names = (
        "recall_at_k",
        "mrr",
        "ndcg_at_k",
        "span_recall_at_k",
        "span_mrr",
        "citation_precision",
        "citation_recall",
        "refusal_precision",
        "refusal_recall",
    )
    print(f"Comparison on {args.split} split, k={args.top_k}")
    print(f"Baseline: {args.baseline_size}/{args.baseline_overlap} ({args.baseline_mode})")
    print(f"Candidate: {args.candidate_size}/{args.candidate_overlap} ({args.candidate_mode})")
    print("metric                 baseline  candidate  delta")
    for name in metric_names:
        baseline = float(metrics["baseline"][name])
        candidate = float(metrics["candidate"][name])
        print(f"{name:22} {baseline:8.3f}  {candidate:9.3f}  {candidate - baseline:+.3f}")


if __name__ == "__main__":
    main()
