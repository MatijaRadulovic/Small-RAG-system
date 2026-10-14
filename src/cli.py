import argparse
import json

from .chunking import build_chunks
from .config import RagConfig
from .corpus import list_corpus_profiles, load_corpus
from .rag import build_grounded_answer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Week 6 RAG homework starter")
    parser.add_argument("--question", required=True, help="Question to ask over the local corpus")
    parser.add_argument("--chunk-size", type=int, default=RagConfig.chunk_size)
    parser.add_argument("--overlap", type=int, default=RagConfig.overlap)
    parser.add_argument("--top-k", type=int, default=RagConfig.top_k)
    parser.add_argument(
        "--retrieval-mode",
        choices=("lexical", "hybrid"),
        default=RagConfig.retrieval_mode,
        help="Lexical BM25 or local embedding plus BM25 retrieval",
    )
    parser.add_argument("--embedding-model", default=RagConfig.embedding_model)
    parser.add_argument("--embedding-cache", default=".cache/fastembed")
    parser.add_argument("--index-cache", default=".cache/rag-index")
    parser.add_argument(
        "--offline-embeddings",
        action="store_true",
        help="Require selected embedding/reranker models to already exist in cache",
    )
    parser.add_argument("--rerank", action="store_true", help="Rerank a broader candidate set")
    parser.add_argument("--candidate-k", type=int, default=RagConfig.candidate_k)
    parser.add_argument("--reranker-model", default=RagConfig.reranker_model)
    parser.add_argument("--max-per-source", type=int, default=RagConfig.max_per_source)
    parser.add_argument(
        "--language",
        choices=("en", "sr", "mixed"),
        default=RagConfig.language,
    )
    parser.add_argument("--semantic-weight", type=float, default=RagConfig.semantic_weight)
    parser.add_argument("--rrf-constant", type=int, default=RagConfig.rrf_constant)
    parser.add_argument(
        "--min-evidence-matches", type=int, default=RagConfig.min_evidence_matches
    )
    parser.add_argument(
        "--min-evidence-coverage", type=float, default=RagConfig.min_evidence_coverage
    )
    parser.add_argument(
        "--corpus-profile",
        default="default",
        choices=list_corpus_profiles(),
        help="Corpus profile: default or training-rag",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print a structured trace without query or passage text",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RagConfig(
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        top_k=args.top_k,
        retrieval_mode=args.retrieval_mode,
        language=args.language,
        max_per_source=args.max_per_source,
        min_evidence_matches=args.min_evidence_matches,
        min_evidence_coverage=args.min_evidence_coverage,
        semantic_weight=args.semantic_weight,
        rrf_constant=args.rrf_constant,
        embedding_model=args.embedding_model,
        rerank=args.rerank,
        candidate_k=args.candidate_k,
        reranker_model=args.reranker_model,
        model_cache_dir=args.embedding_cache,
        index_cache_dir=args.index_cache,
        offline_models=args.offline_embeddings,
    )
    try:
        config.validate()
    except ValueError as error:
        raise SystemExit(f"Invalid RAG configuration: {error}") from error

    chunks = build_chunks(
        load_corpus(args.corpus_profile),
        chunk_size=config.chunk_size,
        overlap=config.overlap,
    )
    try:
        index = config.build_retriever(chunks)
    except (RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    result = build_grounded_answer(
        args.question,
        index,
        top_k=config.top_k,
        language=config.language,
        max_per_source=config.max_per_source,
        min_evidence_matches=config.min_evidence_matches,
        min_evidence_coverage=config.min_evidence_coverage,
    )

    print("Question:")
    print(args.question)
    print("\nAnswer:")
    print(result.answer)
    print("\nCorpus profile:", args.corpus_profile)
    print("Retrieval mode:", config.retrieval_mode)
    if config.rerank:
        print("Reranker:", config.reranker_model, f"(candidate_k={config.candidate_k})")
    print("\nRefused:", result.refused)
    print("Citations:", ", ".join(result.citations) if result.citations else "None")
    if args.debug and result.trace is not None:
        print("\nTrace:")
        print(json.dumps(result.trace.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
