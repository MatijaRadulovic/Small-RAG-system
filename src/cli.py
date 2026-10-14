import argparse

from .chunking import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP, build_chunks
from .corpus import list_corpus_profiles, load_corpus
from .embeddings import DEFAULT_EMBEDDING_MODEL, FastEmbedProvider
from .hybrid_retrieval import HybridRetriever
from .rag import build_grounded_answer
from .retrieval import BM25Index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Week 6 RAG homework starter")
    parser.add_argument("--question", required=True, help="Question to ask over the local corpus")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Chunk size in characters (default: {DEFAULT_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_OVERLAP,
        help=f"Chunk overlap in characters (default: {DEFAULT_OVERLAP})",
    )
    parser.add_argument("--top-k", type=int, default=2, help="Number of retrieved chunks")
    parser.add_argument(
        "--retrieval-mode",
        choices=("lexical", "hybrid"),
        default="lexical",
        help="Lexical BM25 or local embedding plus BM25 retrieval",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="FastEmbed model ID used with --retrieval-mode hybrid",
    )
    parser.add_argument(
        "--embedding-cache",
        help="Directory for downloaded local model files",
    )
    parser.add_argument(
        "--offline-embeddings",
        action="store_true",
        help="Require the embedding model to already exist in its cache",
    )
    parser.add_argument(
        "--max-per-source",
        type=int,
        default=2,
        help="Maximum retrieved chunks from one source document (default: 2)",
    )
    parser.add_argument(
        "--language",
        choices=("en", "sr", "mixed"),
        default="en",
        help="Language stop-word set: English, Serbian, or combined",
    )
    parser.add_argument(
        "--corpus-profile",
        default="default",
        choices=list_corpus_profiles(),
        help="Corpus profile: default or training-rag",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    corpus = load_corpus(args.corpus_profile)
    chunks = build_chunks(corpus, chunk_size=args.chunk_size, overlap=args.overlap)
    if args.retrieval_mode == "hybrid":
        try:
            provider = FastEmbedProvider(
                args.embedding_model,
                cache_dir=args.embedding_cache,
                local_files_only=args.offline_embeddings,
            )
        except RuntimeError as error:
            raise SystemExit(str(error)) from error
        index = HybridRetriever(chunks, provider, language=args.language)
    else:
        index = BM25Index(chunks, language=args.language)
    result = build_grounded_answer(
        args.question,
        index,
        top_k=args.top_k,
        max_per_source=args.max_per_source,
    )

    print("Question:")
    print(args.question)
    print("\nAnswer:")
    print(result.answer)
    print("\nCorpus profile:", args.corpus_profile)
    print("Retrieval mode:", args.retrieval_mode)
    print("\nRefused:", result.refused)
    print("Citations:", ", ".join(result.citations) if result.citations else "None")


if __name__ == "__main__":
    main()
