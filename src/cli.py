import argparse

from .chunking import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP, build_chunks
from .corpus import list_corpus_profiles, load_corpus
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
    index = BM25Index(chunks, language=args.language)
    result = build_grounded_answer(
        args.question,
        index,
        top_k=args.top_k,
    )

    print("Question:")
    print(args.question)
    print("\nAnswer:")
    print(result.answer)
    print("\nCorpus profile:", args.corpus_profile)
    print("\nRefused:", result.refused)
    print("Citations:", ", ".join(result.citations) if result.citations else "None")


if __name__ == "__main__":
    main()
