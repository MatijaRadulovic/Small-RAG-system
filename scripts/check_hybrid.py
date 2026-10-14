"""Run local semantic probes for paraphrases, exact IDs, and Serbian queries."""

import argparse

from src.chunking import Chunk
from src.embeddings import DEFAULT_EMBEDDING_MODEL, FastEmbedProvider
from src.hybrid_retrieval import HybridRetriever
from src.retrieval import retrieve_top_k


PROBES = (
    ("What conveyance moves a person between places?", "vehicle", "synonym paraphrase"),
    ("How do search engines merge their result orderings?", "fusion", "English paraphrase"),
    ("RAG-v2.1", "identifier", "exact technical identifier"),
    ("Kako se spajaju rangirane liste?", "fusion", "Serbian cross-lingual query"),
    ("Kako does reciprocal rank fusion combine lists?", "fusion", "mixed Serbian/English query"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-cache")
    parser.add_argument("--offline-embeddings", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunks = [
        Chunk(
            chunk_id="vehicle",
            doc_id="doc-1",
            title="Vehicle",
            text="An automobile is a road vehicle used to carry people from place to place.",
        ),
        Chunk(
            chunk_id="fusion",
            doc_id="doc-2",
            title="Search Ranking",
            text="Reciprocal rank fusion combines ranked lists from search systems into one ordered result list.",
        ),
        Chunk(
            chunk_id="identifier",
            doc_id="doc-3",
            title="Release",
            text="The application release identifier is RAG-v2.1 and it preserves exact version information.",
        ),
        Chunk(
            chunk_id="rag",
            doc_id="doc-4",
            title="Retrieval",
            text="Retrieval augmented generation finds evidence passages for a language model.",
        ),
    ]
    provider = FastEmbedProvider(
        args.embedding_model,
        cache_dir=args.embedding_cache,
        local_files_only=args.offline_embeddings,
    )
    retriever = HybridRetriever(chunks, provider)

    print(f"Local model: {provider.model_name}; dimensions: {retriever.semantic_index.dimension}")
    print("case | lexical top | hybrid top | semantic cosine | result")
    for query, expected_id, label in PROBES:
        lexical = retrieve_top_k(query, retriever.lexical_index, k=1)
        hybrid = retriever.retrieve_top_k(query, k=1)
        lexical_id = lexical[0].chunk.chunk_id if lexical else "-"
        hybrid_id = hybrid[0].chunk.chunk_id if hybrid else "-"
        cosine = hybrid[0].semantic_score if hybrid else None
        passed = hybrid_id == expected_id
        print(
            f"{label} | {lexical_id} | {hybrid_id} | "
            f"{cosine:.3f}" if cosine is not None else f"{label} | {lexical_id} | - | -",
            "PASS" if passed else "FAIL",
        )
        assert passed, f"{label}: expected {expected_id}, got {hybrid_id}"
    print("PASS: all semantic retrieval probes returned their labeled chunks.")


if __name__ == "__main__":
    main()
