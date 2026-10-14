import pytest

from src.chunking import Chunk
from src.embeddings import Embedding, EmbeddingProvider
from src.hybrid_retrieval import HybridRetriever, SemanticIndex


class FakeEmbeddingProvider:
    model_name = "fake-semantic-test-model"

    @staticmethod
    def _vector(text: str) -> Embedding:
        normalized = text.casefold()
        if "car" in normalized or "automobile" in normalized or "vehicle" in normalized:
            return (1.0, 0.0, 0.0)
        if "rank" in normalized or "fusion" in normalized:
            return (0.0, 1.0, 0.0)
        return (0.0, 0.0, 1.0)

    def embed_passages(self, texts: list[str]) -> list[Embedding]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> Embedding:
        return self._vector(text)


def test_semantic_index_finds_synonym_without_shared_lexical_terms() -> None:
    chunks = [
        Chunk(
            chunk_id="car",
            doc_id="cars",
            title="Transport",
            text="An automobile is a road vehicle for personal transport.",
        ),
        Chunk(
            chunk_id="weather",
            doc_id="weather",
            title="Weather",
            text="Rain and wind are expected tomorrow.",
        ),
    ]
    index = SemanticIndex(chunks, FakeEmbeddingProvider())

    results = index.search("car", k=1)

    assert results[0][0].chunk_id == "car"
    assert results[0][1] == pytest.approx(1.0)


def test_hybrid_fusion_recovers_semantic_match_and_keeps_component_details() -> None:
    chunks = [
        Chunk(
            chunk_id="car",
            doc_id="cars",
            title="Transport",
            text="An automobile is a road vehicle for personal transport.",
        ),
        Chunk(
            chunk_id="exact-noise",
            doc_id="noise",
            title="Car database",
            text="A car catalog stores a car make and model.",
        ),
        Chunk(
            chunk_id="other",
            doc_id="other",
            title="Other",
            text="Rain and wind are expected tomorrow.",
        ),
    ]
    retriever = HybridRetriever(chunks, FakeEmbeddingProvider())

    results = retriever.retrieve_top_k("car", k=3)

    by_id = {item.chunk.chunk_id: item for item in results}
    assert by_id["car"].semantic_rank == 1
    assert by_id["car"].lexical_rank is None
    assert by_id["car"].semantic_score == pytest.approx(1.0)
    assert by_id["exact-noise"].lexical_rank == 1
    assert by_id["exact-noise"].content_score > 0
    assert results == sorted(results, key=lambda item: (-item.score, item.chunk.chunk_id))


def test_hybrid_preserves_exact_identifier_match() -> None:
    chunks = [
        Chunk(chunk_id="exact", doc_id="d1", title="API", text="RAG-v2.1 supports exact identifiers."),
        Chunk(chunk_id="semantic", doc_id="d2", title="Topic", text="Document retrieval can be improved."),
    ]
    retriever = HybridRetriever(chunks, FakeEmbeddingProvider())

    results = retriever.retrieve_top_k("RAG-v2.1", k=2)

    assert results[0].chunk.chunk_id == "exact"
    assert results[0].lexical_rank == 1


def test_semantic_index_checks_vector_count_and_dimensions() -> None:
    class WrongSizeProvider(FakeEmbeddingProvider):
        def embed_passages(self, texts: list[str]) -> list[Embedding]:
            return [(1.0,)] * len(texts)

    chunks = [Chunk(chunk_id="a", doc_id="a", title="A", text="a car")]
    index = SemanticIndex(chunks, WrongSizeProvider())
    with pytest.raises(ValueError, match="dimensions do not match"):
        index.search("car", k=1)

    class MissingPassageProvider(FakeEmbeddingProvider):
        def embed_passages(self, texts: list[str]) -> list[Embedding]:
            return []

    with pytest.raises(ValueError, match="wrong number"):
        SemanticIndex(chunks, MissingPassageProvider())

    class ZeroProvider(FakeEmbeddingProvider):
        def embed_passages(self, texts: list[str]) -> list[Embedding]:
            return [(0.0, 0.0, 0.0)] * len(texts)

    with pytest.raises(ValueError, match="non-zero"):
        SemanticIndex(chunks, ZeroProvider())


def test_hybrid_configuration_is_validated() -> None:
    chunk = Chunk(chunk_id="a", doc_id="a", title="A", text="retrieval")
    with pytest.raises(ValueError, match="semantic_weight"):
        HybridRetriever([chunk], FakeEmbeddingProvider(), semantic_weight=-1)
    with pytest.raises(ValueError, match="semantic_weight"):
        HybridRetriever([chunk], FakeEmbeddingProvider(), semantic_weight=float("nan"))
    with pytest.raises(ValueError, match="rrf_constant"):
        HybridRetriever([chunk], FakeEmbeddingProvider(), rrf_constant=0)


def test_provider_protocol_is_structural() -> None:
    provider: EmbeddingProvider = FakeEmbeddingProvider()
    assert provider.model_name == "fake-semantic-test-model"
