import pytest

from src.chunking import Chunk
from src.config import RagConfig
from src.retrieval import BM25Index


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"chunk_size": 0}, "chunk_size"),
        ({"chunk_size": 20, "overlap": 20}, "overlap"),
        ({"top_k": 0}, "top_k"),
        ({"retrieval_mode": "dense-only"}, "retrieval_mode"),
        ({"language": "fr"}, "language"),
        ({"max_per_source": 0}, "max_per_source"),
        ({"min_evidence_matches": 0}, "min_evidence_matches"),
        ({"min_evidence_coverage": 1.5}, "min_evidence_coverage"),
        ({"semantic_weight": float("nan")}, "semantic_weight"),
        ({"rrf_constant": 0}, "rrf_constant"),
        ({"candidate_k": 0}, "candidate_k"),
    ],
)
def test_rag_config_rejects_invalid_settings(updates, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        RagConfig(**updates).validate()


def test_rag_config_builds_default_lexical_retriever() -> None:
    chunk = Chunk(chunk_id="one@0:9", doc_id="one", title="One", text="retrieval")

    index = RagConfig().build_retriever([chunk])

    assert isinstance(index, BM25Index)
    assert index.chunks == (chunk,)
