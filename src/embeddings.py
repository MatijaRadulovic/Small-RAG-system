"""Optional local sentence-embedding providers."""

from collections.abc import Sequence
import math
from typing import Protocol, TypeAlias


Embedding: TypeAlias = tuple[float, ...]
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class EmbeddingProvider(Protocol):
    """Minimal interface so retrieval is not tied to one model/runtime."""

    model_name: str

    def embed_passages(self, texts: Sequence[str]) -> list[Embedding]: ...

    def embed_query(self, text: str) -> Embedding: ...


def _to_embedding(values: Sequence[float]) -> Embedding:
    vector = tuple(float(value) for value in values)
    if not vector or not all(math.isfinite(value) for value in vector):
        raise ValueError("embedding vectors must contain finite values")
    if not any(value != 0 for value in vector):
        raise ValueError("embedding vectors cannot be all zeros")
    return vector


class FastEmbedProvider:
    """Local ONNX inference through the optional FastEmbed package."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        *,
        cache_dir: str | None = None,
        local_files_only: bool = False,
        threads: int | None = None,
    ) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as error:
            raise RuntimeError(
                "Hybrid retrieval needs the optional local runtime. "
                "Install it with `python3 -m pip install -r requirements-embeddings.txt`."
            ) from error

        self.model_name = model_name
        try:
            self._model = TextEmbedding(
                model_name=model_name,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                threads=threads,
            )
        except Exception as error:
            mode = "offline cache" if local_files_only else "local cache or model download"
            raise RuntimeError(
                f"Could not load embedding model {model_name!r} from the {mode}. "
                "Check the model name, network access, and cache path."
            ) from error

    def embed_passages(self, texts: Sequence[str]) -> list[Embedding]:
        if not texts:
            return []
        return [_to_embedding(values) for values in self._model.passage_embed(list(texts))]

    def embed_query(self, text: str) -> Embedding:
        vectors = list(self._model.query_embed([text]))
        if len(vectors) != 1:
            raise RuntimeError("embedding runtime returned an unexpected query count")
        return _to_embedding(vectors[0])
