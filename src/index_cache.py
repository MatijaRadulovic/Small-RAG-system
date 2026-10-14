"""Small, versioned JSON cache for local semantic index vectors."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import warnings
from collections.abc import Sequence

from .chunking import Chunk
from .embeddings import Embedding


CACHE_SCHEMA_VERSION = 1


def _corpus_version(chunks: Sequence[Chunk]) -> str:
    content = json.dumps(
        [(chunk.chunk_id, chunk.title, chunk.text) for chunk in chunks],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _cache_path(cache_dir: Path, model_name: str, corpus_version: str) -> Path:
    model_version = hashlib.sha256(model_name.encode("utf-8")).hexdigest()
    return cache_dir / f"semantic-v{CACHE_SCHEMA_VERSION}-{model_version}-{corpus_version}.json"


def load_vectors(
    cache_dir: str | Path,
    model_name: str,
    chunks: Sequence[Chunk],
) -> tuple[list[Embedding] | None, str]:
    """Return cached vectors and hit/miss/corrupt status; invalid cache never blocks use."""
    corpus_version = _corpus_version(chunks)
    path = _cache_path(Path(cache_dir), model_name, corpus_version)
    if not path.exists():
        return None, "miss"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("schema_version") != CACHE_SCHEMA_VERSION
            or payload.get("model_name") != model_name
            or payload.get("corpus_version") != corpus_version
            or payload.get("chunk_ids") != [chunk.chunk_id for chunk in chunks]
        ):
            return None, "corrupt"
        raw_vectors = payload["vectors"]
        if not isinstance(raw_vectors, list) or len(raw_vectors) != len(chunks):
            return None, "corrupt"
        vectors = [tuple(float(value) for value in vector) for vector in raw_vectors]
        if not vectors or not vectors[0] or any(
            len(vector) != len(vectors[0])
            or not all(value == value and abs(value) != float("inf") for value in vector)
            or not any(value != 0 for value in vector)
            for vector in vectors
        ):
            return None, "corrupt"
        return vectors, "hit"
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        warnings.warn(f"Ignoring invalid semantic index cache {path}: {error}", RuntimeWarning)
        return None, "corrupt"


def store_vectors(
    cache_dir: str | Path,
    model_name: str,
    chunks: Sequence[Chunk],
    vectors: Sequence[Embedding],
) -> None:
    """Write a cache atomically; inability to cache does not stop retrieval."""
    directory = Path(cache_dir)
    corpus_version = _corpus_version(chunks)
    path = _cache_path(directory, model_name, corpus_version)
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "model_name": model_name,
        "corpus_version": corpus_version,
        "chunk_ids": [chunk.chunk_id for chunk in chunks],
        "vectors": vectors,
    }
    temporary_path: Path | None = None
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=directory, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(payload, temporary, separators=(",", ":"))
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path.replace(path)
    except OSError as error:
        warnings.warn(f"Could not write semantic index cache {path}: {error}", RuntimeWarning)
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
