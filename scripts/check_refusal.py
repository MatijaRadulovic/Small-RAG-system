"""Print and validate Outcome 4 refusal evidence."""

from src.chunking import build_chunks
from src.corpus import load_default_corpus
from src.rag import REFUSAL_MESSAGE, build_grounded_answer
from src.retrieval import BM25Index, retrieve_top_k


QUERY = "What color is retrieval on Mars?"


def main() -> None:
    chunks = build_chunks(load_default_corpus())
    index = BM25Index(chunks)
    retrieved = retrieve_top_k(QUERY, index, k=2)
    result = build_grounded_answer(QUERY, index, top_k=2)

    assert retrieved
    assert result.refused is True
    assert result.answer == REFUSAL_MESSAGE
    assert result.citations == []

    print("OUTCOME 4: REFUSAL CHECK")
    print(f"Unsupported query: {QUERY}")
    print("Retrieved context:")
    for item in retrieved:
        print(f"- {item.chunk.chunk_id} | score={item.score:.3f} | {item.chunk.text}")
    print("\nRefusal output:")
    print(result.answer)
    print("Citations: None")
    print("\nPASS: weak partial overlap is refused without citations.")


if __name__ == "__main__":
    main()
