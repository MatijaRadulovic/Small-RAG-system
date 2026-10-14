# Week 6 RAG Homework Evidence

This evidence reflects the current sentence-aware chunker and BM25 scorer. Re-run the commands below after changing the corpus, chunking settings, or retrieval code because scores and chunk IDs can change.

## 1. Outcome 1: Chunking impact

- **Evaluation:** 42 `(chunk_size, overlap)` configurations, 15 fixed questions, `top_k=3`; relevance is judged by whether the expected source document appears at the specified rank.
- **Config A:** `chunk_size=180`, `overlap=0`
  - Chunks: 16
  - Hit@1: 100%; Hit@3: 100%; MRR: `1.000`
- **Config B:** `chunk_size=100`, `overlap=20`
  - Chunks: 36
  - Hit@1: 93%; Hit@3: 100%; MRR: `0.967`
- **Retrieval difference summary:** Both settings retrieve the expected source within the first three results for every query. `180/0` ranks the expected source first in all 15 cases and creates fewer chunks, so it is the measured default for this corpus and query set. This is not a universal setting; a different corpus or evaluation set may favor other boundaries.
- **Reproduce:** `python3 -m scripts.analyze_chunking`

## 2. Outcome 2: Top-k retrieval

- **Query:** `How do chunk size and overlap improve retrieval quality?`
- **Top-k output:**

| k | Rank | Chunk ID | Score |
| ---: | ---: | --- | ---: |
| 2 | 1 | `doc-2-chunk-2` | 11.124 |
| 2 | 2 | `doc-2-chunk-0` | 2.084 |
| 3 | 1 | `doc-2-chunk-2` | 11.124 |
| 3 | 2 | `doc-2-chunk-0` | 2.084 |
| 3 | 3 | `doc-1-chunk-0` | 1.856 |

- **Why ranking is correct:** The first two passages are from `Chunking and Metadata` and contain the requested chunk size, overlap, and retrieval concepts. The third passage shares fewer relevant terms and scores lower. The checker confirms the requested result count and descending scores.
- **Reproduce:** `python3 -m scripts.check_retrieval`

## 3. Outcome 3: Citation quality

- **Supported query:** `How do citations connect claims to retrieved chunks?`
- **Retrieved chunk IDs:** `doc-5-chunk-0`, `doc-4-chunk-2`
- **Answer with citations:**

```text
Answer:
- Citations connect generated statements to retrieved chunks. Citation identifiers should match the chunk IDs supplied to the answer generator, and the cited text should actually [doc-5-chunk-0]
- A top-k pipeline must return no more than k chunks and order them from highest to lowest similarity. [doc-4-chunk-2]

References:
- [doc-5-chunk-0] Citation Quality
- [doc-4-chunk-2] Retrieval Evaluation
```

- **Verification:** The checker confirms the ordered citation list equals the retrieved chunk IDs. Phase 4 of `improvments_plan.md` will qualify each result independently before answer inclusion; this current output shows why that improvement is needed, since the second passage is related only loosely to the query.
- **Reproduce:** `python3 -m scripts.check_citations`

## 4. Outcome 4: Refusal behavior

- **Unsupported query:** `What color is retrieval on Mars?`
- **Retrieved context:**
  - `doc-1-chunk-0`, score `1.856`: RAG combines retrieval with generation and selects question-relevant passages.
  - `doc-4-chunk-0`, score `1.356`: retrieval evaluation checks whether a relevant source appears among results.
- **Why evidence is insufficient:** The chunks discuss retrieval but contain no information about Mars or a color. The grounding rule requires two matched meaningful terms and at least 50% query-term coverage.
- **Refusal output:** `I do not have enough grounded evidence in the provided corpus to answer that question.`
- **Citations:** None
- **Reproduce:** `python3 -m scripts.check_refusal`

## 5. Outcome 5: Final evidence pair

- **Reproduce both cases:** `python3 -m scripts.generate_evidence`

### Supported case

```text
Question: How do citations connect claims to retrieved chunks?
Retrieved:
- doc-5-chunk-0 | Citation Quality | score=7.600
- doc-4-chunk-2 | Retrieval Evaluation | score=1.966
Citations: doc-5-chunk-0, doc-4-chunk-2
Refused: False
```

The printed answer and full retrieved passages are included in the command output. The evidence check confirms that citations correspond to retrieved IDs.

### Unsupported case

```text
Question: What color is retrieval on Mars?
Retrieved:
- doc-1-chunk-0 | RAG Fundamentals | score=1.856
- doc-4-chunk-0 | Retrieval Evaluation | score=1.356
Answer: I do not have enough grounded evidence in the provided corpus to answer that question.
Citations: None
Refused: True
```

The system refuses because the retrieved text does not answer the color or Mars part of the question.

## Final Verification

```text
./scripts/run_tests.sh
27 passed in 0.19s
```
