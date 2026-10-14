# Week 6 RAG Homework Evidence

This evidence reflects the current sentence-aware chunker, reusable BM25 index, and reciprocal rank fusion. Re-run the commands below after changing the corpus, chunking settings, or retrieval code because rankings and chunk IDs can change.

## 1. Outcome 1: Chunking impact

- **Evaluation:** 42 `(chunk_size, overlap)` configurations, 15 fixed questions, `top_k=3`; relevance is judged by whether the expected source document appears at the specified rank.
- **Config A:** `chunk_size=220`, `overlap=0`
  - Chunks: 14
  - Hit@1: 93.3%; Hit@3: 100%; MRR: `0.967`
- **Config B:** `chunk_size=100`, `overlap=20`
  - Chunks: 36
  - Hit@1: 86.7%; Hit@3: 100%; MRR: `0.922`
- **Retrieval difference summary:** Both settings retrieve the expected source within the first three results for every query. `220/0` ranks the expected source first in 14 of 15 cases and creates fewer chunks, so it is the measured default for this corpus and query set. This is not a universal setting; a different corpus or evaluation set may favor other boundaries.
- **Reproduce:** `python3 -m scripts.analyze_chunking`

## 2. Outcome 2: Top-k retrieval

- **Query:** `How do chunk size and overlap improve retrieval quality?`
- **Top-k output:**

| k | Rank | Chunk ID | Score |
| ---: | ---: | --- | ---: |
| 2 | 1 | `doc-2@219:414` | 0.01639 |
| 2 | 2 | `doc-2@0:218` | 0.01613 |
| 3 | 1 | `doc-2@219:414` | 0.01639 |
| 3 | 2 | `doc-2@0:218` | 0.01613 |
| 3 | 3 | `doc-1@0:201` | 0.01587 |

- **Why ranking is correct:** The first two passages are from `Chunking and Metadata` and contain the requested chunk size, overlap, and retrieval concepts. The third passage shares only `retrieval` and ranks lower. The checker confirms the requested result count and descending fused rank scores; component BM25 scores and matched terms appear in its output.
- **Reproduce:** `python3 -m scripts.check_retrieval`

## 3. Outcome 3: Citation quality

- **Supported query:** `How do citations connect claims to retrieved chunks?`
- **Retrieved chunk IDs:** `doc-5@0:195`, `doc-2@415:519`
- **Answer with citations:**

```text
Answer:
- Citations connect generated statements to retrieved chunks. Citation identifiers should match the chunk IDs supplied to the answer generator, and the cited text should actually support the claim. [doc-5@0:195]
- Metadata such as document ID, title, and section supports filtering, citations, and source traceability. [doc-2@415:519]

References:
- [doc-5@0:195] Citation Quality; section Citation Quality; source characters 0-195
- [doc-2@415:519] Chunking and Metadata; section Chunking and Metadata; source characters 415-519
```

- **Verification:** The checker confirms the ordered citation list equals the retrieved chunk IDs. Phase 4 of `improvments_plan.md` will qualify each result independently before answer inclusion; the second passage is related to citation traceability but less directly supports the specific query.
- **Reproduce:** `python3 -m scripts.check_citations`

## 4. Outcome 4: Refusal behavior

- **Unsupported query:** `What color is retrieval on Mars?`
- **Retrieved context:**
  - `doc-1@0:201`, fused score approximately `0.016`: RAG combines retrieval with generation and selects question-relevant passages.
  - `doc-4@0:156`, fused score approximately `0.016`: retrieval evaluation checks whether a relevant source appears among results.
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
- doc-5@0:195 | Citation Quality | fused score≈0.016
- doc-2@415:519 | Chunking and Metadata | fused score≈0.016
Citations: doc-5@0:195, doc-2@415:519
Refused: False
```

The printed answer and full retrieved passages are included in the command output. The evidence check confirms that citations correspond to retrieved IDs.

### Unsupported case

```text
Question: What color is retrieval on Mars?
Retrieved:
- doc-1@0:201 | RAG Fundamentals | fused score≈0.016
- doc-4@0:156 | Retrieval Evaluation | fused score≈0.016
Answer: I do not have enough grounded evidence in the provided corpus to answer that question.
Citations: None
Refused: True
```

The system refuses because the retrieved text does not answer the color or Mars part of the question.

## Final Verification

```text
./scripts/run_tests.sh
39 passed in 0.15s
```
