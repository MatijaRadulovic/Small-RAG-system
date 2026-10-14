# Week 6 RAG Homework Evidence

## 1. Outcome 1: Chunking impact

- **Query:** `How do chunk size and overlap affect retrieval quality?`
- **Config A:** `chunk_size=100`, `overlap=20`
  - Corpus chunks: 31
  - Expected source: `doc-2`
  - Top result: `doc-2-chunk-4` (`0.464`)
  - Aggregate result over three queries: 100% hit rate, `1.000` MRR, `0.465` mean top score.
- **Config B:** `chunk_size=220`, `overlap=50`
  - Corpus chunks: 14
  - Expected source: `doc-2`
  - Top result: `doc-2-chunk-2` (`0.388`)
  - Aggregate result over three queries: 100% hit rate, `1.000` MRR, `0.335` mean top score.
- **Retrieval difference summary:** Both configurations ranked every expected document first. Config A produced more focused passages and the strongest average top score. Config B created fewer, broader passages, but unrelated terms diluted similarity. `100/20` was selected because its overlap protects boundary context and it performed best after the scorer and corpus improvements.
- **Reproduce:** `python3 -m scripts.analyze_chunking`

## 2. Outcome 2: Top-k retrieval

- **Query:** `How do chunk size and overlap improve retrieval quality?`
- **Top-k output:**

| k | Rank | Chunk ID | Score |
| ---: | ---: | --- | ---: |
| 2 | 1 | `doc-2-chunk-4` | 0.556 |
| 2 | 2 | `doc-2-chunk-2` | 0.192 |
| 3 | 1 | `doc-2-chunk-4` | 0.556 |
| 3 | 2 | `doc-2-chunk-2` | 0.192 |
| 3 | 3 | `doc-5-chunk-1` | 0.153 |

- **Why ranking is correct:** The two highest chunks come from the chunking document and contain the query concepts `chunk`, `size`, `overlap`, and `retrieval quality`. The third chunk shares only the weaker term `chunk`, so it receives a lower score. The checker confirmed exact result limits, descending scores, and the expected source at rank 1.
- **Reproduce:** `python3 -m scripts.check_retrieval`

## 3. Outcome 3: Citation quality

Pending implementation and verification.

## 4. Outcome 4: Refusal behavior

Pending implementation and verification.

## 5. Outcome 5: Final evidence pair

Pending implementation and verification.
