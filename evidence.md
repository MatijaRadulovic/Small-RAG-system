# Week 6 RAG Homework Evidence

This evidence reflects the current sentence-aware chunker, reusable BM25 index, reciprocal rank fusion, and per-result evidence validation. Re-run the commands below after changing the corpus, chunking settings, or retrieval code because rankings and chunk IDs can change.

## 1. Outcome 1: Chunking impact

- **Evaluation:** 42 `(chunk_size, overlap)` configurations, 10 versioned tuning questions, `top_k=3`; relevance is judged by whether the expected source document appears at the specified rank.
- **Config A:** `chunk_size=220`, `overlap=0`
  - Chunks: 14
  - Hit@1: 100%; Hit@3: 100%; MRR: `1.000`
- **Config B:** `chunk_size=100`, `overlap=20`
  - Chunks: 36
  - Hit@1: 90%; Hit@3: 100%; MRR: `0.950`
- **Retrieval difference summary:** Both settings retrieve the expected source within the first three results for every tuning query. `220/0` ranks the expected source first in all 10 cases and creates fewer chunks, so it is the measured default for this corpus and query set. This is not a universal setting; a different corpus or evaluation set may favor other boundaries.
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
- **Diversity check:** The first two are distinct passages from the same source, so both remain eligible; reported lexical diversity scores are `1.00` and `0.93`. A second result from a source is permitted while duplicates and results beyond the default two-per-source limit are filtered.
- **Reproduce:** `python3 -m scripts.check_retrieval`

## 3. Outcome 3: Citation quality

- **Supported query:** `How do citations connect claims to retrieved chunks?`
- **Retrieved chunk IDs:** `doc-5@0:195`, `doc-2@415:519`
- **Evidence assessments:** `doc-5@0:195` matches 4 of 5 terms (80%, sufficient); `doc-2@415:519` matches 1 of 5 (20%, excluded).
- **Answer with citations:**

```text
Answer (based only on the individually supported passages):
- Citations connect generated statements to retrieved chunks. Citation identifiers should match the chunk IDs supplied to the answer generator, and the cited text should actually support the claim. [doc-5@0:195]

References:
- [doc-5@0:195] Citation Quality; section Citation Quality; source characters 0-195
```

- **Verification:** The checker confirms citations exactly match the independently supported retrieved IDs. Weak candidates are assessed and excluded before answer and reference construction.
- **Reproduce:** `python3 -m scripts.check_citations`

## 4. Outcome 4: Refusal behavior

- **Unsupported query:** `What color is retrieval on Mars?`
- **Retrieved context:**
  - `doc-1@0:201`, fused score approximately `0.016`: RAG combines retrieval with generation and selects question-relevant passages.
  - `doc-4@0:156`, fused score approximately `0.016`: retrieval evaluation checks whether a relevant source appears among results.
- **Why evidence is insufficient:** Both chunks match only `retrieval` (1 of 3 meaningful terms, 33% coverage), below the rule of two matches and 50% coverage. Neither passage mentions Mars or a color.
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
Citations: doc-5@0:195
Refused: False
```

The citation-quality passage passes at 80% query-term coverage. The metadata passage fails at 20% and is omitted from the answer. The command prints full retrieved passages, their evidence assessments, and the filtered answer.

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

## Outcome 5 Verification at Original Submission Stage

```text
./scripts/run_tests.sh
57 passed in 0.12s
```

## RAG Improvement Phase 6: Held-Out Evaluation Baseline

The versioned evaluation dataset is `data/rag_eval_v1.jsonl` (10 tuning and 10 held-out cases). It labels expected source documents and exact evidence phrases, answerability, expected facts, and question type. The acceptance policy and metric definitions are in `docs/evaluation_policy.md`.

Command: `python3 -m scripts.evaluate_rag --split heldout`

```text
RAG evaluation: heldout; 10 cases; chunk=220/0; k=3
Retrieval — Recall@3: 1.000; MRR: 0.929; nDCG@3: 0.947; source-span Recall@3: 0.786; span MRR: 0.714
Citations — precision: 1.000; source-span recall: 0.250 (2 citations; 8 expected spans)
Refusals — precision: 0.375; recall: 1.000
```

Command: `python3 -m scripts.compare_evaluation --split heldout`

```text
                       baseline  candidate  delta
Recall@3                  1.000      0.929  -0.071
MRR                       0.929      0.929  +0.000
nDCG@3                    0.947      0.892  -0.055
source-span Recall@3      0.786      0.429  -0.357
citation precision        1.000      0.500  -0.500
```

The default comparison is `220/0` against `100/20`. Although document Recall@3 remains high, `100/20` fails the source-span and citation regression floors. `python3 -m scripts.analyze_chunking` now uses only the tuning split and selects `220/0` at 100% hit@1, 100% hit@3, and MRR `1.000`; these tuning results are not substituted for the held-out report.

## RAG Improvement Phase 7: Hybrid Retrieval

The optional model is `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, loaded locally through FastEmbed. Install `requirements-embeddings.txt`; the first run downloads the model, and later runs support `--offline-embeddings`.

Command: `python3 -m scripts.compare_evaluation --split heldout --baseline-mode lexical --candidate-mode hybrid --baseline-size 220 --candidate-size 220 --baseline-overlap 0 --candidate-overlap 0`

```text
metric                 lexical    hybrid  delta
Recall@3                 1.000     1.000  +0.000
MRR                      0.929     0.929  +0.000
nDCG@3                   0.947     0.947  +0.000
source-span Recall@3     0.786     0.929  +0.143
span MRR                 0.714     0.762  +0.048
citation precision       1.000     1.000  +0.000
citation recall          0.250     0.250  +0.000
refusal precision        0.375     0.375  +0.000
refusal recall           1.000     1.000  +0.000
```

`python3 -m scripts.check_hybrid --offline-embeddings` passes five local probes: synonym paraphrase, English paraphrase, exact version identifier, Serbian query to English text, and mixed Serbian/English query. In one cached CPU run, `python3 -m scripts.benchmark_hybrid --offline-embeddings --repeats 3` measured `291 ms` index build, `66.68 ms` average warm query and `76.33 ms` p95 over 21 queries. The 14-chunk vector matrix has a 21 KiB float32 lower bound; the cached model uses `240.5 MiB`, process peak RSS was `696.8 MiB`, and Python traced peak was `27.7 MiB`. The model initialization reading was `8.41 s`; runtime measures vary by machine.

Current unit suite after Phase 7: `./scripts/run_tests.sh` reports 57 passed in `0.12s`.

## RAG Improvement Phase 8: Candidate Reranking

The optional English cross-encoder is `Xenova/ms-marco-MiniLM-L-6-v2`. It receives the whole question and each passage from the first-stage candidate set; retrieval results retain both first-stage and reranker scores. The current corpus has 14 chunks, so `--candidate-k 14` evaluates all current chunks before returning a diverse top three. Install `requirements-embeddings.txt`; the initial model load needs a download, and later runs can use the populated cache offline.

Command: `PYTHONPATH=. /tmp/rag-week6-phase7/bin/python -m scripts.compare_evaluation --split heldout --baseline-mode hybrid --candidate-mode hybrid --baseline-size 220 --baseline-overlap 0 --candidate-size 220 --candidate-overlap 0 --candidate-rerank --candidate-k 14 --embedding-cache /tmp/rag-week6-embedding-cache`

```text
metric                 hybrid  hybrid + reranker  delta
recall_at_k             1.000       1.000          +0.000
mrr                     0.929       1.000          +0.071
ndcg_at_k               0.947       1.000          +0.053
span_recall_at_k        0.929       0.929          +0.000
span_mrr                0.762       0.929          +0.167
citation_precision      1.000       1.000          +0.000
citation_recall         0.250       0.250          +0.000
refusal_precision       0.375       0.375          +0.000
refusal_recall          1.000       1.000          +0.000
```

Latency command: `PYTHONPATH=. /tmp/rag-week6-phase7/bin/python -m scripts.benchmark_reranking --candidate-k 14 --cache-dir /tmp/rag-week6-embedding-cache --offline`

```text
heldout_queries=10 candidates_mean=14.0
index_build_ms=272.00
reranker_load_ms=222.09
first_stage_retrieval_ms mean=22.58 p95=36.99
cross_encoder_reranking_ms mean=167.45 p95=188.17
```

The held-out ranking gain supports keeping reranking as an opt-in mode; the added ~167 ms average cross-encoder scoring cost is visible and can be avoided by leaving `--rerank` off. Timings depend on hardware, and ten held-out questions are only a small regression set. `./scripts/run_tests.sh` reports 61 passed in `0.08s`.

## RAG Improvement Phase 9: Grounded Answer Synthesis Boundary

No live LLM provider was invoked. `AnswerGenerator` is an adapter protocol; `build_grounded_answer` retains extractive output when no adapter is supplied and falls back to it when model output fails validation. Prompt instructions and source passages are kept in separate messages; passages are JSON-escaped and marked untrusted. A generated factual sentence is accepted only when it includes known chunk IDs and has at least two or half of its meaningful terms in the cited evidence. References are rendered from the actual chunks.

Verification: `./scripts/run_tests.sh` reports 69 passed in `0.12s`; `python3 -m scripts.evaluate_rag --split heldout` preserves the lexical baseline (`Recall@3=1.000`, `MRR=0.929`, `nDCG@3=0.947`, source-span Recall@3 `0.786`, span MRR `0.714`, citation precision `1.000`, citation recall `0.250`, refusal precision `0.375`, refusal recall `1.000`). Unit cases exercise an injected instruction, contradictory source passages, missing/unknown/malformed citations, unsupported claims, absent evidence, and adapter failure. The overlap check is lexical and cannot establish semantic entailment; it can reject a valid paraphrase or accept an unsupported sentence with shared words.

## RAG Improvement Phase 10: Configuration, Observability, and Cache

Run `python3 -m src.cli --question "What does the generator do with retrieved passages?" --debug` to print a request ID, selected chunk IDs, component scores, stage timings, and refusal status without echoing the query or passages inside the trace. `RagConfig` rejects invalid chunking/retrieval/grounding/model settings before pipeline construction. Hybrid mode stores vectors under `.cache/rag-index`, keyed by model and chunk-content fingerprint; bad entries rebuild and the cache can be used with `--offline-embeddings` after model files are present.

On 14 chunks, a cold vector-cache benchmark took `478.5 ms` to build the semantic index; the next process loaded the cached vectors in `65.1 ms`. Mean query latency was `72.95/73.10 ms`, model initialization was `6572.1/6534.9 ms`, and peak RSS was `697.0/683.5 MiB`. Reproduce using the commands in `docs/rag_retrieval_guide.md` with an empty then populated `--index-cache`. The repository does not contain the planned book corpus, so these measurements describe only the bundled starter corpus. Unit tests cover reuse, invalidation after corpus edits, corrupted-cache recovery, invalid configuration, privacy-safe traces, and refusal reasons.

## RAG Improvement Phase 11: Final Integration

The final clean environment used a fresh `/tmp/rag-week6-phase11` virtual environment with `requirements.txt` and `requirements-embeddings.txt` installed. `python -m pytest -q` reports 85 passed in `0.23s`; `python -m scripts.evaluate_rag --split heldout` reproduces the Phase 6 lexical baseline.

On the same 10 held-out cases, the final hybrid + reranker configuration compared with lexical `220/0` produces:

| Metric | Lexical baseline | Hybrid + reranker | Change |
| --- | ---: | ---: | ---: |
| Recall@3 | 1.000 | 1.000 | +0.000 |
| MRR | 0.929 | 1.000 | +0.071 |
| nDCG@3 | 0.947 | 1.000 | +0.053 |
| source-span Recall@3 | 0.786 | 0.929 | +0.143 |
| source-span MRR | 0.714 | 0.929 | +0.214 |
| citation precision | 1.000 | 1.000 | +0.000 |
| citation span recall | 0.250 | 0.250 | +0.000 |
| refusal precision | 0.375 | 0.375 | +0.000 |
| refusal recall | 1.000 | 1.000 | +0.000 |

The hybrid probes pass synonym, paraphrase, identifier, Serbian-to-English, and mixed-language retrieval checks. The supported CLI demo returns citation `doc-1@202:370`; the unsupported demo refuses without citations. A Serbian-only query can retrieve semantically but is then refused by the current lexical evidence gate because no meaningful query terms overlap the English passage. Prompt-injection and conflicting-source behavior are covered by tests using a deterministic generator adapter; no actual LLM backend is configured. This is an intentionally strict refusal, and multilingual answer grounding remains a known limitation.

Chunking comparison: Phase 0 used 180/0 with 15 questions and reported 100% Hit@1/Hit@3 and MRR `1.000`; the final tuning split uses 220/0 with 10 versioned questions and also reports 100% Hit@1/Hit@3 and MRR `1.000` (14 chunks rather than 16). These are different evaluation sets, so the matching values do not establish a direct improvement. Phase 6's held-out lexical metrics and the final held-out hybrid/reranker metrics above are the comparable retrieval evaluation.

The content-aware chunker and book-corpus benchmark remain follow-up work because this repository contains no book corpus. The current evaluation is a small regression set rather than a broad quality guarantee; lexical claim overlap is not semantic entailment.
