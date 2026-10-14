# Homework Work Log

## Scope and Detailed Plan

The attached homework brief defines five outcomes. Outcomes 1 and 2 are complete. Outcomes 3–5 will be implemented and committed separately, with tests and recorded evidence after each outcome.

### Outcome 1: Chunking impact

- [x] Read the homework brief, README, checklist, source modules, and existing tests.
- [x] Identify representative queries with known source documents.
- [x] Add a repeatable analysis script covering several chunk sizes and overlaps.
- [x] Run the baseline analysis and record the metrics and retrieved chunk IDs.
- [x] Compare configurations using hit rate, mean reciprocal rank (MRR), score, and chunk count.
- [x] Select measured defaults and update both chunking functions and CLI arguments.
- [x] Re-run the analysis and explain the observed chunking trade-off.

### Outcome 2: Top-k retrieval

- [x] Extend the default corpus with longer, topic-specific documents.
- [x] Replace the length-biased lexical score with a clearer similarity calculation.
- [x] Ensure results are filtered and sorted from highest to lowest relevance.
- [x] Add checks for scorer behavior, ordering, and the `k` result limit.
- [x] Capture reproducible `top_k=2` and `top_k=3` evidence.
- [x] Run the complete test suite and record the result.

### Outcome 3: Citation quality

- [x] Format supported answers with a clearly separated answer body and References section.
- [x] Include explicit chunk IDs next to every supporting statement.
- [x] Ensure every citation belongs to the exact retrieved result set.
- [x] Add tests for citation presence, reference formatting, and ID correspondence.
- [x] Run the full test suite and a reproducible supported example.
- [x] Record the command, retrieved IDs, answer, and result in `evidence.md` and this log.
- [x] Commit the completed Outcome 3 implementation and evidence.

### Outcome 4: Refusal behavior

- [x] Define a clear evidence-sufficiency rule for weak or unrelated retrieval results.
- [x] Keep refusals free of citations and unsupported claims.
- [x] Add tests for unrelated questions and weak one-term overlap.
- [x] Add a reproducible check that prints retrieved context and the refusal output.
- [x] Run the full test suite and record the result.
- [x] Record the unsupported query, context, and refusal in `evidence.md` and this log.
- [x] Commit the completed Outcome 4 implementation and evidence.

### Outcome 5: Final evidence pair

- [x] Add one command that reproduces a complete supported and unsupported case.
- [x] Show each question, retrieved chunks and scores, final answer, citations, and refusal flag.
- [x] Confirm the supported answer cites only retrieved chunks.
- [x] Confirm the unsupported case refuses with no citations.
- [x] Run the evidence command and full test suite.
- [x] Complete all five sections of `evidence.md` and update the submission checklist status.
- [x] Commit the completed Outcome 5 evidence bundle.

### Submission details

- [x] Record pair contributions and the demo lead (requires student names and assignment).

## Step 1: Baseline and Chunking Experiment

### Why

The original defaults were `chunk_size=140` and `overlap=30`, but the repository contained no measured comparison. A useful decision needs repeatable queries whose expected source documents are known.

### Change

Added `scripts/analyze_chunking.py`. It tests five `(chunk_size, overlap)` pairs against three characteristic questions for RAG fundamentals, chunking, and refusal behavior. It prints every ranked result plus aggregate hit rate, MRR, mean top score, and chunk count.

### Command

```bash
python3 -m scripts.analyze_chunking
```

### Result

All five configurations achieved a 100% hit rate and `1.000` MRR: every expected document ranked first. The aggregate baseline was:

| Chunk size | Overlap | Chunks | Mean top score |
| ---: | ---: | ---: | ---: |
| 80 | 0 | 6 | 0.444 |
| 100 | 20 | 6 | 0.408 |
| 140 | 30 | 6 | 0.315 |
| 180 | 40 | 3 | 0.306 |
| 220 | 50 | 3 | 0.306 |

The `80/0` configuration had the largest score, partly because the original scorer divided matches by chunk length and therefore favored short chunks. Since all configurations had the same ranking quality, `100/20` was selected: it kept the next strongest score while preserving 20 characters across boundaries. Constants in `src/chunking.py` now define these defaults, and `src/cli.py` imports the same constants so the two interfaces cannot drift.

The first attempted command, `.venv/bin/python -m scripts.analyze_chunking`, failed because this copied virtual environment points to a macOS Python path that does not exist on this Linux machine. The successful command was:

```bash
python3 -m scripts.analyze_chunking
```

## Step 2: Measured Defaults

### Why

The baseline showed equal first-place accuracy for every candidate, so score alone was not enough. Zero overlap also risks splitting a useful phrase at a chunk boundary.

### Change

Changed the defaults from `140/30` to `100/20`. Added `DEFAULT_CHUNK_SIZE` and `DEFAULT_OVERLAP` in `src/chunking.py`; both chunking functions and the CLI now share them.

### Result

After the corpus and scorer improvements, a second analysis confirmed the choice. All configurations still achieved `1.000` MRR, while `100/20` produced the highest mean top score (`0.465`). It created 31 focused chunks from the longer corpus. The runner recommends `100/20` using MRR first, then boundary-preserving overlap, mean score, and chunk count.

## Step 3: Longer Evaluation Corpus

### Why

The original three documents were about 140 characters each, too short to show meaningful differences among larger chunk settings.

### Change

Expanded the three original sources into multi-sentence documents and added `doc-4` (retrieval evaluation) and `doc-5` (citation quality). The default corpus now has five documents, each at least 300 characters. The added topics also create plausible competing chunks instead of making every query trivial.

### Result

The experiment now produces between 14 and 32 chunks depending on configuration. The expected source remains rank 1 for all three characteristic queries.

## Step 4: Similarity Scoring

### Why

The old score counted every matching token and divided by chunk length. Repeated terms could inflate relevance, common question words could create false matches, and short chunks received an unfair advantage.

### Change

`src/retrieval.py` now:

1. removes a small set of English stop words;
2. compares unique query and chunk terms using binary cosine similarity;
3. combines 85% text similarity with a 15% title signal;
4. filters by minimum score and sorts by descending score, then chunk ID for deterministic ties.

For query term set `Q` and chunk term set `C`, the core score is `|Q ∩ C| / sqrt(|Q| × |C|)`. This stays between 0 and 1 and rewards shared meaningful terms without counting repetition multiple times.

## Step 5: Outcome 2 Check

### Command

```bash
python3 -m scripts.check_retrieval
```

### Result

Query: `How do chunk size and overlap improve retrieval quality?`

| k | Rank | Chunk | Score |
| ---: | ---: | --- | ---: |
| 2 | 1 | `doc-2-chunk-4` | 0.556 |
| 2 | 2 | `doc-2-chunk-2` | 0.192 |
| 3 | 1 | `doc-2-chunk-4` | 0.556 |
| 3 | 2 | `doc-2-chunk-2` | 0.192 |
| 3 | 3 | `doc-5-chunk-1` | 0.153 |

The script asserted that each result count equals `k`, scores are descending, and the expected chunking document ranks first. It ended with `PASS`.

## Step 6: Verification

Added tests for meaningful-term similarity, the top-k limit and ordering, title relevance, and the longer corpus. Existing citation and refusal tests were retained.

Commands and results:

```bash
python3 -m pytest -q
# 13 passed in 0.06s

python3 -m src.cli --question "How do chunk size and overlap improve retrieval quality?" --top-k 3
# Grounded response; citations begin with doc-2-chunk-4 and doc-2-chunk-2.

python3 -m src.cli --question "Who was emperor of Mars?"
# Refused: True; Citations: None
```

The first attempt to run `./scripts/run_tests.sh` failed because the script called `python`, which is unavailable in this environment. The script was corrected to call `python3`, matching the README setup command, and was run again as the final suite check.

```bash
./scripts/run_tests.sh
# 13 passed in 0.04s
```

## Step 7: Outcome 3 Citation Quality

### Why

The earlier answer placed chunk IDs before raw passages, but it did not clearly separate the answer body from its references. The homework requires explicit citations that can be checked against retrieval output.

### Change

Supported answers now contain an `Answer` section where each evidence statement ends with `[chunk-id]`, followed by a separate `References` section mapping every ID to its document title. The citation list is built from the same retrieval results. `scripts/check_citations.py` independently retrieves the expected chunks and checks exact list equality, formatting, and citation occurrence.

### Commands and result

```bash
python3 -m scripts.check_citations
# Retrieved: doc-5-chunk-0, doc-2-chunk-6
# PASS: answer citations exactly match the retrieved chunk IDs.

./scripts/run_tests.sh
# 14 passed in 0.06s
```

The supported answer cited each retrieved chunk once in its body and once in References. No non-retrieved ID appeared.

## Step 8: Outcome 4 Refusal Behavior

### Why

The earlier implementation answered whenever the best score exceeded `0.02`. An unrelated question with one shared word could pass that threshold and produce an unsupported response.

### Change

Added an explicit sufficiency check. The best result must score at least `0.15` and match at least two meaningful query terms. A query containing only one meaningful term requires that one term. If this rule fails, the system returns the consistent refusal message with an empty citation list. `scripts/check_refusal.py` demonstrates that retrieval can return weak context while the answer layer still refuses.

### Commands and result

```bash
python3 -m scripts.check_refusal
# Retrieved weak matches: doc-4-chunk-0 (0.203), doc-1-chunk-0 (0.174)
# PASS: weak partial overlap is refused without citations.

./scripts/run_tests.sh
# 15 passed in 0.06s

python3 -m scripts.check_citations
# PASS: the supported citation case still works after the stricter rule.
```

## Step 9: Outcome 5 Final Evidence Pair

### Why

The final submission needs one supported and one unsupported case with complete, reproducible retrieval and answer output. Separate commands can make it harder to prove that both cases use the same final implementation.

### Change

Added `scripts/generate_evidence.py`. One execution builds the final corpus and chunks, runs both queries, prints every retrieved chunk with score and text, prints each final answer and citation list, and displays the refusal flag. Assertions confirm that the supported citations exactly equal the retrieved IDs and that the unsupported case refuses without citations.

### Commands and result

```bash
python3 -m scripts.generate_evidence
# PASS: supported and unsupported evidence satisfy the homework checks.

./scripts/run_tests.sh
# 15 passed in 0.05s
```

All individual evidence commands were also rerun. Outcomes 1–5 passed. Pair contribution names and the designated demo lead remain open because that information was not provided.

## Phase 0: BM25 and Boundary-Aware Baseline

### What changed

Added focused tests for sentence and word boundary chunking, empty and long-word inputs, overlap edge cases, BM25 rare-term weighting, repeated-term saturation, length normalization, and empty or unrelated searches. Re-ran the full experiment and all evidence scripts with the newer BM25 scorer.

### Why

The earlier evidence used character slicing and cosine scores, so its scores and chunk IDs no longer described the current implementation. BM25 scores are not on the same scale as cosine similarity, so retaining the prior raw-score refusal threshold would have made grounding decisions arbitrary.

### Decision and verification

The 42-configuration, 15-query experiment selected `chunk_size=180`, `overlap=0`: hit@1 `100%`, hit@3 `100%`, MRR `1.000`, 16 chunks. The former `100/20` configuration scored hit@1 `93%`, hit@3 `100%`, MRR `0.967`, with 36 chunks. Grounding now requires at least two matched meaningful terms and coverage of at least half the query terms (one-term queries need their sole term). This avoids dependence on BM25's corpus-specific numerical scale.

Commands:

```bash
./scripts/run_tests.sh
# 27 passed in 0.19s
python3 -m scripts.analyze_chunking
# recommended 180/0; hit@1=100%, hit@3=100%, MRR=1.000, chunks=16
python3 -m scripts.check_retrieval
python3 -m scripts.check_citations
python3 -m scripts.check_refusal
python3 -m scripts.generate_evidence
python3 -m src.cli --question "What is retrieval augmented generation?"
```

All five evidence scripts passed. The evidence file was refreshed with the current scores and chunk IDs. The independent phases that follow will address per-result qualification and diversity of answer context.

## Phase 1: Unicode and Language-Aware Tokenization

### What changed

Tokenization now applies Unicode NFKC normalization and `casefold()`, recognizes Latin and Cyrillic alphanumeric terms, and preserves identifiers such as `RAG-v2.1`. English, Serbian Latin/Cyrillic, and mixed stop-word sets are selectable with `--language en|sr|mixed`; retrieval and grounding use the same setting.

### Why

The earlier ASCII-only expression dropped Serbian letters and could not support a Serbian corpus. Case normalization also needed to handle decomposed accent characters consistently.

### Decision and verification

Unicode and Serbian retrieval tests pass. Stemming/lemmatization was deferred: the sample corpus is English, so it does not provide a labeled Serbian set to measure morphological matching without risking false conflation. The 15-query English chunking evaluation remains at 100% hit@1 and MRR `1.000`; unsupported evidence still refuses. `./scripts/run_tests.sh` reports 32 passed. The Serbian retrieval unit case uses a constructed Cyrillic chunk because the shipped sample corpus has no Serbian content.

## Phase 2: Reusable BM25 Index and Field Ranking

### What changed

Added `BM25Index`, which tokenizes content and title fields and stores their document frequencies and average lengths once. The CLI and multi-query scripts now reuse one index. Content and title are scored separately and combined by reciprocal rank fusion (RRF), not by adding scores from different scales. `RetrievalResult` now exposes content score, title score, and matched terms; retrieval validates BM25 parameters, `k`, score filters, and title weight. Added a benchmark for build latency, repeated-query latency, vocabulary, indexed terms, and peak traced allocation.

### Why

The previous implementation rebuilt corpus statistics for every query and directly added BM25 to cosine title similarity. Those scores do not share a meaningful scale. RRF combines their rankings, while a small title weight (`0.01`) lets titles help with close comparisons without overpowering a much stronger passage match.

### Decision and verification

The updated 42-configuration evaluation recommends `220/0` for the current corpus: 93.3% hit@1, 100% hit@3, MRR `0.967`, and 14 chunks. The former `100/20` setting has 86.7% hit@1 and MRR `0.922` with 36 chunks under RRF. Benchmark readings are machine-dependent; one run recorded a 12.433 ms index build, 0.289 ms mean repeated query, and 40.0 KiB peak traced allocation. `./scripts/run_tests.sh` reports 35 passed. Top-k, citation, refusal, and final evidence scripts pass with current chunk IDs and RRF outputs.

## Phase 3: Traceable Chunk Metadata and Stable IDs

### What changed

Chunks now store source ID, section, and exact start/end offsets into the original text. IDs use `source_id@start:end`, so they do not depend on chunk list order. Added `extract_source_span()` to verify a cited passage against its original source. References now include section and character offsets.

### Why

An ID alone was not enough to locate or verify source material. Character offsets let a reader inspect the original passage and make citation checks independent of displayed chunk order. The splitter now preserves source whitespace inside a chunk so offsets point to the exact original text.

### Decision and verification

Offsets are zero-based Python Unicode code-point indices with an exclusive end. Tests cover Unicode text, exact source extraction, repeated passages, and stable IDs when an unrelated document is added. Editing text before a span can change its ID and requires regenerating citations. `./scripts/run_tests.sh` reports 39 passed; citation and final-evidence checks pass with IDs such as `doc-5@0:195`.

## Phase 4: Per-Result Evidence Validation

Added `EvidenceAssessment` to report each retrieved passage's fused score, matched meaningful terms, term coverage, pass/fail result, and reason. The answer builder now checks every candidate against passage text, excludes weak passages from the answer and References, refuses if all candidates fail, and labels partial answers to show their restricted evidence scope. The citation demo now prints assessments alongside retrieved chunks.

Why: retrieval returns the closest candidates even for unsupported questions, and a strong top result does not make lower-ranked passages trustworthy. Requiring every cited chunk to pass prevents a weak passage from gaining credibility just by appearing in top-k. Checking body text also prevents a matching title from being mistaken for supporting evidence.

Tests cover a strong first result with two weak lower-ranked results, title-only overlap, full rejection, and assessment details. `./scripts/run_tests.sh` reports 42 passed. `python3 -m scripts.check_citations`, `python3 -m scripts.check_refusal`, and `python3 -m scripts.generate_evidence` all pass. For the supported query, `doc-5@0:195` passes with 80% coverage; `doc-2@415:519` is excluded at 20%. The unsupported query's candidates each match only one of three terms (33%), so it is refused with no citations. Updated `evidence.md`, `README.md`, and the beginner guide to match this behavior.

## Phase 5: Duplicate Control and Result Diversity

Retrieval now collapses exact duplicate text and near-duplicate token sets from the same source (Jaccard threshold `0.85`) and limits results to two chunks per source by default. Callers can configure the cap or disable it with `max_per_source=None`; the CLI exposes `--max-per-source`. Each result reports lexical `redundancy` and `diversity_score` alongside its unchanged BM25/RRF relevance score. The filter still allows two different passages from one document.

Why: overlapping or repeated passages can occupy top-k slots without adding evidence. The source cap gives multi-document questions a chance to retrieve other sources while keeping a small allowance for separate facts in a long document. This is a lexical duplicate-control method; semantic MMR is deferred until an embedding representation exists.

Verification: added tests for exact duplicates, near duplicates, source limits, distinct same-source passages, and invalid settings. `./scripts/run_tests.sh` reports 46 passed. `python3 -m scripts.analyze_chunking` still measures `220/0` at hit@1 93.3%, hit@3 100%, MRR `0.967`; the tested `100/20` configuration remains hit@3 100%, so the source/duplicate filter did not lower measured Recall@3. `python3 -m scripts.check_retrieval`, `python3 -m scripts.check_citations`, `python3 -m scripts.generate_evidence`, and `python3 -m scripts.check_refusal` pass. The CLI accepts the source-limit option; an unsupported query correctly still refuses. The checker displays score plus diversity and redundancy.

## Phase 6: Labeled Retrieval and Grounding Evaluation

Added the versioned dataset `data/rag_eval_v1.jsonl` with ten tuning and ten held-out cases. Labels include relevance-graded documents, concise exact source phrases, answerability, expected facts, categories, and notes. Refactored the chunking grid to use only tuning cases and added `scripts.evaluate_rag` for document Recall@k, MRR, nDCG, exact source-span recall/MRR, citation precision/recall, and refusal precision/recall. Added `scripts.compare_evaluation` for a same-split before/after table.

Why: one hand-picked query cannot reveal whether changes help paraphrases, multi-document questions, weak overlap, or unsupported requests. Document-level recall alone also hid cases where the right source ranked but the evidence passage did not. The held-out source-span and citation metrics check those finer outcomes. Regression floors and metric definitions are recorded in `docs/evaluation_policy.md`.

Commands and results:

```text
./scripts/run_tests.sh
51 passed in 0.07s
python3 -m scripts.analyze_chunking
10 tuning queries; selected 220/0; hit@1=100%, hit@3=100%, MRR=1.000 (14 chunks)
python3 -m scripts.evaluate_rag --split heldout
Recall@3=1.000; MRR=0.929; nDCG@3=0.947; span Recall@3=0.786; span MRR=0.714
Citation precision=1.000; citation span recall=0.250; refusal precision=0.375; refusal recall=1.000
python3 -m scripts.compare_evaluation --split heldout
220/0 vs 100/20: doc Recall@3 1.000 vs 0.929; span Recall@3 0.786 vs 0.429; citation precision 1.000 vs 0.500
```

The comparison shows why passage-level checks matter: `100/20` remains close on document rankings but fails the held-out source-span and citation floors. `evidence.md`, the README, and the retrieval guide now point to the dataset and reproducible evaluation commands. This small English corpus is a regression set, not a statistical quality guarantee; multilingual questions await a Serbian evaluation corpus.

## Phase 7: Local Hybrid Lexical and Semantic Retrieval

Added an `EmbeddingProvider` interface and a lazy `FastEmbedProvider`, then built `SemanticIndex` for cosine ranking and `HybridRetriever` for BM25/semantic reciprocal-rank fusion. Semantic vectors are generated once per chunk and reused for the lifetime of the index. CLI `--retrieval-mode lexical|hybrid` keeps BM25 as the default and fallback; hybrid mode can select an embedding cache and require offline-only model files. Added optional dependency manifest `requirements-embeddings.txt`, evaluation support for both modes, retrieval probes, and a local benchmark.

Why: BM25 is strong on exact terms but misses paraphrases and cross-language meanings. The local model is `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`: the model card lists 50 languages and Apache 2.0; FastEmbed lists 384 dimensions and about 0.22 GB. Serbian is included in the published language list. FastEmbed runs ONNX locally, avoiding an API key or per-query service. Sources: [model card](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2), [FastEmbed supported models](https://qdrant.github.io/fastembed/examples/Supported_Models/), [FastEmbed retrieval guide](https://qdrant.github.io/fastembed/qdrant/Retrieval_with_FastEmbed/).

Verification: installed the optional package in an isolated `/tmp` environment and loaded the model into a temporary cache; no model files or environment packages were added to the repository. `./scripts/run_tests.sh` reports 57 passed in `0.12s`. With the real model and default `220/0` chunks, `python3 -m scripts.compare_evaluation --split heldout --baseline-mode lexical --candidate-mode hybrid --baseline-size 220 --candidate-size 220 --baseline-overlap 0 --candidate-overlap 0` improves exact source-span Recall@3 from `0.786` to `0.929` and span MRR from `0.714` to `0.762`; document Recall@3/MRR/nDCG, citation metrics, and refusal metrics stay the same. `scripts.check_hybrid` passes synonym, English paraphrase, exact identifier, Serbian-only, and mixed Serbian/English probes. Benchmark: model init `8.41 s` from the existing local cache; chunk-index build `291 ms`; mean warm query `66.68 ms`, p95 `76.33 ms`; 14 × 384 float32 vectors have a 21 KiB lower bound; cached model files `240.5 MiB`; process peak RSS `696.8 MiB`; traced Python peak `27.7 MiB`. Timings and RSS are machine-dependent. The 128-token model limit and small evaluation set are documented; embeddings for books may need larger chunks split to stay within that limit.

## Phase 8: Optional Candidate Reranking

Added `RerankingRetriever` as an optional second stage. It gets a broad candidate pool (14 for the current corpus), scores each full question/passage pair with a local FastEmbed cross-encoder, then applies the existing duplicate and source-diversity filters before answer grounding. `RetrievalResult` preserves first-stage and reranker scores separately. CLI and evaluation commands accept `--rerank`, `--candidate-k`, and `--reranker-model`; the default BM25 and hybrid modes remain usable without loading the optional model. Added validation tests for ordering, retained scores, wrong score counts, non-finite values, and invalid candidate limits, plus a benchmark for first-stage and reranking latency.

Why: BM25 and embedding retrieval efficiently find plausible passages, but their initial rank can miss a better match when the whole question is considered. A cross-encoder jointly reads the question and passage and can improve ordering, at added query cost. The default model (`Xenova/ms-marco-MiniLM-L-6-v2`) is an English local model appropriate to this sample corpus; for other languages, choose a suitable model explicitly. Model references: [FastEmbed supported models](https://qdrant.github.io/fastembed/examples/Supported_Models/) and [MS MARCO model card](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2).

Verification: `./scripts/run_tests.sh` reports 61 passed in `0.08s`. Held-out comparison command: `PYTHONPATH=. /tmp/rag-week6-phase7/bin/python -m scripts.compare_evaluation --split heldout --baseline-mode hybrid --candidate-mode hybrid --baseline-size 220 --baseline-overlap 0 --candidate-size 220 --candidate-overlap 0 --candidate-rerank --candidate-k 14 --embedding-cache /tmp/rag-week6-embedding-cache`. With 10 held-out cases, Recall@3 stayed `1.000`, source-span Recall@3 stayed `0.929`, MRR improved `0.929`→`1.000`, nDCG `0.947`→`1.000`, and source-span MRR `0.762`→`0.929`; citation and refusal metrics were unchanged. `PYTHONPATH=. /tmp/rag-week6-phase7/bin/python -m scripts.benchmark_reranking --candidate-k 14 --cache-dir /tmp/rag-week6-embedding-cache --offline` measured index build `272 ms`, cached reranker load `222 ms`, first-stage mean/p95 `22.58/36.99 ms`, and cross-encoder mean/p95 `167.45/188.17 ms`. The reranked CLI supported-query demo returned a grounded answer with a valid citation. An initial offline comparison failed because the reranker had not yet been cached; the model was then downloaded and the offline benchmark passed. Measurements are machine-dependent and the held-out set is small. Commit: `Add optional retrieval reranking`.

## Phase 9: Grounded Answer Synthesis Boundary

Added an `AnswerGenerator` protocol with separate system and user prompts while preserving the extractive answer as the default and fallback. The evidence prompt serializes question and passages as JSON, labels passages as untrusted, and tells the model to ignore instructions embedded in source text. Generated output is checked sentence-by-sentence for approved citations and lexical overlap with cited passages; references are rebuilt from repository chunk metadata rather than model output. If evidence is missing, a citation is unknown, a claim is uncited/unsupported, or the adapter raises an error, the system returns the existing extractive answer (and refuses if retrieval had no sufficient result). `RagAnswer` reports whether output was generated, extractive, or fallback.

Why: a future answer model must not turn arbitrary retrieved text into instructions or cite IDs that were not supplied. A narrow interface lets applications add a provider without coupling core retrieval to a vendor or credentials. The simple overlap check is a guardrail, not semantic entailment; the guide records this limitation.

Verification: added tests for valid generated text, unknown and malformed citations, unsupported and uncited claims, prompt-injection text, conflicting evidence with uncertainty, no-evidence non-invocation, and adapter exceptions. `./scripts/run_tests.sh` reports 69 passed in `0.12s`. `python3 -m scripts.evaluate_rag --split heldout` remains at the lexical baseline: Recall@3 `1.000`, MRR `0.929`, nDCG `0.947`, source-span Recall@3 `0.786`, span MRR `0.714`, citation precision `1.000`, citation span recall `0.250`, refusal precision `0.375`, and refusal recall `1.000`. No actual generator model is installed or called; the verified path uses a deterministic test adapter. Commit: `Add grounded answer synthesis`.

## Phase 10: Validated Configuration, Traces, and Cache

Added `RagConfig` to validate chunking, retrieval, grounding thresholds, model selection, fusion, reranking, and cache settings before the CLI builds a retriever. Added request traces with a generated query ID, retrieval mode, vector-cache status, selected chunk IDs, component scores, refusal reason, and stage timings. The JSON trace deliberately omits question and passage text (and reports matched-term counts only). Added an atomic JSON semantic-vector cache keyed by schema version, model name, and corpus fingerprint; malformed entries are ignored and rebuilt. Cache write failures warn but do not prevent retrieval. `.cache/` is excluded from Git, and the hybrid benchmark reports vector-cache status.

Why: repeat queries should not re-embed an unchanged corpus, and debug output needs enough retrieval detail to explain outcomes without copying sensitive text. A validated settings object prevents conflicting options from failing deep in the pipeline. Cache identity follows the actual chunk content and model, so changed documents produce a new entry.

Verification: added tests for valid and invalid configuration, cache hits, content-based invalidation, corrupt-cache rebuild, and trace contents/privacy. `./scripts/run_tests.sh` reports 85 passed in `0.19s`. `python3 -m src.cli --question "What does the generator do with retrieved passages?" --debug` prints a JSON trace with chunk scores and retrieval/evidence timings; an unsupported query trace includes a refusal reason. An invalid overlap exits with `Invalid RAG configuration: overlap must be >= 0 and smaller than chunk_size`. Offline hybrid CLI reports `index_cache_status=hit` on the cached run. Repeated twice, `PYTHONPATH=. /tmp/rag-week6-phase7/bin/python -m scripts.benchmark_hybrid --embedding-cache /tmp/rag-week6-embedding-cache --index-cache /tmp/rag-week6-index-cache-phase10 --offline-embeddings --repeats 2` measured vector index build at `478.5/65.1 ms`, warm query means `72.95/73.10 ms`, model init `6572.1/6534.9 ms`, and peak RSS `697.0/683.5 MiB`. The planned book corpus is not present in this checkout, so that specific benchmark remains deferred; the script supports rerunning against current local configuration, and the bundled-corpus data is not treated as a book estimate. The optional FastEmbed runtime is isolated under `/tmp`; cache files are not tracked. Commit: `Add RAG observability and caching`.
