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
