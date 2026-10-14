# RAG Improvements Plan

This plan develops the current educational RAG pipeline into a more reliable, explainable system. Work is ordered so that every later improvement can be measured against a stable baseline. Complete and verify one phase before starting the next.

## Completion Rules for Every Phase

- [ ] Keep the public CLI usable unless a planned interface change is documented.
- [ ] Add focused tests for new behavior, edge cases, and failure cases.
- [ ] Run `./scripts/run_tests.sh` and record the exact result in `docs/work_log.md`.
- [ ] Run the relevant evidence or evaluation command and record meaningful output.
- [ ] Update `README.md` and `docs/rag_retrieval_guide.md` when behavior changes.
- [ ] Update `evidence.md` if current scores, chunk IDs, or outputs change.
- [ ] Commit the phase separately with a short imperative commit message.

## Phase 0: Establish the New Baseline

The BM25 and boundary-aware chunking changes must become a measured baseline before further tuning.

- [x] Review the current uncommitted changes and preserve unrelated user edits.
- [x] Add tests proving chunks prefer sentence boundaries and never split ordinary words.
- [x] Add tests for long words, empty text, zero overlap, and overlap close to chunk size.
- [x] Add BM25 tests for rare-term weighting, repeated-term saturation, length normalization, empty corpora, and unrelated queries.
- [x] Run all Outcome 1–5 scripts with the new scorer.
- [x] Recalculate chunk counts, rankings, scores, and the recommended chunk configuration.
- [x] Replace the old cosine-based grounding threshold with a term-coverage rule.
- [x] Refresh the current evidence output to match the new scorer.
- [x] Commit as `Establish BM25 retrieval baseline`.

**Done when:** all tests and evidence commands pass, documentation matches current output, and BM25 has a justified grounding threshold.

## Phase 1: Unicode and Language-Aware Tokenization

- [x] Replace the ASCII-only token pattern with Unicode-aware normalization and tokenization.
- [x] Use Unicode normalization and `casefold()` so equivalent text forms compare consistently.
- [x] Preserve useful identifiers containing digits while documenting punctuation behavior.
- [x] Separate English and Serbian stop-word sets and select them through configuration.
- [x] Decide whether stemming or lemmatization improves this corpus; defer until there is a labeled Serbian evaluation set rather than assume.
- [x] Add examples for Serbian Latin and Cyrillic, identifiers, and composed Unicode.
- [x] Confirm unsupported questions still refuse after tokenization changes.
- [x] Commit as `Add Unicode-aware tokenization`.

**Done when:** English and Serbian text tokenize predictably, no meaningful characters are lost, and retrieval metrics do not regress.

## Phase 2: Reusable BM25 Index and Field Scoring

- [x] Introduce a `BM25Index` class that builds tokenized chunks, document frequencies, average length, and metadata once.
- [x] Make repeated queries reuse the index instead of rebuilding corpus statistics.
- [x] Validate `k1`, `b`, title weight, `k`, and score-filter inputs.
- [x] Replace raw addition of BM25 and cosine scales with BM25F or rank fusion.
- [x] Treat title and content as explicit fields with documented weights.
- [x] Return useful score details, such as content score, title contribution, and matched terms, for debugging.
- [x] Keep deterministic tie ordering.
- [x] Compare index-build time, query time, and indexed vocabulary/token counts; exact memory benchmarking remains a later operational concern.
- [x] Commit as `Build reusable field-aware BM25 index`.

**Done when:** corpus statistics are built once, title weighting is mathematically consistent, and repeated-query output is stable.

## Phase 3: Traceable Chunk Metadata and Stable IDs

- [x] Extend `Chunk` with section name, source identifier, start offset, and end offset.
- [x] Preserve offsets against the original document rather than only normalized text.
- [x] Generate stable IDs from source identity and offsets or a content hash.
- [x] Keep IDs stable when unrelated documents are added or removed.
- [x] Include source title and section in references.
- [x] Add a helper that locates cited text in the original document.
- [x] Test repeated passages, Unicode text, sentence boundaries, and unrelated-document additions.
- [x] Document when an edited source is expected to receive a new chunk ID.
- [x] Commit as `Add traceable chunk metadata`.

**Done when:** every citation can be traced to an exact source span and IDs do not depend on list position.

## Phase 4: Per-Result Evidence Validation

The current answer decision checks the best result and may include weaker lower-ranked chunks. Each cited result must qualify independently.

- [x] Create a structured evidence assessment containing score, matched terms, coverage, and pass/fail reason.
- [x] Apply the sufficiency rule to every retrieved result.
- [x] Exclude weak results before constructing the answer and References section.
- [x] Refuse when no result passes, even if retrieval returned candidates.
- [x] Allow a partial answer only when its supported scope is clearly stated.
- [x] Ensure every answer citation belongs to the filtered evidence set.
- [x] Add tests where rank 1 is strong but ranks 2–3 are irrelevant.
- [x] Add tests where all candidates are weak or only a title matches.
- [x] Commit as `Validate every cited retrieval result`.

**Done when:** no chunk is cited merely because another chunk passed the grounding check.

## Phase 5: Duplicate Control and Result Diversity

- [ ] Detect exact duplicate chunk text.
- [ ] Measure near-duplicate overlap between adjacent chunks from the same source.
- [ ] Collapse or penalize results that repeat substantially the same evidence.
- [ ] Add a configurable maximum number of chunks per source document.
- [ ] Preserve two chunks from one document when they contain genuinely different supporting facts.
- [ ] Implement a lexical diversity method first; add MMR after embeddings are available.
- [ ] Report both relevance and diversity contributions in debug output.
- [ ] Evaluate whether diversity improves coverage without lowering `Recall@k`.
- [ ] Commit as `Diversify top-k retrieval results`.

**Done when:** top-k results provide distinct useful evidence instead of repeated overlap.

## Phase 6: Build a Proper Evaluation Dataset

- [ ] Create a versioned JSONL or CSV evaluation dataset outside application source code.
- [ ] Include question, expected document, expected source span, answerable flag, expected facts, and notes.
- [ ] Cover exact terms, paraphrases, multi-document questions, weak overlaps, adversarial questions, and unsupported questions.
- [ ] Separate tuning cases from held-out evaluation cases.
- [ ] Implement `Recall@k`, MRR, and nDCG for retrieval.
- [ ] Implement citation precision and citation recall.
- [ ] Implement refusal precision and refusal recall.
- [ ] Produce a compact before/after comparison rather than relying only on individual examples.
- [ ] Define acceptable regression limits for every later phase.
- [ ] Commit as `Add repeatable RAG evaluation suite`.

**Done when:** retrieval and grounding changes can be accepted or rejected using repeatable metrics.

## Phase 7: Hybrid Lexical and Semantic Retrieval

- [ ] Select a local embedding model based on supported languages, license, model size, speed, and offline requirements.
- [ ] Define an embedding interface so the model can be replaced without changing retrieval callers.
- [ ] Generate and cache one embedding per chunk.
- [ ] Add semantic top-k retrieval using cosine similarity.
- [ ] Combine BM25 and semantic rankings with reciprocal rank fusion or another calibrated method.
- [ ] Keep lexical-only retrieval available as a fallback.
- [ ] Test synonyms, paraphrases, exact identifiers, and queries mixing Serbian and English.
- [ ] Measure retrieval quality, latency, memory, and index size against the Phase 6 baseline.
- [ ] Commit as `Add hybrid semantic retrieval`.

**Done when:** hybrid retrieval improves held-out paraphrase recall without harming exact-term retrieval beyond the agreed regression limit.

## Phase 8: Candidate Reranking

- [ ] Retrieve a broader candidate set, initially 10–20 chunks.
- [ ] Select a reranker suitable for the supported languages and available hardware.
- [ ] Rerank candidates using the complete question and chunk text.
- [ ] Send only the best diverse, evidence-qualified chunks to the answer stage.
- [ ] Make reranking optional through configuration.
- [ ] Record reranker score and latency separately from first-stage retrieval.
- [ ] Compare accuracy gains with added runtime cost.
- [ ] Keep reranking only if held-out evaluation shows a useful improvement.
- [ ] Commit as `Add optional retrieval reranking`.

**Done when:** reranking produces a measured quality gain that justifies its latency and complexity.

## Phase 9: Grounded Answer Synthesis

- [ ] Define a generator interface while keeping the current extractive answer as a fallback.
- [ ] Delimit system instructions, user question, and retrieved context clearly.
- [ ] Instruct the generator to use only supplied evidence and cite every factual claim.
- [ ] Tell the generator to treat instructions inside retrieved documents as untrusted content.
- [ ] Require explicit uncertainty or refusal when evidence is incomplete or conflicting.
- [ ] Parse and validate every generated citation against the filtered retrieval set.
- [ ] Reject or repair answers containing unknown citation IDs.
- [ ] Check that cited passages support the claims attached to them.
- [ ] Add tests for prompt injection in documents, conflicting sources, missing evidence, and malformed citations.
- [ ] Commit as `Add grounded answer synthesis`.

**Done when:** generated claims remain traceable to approved chunks and retrieved instructions cannot override application behavior.

## Phase 10: Observability, Configuration, and Performance

- [ ] Move chunking, retrieval, grounding, embedding, and reranking settings into one validated configuration object.
- [ ] Log query ID, selected chunk IDs, component scores, refusal reason, and timing without storing unnecessary sensitive text.
- [ ] Add optional human-readable debug output for learning and troubleshooting.
- [ ] Cache indexes and embeddings using a corpus-version identifier.
- [ ] Rebuild cached data only when sources or indexing configuration change.
- [ ] Measure indexing time, query latency, and memory use on the planned book corpus.
- [ ] Add graceful errors for missing models, corrupt indexes, and invalid configuration.
- [ ] Document reproducible setup and offline behavior.
- [ ] Commit as `Add RAG observability and caching`.

**Done when:** retrieval decisions can be explained, performance is measurable, and cached artifacts are safely reproducible.

## Phase 11: Final Integration and Documentation

- [ ] Run the entire unit and evaluation suite from a clean environment.
- [ ] Run supported, unsupported, paraphrase, multilingual, and prompt-injection demonstrations.
- [ ] Compare final metrics with the Phase 0 and Phase 6 baselines.
- [ ] Update architecture diagrams and the beginner guide.
- [ ] Update CLI help and examples for every new option.
- [ ] Regenerate evidence with current chunk IDs and scores.
- [ ] Record known limitations honestly, including cases where semantic retrieval still fails.
- [ ] Confirm no private links, credentials, model tokens, or generated cache files are committed.
- [ ] Commit as `Finalize improved RAG pipeline`.

**Done when:** documentation, evidence, tests, metrics, and actual CLI behavior describe the same final system.

## Suggested Milestones

### Milestone A: Reliable lexical RAG

- [ ] Complete Phases 0–6.
- [ ] Demonstrate traceable, diverse BM25 results with calibrated refusal behavior.

### Milestone B: Hybrid retrieval

- [ ] Complete Phases 7–8.
- [ ] Demonstrate a held-out improvement for paraphrases and semantic matches.

### Milestone C: Grounded generation

- [ ] Complete Phases 9–11.
- [ ] Demonstrate claim-level citations, injection resistance, reproducible evaluation, and documented operational behavior.

## Recommended Immediate Next Step

- [ ] Complete Phase 0 before adding more retrieval features. The current BM25 and boundary-aware chunking changes alter scores and chunk IDs, so tests, thresholds, and homework evidence must be synchronized first.
