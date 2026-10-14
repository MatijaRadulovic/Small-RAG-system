# How This RAG Retriever Works

This guide explains the retrieval part of the project from the beginning. The implementation is deliberately small enough to read, change, and experiment with locally.

## 1. What retrieval does in RAG

RAG means **retrieval augmented generation**. Before an answer is produced, the application searches its own documents for useful passages. Those passages become evidence for the answer.

The pipeline in this repository is:

```text
documents
   ↓
boundary-aware chunks
   ↓
query tokenization
   ↓
BM25 content/title rankings + rank fusion
   ↓
sorted top-k chunks
   ↓
evidence sufficiency check
   ↓
answer with citations, or refusal
```

Retrieval quality matters because the answer stage cannot cite evidence it never received.

## 2. Boundary-aware chunking

Long documents are divided into smaller passages in `src/chunking.py`. A chunk is easier to search than a complete book or article, but careless splitting can damage the text.

The earlier implementation cut at an exact character position. This could produce fragments such as:

```text
Chunk 1: "...source filteri"
Chunk 2: "ng, citations, and traceability..."
```

The new implementation treats `chunk_size` as a maximum target and chooses a better boundary:

1. Look backward for a sentence ending (`.`, `!`, or `?`).
2. Use that sentence ending when the chunk would still contain enough text.
3. Otherwise, look backward for a space and end at a complete word.
4. If one word is longer than the target size, keep the word intact.

For overlap, the next chunk starts approximately `overlap` characters before the previous end, adjusted to the beginning of a word. Overlap repeats some context so a fact near a boundary is less likely to become disconnected from its explanation.

With the defaults:

```python
DEFAULT_CHUNK_SIZE = 220
DEFAULT_OVERLAP = 0
```

the algorithm aims for chunks of about 180 characters and repeats no characters. Actual lengths vary because preserving language boundaries is more valuable than enforcing an exact number. This setting won the current 15-query evaluation; another corpus may benefit from overlap.

Each `Chunk` also stores its source ID, section, and start/end offsets into the original Python string. Offsets use zero-based Unicode code-point positions, with the end offset excluded, matching Python slicing. `extract_source_span()` checks that the cited chunk text still matches that exact source span. A chunk ID such as `book-1@240:418` identifies a source and character range, so adding another document does not change it. Editing text before that range can move its offsets and therefore changes its ID; regenerate citations after source edits.

## 3. Unicode and language-aware tokenization

Retrieval normalizes Unicode text, applies `casefold()` for consistent case handling, and converts it into word-like tokens. The pattern accepts Latin and Cyrillic letters and can keep identifiers such as `rag-v2.1` together. Stop words are configurable as English (`en`), Serbian (`sr`), or the combined set (`mixed`).

```text
Question: "How does the retrieval system find the evidence?"
Tokens:   ["retrieval", "system", "find", "evidence"]
```

Words such as `the`, `does`, and `how` appear in many sentences and usually say little about the topic. Removing them lets important terms have more influence.

The current sample corpus is in English, so English remains the default. For Serbian text, pass `--language sr`; use `--language mixed` when questions or documents combine English and Serbian.

Tokenization is still lexical in BM25 mode: `retrieve`, `retrieval`, and `retrieved` are different tokens. No stemming or lemmatization is enabled. Hybrid mode adds multilingual vector similarity so semantically related words can match even when their spellings differ.

## 4. BM25 scoring

`src/retrieval.py` uses BM25 to score chunk text. BM25 is a classic search algorithm used by many search engines. It improves on simple token counting in three useful ways.

### Rare words matter more

A word that appears in almost every chunk is not very useful for deciding which chunk is best. A rare query word is more informative. BM25 calculates inverse document frequency (IDF) from all chunks:

```text
IDF(term) = log(1 + (N - df + 0.5) / (df + 0.5))
```

- `N` is the number of chunks.
- `df` is the number of chunks containing the term.
- A term found in fewer chunks receives a larger weight.

### Repetition has diminishing returns

If a chunk says `retrieval` ten times, it should not automatically be ten times better than a chunk that says it once. BM25 uses term-frequency saturation: repeated matches help, but each additional repetition adds less value.

### Chunk length is normalized

Long chunks naturally contain more words and have more chances for accidental matches. BM25 compares each chunk length with the average chunk length. The `b` parameter controls this correction.

This project uses conventional starting values:

```python
BM25_K1 = 1.5
BM25_B = 0.75
```

- `k1` controls how quickly repeated terms stop adding much value.
- `b` controls how strongly chunk length is normalized.

These are useful defaults, not universal best values. They should be evaluated on representative questions from the actual corpus.

The `BM25Index` class tokenizes chunk text and titles and calculates document frequencies and average field lengths once. Reusing the index for later questions avoids rebuilding those corpus statistics on each query. The demo CLI builds one index after chunking and then uses it to answer.

## 5. Combining content and title rankings

Document titles contain useful metadata. For example, a chunk from a document titled `Chunking and Metadata` deserves a small extra signal for a question about chunking.

BM25 content and title scores are calculated independently because they have different distributions. Reciprocal rank fusion combines their orderings instead of adding the raw scores:

```text
RRF score = 1 / (60 + content rank) + 0.01 / (60 + title rank)
```

Only fields with a lexical match contribute to the fused score. Content determines the main ranking; the small title weight can help distinguish close results without overpowering a much stronger passage match. The result also retains the separate BM25 field scores and matched content terms for inspection. Fused scores are for ordering and should not be treated as probabilities.

## 6. Top-k selection

After scoring, retrieval:

1. removes results below `min_score`;
2. sorts the remaining chunks from highest to lowest score;
3. skips exact duplicate passages and highly overlapping passages from the same source;
4. limits results to two chunks per source by default;
5. uses chunk ID to make tied results deterministic;
6. returns at most `k` chunks.

Near-duplicate detection uses Jaccard overlap between token sets; its default threshold is `0.85`. The selected result reports `redundancy` and `diversity_score` (one minus the highest overlap with an already selected passage). Its BM25/RRF `score` remains the relevance signal. Set `max_per_source=None` in Python to disable the source cap, or change it in the CLI with `--max-per-source 3`.

`top_k=2` means “return up to two qualifying chunks after duplicate and source filters.” Increasing `k` gives the answer stage more context, but it can also add irrelevant passages. More context is not automatically better. The source cap keeps multi-chunk evidence possible when one document contains distinct facts.

## 7. The grounding check

A search engine always tries to find the closest result, even when the corpus cannot answer the question. Therefore, `src/rag.py` performs a second check before producing an answer.

Each retrieved result receives an `EvidenceAssessment` with its fused retrieval score, matched query terms, query-term coverage, a pass/fail value, and an explanation. The check compares the question with passage text, not just the title, so a matching title cannot support unrelated content. For a multi-term question, each result must:

- match at least two meaningful query terms; and
- cover at least half of the meaningful query terms.

The check uses term coverage instead of a raw BM25 score threshold. BM25 scores depend on the corpus, so a cosine-based cutoff cannot safely be reused after changing the scoring method. A future evaluation set can support a more detailed calibrated rule.

Weak results are removed before answer text and references are built. If every result is weak, the system returns a refusal with no citations. If some qualify, it clearly labels the answer as based only on individually supported passages. This separation is useful:

- retrieval asks, “Which available chunks are closest?”
- grounding asks, “Are those chunks strong enough to support an answer?”

For example, the citation demo retrieves two chunks. The citation-quality passage covers 80% of meaningful query terms and is included; a metadata passage covers 20% and is excluded. This keeps the answer's citations aligned with evidence that passed the same explicit rule.

## 8. Why this is better than the previous scorer

The previous binary cosine scorer was better than raw token counting, but every matching term had the same importance. BM25 distinguishes rare terms from common terms, limits the effect of repetition, and normalizes passage length.

Boundary-aware chunking also improves the material being scored. Complete words and sentences give BM25 meaningful tokens instead of fragments such as `retrie` and `val`.

## 9. What lexical mode still cannot do

BM25 matches words, not concepts. A question containing `automobile` may not retrieve a passage containing only `car`. It also has no stemming, language-specific morphology, typo correction, or learned semantic representation. Use hybrid mode for semantic matching; keep in mind the small current dataset is not enough to establish broad multilingual quality.

For a larger system, common next steps are:

- add stemming or lemmatization;
- add phrase and field weighting;
- rerank the first retrieval results with a stronger model.

BM25 remains valuable even in a hybrid system because it is fast, explainable, and strong for exact names, identifiers, and technical terms.

## 10. How to experiment

Run the existing analysis with different chunk configurations:

```bash
python3 -m scripts.analyze_chunking
```

Inspect top-k output:

```bash
python3 -m scripts.check_retrieval
```

Useful questions to record for each experiment are:

1. Did the expected document rank first?
2. Did a sentence or important phrase get split?
3. Did several returned chunks repeat the same information?
4. Did a rare, meaningful word improve the correct result?
5. Did a weakly related question correctly lead to refusal?

Change one setting at a time and compare the same queries. That makes it possible to explain which change caused the observed difference.

## 11. Use the labeled evaluation set

`data/rag_eval_v1.jsonl` stores fixed questions and labels separately from the application code. Ten questions form the tuning split; another ten form the held-out split. Labels include relevant documents, exact source passages, answerability, expected facts, and the reason a question is included. The held-out cases cover paraphrases, a two-source question, weak lexical overlap, unsupported requests, and an adversarial request.

Run the chunk-size grid on tuning cases, then evaluate the current implementation on held-out cases:

```bash
python3 -m scripts.analyze_chunking
python3 -m scripts.evaluate_rag --split heldout --details
```

Document Recall@k tells whether a relevant source appears; source-span Recall@k tells whether the exact labeled passage appears. MRR rewards earlier relevant results, while nDCG also uses document relevance grades. Citation precision and recall check whether cited chunks contain expected source spans. Refusal precision and recall compare actual refusal decisions with labeled answerability. Current thresholds and limitations are in `docs/evaluation_policy.md`. The set is small and English-only, so its measurements are useful regression signals rather than broad quality guarantees.

## 12. Hybrid lexical and semantic retrieval

Hybrid retrieval keeps BM25 for exact words and identifiers, then adds cosine similarity from local embeddings. It fuses the two rank lists with reciprocal rank fusion, so unrelated raw BM25 and cosine scales are not added directly. `HybridRetriever` builds the lexical statistics and one vector per chunk once, then reuses those vectors for the process lifetime.

The configured model is `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. Its model card lists 50 languages and Apache 2.0; FastEmbed's supported model table reports 384-dimensional vectors and an approximately 0.22 GB model package. The listed language set includes Serbian. FastEmbed runs it through ONNX locally. The model has a 128-token sequence limit, so very long chunks may be truncated; the current 220-character chunks stay below that limit in this corpus. [Model card](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2), [FastEmbed supported models](https://qdrant.github.io/fastembed/examples/Supported_Models/), [FastEmbed retrieval guide](https://qdrant.github.io/fastembed/qdrant/Retrieval_with_FastEmbed/).

Install the optional package and select hybrid mode:

```bash
python3 -m pip install -r requirements-embeddings.txt
python3 -m src.cli --retrieval-mode hybrid --question "What does the generator do with retrieved passages?"
```

The first run downloads the model; later runs can require cached files with `--offline-embeddings`. If the optional dependency or cached model is absent, lexical mode remains available. The held-out evaluation currently improves source-span Recall@3 from `0.786` to `0.929` while keeping document Recall@3, citation, and refusal metrics unchanged. Reproduce with `python3 -m scripts.compare_evaluation --split heldout --baseline-mode lexical --candidate-mode hybrid --baseline-size 220 --candidate-size 220 --baseline-overlap 0 --candidate-overlap 0`. Run `python3 -m scripts.check_hybrid` for synonym, paraphrase, exact-ID, Serbian, and mixed-language probes, and `python3 -m scripts.benchmark_hybrid` for local runtime and memory measurements.
