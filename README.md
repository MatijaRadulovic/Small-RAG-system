# Week 6 RAG Homework Starter

This starter app helps students practice Week 6 concepts from the RAG lecture:

- corpus preparation
- chunking with overlap
- simple retrieval
- answer generation with citations
- refusal behavior when evidence is missing
- per-passage evidence checks that filter weak results before citation
- duplicate filtering and a configurable per-source result limit

## Learning goals

By the end of this homework, a student pair should be able to:

1. Explain how chunking strategy affects retrieval quality.
2. Implement a retrieval pipeline that returns top-k relevant chunks.
3. Return answers with explicit citations to source chunks.
4. Refuse unsupported questions instead of hallucinating.
5. Show evidence for one supported and one unsupported question.

## Project structure

```text
rag-week-6-homework-starter/
  src/
    corpus.py
    chunking.py
    retrieval.py
    rag.py
    cli.py
  tests/
    test_chunking.py
    test_retrieval.py
    test_rag.py
  docs/
    homework-checklist.md
  scripts/
    analyze_chunking.py
    check_retrieval.py
    run_tests.sh
  requirements.txt
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run tests

```bash
./scripts/run_tests.sh
```

## Run homework analysis

```bash
python3 -m scripts.analyze_chunking
python3 -m scripts.check_retrieval
python3 -m scripts.check_citations
python3 -m scripts.check_refusal
python3 -m scripts.generate_evidence
python3 -m scripts.benchmark_retrieval
python3 -m scripts.evaluate_rag --split heldout --details
python3 -m scripts.compare_evaluation --split heldout --candidate-size 100 --candidate-overlap 20
python3 -m scripts.check_hybrid
python3 -m scripts.benchmark_hybrid
python3 -m scripts.benchmark_reranking --offline --cache-dir PATH
```

These commands compare chunking configurations, validate retrieval, citation, and refusal behavior, and generate the final evidence pair.
The versioned evaluation questions are in `data/rag_eval_v1.jsonl`; split definitions, metric meanings, baseline, and regression limits are in `docs/evaluation_policy.md`.

## Run demo CLI

```bash
python3 -m src.cli --question "What is retrieval augmented generation?"
python3 -m src.cli --top-k 4 --max-per-source 2 --question "How do citations work?"
python3 -m src.cli --retrieval-mode hybrid --question "What does the generator do with retrieved passages?"
python3 -m src.cli --retrieval-mode hybrid --rerank --candidate-k 14 --question "What does the generator do with retrieved passages?"
python3 -m src.cli --retrieval-mode hybrid --offline-embeddings --debug --question "What does the generator do with retrieved passages?"
python3 -m src.cli --question "Who won the world cup in 1958?"
python3 -m src.cli --corpus-profile training-rag --question "Why are citations important in RAG?"
python3 -m src.cli --language sr --question "Koji dokument sadrži relevantne podatke?"
```

Hybrid retrieval and cross-encoder reranking are optional. Install their local ONNX runtime with `python3 -m pip install -r requirements-embeddings.txt`. The first run downloads selected models; later runs can use `--offline-embeddings` and `--embedding-cache PATH`. Reranking scores a broader first-stage candidate set (`--candidate-k`) against the full question, then returns the highest-scoring diverse chunks. See `docs/rag_retrieval_guide.md` for evaluation and latency measurements.

The CLI validates its settings through `RagConfig`. Hybrid passage vectors are cached under `.cache/rag-index` using a hash of chunk IDs, titles, text, and model name; changed content or model uses a new cache entry. `--debug` prints a request ID, selected chunk IDs, component scores, stage timings, cache status, and refusal reason without the query or passage text. Local model and index caches are excluded from Git.

Answer generation uses an `AnswerGenerator` protocol in `src/generation.py`; integrations can pass a local or hosted adapter to `build_grounded_answer(..., generator=...)`. The default stays extractive. Generated claims must cite approved chunk IDs and pass a lexical evidence-overlap check or the system falls back to the extractive answer. No model API or credential is required by the starter.

## Corpus profiles

- `default`: expanded corpus for chunking, retrieval, citation, and refusal experiments.
- `training-rag`: profile mirroring examples with focus on citations, references, and LLM-as-a-judge framing.

## Homework tasks for students

1. **Outcome 1 (chunking impact):** tune chunking parameters in `src/chunking.py` and compare retrieval quality for the same query.
2. **Outcome 2 (top-k retrieval):** verify `src/retrieval.py` returns correctly ordered top-k relevant chunks.
3. **Outcome 3 (citations):** ensure `src/rag.py` returns grounded answers with explicit citations to chunk IDs.
4. **Outcome 4 (refusal):** ensure unsupported questions trigger refusal instead of hallucinated answers.
5. **Outcome 5 (evidence):** document one supported and one unsupported query with full retrieval and output evidence.

## Recommended evidence template (`evidence.md`)

Use this exact structure for easy review:

1. **Outcome 1: Chunking impact**
  - Config A:
  - Config B:
  - Retrieval difference summary:
2. **Outcome 2: Top-k retrieval**
  - Query:
  - Top-k output:
  - Why ranking is correct:
3. **Outcome 3: Citation quality**
  - Supported query:
  - Answer with citations:
  - Retrieved chunk IDs:
4. **Outcome 4: Refusal behavior**
  - Unsupported query:
  - Retrieved context:
  - Refusal output:
5. **Outcome 5: Final evidence pair**
  - Supported case command + output
  - Unsupported case command + output

Use `docs/homework-checklist.md` as submission guidance.

## Learn how retrieval works

See [`docs/rag_retrieval_guide.md`](docs/rag_retrieval_guide.md) for a beginner-friendly explanation of boundary-aware chunking, source offsets, BM25 scoring, content/title rank fusion, top-k retrieval, and grounding checks.
