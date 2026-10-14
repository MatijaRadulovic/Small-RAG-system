# RAG Evaluation Dataset and Acceptance Policy

## Dataset

`data/rag_eval_v1.jsonl` is the labeled evaluation set. Every row records a stable ID, tuning or held-out split, question category, expected relevant documents with grades, exact source-span text, answerability, expected facts, and notes. It contains 10 tuning and 10 held-out cases, including exact questions, paraphrases, a two-document question, weak overlap, unsupported requests, and an adversarial request. Source spans are checked against the corpus when the evaluator runs.

The tuning split can guide chunking and retrieval choices. Keep the held-out split unchanged while comparing an implementation; revise it only as a separately versioned dataset. The corpus is English, so this version does not claim multilingual retrieval coverage.

## Metrics

Run `python3 -m scripts.evaluate_rag --split heldout` for the acceptance view. The script reports:

- document `Recall@k`, MRR, and graded nDCG;
- source-span `Recall@k` and span MRR, which catch cases where the right document ranks but the exact passage does not;
- citation precision and source-span citation recall, requiring a citation chunk to contain a labeled source span;
- refusal precision and recall, using the labeled answerable/unsupported flag.

The chunking grid uses only tuning cases: `python3 -m scripts.analyze_chunking`. The evaluator defaults to the current chunk size, overlap, and `k=3`; pass explicit values to compare configurations. `--details` prints each held-out decision.

For a compact before/after table on the same split, run `python3 -m scripts.compare_evaluation --split heldout --baseline-size 220 --baseline-overlap 0 --candidate-size 100 --candidate-overlap 20`.

## Baseline and regression limits

At chunk size `220`, overlap `0`, and `k=3`, the current held-out baseline is: document Recall@3 `1.000`, MRR `0.929`, nDCG@3 `0.947`, source-span Recall@3 `0.786`, span MRR `0.714`, citation precision `1.000`, source-span citation recall `0.250`, refusal precision `0.375`, and refusal recall `1.000`.

For later retrieval or grounding changes, require held-out Recall@3 at least `0.90`, MRR at least `0.85`, nDCG@3 at least `0.85`, source-span Recall@3 at least `0.70`, citation precision at least `0.90`, source-span citation recall at least `0.25`, refusal precision at least `0.375`, and refusal recall `1.00`. Unsupported cases must have zero citations. A quality gain should be kept only when it clears these floors and its latency/cost is justified.

These limits are guardrails for a small learning corpus, not statistical confidence claims. A single held-out case can shift supported-query metrics by about 14 percentage points and refusal metrics by one third. Add more independent questions and sources before treating small changes as general improvements.

As an example, `100/20` retains document Recall@3 `0.929` and MRR `0.929`, but source-span Recall@3 falls to `0.429` and citation precision to `0.500` on this held-out set. It therefore fails the passage and citation floors even though document-level retrieval looks close. Keep the baseline configuration until a later change improves the full set of checks.
