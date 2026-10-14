# Week 6 Homework Checklist (RAG)

Use this checklist when preparing your submission.

## A. Baseline setup

- [x] Project installs and tests run locally.
- [x] You can run the CLI with a custom question.
- [x] You understand where chunking, retrieval, and answer behavior are implemented.

## B. Outcome 1: Explain chunking effect on retrieval quality

- [x] You ran at least two chunking configurations (for example: `chunk_size=140, overlap=30` and one alternative).
- [x] You compared retrieval output for the same question under both configurations.
- [x] You wrote a short explanation of the observed difference in retrieval quality.

## C. Outcome 2: Implement top-k retrieval pipeline

- [x] Retrieval returns top-k chunks sorted by relevance score.
- [x] You show one example output with `top_k=2` and one with `top_k=3`.
- [x] You explain how your scoring approach decides relevance.

## D. Outcome 3: Return answers with explicit citations

- [x] Supported answer includes explicit citations to chunk IDs.
- [x] Citation IDs in answer correspond to retrieved chunks shown in evidence.
- [x] Answer format clearly separates answer body and citations/references.

## E. Outcome 4: Refuse unsupported questions

- [x] For an unsupported question, the system refuses instead of generating an unsupported claim.
- [x] Refusal message is clear and consistent with grounded RAG behavior.
- [x] You explain briefly why retrieved evidence was insufficient.

## F. Outcome 5: Show supported and unsupported evidence

- [x] You provide one supported query with: question, retrieved chunks, answer, citations.
- [x] You provide one unsupported query with: question, retrieved chunks (or empty result), refusal output.
- [x] Evidence for both examples is reproducible from CLI commands or tests.

## G. Pair accountability

- [ ] Pair report includes contribution of both members.
- [ ] The designated demo lead for this week is recorded.

## H. Suggested extension (optional)

- [x] Improve lexical scoring or citation formatting.
- [x] Add at least 3 tests for your extension.

## I. Submission bundle (recommended)

- [x] `evidence.md` with Outcome 1-5 sections.
- [x] CLI commands used for both example queries.
- [x] Test output snapshot.

