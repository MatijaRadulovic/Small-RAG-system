# Week 6 Homework Checklist (RAG)

Use this checklist when preparing your submission.

## A. Baseline setup

- [ ] Project installs and tests run locally.
- [ ] You can run the CLI with a custom question.
- [ ] You understand where chunking, retrieval, and answer behavior are implemented.

## B. Outcome 1: Explain chunking effect on retrieval quality

- [ ] You ran at least two chunking configurations (for example: `chunk_size=140, overlap=30` and one alternative).
- [ ] You compared retrieval output for the same question under both configurations.
- [ ] You wrote a short explanation of the observed difference in retrieval quality.

## C. Outcome 2: Implement top-k retrieval pipeline

- [ ] Retrieval returns top-k chunks sorted by relevance score.
- [ ] You show one example output with `top_k=2` and one with `top_k=3`.
- [ ] You explain how your scoring approach decides relevance.

## D. Outcome 3: Return answers with explicit citations

- [ ] Supported answer includes explicit citations to chunk IDs.
- [ ] Citation IDs in answer correspond to retrieved chunks shown in evidence.
- [ ] Answer format clearly separates answer body and citations/references.

## E. Outcome 4: Refuse unsupported questions

- [ ] For an unsupported question, the system refuses instead of generating an unsupported claim.
- [ ] Refusal message is clear and consistent with grounded RAG behavior.
- [ ] You explain briefly why retrieved evidence was insufficient.

## F. Outcome 5: Show supported and unsupported evidence

- [ ] You provide one supported query with: question, retrieved chunks, answer, citations.
- [ ] You provide one unsupported query with: question, retrieved chunks (or empty result), refusal output.
- [ ] Evidence for both examples is reproducible from CLI commands or tests.

## G. Pair accountability

- [ ] Pair report includes contribution of both members.
- [ ] The designated demo lead for this week is recorded.

## H. Suggested extension (optional)

- [ ] Improve lexical scoring or citation formatting.
- [ ] Add at least 3 tests for your extension.

## I. Submission bundle (recommended)

- [ ] `evidence.md` with Outcome 1-5 sections.
- [ ] CLI commands used for both example queries.
- [ ] Test output snapshot.

