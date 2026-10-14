# Week 6 RAG Homework Starter

This starter app helps students practice Week 6 concepts from the RAG lecture:

- corpus preparation
- chunking with overlap
- simple retrieval
- answer generation with citations
- refusal behavior when evidence is missing
## Learning goals

By the end of this homework, a student pair should be able to:

1. Explain how chunking strategy affects retrieval quality.
2. Implement a retrieval pipeline that returns top-k relevant chunks.
3. Return answers with explicit citations to source chunks.
4. Refuse unsupported questions instead of hallucinating.
5. Show evidence for one supported and one unsupported question.
## Homework tasks for students

1. Outcome 1 (chunking impact): tune chunking parameters in `src/chunking.py` and compare retrieval quality for the same query.
2. Outcome 2 (top-k retrieval): verify `src/retrieval.py` returns correctly ordered top-k relevant chunks.
3. Outcome 3 (citations): ensure `src/rag.py` returns grounded answers with explicit citations to chunk IDs.
4. Outcome 4 (refusal): ensure unsupported questions trigger refusal instead of hallucinated answers.
5. Outcome 5 (evidence): document one supported and one unsupported query with full retrieval and output evidence.
### Recommended evidence template (`evidence.md`)

Use this exact structure for easy review:

1. Outcome 1: Chunking impact
- Config A:
- Config B:
- Retrieval difference summary:
2. Outcome 2: Top-k retrieval
- Query:
- Top-k output:
- Why ranking is correct:
3. Outcome 3: Citation quality
- Supported query:
- Answer with citations:
- Retrieved chunk IDs:
4. Outcome 4: Refusal behavior
- Unsupported query:
- Retrieved context:

- Refusal output:
5. Outcome 5: Final evidence pair
- Supported case command + output
- Unsupported case command + output
### Use `docs/homework-checklist.md` as submission guidance.