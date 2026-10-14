# Repository Guidelines

## Project Structure

- `src/` contains the Python package: `corpus.py` provides sample corpora, `chunking.py` creates overlapping chunks, `retrieval.py` ranks chunks, `rag.py` builds grounded answers or refusals, and `cli.py` exposes the demo.
- `tests/` contains pytest coverage grouped by module (`test_chunking.py`, `test_corpus.py`, `test_retrieval.py`, and `test_rag.py`).
- `docs/homework-checklist.md` describes student submission evidence. `scripts/run_tests.sh` is the project test entry point. Dependencies are listed in `requirements.txt`.

## Build, Test, and Run

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the complete test suite with `./scripts/run_tests.sh` (equivalent to `python -m pytest -q`). Run the CLI with, for example, `python -m src.cli --question "What is retrieval augmented generation?"`. Use `--corpus-profile training-rag` to select the alternate sample corpus, or pass `--chunk-size`, `--overlap`, and `--top-k` to explore retrieval behavior.

## Coding Style

Use Python with four-space indentation, `snake_case` for functions and variables, and `PascalCase` for classes. Keep type annotations consistent with the existing modules and use dataclasses for simple structured records. Keep the pipeline boundaries clear: corpus loading, chunking, retrieval, and answer/refusal behavior belong in their respective modules. No formatter or linter is configured in this starter.

## Testing Guidelines

Tests use pytest and follow `test_*.py` / `test_*` naming. Add or update focused tests alongside the module behavior being changed; cover invalid inputs and both supported and unsupported query behavior where relevant. Run `./scripts/run_tests.sh` before submitting changes.

## Commits and Pull Requests

Git history is not available in this checkout, so no established commit convention could be confirmed. Use short, imperative commit subjects, such as `Add empty-query retrieval coverage`. Pull requests should explain the behavior changed, link related homework/issues when applicable, and include relevant CLI output or evidence for retrieval and citation changes. Record any test command and result.
