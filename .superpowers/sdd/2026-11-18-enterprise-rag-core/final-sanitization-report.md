# Final residual sanitization report

## Scope

Applied only the authorized residual sanitization fix. No vector logic, model IDs, index
generation, corpus data, retry behavior, RAG architecture/streaming logic, deployment
files, services, Docker, system environment, or other projects were changed. No real
provider calls, OCR runs, full-corpus ingestion, or server start/stop commands were used.

## Exact changes

- Added `app/error_sanitization.py` with one shared `sanitized_error_message` function.
  - SQLAlchemy errors become `Database operation failed; check local index storage and
    run verify` with only the safe exception type prefix.
  - Qdrant response errors become `Vector store operation failed; check Qdrant
    availability and configuration`.
  - Configuration errors become `Invalid configuration; check YAML syntax and configured
    values`.
  - OpenAI SDK errors become `Chat provider request failed; check chat credentials, model
    configuration, and connectivity`.
  - HTTPX network errors and `OSError` receive short network/local-operation categories.
  - `RetrievalUnavailable` receives a short hybrid-store/provider category instead of
    exposing nested raw store diagnostics.
  - Existing `BailianError` messages, including `BailianRequestError`, are preserved.
  - CLI application `RuntimeError`/`ValueError` messages not in these operational
    categories retain their prior behavior. Verification's unmapped errors use a generic
    failed-check message rather than raw bodies.
- Updated `app/cli.py` so `_fail` and configuration-failure JSON use the shared sanitizer.
  Added the OpenAI SDK base `OpenAIError` only to the existing outer `ask` exception
  boundary, so failures raised during `ChatOpenAI.astream` are sanitized without changing
  streaming code. Added HTTPX errors to the existing expected CLI error categories.
- Updated `app/verification.py` so `_error_check` uses the same sanitizer for database,
  vector-store, BM25/source, and other structured check failures.
- Added `tests/unit/test_error_sanitization.py` with seven focused tests using sentinel
  SQL, paths, URLs, YAML values, credentials, and authorization headers:
  - SQLAlchemy failure through ingest CLI text.
  - SQLAlchemy failure through verification JSON data.
  - Qdrant failure through a verification check.
  - Actual malformed YAML through `verify` stdout JSON.
  - OpenAI SDK failure raised from the async answer stream through `ask`.
  - Preservation of a safe `BailianRequestError` message.
  - Propagation of an unexpected `TypeError` rather than silently swallowing it.

## RED evidence

Command:

```bash
uv run pytest tests/unit/test_error_sanitization.py -q
```

Result: **exit 1**; **5 failed, 2 passed** before implementation.

The failures directly exposed the raw sentinels:

- SQLAlchemy output contained `SQL_PATH_SENTINEL`, `SQL_SENTINEL`, the raw `SELECT`, and a
  SQLAlchemy URL.
- Verification contained `VERIFY_SQL_SENTINEL` and `VERIFY_PATH_SENTINEL`.
- Qdrant verification contained `QDRANT_SENTINEL`, an internal URL, authorization header,
  and `QDRANT_SECRET`.
- Malformed-YAML JSON contained `YAML_VALUE_SENTINEL`, `YAML_SECRET`, and raw parser
  context.
- The streamed `OpenAIError` escaped the CLI with its raw exception object.
- The safe Bailian-message and programming-error controls already passed.

## GREEN evidence

- `uv run pytest tests/unit/test_error_sanitization.py -q` — **exit 0**, **7 passed**.
  Every sentinel, URL, and authorization header is absent from the asserted CLI/JSON
  output, while category-specific actionable text remains.
- `uv run pytest tests/unit/test_error_sanitization.py tests/unit/test_config.py
  tests/unit/test_bailian.py tests/unit/test_retrieval_service.py -q` — **exit 0**,
  **43 passed**.
- `uv run ruff check app/cli.py app/verification.py app/error_sanitization.py
  tests/unit/test_error_sanitization.py` — final run **exit 0**, `All checks passed!`.
  An earlier combined run exited 1 only because the new test imported unused `pytest`;
  that import was removed and the focused tests were rerun successfully.
- Final `uv run pytest tests/unit/test_error_sanitization.py -q` — **exit 0**, **7 passed**.
- `git diff --check` — **exit 0**, no whitespace errors.

## Residual risks and review notes

- Provider, Qdrant, OCR, and full-corpus integration tests were intentionally not run under
  the task's focused-test/no-server constraints. All failure injections are local test
  doubles; no live behavior is claimed.
- Original exceptions remain exception causes for normal programmatic debugging, but raw
  bodies are not emitted through the sanitized CLI or verification messages.
- Existing safe application error behavior outside the authorized operational categories
  was deliberately preserved rather than broadly rewriting all CLI errors.
- Independent reviewer gate remains required by the orchestrator.
