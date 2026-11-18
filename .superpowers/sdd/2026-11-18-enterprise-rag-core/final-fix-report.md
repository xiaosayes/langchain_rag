# Phase 1 final review fix report

## Scope and review source

This is the single final Phase 1 fix wave. It covers the ten accepted findings and the
requested core documentation only. It does not implement lifecycle cleanup, semantic
relevance/version-conflict evaluation, Web or LangSmith integration, deployment work, or
live Bailian quality evaluation.

The requested `.superpowers/sdd/2026-11-18-enterprise-rag-core/final-review.md` was not
present in this worktree (`ENOENT`). The complete accepted-findings contract supplied in
the implementation task was therefore used as the review source.

## Finding resolutions

### Critical 1 — vector-space compatibility and safe reuse

**Resolved.** Added `app/indexing/vector_space.py`, an immutable value containing provider,
embedding model, dimension, and Qdrant distance. Its canonical SHA-256 fingerprint is
persisted on every new generation. A lightweight SQLite compatibility step adds the
nullable `vector_space_fingerprint` column to existing `index_generations` tables.

`IndexBuilder` requests reusable child hashes only when the active generation fingerprint
exactly matches the current value. A mismatch returns no reusable IDs, so the old Qdrant
collection is not fetched and every child is embedded again. A `NULL` legacy fingerprint
is unknown and also disables reuse. Retrieval checks the active generation fingerprint
before either dense or BM25 work and raises an actionable rebuild error for unknown or
mismatched spaces. Verification has an explicit `vector_space_compatibility` check.
Generation activation remains the existing single transaction that retires the predecessor
and activates the staged generation.

Regression coverage includes same-dimension provider, model, and distance changes with an
assertion that the previous generation was not fetched; legacy non-reuse; query mismatch;
verification mismatch; schema compatibility; and the pre-existing atomic-activation tests.

Files: `app/indexing/vector_space.py`, `app/db/models.py`, `app/db/session.py`,
`app/indexing/repository.py`, `app/indexing/builder.py`, `app/bootstrap.py`,
`app/retrieval/service.py`, `app/verification.py`, `tests/unit/test_vector_space.py`,
`tests/unit/test_index_repository.py`, `tests/unit/test_retrieval_service.py`,
`tests/integration/conftest.py`, `tests/integration/test_index_builder.py`,
`tests/integration/test_rag_cli.py`.

### Important 2 — configured Qdrant distance

**Resolved.** Cosine, Euclid, and dot now map explicitly to their Qdrant enums when a
collection is created. Distance participates in the fingerprint and verification. Real
Qdrant tests cover both non-default values, Euclid and dot.

Files: `app/retrieval/qdrant_store.py`, `app/indexing/vector_space.py`,
`app/verification.py`, `tests/integration/test_qdrant_store.py`.

### Important 3 — dense query embedding degradation

**Resolved.** Query embedding and Qdrant search are now one async dense-stage callable.
That callable and BM25 are gathered with `return_exceptions=True`. An `aembed_query`
failure therefore degrades to BM25 with `degraded=True`; only dense and BM25 both failing
raises `RetrievalUnavailable`.

Files: `app/retrieval/service.py`, `tests/unit/test_retrieval_service.py`.

### Important 4 — verifier false positives

**Resolved.** Corpus verification requires exactly 25 TXT, 25 DOCX, 25 XLSX, and 25 PDF
files, with exactly 20 text and 5 scanned PDFs. The former 100-TXT positive test now builds
that exact shape. Qdrant verification reads all payload child IDs with pagination and
compares them as a set with active SQLite child IDs, in addition to dimension, distance,
and point count. BM25 verification likewise compares persisted child IDs with SQLite IDs,
not just manifest counts. Same-count/wrong-ID regressions fail for both stores.

Files: `app/verification.py`, `app/retrieval/qdrant_store.py`, `app/retrieval/bm25.py`,
`tests/integration/test_rag_cli.py`.

### Important 5 — provider/configuration CLI contracts

**Resolved.** Native Bailian embedding and rerank HTTP boundaries translate 400, 401,
other HTTP status errors, timeouts, and connection errors into sanitized
`BailianRequestError` messages. The 400 message points to model/request parameters; 401
points to `DASHSCOPE_API_KEY`; connection failures point to network/base URL. Backend
response details and credentials are not copied into those messages. Expected SQLAlchemy
and Qdrant client failures are caught explicitly by ingest/rebuild/ask rather than catching
all exceptions. YAML and typed settings failures use `ConfigurationError` (a `ValueError`),
and `verify` turns expected configuration-load failures into a one-check JSON failure on
stdout with exit code 1. Programming errors are not broadly swallowed by the CLI.

Tests cover 400, 401, connection failure, no retry for 400/401, credential/backend-detail
sanitization, and actual malformed YAML to structured stdout JSON.

Files: `app/models/bailian.py`, `app/config.py`, `app/cli.py`,
`tests/unit/test_bailian.py`, `tests/integration/test_rag_cli.py`.

### Accepted Minor 6 — rerank candidate count

**Resolved.** `RetrievalService` accepts `rerank_candidate_count` by injection, and the
composition root passes `settings.rerank.candidate_count`. The configured value is
validated positive. A non-default value of five is covered.

Files: `app/retrieval/service.py`, `app/bootstrap.py`, `app/config.py`,
`tests/unit/test_retrieval_service.py`.

### Accepted Minor 7 — retry jitter and monotonic deadline

**Resolved.** Sync and async retry loops have injectable full jitter, monotonic clock, and
a total retry deadline, in addition to attempt caps and injected sync/async sleep. Delays
are capped at eight seconds and checked against the monotonic remaining time before sleep
or another attempt. Native adapters expose the injection hooks and use configurable
30-second retry deadlines. HTTP 400/401 remain non-retryable. Tests cover exhaustion,
delay capping, async sleep, deterministic jitter, deadline exhaustion, and existing
non-retryable behavior.

The deadline governs retry scheduling between requests; the existing per-request HTTPX
timeout remains responsible for an already in-flight blocking request.

Files: `app/models/retry.py`, `app/models/bailian.py`, `app/config.py`,
`tests/unit/test_retry.py`, `tests/unit/test_bailian.py`.

### Accepted Minor 8 — TXT encoding and PDF OCR selection

**Resolved.** TXT decoding tries exactly UTF-8 with optional BOM and then GB18030, and
reports an encoding error if both fail. PDF text-layer selection now requires at least 20
alphanumeric/meaningful characters and rejects NUL/control/replacement-character corruption
heuristics instead of relying on raw string length. Verification uses the same text-layer
predicate. Focused tests cover GB18030, undecodable input, whitespace/punctuation,
corrupted long layers, and OCR selection. Real CPU OCR integration tests were run because
PDF selection changed.

Files: `app/ingestion/text_office.py`, `app/ingestion/pdf.py`, `app/verification.py`,
`tests/unit/test_text_office_parsers.py`, `tests/unit/test_pdf_parser.py`.

### Accepted Minor 9 — non-finite embeddings

**Resolved.** Every converted embedding component must pass `math.isfinite`; count and
dimension checks remain in place. Raw-JSON regressions cover `NaN`, `Infinity`, and
`-Infinity`.

Files: `app/models/bailian.py`, `tests/unit/test_bailian.py`.

### Accepted Minor 10 — terminal literal bracket

**Resolved.** Citation stream finalization preserves a terminal literal `[` while still
removing incomplete `[S` and `[S12` citation markers and logging their removal.

Files: `app/rag/service.py`, `tests/unit/test_rag_service.py`.

### Documentation

**Resolved.** `docs/core.md` now documents application-owned/trusted BM25 pickle
directories, the approximately 264 MB full-corpus embedding transfer footprint, Windows
short-path workarounds, stdout-only JSON consumption, exact-fingerprint rebuild behavior,
and retained SQLite/Qdrant/BM25 generations with no cleanup in this phase.

## RED / GREEN evidence

### Baseline

- `uv run pytest tests/unit -q` — **exit 0**, 265 passed before changes.

### Focused RED runs

- `uv run pytest tests/unit/test_vector_space.py -q` — **exit 2**, collection failed with
  `ModuleNotFoundError: app.indexing.vector_space`, demonstrating the absent compatibility
  mechanism.
- `uv run pytest tests/unit/test_retry.py tests/unit/test_bailian.py
  tests/unit/test_pdf_parser.py tests/unit/test_text_office_parsers.py
  tests/unit/test_rag_service.py -q` — **exit 2**, collection failed because
  `BailianRequestError` and `has_usable_text_layer` did not exist.
- After the first implementation pass, `uv run pytest tests/unit/test_vector_space.py
  tests/unit/test_retrieval_service.py tests/unit/test_index_repository.py -q` —
  **exit 1**, 23 passed and 3 fixture/legacy-call-shape failures; these were corrected by
  supplying explicit test fingerprints and a valid parent, while preserving retired
  generation compatibility checks.
- First focused real-Qdrant run of `test_qdrant_store.py`, `test_index_builder.py`, and
  `test_rag_cli.py` — **exit 1**, 22 passed and one positive verification failure exposed
  Qdrant's wire distance value `Cosine`; normalizing the enum value fixed it.

### GREEN runs

- `uv run pytest tests/unit/test_retry.py tests/unit/test_bailian.py
  tests/unit/test_pdf_parser.py tests/unit/test_text_office_parsers.py
  tests/unit/test_rag_service.py -q` — **exit 0**, 131 passed.
- `uv run pytest tests/unit -q` — **exit 0**, 291 passed.
- `RUN_INTEGRATION=1 QDRANT_BINARY=$PWD/var/qdrant/1.15.4/qdrant.exe uv run pytest
  tests/integration/test_qdrant_store.py tests/integration/test_index_builder.py
  tests/integration/test_rag_cli.py -q` — **exit 0**, 23 passed using the official
  standalone Qdrant binary.
- `RUN_INTEGRATION=1 QDRANT_BINARY=$PWD/var/qdrant/1.15.4/qdrant.exe uv run pytest
  tests/integration -m "not bailian" -q` — **exit 0**, 114 passed, 3 deselected, one
  Paddle/ccache warning, in 771.79 seconds. OCR tests were included because PDF selection
  changed.
- Final positive verifier rerun — **exit 0**, 1 passed in 4.92 seconds.
- `uv run ruff check .` — **exit 0**, `All checks passed!`.
- `git diff --check` — **exit 0**, no whitespace errors.

## Real-generation compatibility and verification path

An existing pre-wave database receives `NULL` fingerprints during schema compatibility.
That is deliberately **not** treated as compatible. `ask` and `verify` require a rebuild;
`rebuild` computes the current provider/model/dimension/distance fingerprint, finds no
reusable legacy hashes, embeds every child without fetching legacy vectors, stages new
Qdrant/BM25 artifacts, and atomically activates only after validation. Subsequent rebuilds
can reuse unchanged content only under that exact fingerprint.

The old persistent full-corpus generation was not falsely verified under the new contract.
Instead, `test_verify_reports_qdrant_dimension_and_bm25_manifest` built the smallest fresh,
fingerprinted fake-model generation against real standalone Qdrant and persistent BM25,
then obtained a fully successful structured verification with the exact 25/25/25/25 and
20+5 source shape and exact SQLite/Qdrant/BM25 child IDs. A full-corpus legacy rebuild and
live Bailian verification were not run.

## Unresolved residuals / deferred items

- Legacy generations intentionally remain unusable for query/reuse/positive verification
  until the explicit rebuild path above; no automatic migration can infer their historical
  provider/model/distance safely.
- Retired generations continue consuming storage. Production retention/cleanup remains a
  later lifecycle phase, as requested.
- Retry deadlines cannot cancel an already in-flight synchronous HTTP call; HTTPX request
  timeouts bound that operation, while the monotonic deadline bounds retry scheduling.
- Live Bailian APIs were **NOT RUN** (3 opt-in tests deselected). No live Bailian quality or
  relevance claim is made. Semantic version-conflict/relevance-threshold evaluation remains
  deferred, and no arbitrary relevance threshold was introduced.

## Self-review

Reviewed the complete diff and test changes. The implementation preserves atomic
activation, avoids old-vector reads on fingerprint mismatch, keeps degraded retrieval
behavior store-specific, compares exact persisted identities, sanitizes expected provider
errors, and leaves programming exceptions visible. Removed a duplicate test assertion and
normalized Qdrant's distance enum representation discovered by integration testing. No
known blockers remain within the accepted Phase 1 scope.
