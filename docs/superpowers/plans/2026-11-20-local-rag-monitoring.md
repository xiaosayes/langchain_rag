# Local RAG Monitoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox syntax.

**Goal:** Add account-free, fully local complete-chat traces, latency summaries, a read-only React monitoring page, JSONL diagnostics, and manual 30-day retention without changing RAG behavior.

**Architecture:** A fail-open MonitoringService persists run/event records in the existing SQLite database and appends terminal JSONL records. Explicit optional run IDs flow through the existing API, chat, RAG, and retrieval methods; omitted IDs preserve existing behavior. Authenticated read-only APIs feed one React page.

**Tech Stack:** Python 3.11, SQLAlchemy 2, FastAPI, LangChain, pytest, React 19, TypeScript, Ant Design, Node 22.

## Global Constraints

- No external account, LangSmith, Docker, new daemon, paid service, or global environment change.
- Do not alter retrieval ranking, chunking, model settings, generation activation, citations, or SSE event schemas.
- No like/dislike statistics. Existing feedback submission remains untouched.
- Monitoring persistence failures must never break chat; log sanitized warnings only.
- Never persist API keys, passwords, cookies, CSRF tokens, or tracebacks.
- Full question, answer, prompts and retrieved text remain on this server. Default retention is 30 days with explicit manual cleanup only.
- The root guide must explain every added/modified code file, block by block and line by line in accessible Chinese.
- Real Bailian and server validation require explicit user execution; otherwise mark NOT RUN.

### Task 1: Monitoring persistence and fail-open service

**Files:** Modify `app/web/records.py`, `app/db/session.py` if migration registration requires it; create `app/monitoring/__init__.py`, `app/monitoring/service.py`, `tests/unit/test_monitoring_service.py`, `tests/unit/test_monitoring_db.py`.

**Produces:** `MonitoringService(session_factory, state_root)`, `start_run(question, conversation_id) -> str`, `record_event(run_id, stage, status, started_at, duration_ms, details=None)`, `update_preparation(run_id, prepared, prompt_messages)`, `finish_run(run_id, status, answer, error=None)`, `summary(hours)`, `list_runs(limit, status, query)`, `get_run(run_id)`, `purge(before) -> int`.

- [ ] Write failing tests for incremental migration, one terminal run with ordered events, detail queries, duration/P95 summary, status/query filters, and SQL/JSONL failure fail-open.
- [ ] Run `uv run pytest tests/unit/test_monitoring_service.py tests/unit/test_monitoring_db.py -v`; confirm ImportError/assertion failures.
- [ ] Add `ChatRunRecord` and `ChatRunEventRecord` with foreign-key cascade and indexes. Add JSONL terminal snapshots under `state_root/monitoring/chat-trace.jsonl` using atomic append and owner-only permissions where supported.
- [ ] Implement `MonitoringService` with UTC timestamps, `perf_counter` durations supplied by callers, bounded list limits, no traceback logging, and exceptions contained inside monitoring methods.
- [ ] Re-run focused tests and all existing unit tests; Ruff; commit `feat: add local monitoring persistence`.

### Task 2: Chat and retrieval instrumentation

**Files:** Modify `app/api/chat.py`, `app/chat/service.py`, `app/rag/service.py`, `app/retrieval/service.py`, `app/main.py`; create/modify focused chat and retrieval tests.

**Consumes:** Task 1 public service methods. **Produces:** optional `run_id` keyword arguments preserving direct legacy calls.

- [ ] Write failing tests proving success, preparation failure, provider failure and disconnect/cancel produce exactly one terminal trace with partial answers and no SSE schema changes.
- [ ] Test dense, BM25, RRF, rerank, parent recovery and chat stages with nonnegative durations; full prompt and retrieved text are stored locally.
- [ ] Inject one MonitoringService in `create_app`; API starts a run and records preparation errors. Chat/RAG/retrieval accept optional run IDs and instrument stages without changing returned objects.
- [ ] Ensure `MonitoringService` errors cannot replace provider errors or consume a stream. Existing no-monitoring direct service calls remain valid.
- [ ] Run chat/RAG/retrieval tests plus all unit tests and Ruff; commit `feat: trace local RAG chat lifecycle`.

### Task 3: Authenticated monitoring API and retention CLI

**Files:** Create `app/api/monitoring.py`, `app/monitoring/retention.py`, `tests/integration/test_monitoring_api.py`, `tests/unit/test_monitoring_retention.py`; modify `app/main.py`, `app/cli.py`, `app/config.py`, `configs/default.yaml`.

**Produces:** `GET /api/monitoring/summary?hours=24`, `GET /api/monitoring/runs?limit=50&status=&query=`, `GET /api/monitoring/runs/{run_id}`, and `python -m app.cli monitoring-purge --days 30`.

- [ ] Write failing API tests for 401 unauthenticated, authenticated summary/list/detail, invalid status/limit/hours, no credential fields, and no feedback aggregates.
- [ ] Write failing retention tests for old/new DB rows, JSONL compaction, unrelated session/message/feedback survival, and idempotent second purge.
- [ ] Implement read-only routes using existing `get_auth_session`; validate hours 1..8760, limit 1..100, statuses running/succeeded/failed/cancelled.
- [ ] Implement explicit purge by cutoff with DB transaction and JSONL temporary-file replacement; print deleted count. Add `monitoring.retention_days: 30` typed config.
- [ ] Run focused API/retention and existing integration tests, Ruff; commit `feat: add monitoring API and retention`.

### Task 4: React monitoring page

**Files:** Create `frontend/src/pages/MonitoringPage.tsx`, `frontend/src/pages/MonitoringPage.test.tsx`, optionally `frontend/src/api/monitoring.ts`; modify `frontend/src/App.tsx`, `frontend/src/pages/ChatPage.tsx`, `frontend/src/pages/DocumentsPage.tsx`, `frontend/src/styles.css`.

- [ ] Write failing tests for summary cards, successful/failed/cancelled/degraded labels, status filter, recent runs, detail prompt/retrieved text/answer/timeline, API error, and no like/dislike statistics.
- [ ] Implement typed client functions and an authenticated monitoring route using existing AuthProvider/apiFetch.
- [ ] Add navigation links without changing chat/document behavior. Render user content as text, never HTML. No polling is required; provide manual refresh.
- [ ] Run Node 22 tests, lint, typecheck, build and existing static SPA tests; commit `feat: add local monitoring dashboard`.

### Task 5: Complete Chinese guide and final verification

**Files:** Create `MONITORING_IMPLEMENTATION_GUIDE.md`; modify `README.md`, `docs/web.md`; create `.superpowers/sdd/2026-11-20-local-rag-monitoring/progress.md` and task reports.

- [ ] Inventory every production/test/config file added or modified by Tasks 1–4 using `git diff --name-only main...HEAD`.
- [ ] For every code file, document the whole file purpose, each block's function, and each line's concrete role in plain Chinese. Do not limit coverage to key snippets; include tests and configuration changed by this feature. Use anchored source code blocks and line-numbered explanations so readers can map code to prose.
- [ ] Document architecture, request lifecycle, schema, JSONL example with synthetic content, API examples, retention, fail-open, security, deployment update/restart, browser verification, rollback, and explicit real-server NOT RUN boundaries.
- [ ] Run complete Python unit/supported integration/API E2E, Ruff, Node 22 frontend tests/lint/typecheck/build, Chromium E2E if available, `git diff --check`, and clean-tree checks.
- [ ] Independent spec and quality review; only targeted fixes for demonstrated core failures. Commit docs `docs: explain local monitoring implementation line by line` and final fixes separately.

## Integration

After all tasks pass, present merge/push choices. The user already requested push to `https://github.com/xiaosayes/langchain_rag.git`; merge locally first, then push `main` using existing credentials. Never ask for or print credentials.
