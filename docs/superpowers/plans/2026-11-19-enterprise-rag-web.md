# Enterprise RAG Web Application Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a single-administrator FastAPI and React web application with persistent document tasks, one ingestion worker, secure sessions, document management, conversations, cited streaming chat, and local feedback on top of the approved RAG core.

**Architecture:** FastAPI serves the API and production React build while a separate single worker consumes leased SQLite tasks and calls the existing generation builder. Documents are versioned so a failed upload never replaces the active generation; chat uses the existing RAG service with generation-pinned retrieval and SSE. The server deployment remains Conda plus independent Qdrant, with no Docker dependency.

**Tech Stack:** Python 3.11, FastAPI, Uvicorn, SQLAlchemy 2, SQLite WAL, Argon2id, HTTPX, React 19, TypeScript, Vite, Ant Design, Vitest, Testing Library, Playwright, existing LangChain/Qdrant/RAG services.

## Global Constraints

- Preserve every Phase 1 public contract and passing test; do not rewrite retrieval, chunking, model adapters, index generations, or verification logic except for the explicitly assigned progress hook.
- Exactly one administrator account; no user management, roles, RBAC, document ACL, SSO, LDAP, or multi-tenancy.
- Server target is Ubuntu 22.04 with Conda `langchain_test` and an independent Qdrant binary. No server Docker commands, service changes, or deployment actions in this phase.
- Real Bailian calls require `RUN_BAILIAN_TESTS=1` and `DASHSCOPE_API_KEY`; otherwise they remain NOT RUN. Fake providers require both `MODEL_PROVIDER=fake` and `APP_ENV=test`.
- FastAPI defaults to `127.0.0.1:18000`, Qdrant to `127.0.0.1:16333/16334`. No changes to other ports, services, Nginx, Conda environments, or system files.
- Upload only TXT/DOCX/PDF/XLSX, with a separate `kind=qa` path for the Q&A workbook. Maximum 50 MB per file. Evaluation JSONL is never accepted for indexing.
- Sessions use a random opaque cookie token hashed in SQLite and a separate random CSRF token. Only password hashes are stored; no password in command arguments, logs, frontend storage, or API responses.
- One ingestion worker, persistent task leases, staged document versions, and no FastAPI `BackgroundTasks` for indexing.
- Every task uses TDD: capture expected RED, implement minimally, capture GREEN, run focused then full relevant tests, and commit only reviewed files.
- Do not implement LangSmith, semantic evaluation, paid monitoring, deployment scripts, or production lifecycle cleanup in this phase.

---

## File Structure

```text
app/
├── main.py                       # FastAPI composition root
├── worker.py                     # Single ingestion worker entry point
├── config.py                     # Add WebSettings only
├── db/
│   ├── models.py                 # Add web records, preserve core records
│   └── session.py                # Additive PRAGMA migrations
├── web/
│   ├── health.py                 # Liveness/readiness probes
│   ├── auth.py                   # Password/session/CSRF services
│   ├── queue.py                  # Persistent task lease and recovery
│   └── progress.py               # Document/task state transitions
├── api/
│   ├── deps.py                   # Current session and CSRF dependencies
│   ├── auth.py                   # Login/logout/me
│   ├── documents.py              # Documents/upload/content/reindex/delete
│   ├── tasks.py                  # Task status
│   ├── conversations.py          # Conversation list/rename/delete/messages
│   ├── chat.py                   # SSE stream
│   └── feedback.py               # Local feedback upsert
├── documents/
│   ├── storage.py                # Safe versioned raw-file storage
│   ├── service.py                # Upload/version/task orchestration
│   └── schemas.py                # Document/task API schemas
├── conversations/
│   └── service.py                # Conversation persistence
├── chat/
│   ├── schemas.py                # Chat/feedback schemas and events
│   └── service.py                # History/prepare/stream/persist
├── ingestion/
│   └── builder.py                # Optional progress callback only
├── frontend/
│   ├── package.json              # Scripts and exact direct dependencies
│   ├── package-lock.json
│   ├── .nvmrc                    # Node 22 LTS
│   ├── vite.config.ts
│   ├── tsconfig.json
│   ├── index.html
│   └── src/
│       ├── main.tsx
│       ├── App.tsx
│       ├── api/client.ts         # Cookie/CSRF/fetch/SSE client
│       ├── auth/AuthProvider.tsx
│       ├── pages/LoginPage.tsx
│       ├── pages/ChatPage.tsx
│       ├── pages/DocumentsPage.tsx
│       └── components/{CitationList,FeedbackButtons,UploadPanel}.tsx
├── scripts/run_dev.py             # API/worker development process guide
└── tests/
    ├── unit/test_web_*.py
    ├── integration/test_*_api.py
    └── e2e/test_web_flow.py
```

### Task 1: Add FastAPI Composition, Web Settings, and Health Probes

**Files:**
- Modify: `pyproject.toml`, `uv.lock`, `requirements.lock`, `app/config.py`, `configs/default.yaml`
- Create: `app/web/__init__.py`, `app/web/health.py`, `app/main.py`, `tests/unit/test_web_health.py`

**Interfaces:**
- Produces: `WebSettings(host="127.0.0.1", port=18000, session_cookie_name="rag_session", session_ttl_hours=12, secure_cookie=False, max_upload_mb=50, max_chat_concurrency=5, task_lease_seconds=300, raw_file_dir=Path("var/core/raw_files"), frontend_dist=Path("frontend/dist"))`.
- Produces: `create_app(settings: AppSettings | None = None, qdrant_probe: Callable[[], Mapping[str, Any]] | None = None) -> FastAPI`.

- [ ] **Step 1: Write failing health tests**

```python
def test_health_endpoints_expose_no_secrets():
    client = TestClient(create_app(settings=test_settings(), qdrant_probe=lambda: {"status": "ok"}))
    assert client.get("/health/live").json() == {"status": "ok"}
    ready = client.get("/health/ready")
    assert ready.status_code == 200
    assert ready.json()["database"] == "ok"
    assert ready.json()["qdrant"] == "ok"
    assert "dashscope_api_key" not in ready.text
```

Add a failing-probe test expecting HTTP 503 with sanitized `database="error"` or `qdrant="error"` and no raw exception text.

Define `test_settings()` in the test file to return an `AppSettings` with `app_env="test"`, `model_provider="fake"`, and these web values.

- [ ] **Step 2: Run health tests and verify RED**

Run `uv run pytest tests/unit/test_web_health.py -v`.
Expected: import failure for `app.main`.

- [ ] **Step 3: Add dependencies and typed web settings**

Add `fastapi`, `uvicorn[standard]`, `python-multipart`, and `argon2-cffi` to `pyproject.toml`; regenerate `uv.lock` and hashed `requirements.lock` using the Task 1 export command. Add `WebSettings` and `AppSettings.web` with the exact defaults above, including YAML `web` values and positive validators for port/upload/lease/session TTL.

- [ ] **Step 4: Implement the application factory and probes**

```python
def create_app(settings=None, qdrant_probe=None):
    settings = settings or load_settings()
    app = FastAPI(title="Enterprise RAG", version="0.2.0")
    app.state.settings = settings
    app.state.qdrant_probe = qdrant_probe or default_qdrant_probe
    app.include_router(health_router)
    return app
```

`/health/live` touches only the process. `/health/ready` opens SQLite through the existing factory and runs the injected Qdrant probe; no active generation is required for a fresh install to be ready. Return sanitized errors and never include configuration secrets.

- [ ] **Step 5: Verify health tests and lock files**

Run `uv run pytest tests/unit/test_web_health.py -v`, `uv run ruff check app/main.py app/web app/config.py`, and `uv lock --check`.
Expected: all pass, lock contains new packages and no secrets.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock requirements.lock app/config.py configs/default.yaml app/main.py app/web tests/unit/test_web_health.py
git commit -m "feat: add FastAPI composition and health probes"
```

### Task 2: Add Web Persistence and Additive SQLite Migrations

**Files:**
- Modify: `app/db/models.py`, `app/db/session.py`
- Create: `app/web/records.py`, `tests/unit/test_web_db.py`

**Interfaces:**
- Produces: `AdminRecord`, `AuthSessionRecord`, `DocumentVersionRecord`, `TaskRecord`, `ConversationRecord`, `MessageRecord`, `FeedbackRecord`, and `ensure_web_schema(session_factory)`.
- Extends: `DocumentRecord` with `filename`, `media_type`, `status`, `current_version_id`, `error`, `created_at`, `updated_at`, `deleted`.

- [ ] **Step 1: Write failing record and legacy-migration tests**

```python
def test_web_records_and_legacy_document_columns_round_trip(tmp_path):
    factory = create_session_factory(tmp_path / "web.sqlite3")
    ensure_web_schema(factory)
    with factory() as session:
        session.add(DocumentRecord(
            id="doc-1", content_hash="abc", metadata_json={}, filename="a.pdf",
            media_type="application/pdf", status="ready"
        ))
        session.add(DocumentVersionRecord(
            id="v1", document_id="doc-1", version=1, path="raw/a.pdf",
            sha256="abc", status="ready"
        ))
        session.add(TaskRecord(id="t1", document_id="doc-1", kind="ingest", status="queued"))
        session.commit()
        assert session.get(DocumentVersionRecord, "v1").path == "raw/a.pdf"
        assert session.get(TaskRecord, "t1").status == "queued"
```

Create a second database with the old `documents(id, content_hash, metadata_json)` schema and assert `ensure_web_schema` adds columns without deleting old values. Use timezone-aware UTC datetimes and defaults for new columns.

- [ ] **Step 2: Run persistence tests and verify RED**

Run `uv run pytest tests/unit/test_web_db.py -v`.
Expected: missing web record definitions or migration function.

- [ ] **Step 3: Implement records and additive schema creation**

Use `app/web/records.py` for new tables, and add only the listed `DocumentRecord` columns to `app/db/models.py`. Required statuses are document version `uploaded/parsing/chunking/embedding/indexing/ready/failed` and task `queued/running/succeeded/failed`. Task fields must include `id`, `document_id`, `version_id`, `kind`, `status`, `attempts`, `max_attempts`, `payload_json`, `lease_owner`, `lease_expires_at`, `error`, `created_at`, `updated_at`.

Authentication, conversation, message, and feedback tables use the exact fields in the later task interfaces; do not add roles or tenant columns. `MessageRecord.citations_json` stores JSON, `cancelled` is a Boolean defaulting to false, `error` is nullable sanitized text, and `FeedbackRecord.message_id` is unique for single-user upsert.

- [ ] **Step 4: Implement idempotent additive migrations**

`ensure_web_schema` imports all mappings, creates new tables, and uses `PRAGMA table_info(documents)` before issuing one explicit additive statement per missing column with its name, type, and safe default. Never drop or recreate the existing table. Run it once from `create_session_factory` after core schema/fingerprint migration.

- [ ] **Step 5: Verify migrations and commit**

Run `uv run pytest tests/unit/test_web_db.py tests/unit/test_db.py -v` and `uv run ruff check app/db app/web/records.py`.
Expected: new and old records pass. Commit with message `feat: add web persistence and additive migrations`.

### Task 3: Implement Minimal Single-Account Login

**Files:**
- Create: `app/web/auth.py`, `app/api/deps.py`, `app/api/auth.py`, `tests/unit/test_auth.py`, `tests/integration/test_auth_api.py`

**Interfaces:**
- Produces: `AuthService.ensure_admin(username, password)`, `login(username, password) -> tuple[raw_session_token, csrf_token, expires_at]`, `logout(raw_token)`, and `AuthSession(csrf_token, expires_at)`.
- Produces: `get_auth_session(request) -> AuthSession` and `require_csrf(request, session) -> None`.
- Explicitly excludes: login rate limiting, CSRF rotation, interactive admin tools, registration, password reset, user management, roles, and permissions.

- [ ] **Step 1: Write failing minimal-login tests**

Cover one environment-configured account, Argon2id verification, opaque token hashes, login cookie flags, `/me`, logout, expiration, wrong/missing CSRF, and stable CSRF across repeated `/me` calls and two tabs. Assert there is no rate limiter and no interactive account command.

```python
def test_login_and_static_csrf(client, admin):
    response = client.post("/api/auth/login", json={"username": "admin", "password": "correct-password"})
    assert response.status_code == 200
    assert "HttpOnly" in response.headers["set-cookie"]
    first = client.get("/api/auth/me").json()["csrf_token"]
    second = client.get("/api/auth/me").json()["csrf_token"]
    assert first == second
    assert client.post("/api/auth/logout", headers={"X-CSRF-Token": first}).status_code == 204
```

- [ ] **Step 2: Run auth tests and verify RED**

Run `uv run pytest tests/unit/test_auth.py tests/integration/test_auth_api.py -v`.
Expected: missing auth service/routes.

- [ ] **Step 3: Implement minimal password and session services**

Use only `ADMIN_USERNAME` and `ADMIN_PASSWORD` to provision the sole account through `ensure_admin`; store Argon2id hashes, reject any second account, and never log or return the password. Use `secrets.token_urlsafe(32)` for session and CSRF tokens and SHA-256 token hashes in `AuthSessionRecord`. Only the cookie carries the raw session token. Do not create `web_admin.py` or any account-management command.

- [ ] **Step 4: Implement login/logout/me and dependencies**

Login sets `HttpOnly`, `SameSite=lax`, `Path=/`, `Max-Age`, and `Secure` only when `web.secure_cookie=True`. Return the same CSRF token at login and every `/me` call without rotation. Authenticated mutating routes require `X-CSRF-Token` compared with `secrets.compare_digest`. Do not add a rate limiter, password reset, registration, role checks, or permission middleware.

- [ ] **Step 5: Verify minimal login and commit**

Run `uv run pytest tests/unit/test_auth.py tests/integration/test_auth_api.py -v` and `uv run ruff check app/web app/api`.
Expected: all pass with no secret in responses, no rate-limiter code, and no account command. Commit with message `feat: add minimal single-account login`.

### Task 4: Add Persistent Task Leases, Worker Recovery, and Index Progress

**Files:**
- Create: `app/web/queue.py`, `app/web/progress.py`, `app/worker.py`, `tests/unit/test_task_queue.py`, `tests/integration/test_worker_recovery.py`
- Modify: `app/indexing/builder.py`

**Interfaces:**
- Produces: `TaskQueue.enqueue(document_id, version_id, kind)`, `claim_next(owner: str, lease_seconds: int) -> TaskRecord | None`, `claim_batch(owner, lease_seconds) -> list[TaskRecord]`, `complete(task_id)`, `fail(task_id, error)`, `recover_expired() -> int`.
- Extends: `IndexBuilder.build(documents, progress: Callable[[str], None] | None = None) -> str` with unchanged default behavior.

- [ ] **Step 1: Write failing lease/recovery/progress tests**

```python
def test_only_one_worker_claims_task_and_expired_lease_recovers(factory):
    queue = TaskQueue(factory)
    queue.enqueue(document_id="doc-1", version_id="v1", kind="ingest")
    first = queue.claim_next("worker-a", 1)
    assert first is not None
    assert queue.claim_next("worker-b", 1) is None
    with factory() as session:
        task = session.get(TaskRecord, first.id)
        task.lease_expires_at = datetime.now(UTC) - timedelta(seconds=1)
        session.commit()
    assert queue.recover_expired() == 1
    assert queue.claim_next("worker-b", 1).id == first.id
```

Add a batch test proving `claim_batch` atomically returns all queued tasks, and a same-document test proving only the latest pending version is selected for the next build while older queued tasks become `succeeded` with `payload_json["superseded_by"]` set to the newest task ID only after that new generation activates. A failed newest build must not mark older work as successfully superseded.

Add a progress test asserting `build()` emits `chunking`, `embedding`, and `indexing` before activation and never emits `ready` when validation fails.

- [ ] **Step 2: Run queue/progress tests and verify RED**

Run `uv run pytest tests/unit/test_task_queue.py tests/integration/test_worker_recovery.py -v`.
Expected: missing queue/progress behavior.

- [ ] **Step 3: Implement atomic task leases**

Use one SQLite transaction per claim with `UPDATE tasks SET status='running', lease_owner=?, lease_expires_at=?, attempts=attempts+1 WHERE id=(SELECT id FROM tasks WHERE status='queued' ORDER BY created_at LIMIT 1) AND attempts < max_attempts`. `claim_batch` returns every currently queued task after one claim transaction so a single worker can coalesce builds. `complete/fail` require the current owner and `recover_expired` returns expired running tasks to queued without changing active generation.

- [ ] **Step 4: Add optional index progress and worker lifecycle**

Invoke `progress("chunking")`, `progress("embedding")`, and `progress("indexing")` at the existing builder boundaries; do not change vector or activation logic. `app/worker.py` recovers expired tasks, then claims batches, marks `embedding` before the actual embedding stage and `indexing` before Qdrant/BM25 writes, and persists terminal states. It must not start a second worker or use FastAPI background tasks.

- [ ] **Step 5: Verify queue/recovery and commit**

Run the focused tests plus `uv run pytest tests/unit/test_index_repository.py -v` and Ruff. Expected: atomic claims, recovery, and progress pass. Commit with message `feat: add persistent ingestion queue and worker recovery`.

### Task 5: Implement Versioned Document Storage and Ingestion Orchestration

**Files:**
- Create: `app/documents/storage.py`, `app/documents/service.py`, `app/documents/schemas.py`, `tests/unit/test_document_storage.py`, `tests/integration/test_document_service.py`

**Interfaces:**
- Produces: `store_upload(filename, content: BinaryIO, kind: Literal["document", "qa"]) -> StoredUpload`, `DocumentService.upload(filename: str, content: BinaryIO, kind: Literal["document", "qa"]) -> tuple[DocumentVersionRecord, TaskRecord]`, `build_pending(owner: str) -> str`, and `documents_for_tasks(tasks) -> dict[str, list[ParsedBlock]]`.

- [ ] **Step 1: Write failing storage validation tests**

Reject `../secret.txt`, `..\\secret.txt`, absolute paths, EXE/ZIP-disguised files, evaluation JSONL, and files over 50 MB. Verify PDF `%PDF-` and Office ZIP signatures, TXT decoding, and Q&A exact six columns. Compute expected SHA-256 from streamed content without loading the entire upload into memory.

- [ ] **Step 2: Run storage tests and verify RED**

Run `uv run pytest tests/unit/test_document_storage.py -v`.
Expected: missing storage implementation.

- [ ] **Step 3: Implement safe versioned storage**

Reject control characters, reserved names, and either slash variant before taking `Path(filename).name`, then save as `raw_file_dir/document_id/version_id/original_name`. Keep `StoredUpload(document_id, version_id, path, sha256, media_type, kind)`. Upload creates a new version and queued task; it never overwrites `current_version_id` or the accepted raw file.

- [ ] **Step 4: Implement pending generation orchestration**

`documents_for_tasks` combines currently ready versions with the latest pending version per document, excluding deleted/failed versions and `evaluation.jsonl`. It uses `ParserRegistry.default()` for ordinary documents and `QAParser` only when `kind="qa"`. `build_pending` calls the existing `build_index_builder(settings).build(documents_for_tasks(tasks))` after adapters validate credentials, and only on success updates current pointers, task statuses, and document statuses. On failure, previous current versions and active generation remain untouched.

- [ ] **Step 5: Verify success/replacement/failure and commit**

Test first upload, replacement success, replacement parse failure preserving old content, and builder failure preserving active generation. Run focused tests plus Ruff. Commit with message `feat: add versioned document ingestion orchestration`.

### Task 6: Expose Document and Task APIs

**Files:**
- Modify: `app/documents/schemas.py`
- Create: `app/api/documents.py`, `app/api/tasks.py`, `tests/integration/test_documents_api.py`

**Interfaces:**
- Produces: `GET /api/documents`, `POST /api/documents/upload`, `GET /api/documents/{id}`, `GET /api/documents/{id}/content`, `POST /api/documents/{id}/reindex`, `DELETE /api/documents/{id}`, `GET /api/tasks/{id}`.

- [ ] **Step 1: Write failing API tests**

Test authenticated upload/list/detail/download/reindex/delete/task status. Assert upload returns HTTP 202 with `document_id`, `version_id`, and `task_id`; delete also returns 202 and does not immediately remove active content. Add unauthenticated 401, wrong CSRF 403, invalid file 400, oversize 413, and pending-download 409 tests.

- [ ] **Step 2: Run API tests and verify RED**

Run `uv run pytest tests/integration/test_documents_api.py -v`.
Expected: 404 for document routes.

- [ ] **Step 3: Implement typed document schemas and routes**

Return `DocumentOut(id, filename, media_type, status, current_version_id, latest_version_id, parent_count, child_count, source_metadata, error, created_at, updated_at)` and `TaskOut(id, kind, status, attempts, error)`. Populate counts from the active generation only; a pending document has zero counts until activation. All routes require auth and mutations require CSRF. Stream uploads through `store_upload`; do not buffer 50 MB in memory.

- [ ] **Step 4: Implement content download and task transitions**

`content` serves only the accepted current file with `FileResponse` and a safe `Content-Disposition` filename. Reindex/delete enqueue tasks and return 202. Include `TaskQueue` status/error fields sanitized through the existing sanitizer. No active-generation deletion occurs in the route.

- [ ] **Step 5: Verify API behavior and commit**

Run `uv run pytest tests/integration/test_documents_api.py tests/integration/test_auth_api.py -v` and Ruff. Expected: all pass. Commit with message `feat: add document and task management APIs`.

### Task 7: Add Conversation Persistence and Routes

**Files:**
- Create: `app/conversations/service.py`, `app/conversations/schemas.py`, `app/api/conversations.py`, `tests/integration/test_conversations_api.py`

**Interfaces:**
- Produces: `ConversationService.list() -> list[ConversationOut]`, `create(title: str | None)`, `rename(id, title)`, `delete(id)`, and `messages(id, limit=6) -> list[MessageRecord]`.

- [ ] **Step 1: Write failing conversation tests**

Verify create/list/rename/delete and chronological messages. Deleting a conversation removes its messages and feedback through explicit relationships. The same administrator sees all conversations; no user/tenant filtering is added.

- [ ] **Step 2: Run conversation tests and verify RED**

Run `uv run pytest tests/integration/test_conversations_api.py -v`.
Expected: 404 for conversation routes.

- [ ] **Step 3: Implement conversation service and schemas**

Use titles up to 200 characters, defaulting to `新会话`. `messages` returns the latest N in chronological order for chat history, and a separate `list_messages` returns all for display. Return `ConversationOut(id,title,created_at,updated_at)` and `MessageOut(id,role,content,citations,degraded,generation_id,created_at)`.

- [ ] **Step 4: Implement authenticated conversation routes**

`GET /api/sessions`, `POST /api/sessions`, `PATCH /api/sessions/{id}`, `DELETE /api/sessions/{id}`, and `GET /api/sessions/{id}/messages`. Require CSRF on create/rename/delete, and return 404 for unknown IDs.

- [ ] **Step 5: Verify and commit**

Run conversation/auth tests and Ruff. Expected: all pass. Commit with message `feat: add persistent conversations and session routes`.

### Task 8: Implement SSE Chat and Local Feedback

**Files:**
- Create: `app/chat/schemas.py`, `app/chat/service.py`, `app/api/chat.py`, `app/api/feedback.py`, `tests/integration/test_chat_api.py`

**Interfaces:**
- Produces: `ChatService.prepare(question: str, conversation_id: str) -> RAGPreparation`, `stream(conversation_id: str, prepared: RAGPreparation) -> AsyncIterator[ChatEvent]`, and `upsert_feedback(message_id, score, comment) -> FeedbackRecord`.

- [ ] **Step 1: Write failing chat and feedback tests**

Use a fake `RAGService` returning two chunks and two citations. Assert SSE order `start`, `token`, `token`, `citations`, `done`, valid JSON per event, and persisted user/assistant messages with degraded/generation metadata. Test first-turn history, last-six-turn selection, no-generation error, provider error sanitization, partial-answer cancellation, and a sixth concurrent request receiving HTTP 429. Feedback upserts -1/1 and comment up to 2000 characters without a LangSmith run ID.

- [ ] **Step 2: Run chat tests and verify RED**

Run `uv run pytest tests/integration/test_chat_api.py -v`.
Expected: 404 for chat/feedback routes.

- [ ] **Step 3: Implement chat preparation and persistence**

`POST /api/chat/stream` accepts `{conversation_id?: string, question: string}`. Create a conversation when absent, persist the user message, load at most six previous messages before it, read active generation once, and call the existing `RAGService.prepare`. Map stored roles to `HumanMessage`/`AIMessage`. Guard execution with an `asyncio.Semaphore(5)` shared by the application; return HTTP 429 immediately when all five slots are occupied.

- [ ] **Step 4: Implement SSE and disconnect behavior**

Emit `start` with IDs/generation/degraded, then `token`, `citations`, `done`, or sanitized `error`. Accumulate answer text and persist citations only from `RAGPreparation`. On client disconnect, cancel the upstream generator and persist partial content with `cancelled=true`; never present a partial answer as complete.

- [ ] **Step 5: Implement local feedback and verify**

`POST /api/feedback` requires message ID, `score ∈ {-1,1}`, and optional comment ≤2000. Upsert locally with no LangSmith association. Run chat/auth tests and Ruff. Expected: all pass. Commit with message `feat: add cited SSE chat and local feedback`.

### Task 9: Build the React Shell, Login, and Streaming Chat Page

**Files:**
- Create: `frontend/package.json`, `frontend/package-lock.json`, `frontend/.nvmrc`, `frontend/vite.config.ts`, `frontend/tsconfig.json`, `frontend/index.html`, `frontend/src/main.tsx`, `frontend/src/App.tsx`, `frontend/src/api/client.ts`, `frontend/src/auth/AuthProvider.tsx`, `frontend/src/pages/LoginPage.tsx`, `frontend/src/pages/ChatPage.tsx`, `frontend/src/components/CitationList.tsx`, `frontend/src/components/FeedbackButtons.tsx`, `frontend/src/pages/ChatPage.test.tsx`

**Interfaces:**
- Produces: cookie-based `apiFetch`, `login/logout/me`, and `streamChat(request, handlers, signal)`.
- Produces: routes `/login`, `/chat`, `/chat/:id`, and `/documents` placeholder replaced in Task 10.

- [ ] **Step 1: Write failing API/client and chat-component tests**

Mock `/api/auth/me`, login/logout, sessions, and SSE. Assert unauthenticated users redirect to `/login`, successful login stores only CSRF in memory, streamed tokens append in order, citations render, stop aborts, and feedback submits score/comment. No session token or password may appear in localStorage.

- [ ] **Step 2: Run frontend tests and verify RED**

Run `npm --prefix frontend install --package-lock-only`, `npm --prefix frontend ci`, then `npm --prefix frontend test -- ChatPage`.
Expected: missing pages/API client. Node target is 22 LTS; do not require Node on the production server.

- [ ] **Step 3: Implement the typed API client and auth provider**

`apiFetch` uses `credentials: "same-origin"`, JSON headers, and `X-CSRF-Token` for mutations. `streamChat` parses SSE by event boundaries and handles `start/token/citations/done/error` without treating raw event text as HTML. Store CSRF only in React memory.

- [ ] **Step 4: Implement login and chat pages**

Use Ant Design responsive layout with a conversation sidebar, new/rename/delete controls, Markdown answer area, degraded badge, stop button, collapsible citation metadata, and feedback controls. Render source text with `react-markdown` and do not enable raw HTML. Fetch `/api/sessions` after login and refresh after `done`.

- [ ] **Step 5: Verify frontend checks and production build**

Run `npm --prefix frontend run lint`, `npm --prefix frontend run typecheck`, `npm --prefix frontend test`, and `npm --prefix frontend run build`.
Expected: all pass and `frontend/dist` exists. Commit with message `feat: add React authentication and streaming chat UI`.

### Task 10: Build the Document Management Page and Static SPA Delivery

**Files:**
- Create: `frontend/src/pages/DocumentsPage.tsx`, `frontend/src/components/UploadPanel.tsx`, `frontend/src/pages/DocumentsPage.test.tsx`
- Modify: `app/main.py`, `tests/unit/test_web_health.py` or create `tests/unit/test_static_spa.py`

**Interfaces:**
- Produces: document upload/list/download/reindex/delete UI and production FastAPI SPA fallback.

- [ ] **Step 1: Write failing document-page and SPA tests**

Mock document/task APIs and assert upload selection, progress refresh, error display, reindex/delete confirmation, and download URL. Static tests assert `/chat` returns `index.html`, `/api/unknown` remains JSON 404, and missing static assets are not redirected to the SPA.

- [ ] **Step 2: Run page/SPA tests and verify RED**

Run `npm --prefix frontend test -- DocumentsPage` and `uv run pytest tests/unit/test_static_spa.py -v`.
Expected: missing page and fallback.

- [ ] **Step 3: Implement document management UI**

Use Ant Design upload list, file-type hints, 50 MB limit, status tags for queued/running/ready/failed, task error display, reindex/delete confirmations, and polling every 2 seconds only while tasks are active. Stop polling on unmount and never expose filesystem paths or secrets.

- [ ] **Step 4: Implement production static delivery**

When `frontend/dist` exists, mount `/assets` and serve `index.html` for non-API GET paths. `/api`, `/health`, missing `/assets`, and API errors never fall through to HTML. Development API and production static files use the same FastAPI process and origin.

- [ ] **Step 5: Verify page, SPA, and builds**

Run `npm --prefix frontend test -- DocumentsPage`, `npm --prefix frontend run build`, and `uv run pytest tests/unit/test_static_spa.py -v`.
Expected: all pass. Commit with message `feat: add document management UI and SPA delivery`.

### Task 11: Run End-to-End Validation and Document the Web Workflow

**Files:**
- Create: `scripts/run_dev.py`, `scripts/install_qdrant_binary.py`, `tests/e2e/test_web_flow.py`, `frontend/e2e/web-flow.spec.ts`, `docs/web.md`, `README.md`

**Interfaces:**
- Produces: repeatable API and browser smoke flows, development process instructions, and explicit validation evidence boundaries.

- [ ] **Step 1: Write the end-to-end acceptance test**

With `MODEL_PROVIDER=fake`, `APP_ENV=test`, a temporary `RAG_STATE_DIR`, and official standalone Qdrant, use two small TXT fixtures. Assert login → upload → task completion → chat answer/citations → feedback → reindex → delete, plus fresh-install readiness and no active-generation change on a bad upload. Add Playwright for the same critical browser path with visible result statuses.

- [ ] **Step 2: Implement the development process helper and documentation**

`run_dev.py` prints separate commands for API and worker instead of starting an invisible process group. `install_qdrant_binary.py` downloads Qdrant 1.15.4 from the official GitHub API into `var/qdrant`, verifies asset size/type, records the downloaded archive SHA-256, checks the executable reports version 1.15.4, and supports the current Windows development host and Ubuntu 22.04 x86_64 without Docker. `docs/web.md` includes fake/real setup, binary installation, admin bootstrap, npm build, API/worker startup, test commands, upload limits, security boundaries, and no-Docker deployment direction. `README.md` links core/web docs and current phase status without claiming production deployment.

- [ ] **Step 3: Run API end-to-end and browser smoke tests**

Run:

```bash
RUN_INTEGRATION=1 QDRANT_BINARY="$PWD/var/qdrant/1.15.4/qdrant.exe" uv run pytest tests/e2e/test_web_flow.py -v
npm --prefix frontend run build
npm --prefix frontend exec -- playwright install chromium
npm --prefix frontend exec -- playwright test e2e/web-flow.spec.ts
```

Expected: API flow passes. If Chromium cannot install, report browser tests NOT RUN with the exact error rather than claiming browser acceptance.

- [ ] **Step 4: Run final regression and commit**

Run `uv run pytest tests/unit -q`, `RUN_INTEGRATION=1 uv run pytest tests/integration -m "not bailian" -q`, `uv run ruff check .`, `npm --prefix frontend run lint`, `npm --prefix frontend run typecheck`, `npm --prefix frontend test`, and `npm --prefix frontend run build`.
Expected: all relevant tests pass; live Bailian remains NOT RUN. Commit with message `feat: deliver single-admin RAG web application`.

## Phase Completion Evidence

Record actual outputs for:

- Single-account login, cookie, stable CSRF, expiration, and no-login-limiter checks.
- Document version replacement/failure safety and worker lease recovery.
- SSE event order, persisted citations, cancellation, and local feedback.
- React unit/build checks and API/browser critical flows.
- Full unit/integration suites and any tests not run, especially live Bailian or browser limitations.
- No server deployment, Docker, external service, or other environment was modified.
