# Single-admin RAG web workflow

This phase provides a loopback-oriented, single-administrator FastAPI/React application,
a persistent ingestion worker, and repeatable local smoke tests. It is **not a production
deployment**. No server, Docker, service manager, Nginx, or Conda configuration is
installed or changed by this workflow.

## Prerequisites and official Qdrant binary

Use Python 3.11 with `uv`, and Node.js 22 with npm. From the repository root:

```bash
uv sync
npm --prefix frontend ci
```

The supported no-Docker development hosts are Windows x86_64 and Ubuntu 22.04 x86_64.
Install Qdrant from the official GitHub release API:

```bash
uv run python scripts/install_qdrant_binary.py
```

The installer selects only Qdrant **1.15.4**, verifies the GitHub asset name, declared
size and media type, checks the ZIP/gzip signature, rejects unsafe archive members, and
runs the executable's `--version` check. It writes the executable to
`var/qdrant/1.15.4/qdrant.exe` on Windows or `var/qdrant/1.15.4/qdrant` on Ubuntu.
The downloaded archive and `install-manifest.json` remain beside it. The manifest records
the archive SHA-256, size, official URL, media type, and version output. This recorded
digest supports repeatability/auditing; it is not an independent upstream signature or
checksum verification. Python's downloader falls back to the system `curl` using the same
official URL when a development network interrupts Python TLS.

To run a persistent local Qdrant on the application's default port, create the ignored
file `var/qdrant/dev-config.yaml`:

```yaml
storage:
  storage_path: "./var/qdrant/storage"
  snapshots_path: "./var/qdrant/snapshots"
service:
  host: "127.0.0.1"
  http_port: 16333
  grpc_port: 16334
telemetry_disabled: true
```

Keep Qdrant running in its own visible terminal from the repository root:

```bash
# Windows PowerShell
.\var\qdrant\1.15.4\qdrant.exe --config-path var/qdrant/dev-config.yaml --disable-telemetry

# Ubuntu 22.04 x86_64
var/qdrant/1.15.4/qdrant --config-path var/qdrant/dev-config.yaml --disable-telemetry
```

The smoke fixtures instead start their own official binary on isolated ephemeral ports.

## Fake-provider development

Fake embeddings, reranking, and chat are deterministic test doubles. They require **both**
`MODEL_PROVIDER=fake` and `APP_ENV=test`; fake mode with production settings is rejected.
They make no Bailian requests and their answers are not quality-evaluation evidence.

Windows PowerShell example:

```powershell
$env:MODEL_PROVIDER = "fake"
$env:APP_ENV = "test"
$env:RAG_STATE_DIR = "$PWD/var/web-dev"
$env:ADMIN_USERNAME = "admin"
$env:ADMIN_PASSWORD = "replace-with-a-local-development-password"
```

Ubuntu/bash example:

```bash
export MODEL_PROVIDER=fake
export APP_ENV=test
export RAG_STATE_DIR="$PWD/var/web-dev"
export ADMIN_USERNAME=admin
export ADMIN_PASSWORD='replace-with-a-local-development-password'
```

`ADMIN_USERNAME` and `ADMIN_PASSWORD` must be set together. Startup creates or updates
that one administrator in the selected SQLite state. Do not reuse a real password in a
shared development environment. `RAG_STATE_DIR` isolates SQLite and BM25 state. Raw uploads
use `web.raw_file_dir` from `configs/default.yaml` independently; the automated smoke
harnesses explicitly place both state and raw uploads in temporary directories.

Build the static SPA before starting the API:

```bash
npm --prefix frontend run build
uv run python scripts/run_dev.py
```

`run_dev.py` only prints the following separate commands; it never starts a hidden process
group. Run them in two visible terminals with the same environment and Qdrant already
ready:

```bash
uv run uvicorn app.main:create_app --factory --host 127.0.0.1 --port 18000
uv run python -m app.worker
```

Open `http://127.0.0.1:18000`. The API serves `frontend/dist`, including client-route
fallbacks. The worker is a separate, persistent single ingestion process. Uploading,
reindexing, and deleting enqueue SQLite tasks; only a successfully built and activated
generation changes query-visible content. A failed replacement or delete leaves the last
accepted generation/content available.

## Real Bailian configuration

Real mode is explicit and requires credentials; it has not been exercised by the local
fake-provider acceptance flow:

```bash
export DASHSCOPE_API_KEY='your-real-key'
export MODEL_PROVIDER=bailian
export APP_ENV=production
export RAG_STATE_DIR="$PWD/var/core"
export ADMIN_USERNAME=admin
export ADMIN_PASSWORD='use-a-secret-admin-password'
npm --prefix frontend run build
uv run uvicorn app.main:create_app --factory --host 127.0.0.1 --port 18000
# In a second terminal with the same environment:
uv run python -m app.worker
```

Model names, endpoints, chunking, retrieval, Qdrant, upload, session, and web limits remain
in `configs/default.yaml`. Missing real credentials fail before index mutation. The live
Bailian smoke suite is separately opt-in:

```bash
RUN_BAILIAN_TESTS=1 uv run pytest tests/integration/test_bailian_api.py -v
```

Do not claim a live Bailian result unless this command actually ran successfully with a
real key.

## Upload and workflow boundaries

- Ordinary documents: TXT, DOCX, and PDF; knowledge Q&A: XLSX only, with the required six
  columns. Default maximum size is **50 MB per file** (`web.max_upload_mb`).
- TXT must decode as UTF-8/UTF-8-SIG or GB18030. PDF and Office signatures/packages are
  checked. Executable signatures, unsafe paths, reserved names, and `evaluation.jsonl`
  are rejected. These checks are not a malware-scanning service.
- The critical workflow is login → upload two small documents → task completion → chat
  answer and persisted citations → local feedback → reindex → delete.
- Chat uses SSE events in `start`, zero or more `token`, `citations`, `done` order on a
  successful answer. Cancellation and errors persist their terminal/partial state.
- Feedback is stored locally by message ID. It is not sent to LangSmith or an external
  run-tracking service.

## Repeatable validation

Windows PowerShell uses `$env:QDRANT_BINARY = "$PWD/var/qdrant/1.15.4/qdrant.exe"`;
bash uses `export QDRANT_BINARY="$PWD/var/qdrant/1.15.4/qdrant"`. The equivalent Windows
bash command is shown below.

```bash
# Real Qdrant, fake models, temporary state/raw files, and two small TXT fixtures.
RUN_INTEGRATION=1 QDRANT_BINARY="$PWD/var/qdrant/1.15.4/qdrant.exe" \
  uv run pytest tests/e2e/test_web_flow.py -v

npm --prefix frontend run build
npm --prefix frontend exec -- playwright install chromium
QDRANT_BINARY="$PWD/var/qdrant/1.15.4/qdrant.exe" \
  npm --prefix frontend exec -- playwright test e2e/web-flow.spec.ts
```

The Playwright harness starts visible child processes for Qdrant, the API, and the worker,
uses temporary state/raw directories, and removes them afterward. It requires the project
`.venv`, a frontend build, the official binary, and installed Chromium. Playwright prints
passed/failed/skipped statuses. If Chromium cannot install, browser acceptance is **NOT
RUN**; record the exact installation error rather than treating API/unit tests as browser
evidence. Without `QDRANT_BINARY`, the browser spec skips with an explicit reason.

Final local regressions:

```bash
uv run pytest tests/unit -q
RUN_INTEGRATION=1 QDRANT_BINARY="$PWD/var/qdrant/1.15.4/qdrant.exe" \
  uv run pytest tests/integration -m "not bailian" -q
uv run ruff check .
npm --prefix frontend run lint
npm --prefix frontend run typecheck
npm --prefix frontend test
npm --prefix frontend run build
```

## Security and deployment direction

This is a single-account application, not a multi-user identity system. It has no roles,
user self-service, tenant isolation, or login rate limiter. Sessions are server-side,
expire after the configured 12 hours, use an HttpOnly SameSite=Lax cookie, and expose a
stable per-session CSRF token through `/api/auth/me`. Mutations require that token. Enable
`web.secure_cookie` only when the application is actually served over HTTPS. Defaults bind
API and Qdrant to loopback; do not expose either directly to an untrusted network.

Treat SQLite, raw files, and especially BM25 pickle directories as application-owned
trusted state. Qdrant has no authentication configured by this development workflow.
Retired generations are retained; there is no automatic lifecycle cleanup or backup policy.

A future no-Docker, single-host installation can build on the official Qdrant binary, the
built SPA, and explicitly supervised API/worker processes. TLS termination, a reverse
proxy, service supervision, firewall rules, backups, retention, monitoring, secret
management, and production hardening remain separate deployment work. None of those is
claimed, configured, or validated here.
