# Enterprise RAG

A generation-atomic retrieval-augmented generation core with a single-admin FastAPI/React
web workflow and a separate persistent ingestion worker.

## Documentation

- [Core CLI, indexing, retrieval, and verification](docs/core.md)
- [Web setup, official no-Docker Qdrant installation, workflow, and validation](docs/web.md)

## Current phase status

The current phase includes immutable index generations, hybrid Qdrant/BM25 retrieval,
single-account login and CSRF protection, document upload/reindex/delete tasks, cited SSE
chat with local feedback, React static-SPA delivery, and repeatable API/browser smoke
flows. Fake providers are restricted to `MODEL_PROVIDER=fake` with `APP_ENV=test`.

Local validation uses an official standalone Qdrant 1.15.4 binary and temporary test state.
Windows API/Chromium flows passed; live Bailian, real PaddleOCR, and Ubuntu 22.04 execution
remain **NOT RUN**. Production deployment is not implied by this repository or its
fake-provider smoke tests. No production server, Docker deployment, Nginx, service
manager, or Conda environment is configured by this phase.

Start with [docs/web.md](docs/web.md) for exact development and test commands.
