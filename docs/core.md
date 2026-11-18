# Enterprise RAG core

The core CLI builds immutable, generation-scoped hybrid indexes and answers from the
active generation. Mutable SQLite and BM25 state lives under `var/core` by default;
set `RAG_STATE_DIR` to isolate another installation or test run. Qdrant connection and
collection settings are read from `configs/default.yaml`.

## Exact commands

From the repository root:

```bash
uv run python -m app.cli ingest --corpus mock_data/corpus --qa mock_data/qa/knowledge_qa.xlsx
uv run python -m app.cli rebuild --corpus mock_data/corpus --qa mock_data/qa/knowledge_qa.xlsx
uv run python -m app.cli ask --question "差旅住宿标准是多少？"
uv run python -m app.cli verify --corpus mock_data/corpus --qa mock_data/qa/knowledge_qa.xlsx
```

`ingest` and `rebuild` both construct and validate a complete new generation before
atomically activating it. Neither deletes the active generation or its artifacts before
the replacement is ready. Only supported ordinary documents directly inside `--corpus`
and the explicit knowledge Q&A workbook are parsed. `evaluation.jsonl` is held out and
is never ingested.

`ask` prints the generation ID, `Status: ok` or `Status: degraded`, the streamed answer,
and citation labels with document/parent lineage. `verify` emits one JSON object with
structured checks for 100 ordinary documents (including extension and scanned/text PDF
counts), exactly 1000 knowledge Q&A rows, active generation, parent/child integrity,
Qdrant dimension/point count, and BM25 manifest/child count. It exits 1 when any check
fails.

## Fake-model development

Fake adapters are available only when **both** variables are set explicitly:

```bash
export MODEL_PROVIDER=fake
export APP_ENV=test
export RAG_STATE_DIR=var/core-fake
uv run python -m app.cli ingest --corpus mock_data/corpus --qa mock_data/qa/knowledge_qa.xlsx
uv run python -m app.cli ask --question "差旅住宿标准是多少？"
uv run python -m app.cli verify --corpus mock_data/corpus --qa mock_data/qa/knowledge_qa.xlsx
```

The fake embedding/rerank/chat adapters are deterministic local test doubles and make
no Bailian requests. `MODEL_PROVIDER=fake` with `APP_ENV=production` is rejected. Do
not use fake answers as evaluation evidence.

## Real Bailian setup

Production defaults to Bailian. Configure the API key and select real mode explicitly:

```bash
export DASHSCOPE_API_KEY='your-real-key'
export MODEL_PROVIDER=bailian
export APP_ENV=production
export RAG_STATE_DIR=var/core
uv run python -m app.cli ingest --corpus mock_data/corpus --qa mock_data/qa/knowledge_qa.xlsx
uv run python -m app.cli ask --question "差旅住宿标准是多少？"
```

The embedding, rerank, and chat model names/endpoints remain those in
`configs/default.yaml`. Missing credentials fail before a new generation is prepared or
any index artifact is mutated. Real API smoke tests are separately opt-in and require
both the switch and key:

```bash
RUN_BAILIAN_TESTS=1 uv run pytest tests/integration/test_bailian_api.py -v
```

Without both opt-in conditions, those tests skip and no real Bailian result should be
claimed.

## Qdrant and verification tests

The default Qdrant URL is `http://127.0.0.1:16333`. Start Qdrant 1.15.4 there before a
standalone CLI ingest/ask/verify session. The integration fixture can instead start an
official standalone binary on an isolated port:

```bash
export QDRANT_BINARY="$PWD/var/qdrant/1.15.4/qdrant.exe"
RUN_INTEGRATION=1 uv run pytest tests/integration/test_rag_cli.py -v
RUN_INTEGRATION=1 uv run pytest tests/integration -m "not bailian" -v
```

If `QDRANT_BINARY` is unset, the fixture uses Docker with `qdrant/qdrant:1.15.4`.
Integration tests otherwise skip unless `RUN_INTEGRATION=1`.

Complete non-Bailian core checks:

```bash
uv run ruff check .
uv run pytest tests/unit -v
RUN_INTEGRATION=1 uv run pytest tests/integration -m "not bailian" -v
uv run python -m app.cli verify --corpus mock_data/corpus --qa mock_data/qa/knowledge_qa.xlsx
```

The final command verifies the state in `RAG_STATE_DIR` and Qdrant; it does not build an
index implicitly. Run `ingest` first in the same environment. The production fixture
also contains 100 held-out evaluation questions with an 80/20 answerable/unanswerable
split; evaluation data remains outside ingestion.
