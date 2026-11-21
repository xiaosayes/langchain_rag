# Enterprise RAG Core and Mock Corpus Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tested CLI RAG application with deterministic enterprise mock data, multi-format ingestion, OCR, parent-child chunks, generation-consistent Qdrant/BM25 indexes, Bailian model adapters, hybrid retrieval, reranking, parent expansion, and cited streaming answers.

**Architecture:** Parsed documents are normalized into typed source blocks and split into parent/child chunks. Each index build creates a new Qdrant collection and BM25 directory, reuses unchanged vectors, and activates the generation only after validation. The query path combines dense and BM25 results with RRF, calls Bailian rerank, resolves parents from SQLite, and streams a cited response through LangChain models.

**Tech Stack:** Python 3.11, uv, LangChain 1.x, LangChain OpenAI, FastAPI-compatible typed services, SQLAlchemy 2, SQLite, Qdrant 1.15.4, qdrant-client 1.15.x, rank-bm25, jieba, httpx, PaddleOCR, PyMuPDF, python-docx, openpyxl, Pillow, Typer, pytest, Docker for local-only Qdrant integration tests.

## Global Constraints

- Follow `docs/superpowers/specs/2026-11-18-enterprise-rag-langchain-design.md`; do not widen scope to Web/API in this phase.
- Local development uses project `.venv` with Python `>=3.11,<3.12`; the current global Python 3.10 is not used.
- Server dependency metadata uses Conda environment name `langchain_test`; no server command is executed in this phase.
- Docker is allowed only for local Qdrant integration tests; production code and deployment metadata must not require Docker.
- Generate exactly 25 TXT, 25 DOCX, 25 PDF (20 text and 5 scanned), and 25 XLSX ordinary documents, plus 1000 knowledge Q&A and 100 held-out evaluation questions.
- Default models are `deepseek-v4.1-flash-1017`, `text-embedding-v4` with dimension 1024, and `qwen3-rerank`; tests never silently substitute another real model.
- Real API tests require both an explicit `RUN_BAILIAN_TESTS=1` and keys in `.env`; no key is committed or printed.
- Parent chunks default to 1000 characters, child chunks to 250 with 50 overlap, dense/BM25 recall to 30, RRF `k=60`, and final parent count to 6.
- All new Python code passes `uv run ruff check .` and `uv run pytest`; each task commits only its own reviewed files.
- A command that was not run is reported as not run, never as passing.

---

## File Structure

```text
app/
├── __init__.py
├── config.py                    # YAML/env settings and validation
├── bootstrap.py                 # Model/index/RAG composition root
├── verification.py              # Corpus and index integrity reports
├── cli.py                       # Thin ingest/rebuild/ask/verify commands
├── domain/
│   ├── ids.py                   # Stable content-derived IDs
│   └── models.py                # ParsedBlock/ParentChunk/ChildChunk/SearchHit
├── db/
│   ├── base.py                  # SQLAlchemy Base
│   ├── models.py                # Documents, parents, generations
│   └── session.py               # Engine/session factory
├── mockgen/
│   ├── catalog.py               # DocumentSpec/Fact and 100-document catalog
│   ├── renderers.py             # TXT/DOCX/XLSX renderers and dispatcher
│   ├── pdf.py                   # Text/scanned PDF renderers
│   ├── qa.py                    # 1000 Q&A + 100 held-out questions
│   └── generate.py              # Deterministic full corpus and manifest
├── ingestion/
│   ├── base.py                  # DocumentParser protocol
│   ├── registry.py              # Extension-to-parser routing
│   ├── text_office.py           # TXT/DOCX/XLSX/Q&A parsers
│   ├── ocr.py                   # OCR protocol and Paddle implementation
│   ├── pdf.py                   # Text-layer/OCR PDF parser
│   └── chunker.py               # Structure-aware parent/child chunker
├── models/
│   ├── retry.py                 # Bounded transient retry
│   └── bailian.py               # Embedding/Rerank/Chat model factories
├── retrieval/
│   ├── rrf.py                   # Reciprocal rank fusion
│   ├── bm25.py                  # Persistent Chinese BM25 generation
│   ├── qdrant_store.py          # Generation-scoped Qdrant operations
│   └── service.py               # Hybrid retrieval and parent resolution
├── indexing/
│   ├── repository.py            # Generation and parent persistence
│   └── builder.py               # Staged build/vector reuse/activation
└── rag/
    ├── prompts.py               # Grounded answer and rewrite prompts
    └── service.py               # Prepare context/citations and stream answer
configs/default.yaml              # Non-secret defaults
mock_data/                       # Generated, committed corpus and manifests
scripts/download_mock_font.sh    # Pinned open-font acquisition for mock PDFs
tests/unit/                      # No network/external process required
tests/integration/               # Qdrant, OCR and real Bailian markers
docs/core.md                     # CLI and core verification guide
```

### Task 1: Bootstrap Python 3.11, Locked Dependencies, and Typed Configuration

**Files:**
- Create: `pyproject.toml`, `environment.yml`, `.gitignore`, `configs/default.yaml`, `app/__init__.py`, `app/config.py`
- Create: `tests/unit/test_config.py`
- Create: `uv.lock`, `requirements.lock` via commands

**Interfaces:**
- Produces: `load_settings(path: Path = Path("configs/default.yaml")) -> AppSettings`.
- Produces: `AppSettings.embedding.dimension == 1024`, `AppSettings.qdrant.collection_prefix`, and `AppSettings.chunk` for later tasks.

- [ ] **Step 1: Create the isolated project and dependency declarations**

Create `pyproject.toml` with Hatchling packaging, `requires-python = ">=3.11,<3.12"`, runtime dependencies `langchain>=1,<2`, `langchain-openai>=1,<2`, `langchain-text-splitters>=1,<2`, `pydantic>=2.12,<3`, `PyYAML>=6,<7`, `SQLAlchemy>=2,<2.1`, `qdrant-client>=1.15,<1.16`, `httpx>=0.28,<1`, `typer>=0.16,<1`, `rank-bm25>=0.2.2,<1`, `jieba>=0.42,<1`, `pymupdf>=1.26,<2`, `python-docx>=1.2,<2`, `openpyxl>=3.1,<4`, and `Pillow>=11,<13`. Add optional `ocr = ["paddleocr>=3.3,<4", "paddlepaddle>=3.2,<4"]` and dev group `pytest`, `pytest-asyncio`, `pytest-cov`, `respx`, `ruff`, `mypy`, and `testcontainers`.

Create `environment.yml`:

```yaml
name: langchain_test
channels: [conda-forge]
dependencies:
  - python=3.11
  - pip
  - uv
```

Ignore `.venv/`, `.env`, `var/`, Python/test caches, coverage output, `frontend/node_modules/`, and `frontend/dist/`; do not ignore `uv.lock`, `requirements.lock`, or `mock_data/`.

- [ ] **Step 2: Write the failing configuration test**

```python
# tests/unit/test_config.py
from pathlib import Path
from app.config import load_settings


def test_load_settings_reads_defaults_and_secret_env(monkeypatch, tmp_path: Path):
    config = tmp_path / "config.yaml"
    config.write_text(
        "embedding:\n  model: text-embedding-v4\n  dimension: 1024\n"
        "qdrant:\n  collection_prefix: test_rag\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DASHSCOPE_API_KEY", "unit-test-secret")
    settings = load_settings(config)
    assert settings.embedding.dimension == 1024
    assert settings.qdrant.collection_prefix == "test_rag"
    assert settings.dashscope_api_key.get_secret_value() == "unit-test-secret"


def test_embedding_dimension_must_be_supported(tmp_path: Path):
    config = tmp_path / "bad.yaml"
    config.write_text("embedding:\n  dimension: 333\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dimension"):
        load_settings(config)
```

Import `pytest` in the test. Supported dimensions are `{64, 128, 256, 512, 768, 1024, 1536, 2048}`.

- [ ] **Step 3: Run the test and verify RED**

Run `uv sync --extra ocr --group dev`, then `uv run pytest tests/unit/test_config.py -v`.
Expected: collection or import failure because `app.config` does not exist.

- [ ] **Step 4: Implement minimal typed settings**

Use Pydantic models for `ChunkSettings`, `EmbeddingSettings`, `RerankSettings`, `RetrievalSettings`, `QdrantSettings`, and `AppSettings`. Load YAML with `yaml.safe_load`, merge `APP_ENV`, `MODEL_PROVIDER`, and environment secrets without serializing secrets, and validate dimensions with a field validator. `APP_ENV` defaults to `production`, `MODEL_PROVIDER` defaults to `bailian`, and both are represented by constrained string fields on `AppSettings`. Defaults must match the design:

```yaml
# configs/default.yaml
model_provider: bailian
llm:
  model: deepseek-v4.1-flash-1017
  base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
  temperature: 0.1
  max_output_tokens: 2048
  timeout_seconds: 60
  max_retries: 3
embedding:
  model: text-embedding-v4
  base_url: https://dashscope.aliyuncs.com
  dimension: 1024
  batch_size: 10
  timeout_seconds: 60
  max_retries: 3
rerank:
  model: qwen3-rerank
  base_url: https://dashscope.aliyuncs.com
  candidate_count: 30
  timeout_seconds: 30
  max_retries: 3
qdrant:
  url: http://127.0.0.1:16333
  collection_prefix: enterprise_rag
  distance: cosine
chunk:
  parent_size: 1000
  child_size: 250
  child_overlap: 50
retrieval:
  dense_top_k: 30
  bm25_top_k: 30
  rrf_k: 60
  final_parent_count: 6
  max_parents_per_document: 2
  max_context_characters: 12000
```

- [ ] **Step 5: Verify configuration and lock dependencies**

Run `uv run pytest tests/unit/test_config.py -v`, `uv run ruff check app tests`, `uv lock`, and `uv export --format requirements-txt --extra ocr --no-dev --no-emit-project --output-file requirements.lock`.
Expected: tests and lint pass; both lock files exist and contain no secret values.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml environment.yml .gitignore configs app tests uv.lock requirements.lock
git commit -m "build: bootstrap typed RAG core"
```

### Task 2: Define Stable Domain Contracts and SQLite Persistence

**Files:**
- Create: `app/domain/ids.py`, `app/domain/models.py`, `app/db/base.py`, `app/db/models.py`, `app/db/session.py`
- Test: `tests/unit/test_domain.py`, `tests/unit/test_db.py`

**Interfaces:**
- Consumes: `AppSettings` from Task 1.
- Produces: `SourceLocation`, `ParsedBlock`, `ParentChunk`, `ChildChunk`, `SearchHit(child_id, parent_id, document_id, text, score, source, metadata)`, `stable_id(namespace: str, *parts: str) -> str`, `create_session_factory(path: Path)`, and SQLAlchemy `DocumentRecord`, `ParentChunkRecord`, `IndexGenerationRecord`.

- [ ] **Step 1: Write failing ID and persistence tests**

```python
def test_stable_id_is_namespaced_and_repeatable():
    first = stable_id("child", "doc-1", "parent-1", "0", "报销")
    assert first == stable_id("child", "doc-1", "parent-1", "0", "报销")
    assert first != stable_id("parent", "doc-1", "parent-1", "0", "报销")
    assert first.startswith("child_")


def test_generation_and_parent_records_round_trip(tmp_path):
    session_factory = create_session_factory(tmp_path / "rag.sqlite3")
    with session_factory() as session:
        generation = IndexGenerationRecord(id="gen-1", status="active")
        session.add(generation)
        session.add(ParentChunkRecord(
            id="parent-1", generation_id="gen-1", document_id="doc-1",
            ordinal=0, text="差旅标准为每天500元", content_hash="abc", metadata_json={}
        ))
        session.commit()
        stored = session.get(ParentChunkRecord, ("parent-1", "gen-1"))
        assert stored.text == "差旅标准为每天500元"
```

Use a composite primary key `(id, generation_id)` for parent records so old generations remain queryable. `IndexGenerationRecord.status` is one of `building`, `active`, `retired`, `failed`.

- [ ] **Step 2: Verify RED**

Run `uv run pytest tests/unit/test_domain.py tests/unit/test_db.py -v`.
Expected: import failures for domain and DB modules.

- [ ] **Step 3: Implement immutable domain values and relational records**

Implement `stable_id` as SHA-256 over unit-separator-delimited UTF-8 parts, prefixed with the namespace and 24 lowercase hex characters. Define frozen dataclasses:

```python
@dataclass(frozen=True)
class ParsedBlock:
    document_id: str
    sequence: int
    text: str
    location: SourceLocation
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class ChildChunk:
    id: str
    parent_id: str
    document_id: str
    ordinal: int
    text: str
    content_hash: str
    metadata: dict[str, Any]
```

Add equivalent `ParentChunk` and `SearchHit` fields from the file map; `SearchHit.source` is `dense`, `bm25`, or `fused`. Configure SQLite foreign keys and WAL in `create_session_factory`, and create tables through `Base.metadata.create_all` for this phase.

- [ ] **Step 4: Verify GREEN**

Run `uv run pytest tests/unit/test_domain.py tests/unit/test_db.py -v` and `uv run ruff check app tests`.
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/domain app/db tests/unit/test_domain.py tests/unit/test_db.py
git commit -m "feat: add stable RAG domain and persistence contracts"
```

### Task 3: Generate the 75 Non-PDF Ordinary Documents

**Files:**
- Create: `app/mockgen/catalog.py`, `app/mockgen/renderers.py`
- Test: `tests/unit/test_mock_catalog.py`, `tests/integration/test_mock_office_renderers.py`

**Interfaces:**
- Produces: `Fact(question, paraphrase, answer, evidence)`, `DocumentSpec(id, domain, title, fmt, scanned, facts, version, effective_date)`, `make_spec(index: int, fmt: str, scanned: bool) -> DocumentSpec`, `build_catalog() -> list[DocumentSpec]`, and `render_document(spec, output_dir) -> Path`.
- Consumes: no model APIs; all facts are deterministic synthetic source data.

- [ ] **Step 1: Write failing catalog tests**

```python
def test_catalog_has_exact_shape_and_unique_facts():
    catalog = build_catalog()
    assert len(catalog) == 100
    assert Counter(spec.fmt for spec in catalog) == {
        "txt": 25, "docx": 25, "pdf": 25, "xlsx": 25
    }
    assert sum(spec.scanned for spec in catalog) == 5
    assert len({spec.id for spec in catalog}) == 100
    assert all(len(spec.facts) == 10 for spec in catalog)
    assert all(spec.facts for spec in catalog)


def test_catalog_is_deterministic():
    assert build_catalog() == build_catalog()
```

- [ ] **Step 2: Run the catalog test and verify RED**

Run `uv run pytest tests/unit/test_mock_catalog.py -v`.
Expected: import failure for `app.mockgen.catalog`.

- [ ] **Step 3: Implement the deterministic catalog**

Implement seven domain template groups and a fixed seed. Assign formats in contiguous blocks of 25, mark only PDF specs 20–24 as scanned, and generate ten explicit facts per document. Every fact must contain a distinct question, a non-identical paraphrase, a concise answer, and evidence literally rendered into the source document.

```python
def build_catalog() -> list[DocumentSpec]:
    formats = ["txt"] * 25 + ["docx"] * 25 + ["pdf"] * 25 + ["xlsx"] * 25
    return [
        make_spec(index=index, fmt=fmt, scanned=fmt == "pdf" and index % 25 >= 20)
        for index, fmt in enumerate(formats)
    ]
```

`make_spec` uses `random.Random(20261118 + index)` only for choosing domain-specific numeric values; question, answer, evidence, version, and effective date remain explicit fields rather than opaque generated prose.

- [ ] **Step 4: Write failing renderer tests**

Generate one DOCX and one XLSX into `tmp_path`; reopen them using `Document` and `openpyxl.load_workbook`. Assert the title, version, effective date, and all ten evidence strings are visible. Add a TXT UTF-8 round-trip test.

```python
def test_docx_renderer_preserves_all_evidence(tmp_path):
    spec = next(item for item in build_catalog() if item.fmt == "docx")
    path = render_document(spec, tmp_path)
    document = Document(path)
    text = "\n".join(paragraph.text for paragraph in document.paragraphs)
    text += "\n" + "\n".join(
        cell.text for table in document.tables for row in table.rows for cell in row.cells
    )
    assert spec.title in text
    assert all(fact.evidence in text for fact in spec.facts)
```

- [ ] **Step 5: Run renderer tests and verify RED**

Run `uv run pytest tests/integration/test_mock_office_renderers.py -v`.
Expected: import or dispatch failure because the renderers are not implemented.

- [ ] **Step 6: Implement TXT/DOCX/XLSX rendering**

`render_document` dispatches only non-PDF formats in this task and raises `ValueError("PDF renderer is not registered")` for PDF. DOCX uses headings, paragraphs, and at least one table; XLSX uses an overview Sheet and a facts Sheet with header, version, effective date, and row numbers. Never execute formulas.

```python
def render_document(spec: DocumentSpec, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{spec.id}.{spec.fmt}"
    renderers = {"txt": render_txt, "docx": render_docx, "xlsx": render_xlsx}
    if spec.fmt not in renderers:
        raise ValueError("PDF renderer is not registered")
    renderers[spec.fmt](spec, path)
    return path
```

- [ ] **Step 7: Run renderer and catalog tests**

Run `uv run pytest tests/unit/test_mock_catalog.py tests/integration/test_mock_office_renderers.py -v`.
Expected: 100-spec catalog passes and non-PDF renderer paths pass.

- [ ] **Step 8: Commit**

```bash
git add app/mockgen tests/unit/test_mock_catalog.py tests/integration/test_mock_office_renderers.py
git commit -m "feat: add deterministic enterprise document catalog"
```

### Task 4: Render 25 PDFs and the Complete 100-Document Manifest

**Files:**
- Create: `scripts/download_mock_font.sh`, `assets/fonts/NotoSansCJKsc-Regular.otf`, `assets/fonts/OFL.txt`
- Create: `app/mockgen/pdf.py`, `app/mockgen/generate.py`
- Test: `tests/integration/test_mock_pdf.py`, `tests/integration/test_mock_manifest.py`

**Interfaces:**
- Consumes: `DocumentSpec` and renderer dispatcher from Task 3.
- Produces: `render_pdf(spec, output_dir, font_path) -> Path` and `generate_corpus(output_dir: Path, font_path: Path) -> CorpusManifest`.

- [ ] **Step 1: Add the openly licensed font and write failing PDF tests**

Download `NotoSansCJKsc-Regular.otf` and its OFL license from `notofonts/noto-cjk` using `curl --fail --location`, record their SHA-256 values in `assets/fonts/SHA256SUMS`, and make `scripts/download_mock_font.sh` verify those checksums. Commit the font and license so later generation does not depend on a moving upstream branch. Test one text PDF with `fitz.open`: every evidence string is extractable. Test one scanned PDF: extracted text is empty or below 20 characters and at least one page image exists.

```python
def test_scanned_mock_pdf_has_image_but_no_text_layer(tmp_path):
    spec = next(item for item in build_catalog() if item.scanned)
    path = render_pdf(spec, tmp_path, FONT_PATH)
    with fitz.open(path) as document:
        assert len("".join(page.get_text() for page in document).strip()) < 20
        assert any(page.get_images() for page in document)
```

- [ ] **Step 2: Run PDF tests and verify RED**

Run `uv run pytest tests/integration/test_mock_pdf.py -v`.
Expected: import failure for `app.mockgen.pdf`.

- [ ] **Step 3: Implement PDF rendering**

For text PDFs, embed selectable Chinese text using PyMuPDF's CJK font or the downloaded Noto font. For scanned PDFs, render evidence into RGB images with Pillow and insert only images into PDF pages; do not leave a hidden text layer. Include page numbers in rendered content.

```python
def render_pdf(spec: DocumentSpec, output_dir: Path, font_path: Path) -> Path:
    path = output_dir / f"{spec.id}.pdf"
    with fitz.open() as document:
        for page_number, lines in enumerate(paginate_spec(spec), start=1):
            page = document.new_page(width=595, height=842)
            if spec.scanned:
                png = render_lines_to_png(lines, page_number, font_path)
                page.insert_image(page.rect, stream=png)
            else:
                insert_selectable_chinese(page, lines, page_number, font_path)
        document.set_metadata({"title": spec.title, "creationDate": "D:20261118000000Z"})
        document.save(path)
    return path
```

`paginate_spec`, `render_lines_to_png`, and `insert_selectable_chinese` are small helpers in `app/mockgen/pdf.py`; their unit contract is respectively page grouping, RGB PNG without text layers, and extractable text placement.

- [ ] **Step 4: Write failing full-manifest test**

```python
def test_generated_manifest_has_all_required_documents(tmp_path):
    manifest = generate_corpus(tmp_path, FONT_PATH)
    assert manifest.total_documents == 100
    assert manifest.counts == {"txt": 25, "docx": 25, "pdf": 25, "xlsx": 25}
    assert manifest.scanned_pdf_count == 5
    assert all((tmp_path / item.relative_path).is_file() for item in manifest.documents)
    assert len({item.sha256 for item in manifest.documents}) == 100
```

- [ ] **Step 5: Run the manifest test and verify RED**

Run `uv run pytest tests/integration/test_mock_manifest.py -v`.
Expected: import failure for `app.mockgen.generate`.

- [ ] **Step 6: Implement deterministic full generation**

Generate into `mock_data/corpus`, store machine-readable source specs under `mock_data/source`, and write `mock_data/manifest.json` with relative paths, SHA-256, format, scanned status, domain, version, and effective date. A second generation into another directory must produce identical catalog values, normalized document text, counts, and source references; per-file SHA-256 values identify each generated artifact and are not assumed to be stable across library-level binary metadata changes.

- [ ] **Step 7: Generate and verify the full corpus**

Run `uv run python -m app.mockgen.generate --output mock_data --font assets/fonts/NotoSansCJKsc-Regular.otf`, then `uv run pytest tests/integration/test_mock_pdf.py tests/integration/test_mock_manifest.py -v`.
Expected: exactly 100 files and all assertions pass.

- [ ] **Step 8: Commit source, generator, and generated corpus**

```bash
git add app/mockgen scripts/download_mock_font.sh assets/fonts tests mock_data
git commit -m "feat: generate 100 multi-format enterprise documents"
```

### Task 5: Generate 1000 Knowledge Q&A and 100 Held-Out Evaluation Questions

**Files:**
- Create: `app/mockgen/qa.py`, `mock_data/qa/knowledge_qa.xlsx`, `mock_data/eval/evaluation.jsonl`
- Modify: `app/mockgen/generate.py`, `mock_data/manifest.json`
- Test: `tests/integration/test_mock_qa.py`

**Interfaces:**
- Consumes: the 100 `DocumentSpec` objects and ten facts per document.
- Produces: `generate_qa(catalog, output_dir) -> QAManifest`; XLSX columns `qa_id`, `question`, `answer`, `source_document_id`, `evidence`, `domain`; JSONL evaluation fields `eval_id`, `question`, `answerable`, `expected_answer`, `source_document_ids`, `evidence`.

- [ ] **Step 1: Write failing count, lineage, and leakage tests**

```python
def test_qa_counts_lineage_and_holdout(tmp_path):
    generate_qa(build_catalog(), tmp_path)
    rows = list(load_workbook(tmp_path / "qa/knowledge_qa.xlsx").active.values)
    evals = [json.loads(line) for line in (tmp_path / "eval/evaluation.jsonl").read_text().splitlines()]
    assert len(rows) == 1001
    assert len({row[0] for row in rows[1:]}) == 1000
    assert len(evals) == 100
    assert sum(item["answerable"] for item in evals) == 80
    knowledge_questions = {row[1] for row in rows[1:]}
    assert all(item["question"] not in knowledge_questions for item in evals)
    assert all(set(item["source_document_ids"]) <= {spec.id for spec in build_catalog()} for item in evals if item["answerable"])
```

Also assert every knowledge row has a known source ID and non-empty answer/evidence.

- [ ] **Step 2: Run Q&A tests and verify RED**

Run `uv run pytest tests/integration/test_mock_qa.py -v`.
Expected: import failure for `app.mockgen.qa`.

- [ ] **Step 3: Implement Q&A generation**

Create exactly ten knowledge Q&A per ordinary document. Select 80 answerable evaluation items from distinct documents using fact paraphrases, and create 20 enterprise-context but unsupported questions. Do not index `evaluation.jsonl` in later ingestion commands.

```python
knowledge_rows = [
    {
        "qa_id": f"{spec.id}-Q{fact_index:02d}",
        "question": fact.question,
        "answer": fact.answer,
        "source_document_id": spec.id,
        "evidence": fact.evidence,
        "domain": spec.domain,
    }
    for spec in catalog
    for fact_index, fact in enumerate(spec.facts, start=1)
]
assert len(knowledge_rows) == 1000
```

Write these rows with `openpyxl`, then create answerable evaluation entries from the first fact of 80 distinct specs and append 20 explicit unsupported entries with `source_document_ids=[]` and `expected_answer=""`.

- [ ] **Step 4: Generate and verify artifacts**

Run the generator and verification commands:

```bash
uv run python -m app.mockgen.generate --output mock_data --font assets/fonts/NotoSansCJKsc-Regular.otf
uv run pytest tests/integration/test_mock_qa.py -v
uv run python -c 'import json; from pathlib import Path; from openpyxl import load_workbook; m=json.loads(Path("mock_data/manifest.json").read_text()); q=load_workbook("mock_data/qa/knowledge_qa.xlsx").active.max_row-1; e=[json.loads(x) for x in Path("mock_data/eval/evaluation.jsonl").read_text().splitlines()]; print("ordinary_documents={} knowledge_qa={} evaluation={} answerable={} unanswerable={}".format(m["total_documents"], q, len(e), sum(x["answerable"] for x in e), sum(not x["answerable"] for x in e)))'
```

Expected: all Q&A tests pass and the final line is `ordinary_documents=100 knowledge_qa=1000 evaluation=100 answerable=80 unanswerable=20`.

- [ ] **Step 5: Commit**

```bash
git add app/mockgen/qa.py app/mockgen/generate.py tests/integration/test_mock_qa.py mock_data
git commit -m "feat: add traceable QA and held-out evaluation corpus"
```

### Task 6: Parse TXT, DOCX, XLSX, and Q&A into Typed Source Blocks

**Files:**
- Create: `app/ingestion/base.py`, `app/ingestion/registry.py`, `app/ingestion/text_office.py`
- Test: `tests/unit/test_text_office_parsers.py`

**Interfaces:**
- Consumes: `ParsedBlock` and `SourceLocation` from Task 2, generated files from Tasks 3–5.
- Produces: `DocumentParser.parse(path: Path, document_id: str) -> list[ParsedBlock]`, `ParserRegistry.parse(path, document_id)`, and `QAParser.parse(path: Path, document_id: str = "knowledge_qa")`.

- [ ] **Step 1: Write failing parser tests**

Parametrize one TXT, DOCX, and XLSX sample; assert document IDs, monotonically increasing sequence numbers, non-empty text, and source locations. Assert XLSX blocks contain Sheet and row ranges, DOCX headings are preserved in `heading_path`, and Q&A rows produce one block per row with `metadata["record_type"] == "qa"` plus the original source document ID.

```python
def test_xlsx_parser_preserves_sheet_and_rows():
    spec = next(item for item in build_catalog() if item.fmt == "xlsx")
    path = Path("mock_data/corpus") / f"{spec.id}.xlsx"
    blocks = ParserRegistry.default().parse(path, spec.id)
    assert [block.sequence for block in blocks] == list(range(len(blocks)))
    assert all(block.text.strip() for block in blocks)
    assert any(block.location.sheet == "facts" and block.location.row_start for block in blocks)
```

- [ ] **Step 2: Run parser tests and verify RED**

Run `uv run pytest tests/unit/test_text_office_parsers.py -v`.
Expected: import failure for parser modules.

- [ ] **Step 3: Implement parsers**

Implement the protocol and suffix registry. Strip blank paragraphs without merging unrelated sections. DOCX reads both paragraphs and tables; XLSX uses values only and groups consecutive rows without dropping headers; QAParser validates exactly the six columns defined in Task 5 and rejects duplicate `qa_id`.

```python
class ParserRegistry:
    def __init__(self, parsers: Iterable[DocumentParser]):
        self._parsers = {suffix: parser for parser in parsers for suffix in parser.suffixes}

    @classmethod
    def default(cls) -> "ParserRegistry":
        return cls([TextParser(), DocxParser(), XlsxParser()])

    def parse(self, path: Path, document_id: str) -> list[ParsedBlock]:
        parser = self._parsers.get(path.suffix.lower())
        if parser is None:
            raise UnsupportedDocumentType(path.suffix)
        return parser.parse(path, document_id)
```

Task 7 registers `PDFParser` in this factory while keeping the public `parse` signature unchanged.

- [ ] **Step 4: Verify against generated files**

Run `uv run pytest tests/unit/test_text_office_parsers.py -v`. Include this corpus-wide case in the test file:

```python
@pytest.mark.parametrize("spec", [item for item in build_catalog() if item.fmt != "pdf"])
def test_every_generated_office_document_parses(spec):
    path = Path("mock_data/corpus") / f"{spec.id}.{spec.fmt}"
    blocks = ParserRegistry.default().parse(path, spec.id)
    assert blocks
    combined = "\n".join(block.text for block in blocks)
    assert all(fact.evidence in combined for fact in spec.facts)
```

Expected: all 75 ordinary non-PDF documents parse without an exception or empty output.

- [ ] **Step 5: Commit**

```bash
git add app/ingestion tests/unit/test_text_office_parsers.py
git commit -m "feat: parse text office and QA documents"
```

### Task 7: Parse Text and Scanned PDFs with Selective OCR

**Files:**
- Create: `app/ingestion/ocr.py`, `app/ingestion/pdf.py`
- Modify: `app/ingestion/registry.py`
- Test: `tests/unit/test_pdf_parser.py`, `tests/integration/test_pdf_ocr.py`

**Interfaces:**
- Produces: `OCREngine.recognize(image_png: bytes) -> str`, `PaddleOCREngine`, and `PDFParser.parse(path, document_id)`.

- [ ] **Step 1: Write failing selective-OCR tests with a fake engine**

Create a `FakeOCR` returning a known sentence. For a text PDF, assert no OCR calls and correct page numbers. For a scanned PDF, assert one OCR call per image-only page, returned text is included, and each block retains the page number. Add a test where a page with at least 20 meaningful extracted characters does not call OCR.

```python
class FakeOCR:
    def __init__(self):
        self.calls = 0

    def recognize(self, image_png: bytes) -> str:
        self.calls += 1
        assert image_png.startswith(b"\x89PNG")
        return "扫描件中的报销标准为每天500元"


def test_scanned_pdf_uses_ocr_and_keeps_page_numbers():
    spec = next(item for item in build_catalog() if item.scanned)
    ocr = FakeOCR()
    blocks = PDFParser(ocr=ocr).parse(Path("mock_data/corpus") / f"{spec.id}.pdf", spec.id)
    assert ocr.calls > 0
    assert "每天500元" in "\n".join(block.text for block in blocks)
    assert all(block.location.page is not None for block in blocks)
```

Use `PDFParser(ocr: OCREngine | None = None)`; `None` lazily creates the Paddle engine only when OCR is actually required.

- [ ] **Step 2: Run fake-OCR tests and verify RED**

Run `uv run pytest tests/unit/test_pdf_parser.py -v`.
Expected: import failure for PDF/OCR modules.

- [ ] **Step 3: Implement PDF/OCR adapters**

Implement lazy PaddleOCR import so non-OCR tests need no running model. Render only OCR-required pages at 200 DPI, pass PNG bytes to the engine, normalize line ordering, and expose clear `OCRProcessingError` with document/page context. Do not catch and hide all parser exceptions.

```python
for page_index, page in enumerate(document):
    extracted = page.get_text("text").strip()
    if len(extracted) >= 20:
        text = extracted
    else:
        pixmap = page.get_pixmap(dpi=200)
        try:
            text = self.ocr_engine.recognize(pixmap.tobytes("png")).strip()
        except Exception as exc:
            raise OCRProcessingError(f"{path.name}: page {page_index + 1}") from exc
    if text:
        blocks.append(ParsedBlock(
            document_id=document_id, sequence=len(blocks), text=text,
            location=SourceLocation(page=page_index + 1), metadata={"parser": "pdf"}
        ))
```

- [ ] **Step 4: Run fake-OCR and real OCR integration tests**

Run `uv run pytest tests/unit/test_pdf_parser.py -v`; expected PASS. Run `uv run pytest tests/integration/test_pdf_ocr.py -m ocr -v` with CPU PaddleOCR against the five generated scanned PDFs; expected: each document yields its title and at least 90% character coverage for its ten evidence strings after whitespace normalization. If a real OCR assertion fails, preserve the extracted text and page image as test artifacts before changing preprocessing.

- [ ] **Step 5: Commit**

```bash
git add app/ingestion tests/unit/test_pdf_parser.py tests/integration/test_pdf_ocr.py
git commit -m "feat: add selective scanned PDF OCR"
```

### Task 8: Implement Structure-Aware Parent-Child Chunking

**Files:**
- Create: `app/ingestion/chunker.py`
- Test: `tests/unit/test_chunker.py`

**Interfaces:**
- Consumes: `ParsedBlock`, chunk settings, stable IDs.
- Produces: `chunk_blocks(document_id: str, blocks: list[ParsedBlock], settings: ChunkSettings) -> tuple[list[ParentChunk], list[ChildChunk]]`.

- [ ] **Step 1: Write failing boundary and metadata tests**

Cover: a long section creates parents near 1000 characters and children near 250 with 50 overlap; every child references an existing parent; IDs repeat exactly across two calls; no empty chunks; page/Sheet/heading metadata survives. Add a Q&A block longer than 250 characters and assert its question and answer remain in one atomic parent and one child rather than being separated.

```python
def test_chunking_is_repeatable_and_references_real_parents():
    blocks = [ParsedBlock(
        document_id="doc-1", sequence=0, text="报销制度。" * 400,
        location=SourceLocation(heading_path=("财务", "报销")), metadata={}
    )]
    settings = ChunkSettings(parent_size=1000, child_size=250, child_overlap=50)
    parents1, children1 = chunk_blocks("doc-1", blocks, settings)
    parents2, children2 = chunk_blocks("doc-1", blocks, settings)
    assert (parents1, children1) == (parents2, children2)
    assert {child.parent_id for child in children1} <= {parent.id for parent in parents1}
    assert all(child.text.strip() for child in children1)
```

- [ ] **Step 2: Verify RED**

Run `uv run pytest tests/unit/test_chunker.py -v`.
Expected: import failure for `chunk_blocks`.

- [ ] **Step 3: Implement minimal chunking**

Group structural blocks into parents without crossing unrelated headings, Sheets, or pages unless a single block must be split. Use `RecursiveCharacterTextSplitter` for oversized plain-text blocks with configured character lengths and separators `["\n\n", "\n", "。", "；", "，", ""]`. Treat Q&A as atomic and reject a Q&A block exceeding configured model limits with a descriptive validation error instead of splitting its answer from its question. Hash normalized text with SHA-256 and use stable parent/child IDs.

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.child_size,
    chunk_overlap=settings.child_overlap,
    length_function=len,
    separators=["\n\n", "\n", "。", "；", "，", ""],
)
for parent in parents:
    child_texts = (
        [parent.text]
        if parent.metadata.get("record_type") == "qa"
        else splitter.split_text(parent.text)
    )
    children.extend(make_child(parent, ordinal, text) for ordinal, text in enumerate(child_texts))
```

`make_child(parent: ParentChunk, ordinal: int, text: str) -> ChildChunk` normalizes whitespace, rejects empty output, hashes the normalized text, and copies the parent's source metadata.

- [ ] **Step 4: Verify all chunk invariants**

Run `uv run pytest tests/unit/test_chunker.py -v`, including the corpus smoke test that chunks all parsed non-scanned documents and asserts every child has a parent and every parent has at least one child.
Expected: PASS and zero empty chunks.

- [ ] **Step 5: Commit**

```bash
git add app/ingestion/chunker.py tests/unit/test_chunker.py
git commit -m "feat: add structure-aware parent child chunking"
```

### Task 9: Implement Bailian Embedding, Rerank, and LangChain Chat Adapters

**Files:**
- Create: `app/models/retry.py`, `app/models/bailian.py`
- Test: `tests/unit/test_bailian.py`, `tests/integration/test_bailian_api.py`

**Interfaces:**
- Consumes: validated model settings and `DASHSCOPE_API_KEY`.
- Produces: `BailianEmbeddings(settings: AppSettings)` implementing LangChain `Embeddings`, `BailianReranker(settings: AppSettings)`, `BailianReranker.rerank(query: str, documents: list[str], top_n: int) -> list[RerankResult]`, `BailianReranker.arerank(query: str, documents: list[str], top_n: int) -> list[RerankResult]`, `create_chat_model(settings: AppSettings) -> ChatOpenAI`, and `RerankResult(index: int, relevance_score: float)`.

- [ ] **Step 1: Write failing contract tests with mocked HTTP**

Use `respx` for native DashScope endpoints. Embedding response contains out-of-order `text_index` values; assert results are restored to input order and dimension validated. Rerank response contains `output.results`; assert index/score parsing and `top_n`. Add 429-then-success retry and 400-no-retry tests. Assert Authorization headers are correct without printing their values.

```python
@respx.mock
def test_embeddings_restore_input_order(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "unit-test-secret")
    settings = load_settings()
    route = respx.post(settings.embedding.base_url + EMBEDDING_PATH).mock(
        return_value=httpx.Response(200, json={"output": {"embeddings": [
            {"text_index": 1, "embedding": [0.0] * 1024},
            {"text_index": 0, "embedding": [1.0] * 1024},
        ]}})
    )
    vectors = BailianEmbeddings(settings).embed_documents(["first", "second"])
    assert vectors[0][0] == 1.0
    assert vectors[1][0] == 0.0
    assert route.calls[0].request.headers["authorization"] == "Bearer unit-test-secret"
```

- [ ] **Step 2: Run model adapter tests and verify RED**

Run `uv run pytest tests/unit/test_bailian.py -v`.
Expected: import failure for `app.models.bailian`.

- [ ] **Step 3: Implement adapters**

Implement native endpoints:

```text
POST /api/v1/services/embeddings/text-embedding/text-embedding
POST /api/v1/services/rerank/text-rerank/text-rerank
```

Embedding JSON uses `input.texts` and `parameters.dimension`; rerank JSON uses `input.query`, `input.documents`, and `parameters.top_n`. Retry only connect/read timeouts, 408, 409, 429, and 5xx with capped exponential backoff and injectable sleep. `create_chat_model` uses the configured OpenAI-compatible base URL, exact model ID, temperature, max tokens, timeout, and retries.

```python
payload = response.json()
items = sorted(payload["output"]["embeddings"], key=lambda item: item["text_index"])
vectors = [[float(value) for value in item["embedding"]] for item in items]
if len(vectors) != len(texts) or any(len(vector) != self.dimension for vector in vectors):
    raise BailianResponseError("Embedding count or dimension does not match the request")
return vectors
```

Use the same response validation pattern for rerank: result indices must be within the input document range, scores must be finite floats, and duplicate indices are rejected.

- [ ] **Step 4: Write real API smoke tests**

Real tests are skipped unless `RUN_BAILIAN_TESTS=1` and `DASHSCOPE_API_KEY` exist. Assert embedding length 1024, rerank places a directly relevant document above an unrelated one, and chat `astream` yields non-empty chunks from the exact configured model.

```python
@pytest.mark.bailian
def test_real_embedding_has_configured_dimension():
    if os.environ.get("RUN_BAILIAN_TESTS") != "1":
        pytest.skip("real Bailian tests are disabled")
    if not os.environ.get("DASHSCOPE_API_KEY"):
        pytest.skip("DASHSCOPE_API_KEY is not configured")
    vector = BailianEmbeddings(load_settings()).embed_query("差旅报销")
    assert len(vector) == 1024
    assert all(math.isfinite(value) for value in vector)
```

Add equivalent real rerank and streaming-chat tests without logging response headers or secrets.

- [ ] **Step 5: Verify mocked contracts and optional real APIs**

Run `uv run pytest tests/unit/test_bailian.py -v`; expected PASS. If keys were supplied, run `RUN_BAILIAN_TESTS=1 uv run pytest tests/integration/test_bailian_api.py -m bailian -v`. Otherwise report the real API tests as not run. Record failures verbatim; never switch model IDs to make a test pass.

- [ ] **Step 6: Commit**

```bash
git add app/models tests/unit/test_bailian.py tests/integration/test_bailian_api.py
git commit -m "feat: add configurable Bailian model adapters"
```

### Task 10: Implement Qdrant Generations, Persistent Chinese BM25, and RRF

**Files:**
- Create: `app/retrieval/rrf.py`, `app/retrieval/bm25.py`, `app/retrieval/qdrant_store.py`
- Test: `tests/unit/test_rrf.py`, `tests/unit/test_bm25.py`, `tests/integration/test_qdrant_store.py`

**Interfaces:**
- Consumes: `ChildChunk`, `SearchHit`, Qdrant settings.
- Produces: `reciprocal_rank_fusion(result_sets, k=60) -> list[SearchHit]`, `PersistentBM25.build(chunks, generation_dir)`, `.search(query, top_k)`, `QdrantGenerationStore.create/upsert/search/fetch_vectors/delete`.

- [ ] **Step 1: Write failing RRF and Chinese BM25 tests**

RRF test uses two overlapping ranked lists and calculates expected scores explicitly as `1/(60+rank)`. BM25 test builds three Chinese chunks, searches `差旅报销标准`, reloads the generation directory, and asserts the same top child ID and no dependency on original in-memory chunks. Assert deleted/replaced generations cannot be searched after their directory is removed.

```python
def test_rrf_combines_one_based_ranks():
    dense = [make_hit("a", "dense"), make_hit("b", "dense")]
    bm25 = [make_hit("b", "bm25"), make_hit("c", "bm25")]
    fused = reciprocal_rank_fusion([dense, bm25], k=60)
    assert [hit.child_id for hit in fused] == ["b", "a", "c"]
    assert fused[0].score == pytest.approx(2 / 61)
    assert fused[1].score == pytest.approx(1 / 61)
    assert fused[2].score == pytest.approx(1 / 62)
```

Define `make_hit(child_id: str, source: str) -> SearchHit` in the test with fixed parent/document/text values and `score=0.0`.

- [ ] **Step 2: Run RRF/BM25 tests and verify RED**

Run `uv run pytest tests/unit/test_rrf.py tests/unit/test_bm25.py -v`.
Expected: import failure for retrieval modules.

- [ ] **Step 3: Implement RRF/BM25**

Use jieba tokenization plus alphanumeric lowercase tokens, ignore whitespace tokens, persist BM25 state and child payloads inside the generation directory, and reject loading a manifest with an incompatible tokenizer/schema version. RRF merges by child ID, preserves each source rank/score in metadata, and sorts by fused score then stable child ID.

```python
def reciprocal_rank_fusion(result_sets: list[list[SearchHit]], k: int = 60) -> list[SearchHit]:
    accumulated: dict[str, dict[str, Any]] = {}
    for results in result_sets:
        for rank, hit in enumerate(results, start=1):
            entry = accumulated.setdefault(
                hit.child_id, {"hit": hit, "score": 0.0, "ranks": {}, "source_scores": {}}
            )
            entry["score"] += 1.0 / (k + rank)
            entry["ranks"][hit.source] = rank
            entry["source_scores"][hit.source] = hit.score
    fused = [
        replace(
            entry["hit"], score=entry["score"], source="fused",
            metadata={
                **entry["hit"].metadata,
                "ranks": entry["ranks"],
                "source_scores": entry["source_scores"],
            },
        )
        for entry in accumulated.values()
    ]
    return sorted(fused, key=lambda hit: (-hit.score, hit.child_id))
```

- [ ] **Step 4: Write failing real Qdrant integration tests**

Use `testcontainers` with image `qdrant/qdrant:1.15.4`, mapped to an ephemeral localhost port. Create a 3-dimension test generation (real model dimensions are not required for unit infrastructure), upsert three chunks with known vectors, search the nearest vector, fetch vectors by IDs, and assert collection deletion. Mark with `@pytest.mark.integration` and fail clearly if `RUN_INTEGRATION=1` is set but Docker is unavailable.

```python
@pytest.mark.integration
def test_qdrant_generation_round_trip(qdrant_url):
    store = QdrantGenerationStore(QdrantSettings(url=qdrant_url))
    store.create("gen-1", dimension=3)
    store.upsert("gen-1", three_test_chunks(), [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    hits = store.search("gen-1", [0.9, 0.1, 0], top_k=1)
    assert hits[0].child_id == "child-a"
    assert store.fetch_vectors("gen-1", ["child-a"])["child-a"] == [1, 0, 0]
    store.delete("gen-1")
```

The `qdrant_url` fixture and `three_test_chunks()` helper are created in `tests/integration/conftest.py`; the helper returns three valid `ChildChunk` values with IDs `child-a`, `child-b`, and `child-c`.

- [ ] **Step 5: Run the Qdrant integration test and verify RED**

Run `RUN_INTEGRATION=1 uv run pytest tests/integration/test_qdrant_store.py -v`.
Expected: import failure for `app.retrieval.qdrant_store`.

- [ ] **Step 6: Implement Qdrant generation adapter**

Use deterministic UUID point IDs derived from child IDs, cosine distance, payload indexes for document/parent IDs, and generation-specific collection names. `fetch_vectors` must return vectors without changing their values for reuse.

```python
collection = self.collection_name(generation_id)
self.client.create_collection(
    collection_name=collection,
    vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
)
self.client.create_payload_index(collection, "document_id", models.PayloadSchemaType.KEYWORD)
self.client.create_payload_index(collection, "parent_id", models.PayloadSchemaType.KEYWORD)
```

Production code must not invoke Docker; Docker usage is confined to `tests/integration/conftest.py`.

- [ ] **Step 7: Verify all retrieval storage tests**

Run `uv run pytest tests/unit/test_rrf.py tests/unit/test_bm25.py tests/integration/test_qdrant_store.py -v` with `RUN_INTEGRATION=1`.
Expected: RRF, BM25 persistence, and Qdrant round trips pass.

- [ ] **Step 8: Commit**

```bash
git add app/retrieval tests/unit/test_rrf.py tests/unit/test_bm25.py tests/integration
git commit -m "feat: add generation scoped hybrid indexes"
```

### Task 11: Build and Atomically Activate Reusable Index Generations

**Files:**
- Create: `app/indexing/repository.py`, `app/indexing/builder.py`
- Modify: `app/db/models.py`
- Test: `tests/unit/test_index_repository.py`, `tests/integration/test_index_builder.py`

**Interfaces:**
- Consumes: parsers, chunker, model adapters, both retrieval stores.
- Produces: `IndexRepository(session_factory)`, `IndexRepository.active_generation() -> str | None`, `.parents(generation_id: str, parent_ids: list[str]) -> list[ParentChunk]`, `.reusable_vectors(children: list[ChildChunk]) -> set[str]`, `IndexBuilder(repository, qdrant, bm25_root, embeddings, chunk_settings)`, and `IndexBuilder.build(documents: dict[str, list[ParsedBlock]]) -> str`.

- [ ] **Step 1: Write failing activation and vector-reuse tests**

Build generation 1 from two documents using counting deterministic fake embeddings and real Qdrant/BM25. Activate it and assert `active_generation()` returns the new ID. Build generation 2 with one unchanged document and one changed document; assert embedding calls include only changed/new child texts. Keep a handle to generation 1 and prove its parents remain available after generation 2 activates. Force BM25 validation failure and assert the previous active generation remains unchanged.

```python
@pytest.mark.integration
def test_second_generation_reuses_unchanged_vectors(index_builder, fake_embeddings):
    documents_v1 = {"doc-a": parsed_blocks("doc-a", "差旅标准为500元"), "doc-b": parsed_blocks("doc-b", "年假为10天")}
    first = index_builder.build(documents_v1)
    fake_embeddings.embedded_texts.clear()
    documents_v2 = dict(documents_v1, **{"doc-b": parsed_blocks("doc-b", "年假为15天")})
    second = index_builder.build(documents_v2)
    assert second != first
    assert "差旅标准为500元" not in fake_embeddings.embedded_texts
    assert "年假为15天" in fake_embeddings.embedded_texts
```

Create `index_builder`, `fake_embeddings`, and `parsed_blocks` fixtures/helpers in `tests/integration/conftest.py`; the builder fixture uses a temporary SQLite database, real test Qdrant, temporary BM25 root, and small chunk sizes.

- [ ] **Step 2: Run index repository tests and verify RED**

Run `uv run pytest tests/unit/test_index_repository.py -v`.
Expected: import failure for `app.indexing.repository`.

- [ ] **Step 3: Implement repository transactions**

Add generation-aware document and parent rows. `prepare_generation` inserts `building`; `activate_generation` validates staged parent rows, marks the previous generation `retired`, and changes exactly one row to `active` in a single SQLite transaction. Repository queries always require explicit generation IDs; there is no implicit mixture of active and retired parents.

```python
with self.session_factory.begin() as session:
    staged = session.get(IndexGenerationRecord, generation_id)
    if staged is None or staged.status != "building":
        raise GenerationStateError(generation_id)
    previous = session.scalars(
        select(IndexGenerationRecord).where(IndexGenerationRecord.status == "active")
    ).one_or_none()
    if previous is not None:
        previous.status = "retired"
    staged.status = "active"
```

A partial unique index on `status = "active"` or a guarded transaction must prevent two active rows. `activate_generation` also inserts the supplied staged parent records inside this transaction.

- [ ] **Step 4: Run the index builder test and verify RED**

Run `RUN_INTEGRATION=1 uv run pytest tests/integration/test_index_builder.py -v`.
Expected: import failure for `app.indexing.builder`.

- [ ] **Step 5: Implement staged index builder**

`IndexBuilder.build` performs: chunk all desired documents, create a unique generation ID, compare child content hashes with the previous generation, fetch/copy unchanged vectors, embed only missing vectors in configured batches, build a fresh BM25 directory, validate counts/dimensions/parent references, persist parents, and activate. Any exception marks only the new generation `failed` and leaves its artifacts for diagnostics without making them searchable.

```python
def build(self, documents: dict[str, list[ParsedBlock]]) -> str:
    generation_id = str(uuid4())
    self.repository.prepare_generation(generation_id)
    try:
        parents, children = chunk_all(documents, self.chunk_settings)
        reusable = self.repository.reusable_vectors(children)
        previous_generation = self.repository.active_generation()
        vectors = (
            self.qdrant.fetch_vectors(previous_generation, list(reusable))
            if previous_generation is not None and reusable
            else {}
        )
        missing = [child for child in children if child.id not in vectors]
        vectors.update(self.embed_new(missing))
        self.qdrant.create(generation_id, self.embeddings.dimension)
        self.qdrant.upsert(generation_id, children, [vectors[child.id] for child in children])
        PersistentBM25.build(children, self.bm25_root / generation_id)
        self.validate(generation_id, parents, children, vectors)
        self.repository.activate_generation(generation_id, parents)
        return generation_id
    except Exception:
        self.repository.mark_failed(generation_id)
        raise
```

For the first generation, `active_generation()` may return `None`, so both repository and builder signatures use `active_generation() -> str | None`; `ask` converts `None` into a clear empty-knowledge-base error. `reusable_vectors` returns IDs whose content hashes match the previous generation, and `embed_new` returns a `dict[str, list[float]]` in input order.

- [ ] **Step 6: Verify repeatability and failure safety**

Run repository unit tests and the real Qdrant integration test twice. Expected: identical source data produces equivalent searchable content, second build reuses unchanged vectors, failed builds never change active generation, and no active generation points to a missing collection or BM25 manifest.

- [ ] **Step 7: Commit**

```bash
git add app/indexing app/db/models.py tests/unit/test_index_repository.py tests/integration/test_index_builder.py tests/integration/conftest.py
git commit -m "feat: add atomic reusable index generations"
```

### Task 12: Implement Hybrid Retrieval and Parent Resolution

**Files:**
- Create: `app/retrieval/service.py`
- Test: `tests/unit/test_retrieval_service.py`

**Interfaces:**
- Consumes: active generation repository, both indexes, model adapters, and parent records.
- Produces: `RetrievalService(repository, qdrant, bm25_root, embeddings, reranker, retrieval_settings)` and `RetrievalService.asearch(query: str, generation_id: str) -> RetrievalResult`.
- Produces: `RetrievalResult(hits: list[SearchHit], documents: list[Document], degraded: bool)` and typed `RetrievalUnavailable`.

- [ ] **Step 1: Write failing fusion/rerank/parent tests**

Fake dense and BM25 stores return overlapping children. Fake rerank reverses the fused order. Assert `RetrievalResult.hits` follows rerank order, duplicate parents are removed, no more than two parents per document and six total are returned, and context never exceeds 12000 characters. Simulate dense and rerank failures separately and assert `degraded=True`; simulate both retrieval stores failing and assert `RetrievalUnavailable`.

```python
@pytest.mark.asyncio
async def test_rerank_order_controls_parent_selection(retrieval_service):
    result = await retrieval_service.asearch("报销标准", "gen-test")
    assert result.hits[0].child_id == "child-from-second-store"
    assert len(result.documents) <= 6
    assert all(document.metadata["generation_id"] == "gen-test" for document in result.documents)
    assert sum(len(document.page_content) for document in result.documents) <= 12000
```

The `retrieval_service` fixture wires fake stores where `child-from-second-store` wins only after rerank; it uses real parent resolution against a temporary SQLite generation.

- [ ] **Step 2: Run tests and verify RED**

Run `uv run pytest tests/unit/test_retrieval_service.py -v`.
Expected: import failure for `app.retrieval.service`.

- [ ] **Step 3: Implement retrieval service**

Execute dense and BM25 in `asyncio.gather`, use `reciprocal_rank_fusion`, pass at most 30 candidates to rerank, resolve parents only from the requested generation, and preserve source metadata in LangChain `Document` objects. Capture each stage exception independently. If dense alone fails, continue with BM25; if BM25 alone fails, continue with dense; if rerank fails, use fused order. Always set the degradation flag and preserve the underlying error category for later observability.

```python
async def asearch(self, query: str, generation_id: str) -> RetrievalResult:
    vector = await self.embeddings.aembed_query(query)
    dense_result, bm25_result = await asyncio.gather(
        asyncio.to_thread(self.qdrant.search, generation_id, vector, self.settings.dense_top_k),
        asyncio.to_thread(self.load_bm25(generation_id).search, query, self.settings.bm25_top_k),
        return_exceptions=True,
    )
    result_sets, errors = successful_result_sets(dense=dense_result, bm25=bm25_result)
    if not result_sets:
        raise RetrievalUnavailable(str(errors))
    fused = reciprocal_rank_fusion(result_sets, self.settings.rrf_k)
    ranked, rerank_failed = await self.rerank_or_original(query, fused[:30])
    parents = self.repository.parents(generation_id, [hit.parent_id for hit in ranked])
    documents = select_parent_documents(parents, self.settings)
    return RetrievalResult(ranked, documents, degraded=bool(errors) or rerank_failed)
```

`successful_result_sets` rejects exception values without losing successful results, and `select_parent_documents` performs parent deduplication, per-document limits, stable rerank ordering, and the character budget.

- [ ] **Step 4: Verify GREEN**

Run `uv run pytest tests/unit/test_retrieval_service.py -v` and `uv run ruff check app/retrieval tests/unit/test_retrieval_service.py`.
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/retrieval/service.py tests/unit/test_retrieval_service.py
git commit -m "feat: add hybrid retrieval with parent resolution"
```

### Task 13: Implement Grounded Cited Streaming RAG

**Files:**
- Create: `app/rag/prompts.py`, `app/rag/service.py`
- Test: `tests/unit/test_rag_service.py`

**Interfaces:**
- Consumes: `RetrievalService.asearch` and `RetrievalResult` from Task 12 plus a LangChain chat model from Task 9.
- Produces: `Citation(label: str, document_id: str, parent_id: str, text: str, metadata: dict[str, Any])`.
- Produces: `RAGPreparation(original_question: str, retrieval_query: str, documents: list[Document], citations: list[Citation], degraded: bool, generation_id: str, insufficient_context: bool)`.
- Produces: `RAGService(chat_model, retrieval_service, repository)`, `.prepare(question: str, history: list[BaseMessage], generation_id: str) -> RAGPreparation`, and `.stream_answer(prepared: RAGPreparation) -> AsyncIterator[str]`.

- [ ] **Step 1: Write failing grounded-answer tests**

Use a fake streaming LangChain chat model. Assert first-turn questions skip rewrite, follow-up questions invoke rewrite once, prompt contains source labels and parent text, citations map only to supplied parents, and empty/insufficient context produces the configured refusal rather than a model answer. Assert answer streaming yields chunks in order and preparation exposes `degraded` and `generation_id`.

```python
@pytest.mark.asyncio
async def test_first_turn_skips_rewrite_and_streams_grounded_answer(rag_service, fake_chat):
    prepared = await rag_service.prepare("住宿标准？", [], "gen-test")
    assert prepared.retrieval_query == "住宿标准？"
    assert fake_chat.rewrite_calls == 0
    chunks = [chunk async for chunk in rag_service.stream_answer(prepared)]
    assert "".join(chunks) == "住宿标准为每天500元[S1]。"
    assert prepared.citations[0].label == "S1"
    assert prepared.generation_id == "gen-test"
```

- [ ] **Step 2: Run tests and verify RED**

Run `uv run pytest tests/unit/test_rag_service.py -v`.
Expected: import failure for `app.rag.service`.

- [ ] **Step 3: Implement prompts and RAG service**

The answer prompt requires Chinese, source-grounded statements, `[S1]`-style citations, current-version precedence, and explicit knowledge insufficiency. `prepare` performs optional rewrite and retrieval, labels each supplied parent exactly once, and builds citations solely from those parents. `stream_answer` streams only final generation and never invents citation labels when the model emits an unknown source; unknown labels are removed and recorded as a validation warning.

```python
def make_preparation(
    original_question: str, retrieval_query: str, result: RetrievalResult, generation_id: str
) -> RAGPreparation:
    citations = [
        Citation(f"S{index}", doc.metadata["document_id"], doc.metadata["parent_id"],
                 doc.page_content, doc.metadata)
        for index, doc in enumerate(result.documents, start=1)
    ]
    return RAGPreparation(
        original_question, retrieval_query, result.documents, citations,
        result.degraded, generation_id, insufficient_context=not result.documents
    )
```

For zero documents, `prepare` returns an empty preparation with `insufficient_context=True`; add that Boolean field to `RAGPreparation` and make `stream_answer` yield only `“根据当前知识库，无法可靠回答该问题。”` without invoking the chat model.

- [ ] **Step 4: Verify GREEN**

Run `uv run pytest tests/unit/test_rag_service.py -v` and `uv run ruff check app/rag tests/unit/test_rag_service.py`.
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/rag tests/unit/test_rag_service.py
git commit -m "feat: add grounded cited streaming RAG"
```

### Task 14: Deliver the CLI and Complete Core Verification

**Files:**
- Create: `app/bootstrap.py`, `app/verification.py`, `app/cli.py`, `docs/core.md`
- Test: `tests/integration/test_rag_cli.py`

**Interfaces:**
- Consumes: parser registry, `IndexBuilder`, `IndexRepository`, `RAGService`, and all settings from earlier tasks.
- Produces: `build_rag_service(settings) -> RAGService`, `CoreVerifier.run() -> VerificationReport`, and thin Typer commands `ingest`, `rebuild`, `ask`, `verify` for use by the Web and deployment phases.

- [ ] **Step 1: Write failing CLI integration tests**

Use Typer's test runner and fake model providers to ingest two small documents, activate an index, ask a grounded question, and verify output contains the answer and source label. Add a `verify` test that fails when the Q&A file has 999 rows and another that fails when a child references a missing parent. Assert real model mode is never selected by a unit/integration fake fixture.

```python
def test_verify_rejects_short_qa_file(cli_environment, tmp_path):
    qa_path = tmp_path / "qa.xlsx"
    write_qa_workbook(qa_path, row_count=999)
    result = runner.invoke(
        cli_app,
        ["verify", "--corpus", str(cli_environment.corpus), "--qa", str(qa_path)],
        env={"MODEL_PROVIDER": "fake", "APP_ENV": "test"},
    )
    assert result.exit_code == 1
    assert "expected 1000" in result.stdout
```

`cli_environment` builds an isolated temporary corpus and SQLite/Qdrant/BM25 state; `write_qa_workbook` is a test helper that writes the six required columns and the requested number of valid rows.

- [ ] **Step 2: Run tests and verify RED**

Run `uv run pytest tests/integration/test_rag_cli.py -v`.
Expected: import failure for `app.cli`.

- [ ] **Step 3: Implement Typer commands and core documentation**

Commands must support:

```bash
uv run python -m app.cli ingest --corpus mock_data/corpus --qa mock_data/qa/knowledge_qa.xlsx
uv run python -m app.cli ask --question "差旅住宿标准是多少？"
uv run python -m app.cli verify --corpus mock_data/corpus --qa mock_data/qa/knowledge_qa.xlsx
```

`ingest` uses the parser registry and generation builder; `rebuild` forces a new generation without deleting the active one first; `ask` uses the active generation and prints streamed text followed by citations; `verify` checks actual corpus counts, Q&A counts, active generation, child/parent consistency, Qdrant dimension, and BM25 manifest. `docs/core.md` records fake-model development, real API setup, Qdrant test setup, and exact commands.

```python
@cli_app.command()
def verify(corpus: Path, qa: Path) -> None:
    settings = load_settings()
    report = CoreVerifier(settings, corpus, qa).run()
    typer.echo(report.to_json())
    if not report.ok:
        raise typer.Exit(code=1)

@cli_app.command()
def ask(question: str) -> None:
    service = build_rag_service(load_settings())
    asyncio.run(stream_cli_answer(service, question))
```

`CoreVerifier.run() -> VerificationReport` is synchronous and returns structured checks rather than exiting internally. `build_rag_service` reads `settings.model_provider`. It selects Fake only when both `MODEL_PROVIDER=fake` and `APP_ENV=test`; test fixtures set both explicitly. Production defaults to Bailian, a Fake request outside the test environment is rejected, and missing production keys fail before any index mutation.

- [ ] **Step 4: Run the complete core verification**

Run:

```bash
uv run ruff check .
uv run pytest tests/unit -v
RUN_INTEGRATION=1 uv run pytest tests/integration -m "not bailian" -v
uv run python -m app.cli verify --corpus mock_data/corpus --qa mock_data/qa/knowledge_qa.xlsx
```

Expected: lint, unit tests, non-Bailian integration tests, and CLI verification pass; actual output reports 100 ordinary documents and 1000 knowledge Q&A. If real keys were supplied, separately run `RUN_BAILIAN_TESTS=1 uv run pytest tests/integration/test_bailian_api.py -v` and report its actual result.

- [ ] **Step 5: Commit**

```bash
git add app/bootstrap.py app/verification.py app/cli.py docs/core.md tests/integration/test_rag_cli.py
git commit -m "feat: deliver testable hybrid RAG CLI"
```

## Phase Completion Evidence

Before starting the Web phase, record command output for:

- Actual file counts by extension and scanned/text PDF counts.
- Exactly 1000 knowledge Q&A and 100 evaluation questions with 80/20 answerability.
- Parent/child integrity and active-generation recovery tests.
- Dense + BM25 + RRF + rerank + parent resolution tests.
- Non-Bailian tests and, when keys are available, real Bailian smoke tests.
- Known limitations and any tests not executable locally, especially server-specific behavior.
