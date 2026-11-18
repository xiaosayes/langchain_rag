import json
from collections import Counter

from openpyxl import load_workbook

from app.mockgen.catalog import build_catalog
from app.mockgen.qa import generate_qa


def _load_qa_and_evaluation(tmp_path):
    manifest = generate_qa(build_catalog(), tmp_path)
    workbook = load_workbook(tmp_path / "qa/knowledge_qa.xlsx", read_only=True)
    try:
        rows = list(workbook.active.values)
    finally:
        workbook.close()
    evaluations = [
        json.loads(line)
        for line in (tmp_path / "eval/evaluation.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    return manifest, rows, evaluations


def test_qa_counts_lineage_and_holdout(tmp_path):
    catalog = build_catalog()
    manifest, rows, evaluations = _load_qa_and_evaluation(tmp_path)

    assert rows[0] == (
        "qa_id",
        "question",
        "answer",
        "source_document_id",
        "evidence",
        "domain",
    )
    assert len(rows) == 1001
    assert len({row[0] for row in rows[1:]}) == 1000
    assert len({row[1] for row in rows[1:]}) == 1000
    assert Counter(row[3] for row in rows[1:]) == Counter(
        {spec.id: 10 for spec in catalog}
    )

    known_sources = {spec.id: spec for spec in catalog}
    for row in rows[1:]:
        qa_id, question, answer, source_id, evidence, domain = row
        assert source_id in known_sources
        assert all(isinstance(value, str) and value for value in row)
        spec = known_sources[source_id]
        fact_index = int(qa_id.rsplit("-Q", 1)[1]) - 1
        fact = spec.facts[fact_index]
        assert qa_id == f"{source_id}-Q{fact_index + 1:02d}"
        assert (question, answer, evidence, domain) == (
            fact.question,
            fact.answer,
            fact.evidence,
            spec.domain,
        )

    assert len(evaluations) == 100
    assert len({item["eval_id"] for item in evaluations}) == 100
    assert len({item["question"] for item in evaluations}) == 100
    assert sum(item["answerable"] for item in evaluations) == 80
    knowledge_questions = {row[1] for row in rows[1:]}
    assert all(item["question"] not in knowledge_questions for item in evaluations)

    answerable = [item for item in evaluations if item["answerable"]]
    unsupported = [item for item in evaluations if not item["answerable"]]
    assert len({item["source_document_ids"][0] for item in answerable}) == 80
    assert all(
        set(item)
        == {
            "eval_id",
            "question",
            "answerable",
            "expected_answer",
            "source_document_ids",
            "evidence",
        }
        for item in evaluations
    )
    for item in answerable:
        assert len(item["source_document_ids"]) == 1
        source_id = item["source_document_ids"][0]
        assert source_id in known_sources
        fact = known_sources[source_id].facts[0]
        assert item["question"] == fact.paraphrase
        assert item["expected_answer"] == fact.answer
        assert item["evidence"] == fact.evidence
    assert all(
        item["question"]
        and item["source_document_ids"] == []
        and item["expected_answer"] == ""
        and item["evidence"] == ""
        for item in unsupported
    )

    assert manifest.knowledge_qa_count == 1000
    assert manifest.evaluation_count == 100
    assert manifest.answerable_count == 80
    assert manifest.unanswerable_count == 20
