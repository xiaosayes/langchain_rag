from collections import Counter

import pytest

from app.mockgen.catalog import build_catalog, make_spec


def test_catalog_has_exact_shape_and_unique_facts():
    catalog = build_catalog()

    assert len(catalog) == 100
    assert Counter(spec.fmt for spec in catalog) == {
        "txt": 25,
        "docx": 25,
        "pdf": 25,
        "xlsx": 25,
    }
    assert sum(spec.scanned for spec in catalog) == 5
    assert len({spec.id for spec in catalog}) == 100
    assert all(len(spec.facts) == 10 for spec in catalog)
    assert all(spec.facts for spec in catalog)
    assert len({spec.domain for spec in catalog}) == 7
    assert {spec.domain for spec in catalog} == {
        "财务管理",
        "人力资源",
        "采购管理",
        "信息技术",
        "法务合规",
        "行政运营",
        "产品制度",
    }
    assert len({fact.question for spec in catalog for fact in spec.facts}) == 1000
    assert all(
        fact.question != fact.paraphrase
        and fact.answer
        and fact.evidence
        and fact.answer in fact.evidence
        for spec in catalog
        for fact in spec.facts
    )


def test_catalog_is_deterministic():
    assert build_catalog() == build_catalog()
    assert make_spec(index=7, fmt="txt", scanned=False) == make_spec(
        index=7, fmt="txt", scanned=False
    )


def test_formats_are_contiguous_and_only_last_five_pdfs_are_scanned():
    catalog = build_catalog()

    assert [spec.fmt for spec in catalog] == (
        ["txt"] * 25 + ["docx"] * 25 + ["pdf"] * 25 + ["xlsx"] * 25
    )
    assert [spec.id for spec in catalog if spec.scanned] == [
        spec.id for spec in catalog[70:75]
    ]
    assert all(not spec.scanned for spec in catalog if spec.fmt != "pdf")


@pytest.mark.parametrize("index", [0, 6, 24, 25, 50, 75, 99])
def test_each_document_has_ten_distinct_traceable_facts(index):
    spec = build_catalog()[index]

    assert len({fact.question for fact in spec.facts}) == 10
    assert len({fact.evidence for fact in spec.facts}) == 10
    assert spec.version
    assert spec.effective_date
    assert spec.title in spec.facts[0].evidence
