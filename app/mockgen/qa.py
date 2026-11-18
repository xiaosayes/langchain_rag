"""Generate source-grounded knowledge Q&A and held-out evaluation questions."""

import json
from dataclasses import dataclass
from pathlib import Path

from openpyxl import Workbook

from app.mockgen.catalog import DocumentSpec

_QA_COLUMNS = (
    "qa_id",
    "question",
    "answer",
    "source_document_id",
    "evidence",
    "domain",
)

_UNSUPPORTED_QUESTIONS = (
    "员工因私出国（境）审批需要提前多少个工作日提交申请？",
    "核心研发代码跨境传输的审批时限是多少小时？",
    "公司董事会会议材料的最低保密保存年限是多少年？",
    "海外子公司开立银行账户需要几名高管共同审批？",
    "商业航天项目供应商必须具备多少年的行业经验？",
    "员工使用个人生成式人工智能工具处理客户数据的罚款上限是多少元？",
    "数据中心遭遇区域性断电后必须在多少分钟内切换至异地园区？",
    "新建生物实验室的危险废弃物暂存期限是多少天？",
    "领导干部直系亲属经商办企业信息应当每多少个月申报一次？",
    "重大并购交易签署前的反垄断内部审查需要多少个工作日？",
    "公务接待中进口酒水的单人消费上限是多少元？",
    "员工申请无息购房借款的最高额度是多少万元？",
    "自动驾驶道路测试事故必须在多少小时内向董事会报告？",
    "涉密项目组成员离岗后的脱密期是多少个月？",
    "企业捐赠单笔超过多少万元必须经过职工代表大会审议？",
    "海外员工极端天气停工补贴按每天多少元发放？",
    "客户数据永久删除申请的内部处理时限是多少小时？",
    "量子计算研究设备的年度强制校准次数是多少次？",
    "公司吉祥物形象对外授权合同的有效期是多少年？",
    "员工参加国际空间技术会议的签证费用报销比例是多少？",
)


@dataclass(frozen=True)
class QAManifest:
    """Counts for the generated knowledge and held-out evaluation artifacts."""

    knowledge_qa_count: int
    evaluation_count: int
    answerable_count: int
    unanswerable_count: int


def _write_knowledge_qa(catalog: list[DocumentSpec], path: Path) -> int:
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

    workbook = Workbook(write_only=True)
    worksheet = workbook.create_sheet("knowledge_qa")
    worksheet.append(_QA_COLUMNS)
    for row in knowledge_rows:
        worksheet.append(tuple(row[column] for column in _QA_COLUMNS))
    workbook.save(path)
    return len(knowledge_rows)


def _write_evaluation(catalog: list[DocumentSpec], path: Path) -> tuple[int, int]:
    answerable = [
        {
            "eval_id": f"EVAL-{index:03d}",
            "question": spec.facts[0].paraphrase,
            "answerable": True,
            "expected_answer": spec.facts[0].answer,
            "source_document_ids": [spec.id],
            "evidence": spec.facts[0].evidence,
        }
        for index, spec in enumerate(catalog[:80], start=1)
    ]
    unsupported = [
        {
            "eval_id": f"EVAL-{index:03d}",
            "question": question,
            "answerable": False,
            "expected_answer": "",
            "source_document_ids": [],
            "evidence": "",
        }
        for index, question in enumerate(_UNSUPPORTED_QUESTIONS, start=81)
    ]
    evaluations = answerable + unsupported
    assert len(evaluations) == 100
    path.write_text(
        "".join(
            json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n"
            for item in evaluations
        ),
        encoding="utf-8",
        newline="\n",
    )
    return len(answerable), len(unsupported)


def generate_qa(catalog: list[DocumentSpec], output_dir: Path) -> QAManifest:
    """Write ten knowledge rows per document and 100 held-out evaluations."""
    qa_dir = output_dir / "qa"
    eval_dir = output_dir / "eval"
    qa_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    knowledge_qa_count = _write_knowledge_qa(catalog, qa_dir / "knowledge_qa.xlsx")
    answerable_count, unanswerable_count = _write_evaluation(
        catalog, eval_dir / "evaluation.jsonl"
    )
    return QAManifest(
        knowledge_qa_count=knowledge_qa_count,
        evaluation_count=answerable_count + unanswerable_count,
        answerable_count=answerable_count,
        unanswerable_count=unanswerable_count,
    )
