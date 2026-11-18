"""Deterministic synthetic specifications for the enterprise document corpus."""

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Fact:
    """A question, its paraphrase, and source-grounded answer data."""

    question: str
    paraphrase: str
    answer: str
    evidence: str


@dataclass(frozen=True)
class DocumentSpec:
    """All source data needed to render one synthetic enterprise document."""

    id: str
    domain: str
    title: str
    fmt: str
    scanned: bool
    facts: tuple[Fact, ...]
    version: str
    effective_date: str


@dataclass(frozen=True)
class _Metric:
    label: str
    unit: str
    minimum: int
    maximum: int


@dataclass(frozen=True)
class _DomainTemplate:
    domain: str
    subject: str
    metrics: tuple[_Metric, ...]


def _metric(label: str, unit: str, minimum: int, maximum: int) -> _Metric:
    return _Metric(label, unit, minimum, maximum)


_DOMAIN_TEMPLATES = (
    _DomainTemplate(
        domain="财务管理",
        subject="费用与预算管理规范",
        metrics=(
            _metric("单笔差旅报销上限", "元", 3000, 9000),
            _metric("日常费用审批时限", "个工作日", 2, 7),
            _metric("年度预算预留比例", "%", 3, 12),
            _metric("发票归档保存年限", "年", 5, 15),
            _metric("部门月度预算核查次数", "次", 1, 4),
            _metric("备用金申请提前期", "个工作日", 2, 6),
            _metric("付款复核间隔上限", "小时", 8, 48),
            _metric("固定资产盘点频次", "次/年", 1, 4),
            _metric("预算调整审批人数", "人", 2, 5),
            _metric("财务档案调阅期限", "个工作日", 3, 10),
        ),
    ),
    _DomainTemplate(
        domain="人力资源",
        subject="员工发展与福利管理办法",
        metrics=(
            _metric("新员工入职培训时长", "小时", 8, 40),
            _metric("年度带薪培训天数", "天", 2, 10),
            _metric("试用期目标复盘次数", "次", 1, 5),
            _metric("导师辅导周期", "周", 4, 16),
            _metric("内部课程积分要求", "分", 10, 60),
            _metric("绩效反馈完成时限", "个工作日", 3, 10),
            _metric("岗位轮换最短周期", "个月", 3, 18),
            _metric("年度健康体检项目数", "项", 8, 25),
            _metric("技能认证补贴上限", "元", 500, 5000),
            _metric("员工建议处理时限", "个工作日", 3, 12),
        ),
    ),
    _DomainTemplate(
        domain="采购管理",
        subject="供应商与采购执行细则",
        metrics=(
            _metric("常规采购候选供应商数量", "家", 2, 6),
            _metric("采购需求审批时限", "个工作日", 2, 8),
            _metric("供应商年度评审次数", "次", 1, 4),
            _metric("样品验收观察期", "天", 3, 20),
            _metric("合同付款账期", "天", 15, 90),
            _metric("紧急采购复核人数", "人", 2, 5),
            _metric("到货抽检最低比例", "%", 5, 30),
            _metric("供应商整改期限", "个工作日", 5, 20),
            _metric("采购档案保存年限", "年", 5, 15),
            _metric("框架协议有效月数", "个月", 6, 36),
        ),
    ),
    _DomainTemplate(
        domain="信息技术",
        subject="信息系统运行与安全规范",
        metrics=(
            _metric("关键系统备份频率", "次/日", 1, 6),
            _metric("账号权限复核周期", "天", 30, 180),
            _metric("严重故障响应时限", "分钟", 5, 60),
            _metric("操作日志保存月数", "个月", 6, 36),
            _metric("终端补丁安装期限", "个工作日", 1, 10),
            _metric("恢复演练年度次数", "次", 1, 6),
            _metric("密码最小长度", "位", 8, 20),
            _metric("离岗账号停用时限", "小时", 1, 24),
            _metric("变更冻结通知提前期", "小时", 12, 72),
            _metric("安全告警复核时限", "分钟", 10, 120),
        ),
    ),
    _DomainTemplate(
        domain="法务合规",
        subject="合同审查与合规管理规则",
        metrics=(
            _metric("标准合同审查时限", "个工作日", 2, 8),
            _metric("重大合同会签人数", "人", 2, 6),
            _metric("合同原件保存年限", "年", 5, 20),
            _metric("合规培训年度时长", "小时", 4, 24),
            _metric("印章使用登记复核频率", "次/月", 1, 4),
            _metric("法律风险反馈时限", "个工作日", 1, 7),
            _metric("授权文件有效月数", "个月", 3, 24),
            _metric("争议事项报告时限", "小时", 4, 48),
            _metric("合规抽查最低比例", "%", 5, 30),
            _metric("制度年度复审次数", "次", 1, 4),
        ),
    ),
    _DomainTemplate(
        domain="行政运营",
        subject="办公场所与行政服务标准",
        metrics=(
            _metric("会议室预约提前期", "小时", 1, 48),
            _metric("访客登记信息保存月数", "个月", 3, 24),
            _metric("办公用品领用周期", "天", 7, 45),
            _metric("公共区域巡检次数", "次/日", 1, 6),
            _metric("维修工单响应时限", "小时", 1, 12),
            _metric("公务车辆申请提前期", "小时", 2, 48),
            _metric("工位调整通知期限", "个工作日", 2, 10),
            _metric("消防设施检查频率", "次/月", 1, 4),
            _metric("行政仓库盘点频次", "次/季度", 1, 3),
            _metric("快递滞留提醒时限", "小时", 4, 48),
        ),
    ),
    _DomainTemplate(
        domain="市场销售",
        subject="客户服务与销售过程管理规范",
        metrics=(
            _metric("客户首次响应时限", "小时", 1, 12),
            _metric("重点客户回访频率", "次/月", 1, 8),
            _metric("销售线索分配时限", "小时", 2, 24),
            _metric("报价审批完成时限", "个工作日", 1, 6),
            _metric("客户投诉闭环期限", "个工作日", 2, 10),
            _metric("演示环境准备提前期", "小时", 4, 48),
            _metric("商机阶段复核频率", "次/周", 1, 5),
            _metric("客户资料保存年限", "年", 3, 10),
            _metric("市场活动复盘时限", "个工作日", 2, 10),
            _metric("服务满意度最低目标", "%", 80, 99),
        ),
    ),
)

_EFFECTIVE_DATES = (
    "2026-01-15",
    "2026-02-15",
    "2026-03-15",
    "2026-04-15",
    "2026-05-15",
    "2026-06-15",
    "2026-07-15",
)


def make_spec(index: int, fmt: str, scanned: bool) -> DocumentSpec:
    """Build one explicit document specification using its fixed local seed."""
    template = _DOMAIN_TEMPLATES[index % len(_DOMAIN_TEMPLATES)]
    rng = random.Random(20261118 + index)
    document_id = f"DOC-{index:03d}"
    title = f"{template.subject}（{index + 1:03d}）"
    facts = []

    for metric in template.metrics:
        value = rng.randint(metric.minimum, metric.maximum)
        answer = f"{metric.label}为{value}{metric.unit}。"
        facts.append(
            Fact(
                question=f"{title}的{metric.label}是多少？",
                paraphrase=f"请说明{title}中{metric.label}的具体数值。",
                answer=answer,
                evidence=f"{title}明确规定，{answer}",
            )
        )

    return DocumentSpec(
        id=document_id,
        domain=template.domain,
        title=title,
        fmt=fmt,
        scanned=scanned,
        facts=tuple(facts),
        version=f"{1 + index % 3}.0",
        effective_date=_EFFECTIVE_DATES[index % len(_EFFECTIVE_DATES)],
    )


def build_catalog() -> list[DocumentSpec]:
    """Return the complete, stable 100-document catalog in format blocks."""
    formats = ["txt"] * 25 + ["docx"] * 25 + ["pdf"] * 25 + ["xlsx"] * 25
    return [
        make_spec(index=index, fmt=fmt, scanned=fmt == "pdf" and index % 25 >= 20)
        for index, fmt in enumerate(formats)
    ]
