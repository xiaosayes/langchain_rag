"""Chinese prompts for retrieval query rewriting and grounded answering."""

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

GROUNDED_ANSWER_PROMPT = """你是企业知识库问答助手，必须始终使用中文回答。
只能依据下面提供的来源作答，不得使用外部知识，不得编造事实、来源或引用。
每个由来源支撑的陈述都应紧跟对应来源的引用标记，例如[S1]；只能使用实际提供的标记。
同一主题存在多个制度版本时，优先采用当前版本，不得让旧版本覆盖当前版本。
如果来源信息不足，无法可靠回答问题，请明确说明知识不足，不要猜测，并只回答：
根据当前知识库，无法可靠回答该问题。"""

REWRITE_QUERY_PROMPT = """你是检索查询改写助手。请根据对话历史，将用户最新问题改写为语义完整、可独立检索的中文查询。
保留最新问题中的实体、条件和限制，只补充必要的上下文，不要回答问题，不要添加新事实。
只输出改写后的查询文本。"""


def build_rewrite_messages(
    question: str, history: list[BaseMessage]
) -> list[BaseMessage]:
    """Build the single rewrite request for a follow-up question."""
    return [
        SystemMessage(content=REWRITE_QUERY_PROMPT),
        *history,
        HumanMessage(content=f"请改写这个最新问题：\n{question}"),
    ]


def build_grounded_messages(
    question: str, documents: list[Document]
) -> list[BaseMessage]:
    """Label each supplied parent once and construct the grounded answer request."""
    sources = "\n\n".join(
        f"[S{index}]\n{document.page_content}"
        for index, document in enumerate(documents, start=1)
    )
    return [
        SystemMessage(content=GROUNDED_ANSWER_PROMPT),
        HumanMessage(content=f"用户问题：\n{question}\n\n知识库来源：\n{sources}"),
    ]
