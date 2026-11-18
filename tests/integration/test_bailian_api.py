import math
import os

import pytest

from app.config import AppSettings, load_settings
from app.models.bailian import BailianEmbeddings, BailianReranker, create_chat_model

pytestmark = pytest.mark.bailian


def require_real_bailian() -> AppSettings:
    if os.environ.get("RUN_BAILIAN_TESTS") != "1":
        pytest.skip("real Bailian tests are disabled")
    if not os.environ.get("DASHSCOPE_API_KEY"):
        pytest.skip("DASHSCOPE_API_KEY is not configured")
    return load_settings()


def test_real_embedding_has_configured_dimension():
    settings = require_real_bailian()

    vector = BailianEmbeddings(settings).embed_query("差旅报销")

    assert len(vector) == 1024
    assert all(math.isfinite(value) for value in vector)


def test_real_rerank_prefers_directly_relevant_document():
    settings = require_real_bailian()

    results = BailianReranker(settings).rerank(
        "差旅报销需要哪些材料",
        ["差旅报销需要发票和行程单。", "今天天气晴朗，适合散步。"],
        2,
    )

    assert len(results) == 2
    assert results[0].index == 0
    assert math.isfinite(results[0].relevance_score)


@pytest.mark.asyncio
async def test_real_chat_astream_yields_content_from_configured_model():
    settings = require_real_bailian()
    chat = create_chat_model(settings)

    chunks = []
    async for chunk in chat.astream("请只回答：测试成功"):
        if chunk.content and str(chunk.content).strip():
            chunks.append(chunk)

    assert chat.model_name == settings.llm.model
    assert chunks
