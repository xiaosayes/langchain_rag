#!/usr/bin/env python3
"""Small real-Bailian smoke check using only runtime project dependencies."""

from __future__ import annotations

import asyncio
import math
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import load_settings
from app.models.bailian import (
    BailianEmbeddings,
    BailianReranker,
    create_chat_model,
)


async def check_chat() -> None:
    settings = load_settings()
    chat = create_chat_model(settings)
    chunks: list[str] = []
    async for chunk in chat.astream("请只回答：测试成功"):
        if chunk.content and str(chunk.content).strip():
            chunks.append(str(chunk.content))
    if not chunks:
        raise RuntimeError("Bailian chat returned no content")
    print("Chat streaming: OK")


def main() -> None:
    if os.environ.get("MODEL_PROVIDER") != "bailian":
        raise SystemExit("MODEL_PROVIDER must be bailian")
    if os.environ.get("APP_ENV") != "production":
        raise SystemExit("APP_ENV must be production")
    if not os.environ.get("DASHSCOPE_API_KEY"):
        raise SystemExit("DASHSCOPE_API_KEY is not configured")

    settings = load_settings()
    vector = BailianEmbeddings(settings).embed_query("差旅报销")
    if len(vector) != 1024 or not all(math.isfinite(value) for value in vector):
        raise RuntimeError("Unexpected embedding response")
    print("Embedding 1024 dimensions: OK")

    results = BailianReranker(settings).rerank(
        "差旅报销需要哪些材料",
        ["差旅报销需要发票和行程单。", "今天天气晴朗，适合散步。"],
        2,
    )
    if len(results) != 2 or results[0].index != 0:
        raise RuntimeError("Unexpected rerank response")
    print("Rerank relevance ordering: OK")

    asyncio.run(check_chat())
    print("Real Bailian smoke: PASS")


if __name__ == "__main__":
    main()
