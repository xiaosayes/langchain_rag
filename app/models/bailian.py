"""Native Bailian embedding/rerank adapters and LangChain chat configuration."""

import asyncio
import math
import time
from dataclasses import dataclass
from typing import Any

import httpx
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI

from app.config import AppSettings
from app.models.retry import (
    AsyncSleep,
    SyncSleep,
    apost_with_retries,
    post_with_retries,
)

EMBEDDING_PATH = "/api/v1/services/embeddings/text-embedding/text-embedding"
RERANK_PATH = "/api/v1/services/rerank/text-rerank/text-rerank"


class BailianError(RuntimeError):
    """Base error for Bailian adapter failures."""


class BailianConfigurationError(BailianError):
    """Raised when required Bailian credentials are absent."""


class BailianResponseError(BailianError):
    """Raised when a successful Bailian response violates its contract."""


@dataclass(frozen=True)
class RerankResult:
    index: int
    relevance_score: float


def _authorization_headers(settings: AppSettings) -> dict[str, str]:
    api_key = settings.dashscope_api_key
    if api_key is None or not api_key.get_secret_value():
        raise BailianConfigurationError("DASHSCOPE_API_KEY is required")
    return {
        "Authorization": f"Bearer {api_key.get_secret_value()}",
        "Content-Type": "application/json",
    }


def _response_payload(response: httpx.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError as error:
        raise BailianResponseError("Bailian returned invalid JSON") from error
    if not isinstance(payload, dict):
        raise BailianResponseError("Bailian response must be a JSON object")
    return payload


class BailianEmbeddings(Embeddings):
    """LangChain embeddings backed by the native DashScope endpoint."""

    def __init__(
        self,
        settings: AppSettings,
        *,
        sleep: SyncSleep = time.sleep,
        async_sleep: AsyncSleep = asyncio.sleep,
    ) -> None:
        self.settings = settings
        self.dimension = settings.embedding.dimension
        self._url = settings.embedding.base_url.rstrip("/") + EMBEDDING_PATH
        self._headers = _authorization_headers(settings)
        self._sleep = sleep
        self._async_sleep = async_sleep

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        batch_size = self.settings.embedding.batch_size
        with httpx.Client(timeout=self.settings.embedding.timeout_seconds) as client:
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                response = post_with_retries(
                    client,
                    self._url,
                    json=self._request_payload(batch),
                    headers=self._headers,
                    max_retries=self.settings.embedding.max_retries,
                    sleep=self._sleep,
                )
                vectors.extend(self._parse_response(response, len(batch)))
        return vectors

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        batch_size = self.settings.embedding.batch_size
        async with httpx.AsyncClient(
            timeout=self.settings.embedding.timeout_seconds
        ) as client:
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                response = await apost_with_retries(
                    client,
                    self._url,
                    json=self._request_payload(batch),
                    headers=self._headers,
                    max_retries=self.settings.embedding.max_retries,
                    sleep=self._async_sleep,
                )
                vectors.extend(self._parse_response(response, len(batch)))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    async def aembed_query(self, text: str) -> list[float]:
        return (await self.aembed_documents([text]))[0]

    def _request_payload(self, texts: list[str]) -> dict[str, Any]:
        return {
            "model": self.settings.embedding.model,
            "input": {"texts": texts},
            "parameters": {"dimension": self.dimension},
        }

    def _parse_response(
        self, response: httpx.Response, expected_count: int
    ) -> list[list[float]]:
        payload = _response_payload(response)
        error = BailianResponseError(
            "Embedding count or dimension does not match the request"
        )
        try:
            items = payload["output"]["embeddings"]
        except (KeyError, TypeError) as exception:
            raise error from exception
        if not isinstance(items, list) or len(items) != expected_count:
            raise error

        vectors: list[list[float] | None] = [None] * expected_count
        for item in items:
            try:
                text_index = item["text_index"]
                embedding = item["embedding"]
            except (KeyError, TypeError) as exception:
                raise error from exception
            if (
                type(text_index) is not int
                or not 0 <= text_index < expected_count
                or vectors[text_index] is not None
                or not isinstance(embedding, list)
            ):
                raise error
            try:
                vector = [float(value) for value in embedding]
            except (TypeError, ValueError) as exception:
                raise error from exception
            if len(vector) != self.dimension:
                raise error
            vectors[text_index] = vector

        if any(vector is None for vector in vectors):
            raise error
        return [vector for vector in vectors if vector is not None]


class BailianReranker:
    """Synchronous and asynchronous native DashScope reranking."""

    def __init__(
        self,
        settings: AppSettings,
        *,
        sleep: SyncSleep = time.sleep,
        async_sleep: AsyncSleep = asyncio.sleep,
    ) -> None:
        self.settings = settings
        self._url = settings.rerank.base_url.rstrip("/") + RERANK_PATH
        self._headers = _authorization_headers(settings)
        self._sleep = sleep
        self._async_sleep = async_sleep

    def rerank(
        self, query: str, documents: list[str], top_n: int
    ) -> list[RerankResult]:
        with httpx.Client(timeout=self.settings.rerank.timeout_seconds) as client:
            response = post_with_retries(
                client,
                self._url,
                json=self._request_payload(query, documents, top_n),
                headers=self._headers,
                max_retries=self.settings.rerank.max_retries,
                sleep=self._sleep,
            )
        return self._parse_response(response, len(documents), top_n)

    async def arerank(
        self, query: str, documents: list[str], top_n: int
    ) -> list[RerankResult]:
        async with httpx.AsyncClient(
            timeout=self.settings.rerank.timeout_seconds
        ) as client:
            response = await apost_with_retries(
                client,
                self._url,
                json=self._request_payload(query, documents, top_n),
                headers=self._headers,
                max_retries=self.settings.rerank.max_retries,
                sleep=self._async_sleep,
            )
        return self._parse_response(response, len(documents), top_n)

    def _request_payload(
        self, query: str, documents: list[str], top_n: int
    ) -> dict[str, Any]:
        return {
            "model": self.settings.rerank.model,
            "input": {"query": query, "documents": documents},
            "parameters": {"top_n": top_n},
        }

    @staticmethod
    def _parse_response(
        response: httpx.Response, document_count: int, top_n: int
    ) -> list[RerankResult]:
        payload = _response_payload(response)
        error = BailianResponseError("Invalid rerank results in Bailian response")
        try:
            items = payload["output"]["results"]
        except (KeyError, TypeError) as exception:
            raise error from exception
        if not isinstance(items, list) or len(items) > top_n:
            raise error

        results: list[RerankResult] = []
        seen_indices: set[int] = set()
        for item in items:
            try:
                index = item["index"]
                raw_score = item["relevance_score"]
            except (KeyError, TypeError) as exception:
                raise error from exception
            if (
                type(index) is not int
                or not 0 <= index < document_count
                or index in seen_indices
            ):
                raise error
            try:
                score = float(raw_score)
            except (TypeError, ValueError) as exception:
                raise error from exception
            if not math.isfinite(score):
                raise error
            seen_indices.add(index)
            results.append(RerankResult(index=index, relevance_score=score))
        return results


def create_chat_model(settings: AppSettings) -> ChatOpenAI:
    """Build a chat model without substituting any configured model setting."""
    api_key = settings.dashscope_api_key
    if api_key is None or not api_key.get_secret_value():
        raise BailianConfigurationError("DASHSCOPE_API_KEY is required")
    return ChatOpenAI(
        model=settings.llm.model,
        api_key=api_key,
        base_url=settings.llm.base_url,
        temperature=settings.llm.temperature,
        max_completion_tokens=settings.llm.max_output_tokens,
        timeout=settings.llm.timeout_seconds,
        max_retries=settings.llm.max_retries,
    )
