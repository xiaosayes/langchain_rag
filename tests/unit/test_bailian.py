import json

import httpx
import pytest
import respx

from app.config import load_settings
from app.models.bailian import (
    EMBEDDING_PATH,
    RERANK_PATH,
    BailianEmbeddings,
    BailianRequestError,
    BailianReranker,
    BailianResponseError,
    create_chat_model,
)


def make_settings(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "unit-test-secret")
    return load_settings()


@respx.mock
def test_embeddings_restore_input_order(monkeypatch, caplog):
    settings = make_settings(monkeypatch)
    route = respx.post(settings.embedding.base_url + EMBEDDING_PATH).mock(
        return_value=httpx.Response(
            200,
            json={
                "output": {
                    "embeddings": [
                        {"text_index": 1, "embedding": [0.0] * 1024},
                        {"text_index": 0, "embedding": [1.0] * 1024},
                    ]
                }
            },
        )
    )

    vectors = BailianEmbeddings(settings).embed_documents(["first", "second"])

    assert len(vectors) == 2
    assert all(len(vector) == 1024 for vector in vectors)
    assert vectors[0][0] == 1.0
    assert vectors[1][0] == 0.0
    request = route.calls[0].request
    assert request.headers["authorization"] == "Bearer unit-test-secret"
    assert request.read() == (
        b'{"model":"text-embedding-v4","input":{"texts":["first","second"]},'
        b'"parameters":{"dimension":1024}}'
    )
    assert "unit-test-secret" not in caplog.text


@respx.mock
@pytest.mark.parametrize(
    "items",
    [
        [{"text_index": 0, "embedding": [0.0] * 1024}],
        [
            {"text_index": 0, "embedding": [0.0] * 1024},
            {"text_index": 1, "embedding": [0.0] * 1023},
        ],
        [
            {"text_index": 0, "embedding": [0.0] * 1024},
            {"text_index": 0, "embedding": [1.0] * 1024},
        ],
        [
            {"text_index": 0, "embedding": [0.0] * 1024},
            {"text_index": 2, "embedding": [1.0] * 1024},
        ],
    ],
)
def test_embeddings_reject_invalid_count_dimension_or_indices(monkeypatch, items):
    settings = make_settings(monkeypatch)
    respx.post(settings.embedding.base_url + EMBEDDING_PATH).mock(
        return_value=httpx.Response(200, json={"output": {"embeddings": items}})
    )

    with pytest.raises(BailianResponseError):
        BailianEmbeddings(settings).embed_documents(["first", "second"])


@respx.mock
def test_embeddings_respect_configured_batch_size(monkeypatch):
    settings = make_settings(monkeypatch)
    settings = settings.model_copy(
        update={"embedding": settings.embedding.model_copy(update={"batch_size": 1})}
    )
    route = respx.post(settings.embedding.base_url + EMBEDDING_PATH).mock(
        side_effect=[
            httpx.Response(
                200,
                json={
                    "output": {
                        "embeddings": [{"text_index": 0, "embedding": [1.0] * 1024}]
                    }
                },
            ),
            httpx.Response(
                200,
                json={
                    "output": {
                        "embeddings": [{"text_index": 0, "embedding": [2.0] * 1024}]
                    }
                },
            ),
        ]
    )

    vectors = BailianEmbeddings(settings).embed_documents(["first", "second"])

    assert [vector[0] for vector in vectors] == [1.0, 2.0]
    assert route.call_count == 2


@respx.mock
def test_embeddings_retry_429_then_success(monkeypatch):
    settings = make_settings(monkeypatch)
    route = respx.post(settings.embedding.base_url + EMBEDDING_PATH).mock(
        side_effect=[
            httpx.Response(429, json={"message": "rate limited"}),
            httpx.Response(
                200,
                json={
                    "output": {
                        "embeddings": [{"text_index": 0, "embedding": [1.0] * 1024}]
                    }
                },
            ),
        ]
    )
    sleeps = []

    vector = BailianEmbeddings(settings, sleep=sleeps.append).embed_query("query")

    assert len(vector) == 1024
    assert route.call_count == 2
    assert len(sleeps) == 1
    assert sleeps[0] > 0


@respx.mock
def test_embeddings_do_not_retry_400(monkeypatch):
    settings = make_settings(monkeypatch)
    route = respx.post(settings.embedding.base_url + EMBEDDING_PATH).mock(
        return_value=httpx.Response(400, json={"message": "bad request"})
    )
    sleeps = []

    with pytest.raises(BailianRequestError, match="model|parameters") as error:
        BailianEmbeddings(settings, sleep=sleeps.append).embed_query("query")

    assert "bad request" not in str(error.value)
    assert "unit-test-secret" not in str(error.value)
    assert route.call_count == 1
    assert sleeps == []


@respx.mock
def test_embeddings_retry_read_timeout_but_not_connect_error(monkeypatch):
    settings = make_settings(monkeypatch)
    route = respx.post(settings.embedding.base_url + EMBEDDING_PATH).mock(
        side_effect=[
            httpx.ReadTimeout("read timed out"),
            httpx.Response(
                200,
                json={
                    "output": {
                        "embeddings": [{"text_index": 0, "embedding": [1.0] * 1024}]
                    }
                },
            ),
        ]
    )
    sleeps = []

    vector = BailianEmbeddings(settings, sleep=sleeps.append).embed_query("query")

    assert len(vector) == 1024
    assert route.call_count == 2
    assert len(sleeps) == 1

    route.mock(side_effect=httpx.ConnectError("connection failed"))
    with pytest.raises(BailianRequestError, match="connect|network"):
        BailianEmbeddings(settings, sleep=sleeps.append).embed_query("query")
    assert route.call_count == 3


@respx.mock
def test_embeddings_401_is_sanitized_actionable_and_not_retried(monkeypatch):
    settings = make_settings(monkeypatch)
    route = respx.post(settings.embedding.base_url + EMBEDDING_PATH).mock(
        return_value=httpx.Response(401, json={"message": "secret backend detail"})
    )
    sleeps = []

    with pytest.raises(BailianRequestError, match="DASHSCOPE_API_KEY") as error:
        BailianEmbeddings(settings, sleep=sleeps.append).embed_query("query")

    assert route.call_count == 1
    assert sleeps == []
    assert "secret backend detail" not in str(error.value)
    assert "unit-test-secret" not in str(error.value)


@respx.mock
@pytest.mark.parametrize("non_finite", ["NaN", "Infinity", "-Infinity"])
def test_embeddings_reject_non_finite_components_from_raw_json(
    monkeypatch, non_finite
):
    settings = make_settings(monkeypatch)
    vector = "[" + ",".join(["0.0"] * 1023 + [non_finite]) + "]"
    content = (
        '{"output":{"embeddings":[{"text_index":0,"embedding":'
        + vector
        + "}]}}"
    )
    respx.post(settings.embedding.base_url + EMBEDDING_PATH).mock(
        return_value=httpx.Response(
            200, content=content, headers={"Content-Type": "application/json"}
        )
    )

    with pytest.raises(BailianResponseError):
        BailianEmbeddings(settings).embed_query("query")


@respx.mock
def test_rerank_parses_results_and_sends_top_n(monkeypatch):
    settings = make_settings(monkeypatch)
    route = respx.post(settings.rerank.base_url + RERANK_PATH).mock(
        return_value=httpx.Response(
            200,
            json={
                "output": {
                    "results": [
                        {"index": 1, "relevance_score": 0.75},
                        {"index": 0, "relevance_score": 0.25},
                    ]
                }
            },
        )
    )

    results = BailianReranker(settings).rerank("query", ["first", "second"], 2)

    assert [(result.index, result.relevance_score) for result in results] == [
        (1, 0.75),
        (0, 0.25),
    ]
    request = route.calls[0].request
    assert request.headers["authorization"] == "Bearer unit-test-secret"
    assert request.read() == (
        b'{"model":"qwen3-rerank","input":{"query":"query",'
        b'"documents":["first","second"]},"parameters":{"top_n":2}}'
    )


@respx.mock
@pytest.mark.asyncio
async def test_arerank_uses_async_http(monkeypatch):
    settings = make_settings(monkeypatch)
    route = respx.post(settings.rerank.base_url + RERANK_PATH).mock(
        return_value=httpx.Response(
            200,
            json={"output": {"results": [{"index": 0, "relevance_score": 1.0}]}},
        )
    )

    results = await BailianReranker(settings).arerank("query", ["document"], 1)

    assert results[0].index == 0
    assert results[0].relevance_score == 1.0
    assert route.call_count == 1


@respx.mock
@pytest.mark.parametrize(
    "items",
    [
        [{"index": 2, "relevance_score": 0.5}],
        [{"index": -1, "relevance_score": 0.5}],
        [{"index": 0, "relevance_score": float("nan")}],
        [{"index": 0, "relevance_score": float("inf")}],
        [
            {"index": 0, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.8},
        ],
    ],
)
def test_rerank_rejects_invalid_indices_and_scores(monkeypatch, items):
    settings = make_settings(monkeypatch)
    respx.post(settings.rerank.base_url + RERANK_PATH).mock(
        return_value=httpx.Response(
            200,
            content=json.dumps({"output": {"results": items}}),
            headers={"Content-Type": "application/json"},
        )
    )

    with pytest.raises(BailianResponseError):
        BailianReranker(settings).rerank("query", ["first", "second"], 2)


def test_create_chat_model_uses_exact_configuration(monkeypatch):
    settings = make_settings(monkeypatch)

    chat = create_chat_model(settings)

    assert chat.model_name == settings.llm.model
    assert chat.openai_api_base == settings.llm.base_url
    assert chat.temperature == settings.llm.temperature
    assert chat.max_tokens == settings.llm.max_output_tokens
    assert chat.request_timeout == settings.llm.timeout_seconds
    assert chat.max_retries == settings.llm.max_retries
    assert chat.openai_api_key is not None
    assert chat.openai_api_key.get_secret_value() == "unit-test-secret"
