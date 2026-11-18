from pathlib import Path

import pytest

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
