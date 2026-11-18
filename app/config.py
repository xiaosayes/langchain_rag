"""Typed application settings loaded from YAML and environment overrides."""

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, SecretStr, field_validator


class ConfigurationError(ValueError):
    """Raised for expected YAML and typed configuration failures."""


SUPPORTED_EMBEDDING_DIMENSIONS = frozenset(
    {64, 128, 256, 512, 768, 1024, 1536, 2048}
)


class LLMSettings(BaseModel):
    model: str = "deepseek_v4.1_flash_1017"
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    temperature: float = 0.1
    max_output_tokens: int = 2048
    timeout_seconds: int = 60
    max_retries: int = 3


class ChunkSettings(BaseModel):
    parent_size: int = 1000
    child_size: int = 250
    child_overlap: int = 50


class EmbeddingSettings(BaseModel):
    model: str = "text-embedding-v4"
    base_url: str = "https://dashscope.aliyuncs.com"
    dimension: int = 1024
    batch_size: int = 10
    timeout_seconds: int = 60
    max_retries: int = 3
    retry_total_deadline_seconds: float = Field(default=30, gt=0)

    @field_validator("dimension")
    @classmethod
    def validate_dimension(cls, value: int) -> int:
        if value not in SUPPORTED_EMBEDDING_DIMENSIONS:
            supported = ", ".join(
                str(dimension)
                for dimension in sorted(SUPPORTED_EMBEDDING_DIMENSIONS)
            )
            raise ValueError(
                f"Unsupported embedding dimension {value}; supported: {supported}"
            )
        return value


class RerankSettings(BaseModel):
    model: str = "qwen3-rerank"
    base_url: str = "https://dashscope.aliyuncs.com"
    candidate_count: int = Field(default=30, gt=0)
    timeout_seconds: int = 30
    max_retries: int = 3
    retry_total_deadline_seconds: float = Field(default=30, gt=0)


class RetrievalSettings(BaseModel):
    dense_top_k: int = 30
    bm25_top_k: int = 30
    rrf_k: int = 60
    final_parent_count: int = 6
    max_parents_per_document: int = 2
    max_context_characters: int = 12000


class QdrantSettings(BaseModel):
    url: str = "http://127.0.0.1:16333"
    collection_prefix: str = "enterprise_rag"
    distance: Literal["cosine", "euclid", "dot"] = "cosine"


class AppSettings(BaseModel):
    app_env: Literal["production", "test"] = "production"
    model_provider: Literal["bailian", "fake"] = "bailian"
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    rerank: RerankSettings = Field(default_factory=RerankSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    chunk: ChunkSettings = Field(default_factory=ChunkSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    dashscope_api_key: SecretStr | None = None


def load_settings(path: Path = Path("configs/default.yaml")) -> AppSettings:
    """Load settings from YAML, then apply supported environment overrides."""
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as error:
        raise ConfigurationError(f"Invalid YAML configuration: {error}") from error
    if not isinstance(loaded, dict):
        raise ConfigurationError("Configuration root must be a mapping")

    data: dict[str, Any] = dict(loaded)
    environment_fields = {
        "APP_ENV": "app_env",
        "MODEL_PROVIDER": "model_provider",
        "DASHSCOPE_API_KEY": "dashscope_api_key",
    }
    for environment_name, field_name in environment_fields.items():
        value = os.environ.get(environment_name)
        if value is not None:
            data[field_name] = value

    try:
        return AppSettings.model_validate(data)
    except ValueError as error:
        raise ConfigurationError(str(error)) from error
