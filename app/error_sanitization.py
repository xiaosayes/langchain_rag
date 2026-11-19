"""Short, actionable messages for expected operational failures."""

import httpx
from openai import OpenAIError
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from sqlalchemy.exc import SQLAlchemyError

from app.config import ConfigurationError
from app.models.bailian import BailianError
from app.retrieval.service import RetrievalUnavailable


def sanitized_error_message(
    error: Exception, *, preserve_application_message: bool = False
) -> str:
    """Return a category-specific message without raw diagnostics or credentials."""
    error_type = type(error).__name__
    if isinstance(error, BailianError):
        return str(error)
    if isinstance(error, ConfigurationError):
        return (
            f"{error_type}: Invalid configuration; check YAML syntax and "
            "configured values"
        )
    if isinstance(error, SQLAlchemyError):
        return (
            f"{error_type}: Database operation failed; check local index storage "
            "and run verify"
        )
    if isinstance(error, (ResponseHandlingException, UnexpectedResponse)):
        return (
            f"{error_type}: Vector store operation failed; check Qdrant "
            "availability and configuration"
        )
    if isinstance(error, OpenAIError):
        return (
            f"{error_type}: Chat provider request failed; check chat credentials, "
            "model configuration, and connectivity"
        )
    if isinstance(error, RetrievalUnavailable):
        return (
            f"{error_type}: Hybrid retrieval unavailable; check Qdrant, BM25, "
            "and the embedding provider"
        )
    if isinstance(error, httpx.HTTPError):
        return (
            f"{error_type}: Network request failed; check connectivity and "
            "service availability"
        )
    if isinstance(error, OSError):
        return (
            f"{error_type}: Local file or network operation failed; check "
            "configured paths, permissions, and connectivity"
        )
    if preserve_application_message and isinstance(error, (RuntimeError, ValueError)):
        return str(error)
    return f"{error_type}: Operation failed; check inputs and stored generation state"
