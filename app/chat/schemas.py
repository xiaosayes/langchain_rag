"""Typed chat requests and server-sent event payloads."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    conversation_id: str | None = None
    question: str = Field(min_length=1)


@dataclass(frozen=True)
class ChatEvent:
    event: Literal["start", "token", "citations", "done", "error"]
    data: dict[str, Any]


class FeedbackRequest(BaseModel):
    message_id: str = Field(min_length=1)
    score: Literal[-1, 1]
    comment: str | None = Field(default=None, max_length=2000)


class FeedbackOut(BaseModel):
    id: str
    message_id: str
    score: int
    comment: str | None
    created_at: datetime
