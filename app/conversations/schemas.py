"""Public request and response schemas for persisted conversations."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

MAX_TITLE_LENGTH = 200


class ConversationCreate(BaseModel):
    title: str | None = Field(default=None, max_length=MAX_TITLE_LENGTH)


class ConversationRename(BaseModel):
    title: str = Field(max_length=MAX_TITLE_LENGTH)


class ConversationOut(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime


class MessageOut(BaseModel):
    id: str
    role: str
    content: str
    citations: list[dict[str, Any]] = Field(default_factory=list)
    degraded: bool
    generation_id: str | None
    created_at: datetime
