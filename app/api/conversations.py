"""Authenticated conversation and message-history routes."""

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from app.api.deps import get_auth_session, require_csrf
from app.conversations.schemas import (
    ConversationCreate,
    ConversationOut,
    ConversationRename,
    MessageOut,
)
from app.conversations.service import ConversationNotFoundError, ConversationService
from app.web.records import MessageRecord

conversations_router = APIRouter(
    prefix="/api/sessions",
    tags=["conversations"],
    dependencies=[Depends(get_auth_session)],
)


def _service(request: Request) -> ConversationService:
    return request.app.state.conversation_service


def _not_found() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Conversation not found",
    )


def _message_out(message: MessageRecord) -> MessageOut:
    return MessageOut(
        id=message.id,
        role=message.role,
        content=message.content,
        citations=list(message.citations_json),
        degraded=message.degraded,
        generation_id=message.generation_id,
        created_at=message.created_at,
    )


@conversations_router.get("", response_model=list[ConversationOut])
def list_conversations(request: Request) -> list[ConversationOut]:
    return _service(request).list()


@conversations_router.post(
    "",
    response_model=ConversationOut,
    dependencies=[Depends(require_csrf)],
)
def create_conversation(
    payload: ConversationCreate, request: Request
) -> ConversationOut:
    return _service(request).create(payload.title)


@conversations_router.patch(
    "/{conversation_id}",
    response_model=ConversationOut,
    dependencies=[Depends(require_csrf)],
)
def rename_conversation(
    conversation_id: str, payload: ConversationRename, request: Request
) -> ConversationOut:
    try:
        return _service(request).rename(conversation_id, payload.title)
    except ConversationNotFoundError as error:
        raise _not_found() from error


@conversations_router.delete(
    "/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(require_csrf)],
)
def delete_conversation(conversation_id: str, request: Request) -> Response:
    try:
        _service(request).delete(conversation_id)
    except ConversationNotFoundError as error:
        raise _not_found() from error
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@conversations_router.get(
    "/{conversation_id}/messages", response_model=list[MessageOut]
)
def list_messages(conversation_id: str, request: Request) -> list[MessageOut]:
    try:
        messages = _service(request).list_messages(conversation_id)
    except ConversationNotFoundError as error:
        raise _not_found() from error
    return [_message_out(message) for message in messages]
