"""Authenticated, concurrency-limited server-sent event chat."""

import asyncio
import json
from collections.abc import AsyncIterator, Callable

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from app.api.deps import get_auth_session, require_csrf
from app.chat.schemas import ChatEvent, ChatRequest
from app.chat.service import ChatService, NoActiveGenerationError
from app.conversations.service import ConversationNotFoundError
from app.error_sanitization import sanitized_error_message

chat_router = APIRouter(
    prefix="/api/chat",
    tags=["chat"],
    dependencies=[Depends(get_auth_session)],
)


def _encode_event(event: ChatEvent) -> bytes:
    data = json.dumps(event.data, ensure_ascii=False)
    return f"event: {event.event}\ndata: {data}\n\n".encode()


def _public_error(error: Exception) -> str:
    if isinstance(error, NoActiveGenerationError):
        return str(error)
    return sanitized_error_message(error)


async def _encode_stream(
    events: AsyncIterator[ChatEvent], release: Callable[[], None]
) -> AsyncIterator[bytes]:
    try:
        async for event in events:
            yield _encode_event(event)
    finally:
        await events.aclose()
        release()


async def _preparation_error(
    error: Exception, release: Callable[[], None]
) -> AsyncIterator[bytes]:
    try:
        yield _encode_event(ChatEvent("error", {"message": _public_error(error)}))
    finally:
        release()


def _response(content: AsyncIterator[bytes]) -> StreamingResponse:
    return StreamingResponse(
        content,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


@chat_router.post("/stream", dependencies=[Depends(require_csrf)])
async def stream_chat(payload: ChatRequest, request: Request) -> StreamingResponse:
    semaphore: asyncio.Semaphore = request.app.state.chat_semaphore
    if semaphore.locked():
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Chat concurrency limit reached",
        )
    # There is no suspension between the locked check and this immediate acquire,
    # so requests on the event loop cannot race past the limit.
    await semaphore.acquire()
    service: ChatService = request.app.state.chat_service
    try:
        conversation_id = payload.conversation_id
        if conversation_id is None:
            conversation_id = service.conversation_service.create().id
        prepared = await service.prepare(payload.question, conversation_id)
    except ConversationNotFoundError as error:
        semaphore.release()
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found"
        ) from error
    except Exception as error:  # noqa: BLE001 - SSE is the public error boundary
        return _response(_preparation_error(error, semaphore.release))
    except BaseException:
        semaphore.release()
        raise

    return _response(
        _encode_stream(
            service.stream(conversation_id, prepared), semaphore.release
        )
    )
