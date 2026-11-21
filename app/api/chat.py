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
from app.monitoring.service import MonitoringService

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


def _start_run(
    monitoring: MonitoringService, question: str, conversation_id: str
) -> str | None:
    try:
        return monitoring.start_run(question, conversation_id)
    except Exception:  # noqa: BLE001 - diagnostics must not affect chat
        return None


def _finish_run(
    monitoring: MonitoringService,
    run_id: str | None,
    status: str,
    error: str | None = None,
) -> None:
    if run_id is None:
        return
    try:
        monitoring.finish_run(run_id, status, "", error)
    except Exception:  # noqa: BLE001, S110 - preserve the original chat outcome
        pass


async def _encode_stream(
    events: AsyncIterator[ChatEvent], release: Callable[[], None]
) -> AsyncIterator[bytes]:
    try:
        async for event in events:
            yield _encode_event(event)
    finally:
        try:
            await events.aclose()
        finally:
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
    monitoring: MonitoringService = request.app.state.monitoring_service
    run_id: str | None = None
    try:
        conversation_id = payload.conversation_id
        if conversation_id is None:
            conversation_id = service.conversation_service.create().id
        run_id = _start_run(monitoring, payload.question, conversation_id)
        if run_id is None:
            prepared = await service.prepare(payload.question, conversation_id)
        else:
            prepared = await service.prepare(
                payload.question, conversation_id, run_id=run_id
            )
    except ConversationNotFoundError as error:
        _finish_run(monitoring, run_id, "failed", _public_error(error))
        semaphore.release()
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found"
        ) from error
    except Exception as error:  # noqa: BLE001 - SSE is the public error boundary
        _finish_run(monitoring, run_id, "failed", _public_error(error))
        return _response(_preparation_error(error, semaphore.release))
    except BaseException:
        _finish_run(monitoring, run_id, "cancelled")
        semaphore.release()
        raise

    if run_id is None:
        events = service.stream(conversation_id, prepared)
    else:
        events = service.stream(conversation_id, prepared, run_id=run_id)
    return _response(_encode_stream(events, semaphore.release))
