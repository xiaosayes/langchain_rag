"""Authenticated local message-feedback route."""

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.api.deps import get_auth_session, require_csrf
from app.chat.schemas import FeedbackOut, FeedbackRequest
from app.chat.service import ChatService, MessageNotFoundError

feedback_router = APIRouter(
    prefix="/api/feedback",
    tags=["feedback"],
    dependencies=[Depends(get_auth_session)],
)


@feedback_router.post(
    "", response_model=FeedbackOut, dependencies=[Depends(require_csrf)]
)
def upsert_feedback(payload: FeedbackRequest, request: Request) -> FeedbackOut:
    service: ChatService = request.app.state.chat_service
    try:
        feedback = service.upsert_feedback(
            payload.message_id, payload.score, payload.comment
        )
    except MessageNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Message not found"
        ) from error
    return FeedbackOut(
        id=feedback.id,
        message_id=feedback.message_id,
        score=feedback.score,
        comment=feedback.comment,
        created_at=feedback.created_at,
    )
