"""Fail-open persistence and queries for local chat diagnostic traces."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import UTC, datetime, timedelta
from math import ceil
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from sqlalchemy import delete, select
from sqlalchemy.orm import Session, sessionmaker

from app.web.records import ChatRunEventRecord, ChatRunRecord

if TYPE_CHECKING:
    from app.rag.service import RAGPreparation

logger = logging.getLogger(__name__)

_RUN_STATUSES = frozenset({"running", "succeeded", "failed", "cancelled"})
_TERMINAL_STATUSES = frozenset({"succeeded", "failed", "cancelled"})


class MonitoringService:
    """Persist local traces without allowing monitoring failures to escape."""

    def __init__(
        self, session_factory: sessionmaker[Session], state_root: Path
    ) -> None:
        self.session_factory = session_factory
        self.state_root = Path(state_root)
        self.trace_path = self.state_root / "monitoring" / "chat-trace.jsonl"
        self._clock_lock = Lock()
        self._jsonl_lock = Lock()
        self._started_clocks: dict[str, float] = {}

    @staticmethod
    def _warn(operation: str, error: Exception) -> None:
        """Log only the operation and exception category, never diagnostics."""
        logger.warning("Monitoring %s failed (%s)", operation, type(error).__name__)

    def start_run(self, question: str, conversation_id: str | None) -> str:
        """Start a running trace and return its ID even if storage is unavailable."""
        run_id = uuid4().hex
        with self._clock_lock:
            self._started_clocks[run_id] = perf_counter()
        try:
            with self.session_factory.begin() as session:
                session.add(
                    ChatRunRecord(
                        id=run_id,
                        conversation_id=conversation_id,
                        question=question,
                        status="running",
                        created_at=datetime.now(UTC),
                    )
                )
        except Exception as error:  # noqa: BLE001 - monitoring must fail open
            self._warn("start", error)
        return run_id

    def record_event(
        self,
        run_id: str,
        stage: str,
        status: str,
        started_at: datetime,
        duration_ms: float,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Record a stage using its UTC start and caller-measured duration."""
        try:
            duration = float(duration_ms)
            if duration < 0:
                raise ValueError("duration_ms must be nonnegative")
            ended_at = started_at + timedelta(milliseconds=duration)
            with self.session_factory.begin() as session:
                session.add(
                    ChatRunEventRecord(
                        run_id=run_id,
                        stage=stage,
                        status=status,
                        started_at=started_at,
                        ended_at=ended_at,
                        duration_ms=duration,
                        details_json=details or {},
                    )
                )
        except Exception as error:  # noqa: BLE001 - monitoring must fail open
            self._warn("event", error)

    def update_preparation(
        self,
        run_id: str,
        prepared: RAGPreparation,
        prompt_messages: list[Any],
    ) -> None:
        """Attach the complete local prompt and retrieved context to a run."""
        try:
            messages = [self._prompt_message(message) for message in prompt_messages]
            retrieved_text = "\n\n".join(
                document.page_content for document in prepared.documents
            )
            with self.session_factory.begin() as session:
                run = session.get(ChatRunRecord, run_id)
                if run is None or run.status != "running":
                    return
                run.retrieval_query = prepared.retrieval_query
                run.prompt_messages_json = messages
                run.retrieved_text = retrieved_text
                run.generation_id = prepared.generation_id
                run.degraded = prepared.degraded
        except Exception as error:  # noqa: BLE001 - monitoring must fail open
            self._warn("preparation", error)

    @staticmethod
    def _prompt_message(message: Any) -> dict[str, Any]:
        if isinstance(message, dict):
            return {
                "type": str(message.get("type", "message")),
                "content": message.get("content", ""),
            }
        return {
            "type": str(getattr(message, "type", type(message).__name__)),
            "content": getattr(message, "content", str(message)),
        }

    def finish_run(
        self,
        run_id: str,
        status: str,
        answer: str,
        error: str | None = None,
    ) -> None:
        """Finish a running run once, then append its complete JSONL snapshot."""
        if status not in _TERMINAL_STATUSES:
            raise ValueError("status must be succeeded, failed, or cancelled")
        with self._clock_lock:
            started_clock = self._started_clocks.pop(run_id, None)
        ended_at = datetime.now(UTC)
        duration_ms = (
            max(0.0, (perf_counter() - started_clock) * 1000)
            if started_clock is not None
            else None
        )
        try:
            snapshot: dict[str, Any] | None = None
            with self.session_factory.begin() as session:
                run = session.get(ChatRunRecord, run_id)
                if run is None or run.status != "running":
                    return
                run.status = status
                run.answer = answer
                run.error = error
                run.ended_at = ended_at
                run.duration_ms = duration_ms
                session.flush()
                events = session.scalars(
                    select(ChatRunEventRecord)
                    .where(ChatRunEventRecord.run_id == run_id)
                    .order_by(
                        ChatRunEventRecord.started_at,
                        ChatRunEventRecord.id,
                    )
                ).all()
                snapshot = self._run_detail(run, events)
            if snapshot is not None:
                self._append_terminal_snapshot(snapshot)
        except Exception as monitoring_error:  # noqa: BLE001 - fail open
            self._warn("finish", monitoring_error)

    def summary(self, hours: int) -> dict[str, int | float]:
        """Return counts and nearest-rank latency statistics for recent runs."""
        if not 1 <= hours <= 8760:
            raise ValueError("hours must be between 1 and 8760")
        empty: dict[str, int | float] = {
            "hours": hours,
            "total": 0,
            "running": 0,
            "succeeded": 0,
            "failed": 0,
            "cancelled": 0,
            "degraded": 0,
            "average_duration_ms": 0.0,
            "p95_duration_ms": 0.0,
        }
        try:
            cutoff = datetime.now(UTC) - timedelta(hours=hours)
            with self.session_factory() as session:
                rows = session.execute(
                    select(
                        ChatRunRecord.status,
                        ChatRunRecord.degraded,
                        ChatRunRecord.duration_ms,
                    ).where(ChatRunRecord.created_at >= cutoff)
                ).all()
            for status, degraded, _duration_ms in rows:
                empty["total"] += 1
                if status in _RUN_STATUSES:
                    empty[status] += 1
                if degraded:
                    empty["degraded"] += 1
            durations = sorted(
                float(duration_ms)
                for _status, _degraded, duration_ms in rows
                if duration_ms is not None
            )
            if durations:
                empty["average_duration_ms"] = sum(durations) / len(durations)
                empty["p95_duration_ms"] = durations[ceil(len(durations) * 0.95) - 1]
            return empty
        except Exception as error:  # noqa: BLE001 - monitoring must fail open
            self._warn("summary", error)
            return empty

    def list_runs(
        self,
        limit: int = 50,
        status: str | None = None,
        query: str | None = None,
    ) -> list[dict[str, Any]]:
        """List newest runs, optionally filtering status or an exact record ID."""
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        status = status or None
        query = query or None
        if status is not None and status not in _RUN_STATUSES:
            raise ValueError("unsupported run status")
        try:
            statement = select(ChatRunRecord)
            if status is not None:
                statement = statement.where(ChatRunRecord.status == status)
            if query is not None:
                statement = statement.where(
                    (ChatRunRecord.id == query)
                    | (ChatRunRecord.conversation_id == query)
                    | (ChatRunRecord.message_id == query)
                )
            statement = statement.order_by(
                ChatRunRecord.created_at.desc(), ChatRunRecord.id.desc()
            ).limit(limit)
            with self.session_factory() as session:
                runs = session.scalars(statement).all()
                return [self._run_summary(run) for run in runs]
        except Exception as error:  # noqa: BLE001 - monitoring must fail open
            self._warn("list", error)
            return []

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Return one complete run and its ordered stage timeline."""
        try:
            with self.session_factory() as session:
                run = session.get(ChatRunRecord, run_id)
                if run is None:
                    return None
                events = session.scalars(
                    select(ChatRunEventRecord)
                    .where(ChatRunEventRecord.run_id == run_id)
                    .order_by(
                        ChatRunEventRecord.started_at,
                        ChatRunEventRecord.id,
                    )
                ).all()
                return self._run_detail(run, events)
        except Exception as error:  # noqa: BLE001 - monitoring must fail open
            self._warn("detail", error)
            return None

    @staticmethod
    def _run_summary(run: ChatRunRecord) -> dict[str, Any]:
        return {
            "id": run.id,
            "conversation_id": run.conversation_id,
            "message_id": run.message_id,
            "question": run.question,
            "status": run.status,
            "degraded": run.degraded,
            "created_at": run.created_at,
            "ended_at": run.ended_at,
            "duration_ms": run.duration_ms,
        }

    @staticmethod
    def _event_detail(event: ChatRunEventRecord) -> dict[str, Any]:
        return {
            "id": event.id,
            "run_id": event.run_id,
            "stage": event.stage,
            "status": event.status,
            "started_at": event.started_at,
            "ended_at": event.ended_at,
            "duration_ms": event.duration_ms,
            "details": event.details_json,
        }

    @classmethod
    def _run_detail(
        cls, run: ChatRunRecord, events: list[ChatRunEventRecord]
    ) -> dict[str, Any]:
        detail = cls._run_summary(run)
        detail.update(
            {
                "retrieval_query": run.retrieval_query,
                "prompt_messages": run.prompt_messages_json,
                "retrieved_text": run.retrieved_text,
                "answer": run.answer,
                "generation_id": run.generation_id,
                "error": run.error,
                "events": [cls._event_detail(event) for event in events],
            }
        )
        return detail

    @staticmethod
    def _json_snapshot(detail: dict[str, Any]) -> dict[str, Any]:
        snapshot = {
            key: value.isoformat() if isinstance(value, datetime) else value
            for key, value in detail.items()
            if key != "events"
        }
        snapshot["events"] = [
            {
                key: value.isoformat() if isinstance(value, datetime) else value
                for key, value in event.items()
            }
            for event in detail["events"]
        ]
        return snapshot

    def _append_terminal_snapshot(self, detail: dict[str, Any]) -> None:
        """Append one complete terminal JSON value with a single atomic write."""
        payload = (
            json.dumps(
                self._json_snapshot(detail), ensure_ascii=False, separators=(",", ":")
            )
            + "\n"
        ).encode("utf-8")
        with self._jsonl_lock:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            descriptor = os.open(
                self.trace_path,
                os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                0o600,
            )
            try:
                if os.write(descriptor, payload) != len(payload):
                    raise OSError("Incomplete monitoring snapshot append")
                os.fsync(descriptor)
                os.chmod(self.trace_path, 0o600)
            finally:
                os.close(descriptor)

    def purge(self, before: datetime) -> int:
        """Delete old runs and their JSONL snapshots; repeat calls are safe."""
        try:
            with self.session_factory.begin() as session:
                run_ids = set(
                    session.scalars(
                        select(ChatRunRecord.id).where(
                            ChatRunRecord.created_at < before
                        )
                    ).all()
                )
                if run_ids:
                    session.execute(
                        delete(ChatRunRecord).where(ChatRunRecord.id.in_(run_ids))
                    )
        except Exception as error:  # noqa: BLE001 - monitoring must fail open
            self._warn("purge database", error)
            return 0

        try:
            self._purge_jsonl(before, run_ids)
        except Exception as error:  # noqa: BLE001 - monitoring must fail open
            self._warn("purge JSONL", error)
        return len(run_ids)

    def _purge_jsonl(self, before: datetime, run_ids: set[str]) -> None:
        """Atomically compact snapshots, preserving malformed or unrelated lines."""
        with self._jsonl_lock:
            if not self.trace_path.exists():
                return
            kept_lines: list[str] = []
            for line in self.trace_path.read_text(encoding="utf-8").splitlines(
                keepends=True
            ):
                try:
                    snapshot = json.loads(line)
                    created_at = datetime.fromisoformat(snapshot["created_at"])
                    if created_at.tzinfo is None:
                        created_at = created_at.replace(tzinfo=UTC)
                    if snapshot.get("id") in run_ids or created_at < before:
                        continue
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    # Never discard an unrelated or manually edited diagnostic line.
                    pass
                kept_lines.append(line)

            temporary_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=self.trace_path.parent,
                    prefix="chat-trace-",
                    suffix=".tmp",
                    delete=False,
                ) as temporary:
                    temporary_path = Path(temporary.name)
                    temporary.writelines(kept_lines)
                    temporary.flush()
                    os.fsync(temporary.fileno())
                os.chmod(temporary_path, 0o600)
                os.replace(temporary_path, self.trace_path)
            except Exception:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
                raise
