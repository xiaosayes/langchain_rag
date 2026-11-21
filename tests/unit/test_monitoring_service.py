import json
import logging
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from sqlalchemy import select

from app.db.session import create_session_factory
from app.monitoring.service import MonitoringService
from app.web.records import ChatRunEventRecord, ChatRunRecord


def make_service(tmp_path: Path) -> MonitoringService:
    return MonitoringService(
        create_session_factory(tmp_path / "index.sqlite3"), tmp_path / "state"
    )


def make_preparation() -> SimpleNamespace:
    return SimpleNamespace(
        retrieval_query="rewritten question",
        documents=[
            Document(page_content="first retrieved text"),
            Document(page_content="second retrieved text"),
        ],
        degraded=True,
        generation_id="generation-1",
    )


def add_run(
    tmp_path: Path,
    run_id: str,
    *,
    status: str = "succeeded",
    created_at: datetime | None = None,
    duration_ms: float | None = 100,
    conversation_id: str | None = None,
    message_id: str | None = None,
    degraded: bool = False,
) -> None:
    factory = create_session_factory(tmp_path / "index.sqlite3")
    with factory.begin() as session:
        session.add(
            ChatRunRecord(
                id=run_id,
                conversation_id=conversation_id,
                message_id=message_id,
                question=f"question for {run_id}",
                status=status,
                created_at=created_at or datetime.now(UTC),
                ended_at=created_at or datetime.now(UTC),
                duration_ms=duration_ms,
                degraded=degraded,
            )
        )


def test_terminal_run_has_preparation_ordered_events_and_one_jsonl_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    ticks = iter([1000.0, 1000.25])
    monkeypatch.setattr("app.monitoring.service.perf_counter", lambda: next(ticks))
    service = make_service(tmp_path)
    run_id = service.start_run("original question", "conversation-1")
    started_at = datetime(2026, 11, 20, 10, 0, tzinfo=UTC)

    service.record_event(
        run_id,
        "dense",
        "succeeded",
        started_at,
        12.5,
        {"candidates": 3},
    )
    service.record_event(
        run_id, "rerank", "succeeded", started_at + timedelta(seconds=1), 20
    )
    service.update_preparation(
        run_id,
        make_preparation(),
        [SystemMessage(content="system prompt"), HumanMessage(content="question")],
    )
    service.finish_run(run_id, "succeeded", "final answer")

    detail = service.get_run(run_id)
    assert detail is not None
    assert detail["question"] == "original question"
    assert detail["retrieval_query"] == "rewritten question"
    assert detail["retrieved_text"] == ("first retrieved text\n\nsecond retrieved text")
    assert detail["prompt_messages"] == [
        {"type": "system", "content": "system prompt"},
        {"type": "human", "content": "question"},
    ]
    assert detail["generation_id"] == "generation-1"
    assert detail["degraded"] is True
    assert detail["answer"] == "final answer"
    assert detail["status"] == "succeeded"
    assert detail["duration_ms"] == pytest.approx(250)
    assert detail["created_at"].tzinfo is not None
    assert detail["ended_at"].tzinfo is not None
    assert [event["stage"] for event in detail["events"]] == ["dense", "rerank"]
    assert detail["events"][0]["details"] == {"candidates": 3}
    assert detail["events"][0]["ended_at"] == started_at + timedelta(milliseconds=12.5)

    trace_path = tmp_path / "state" / "monitoring" / "chat-trace.jsonl"
    snapshots = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert len(snapshots) == 1
    assert snapshots[0]["id"] == run_id
    assert snapshots[0]["status"] == "succeeded"
    assert [event["stage"] for event in snapshots[0]["events"]] == [
        "dense",
        "rerank",
    ]
    if os.name == "posix":
        assert trace_path.stat().st_mode & 0o777 == 0o600

    service.finish_run(run_id, "failed", "replacement", "do not replace")
    assert service.get_run(run_id)["answer"] == "final answer"
    assert len(trace_path.read_text().splitlines()) == 1


def test_summary_counts_statuses_and_computes_average_and_nearest_rank_p95(
    tmp_path: Path,
):
    service = make_service(tmp_path)
    now = datetime.now(UTC)
    for index, duration in enumerate([100, 200, 300, 400]):
        add_run(
            tmp_path,
            f"run-{index}",
            created_at=now - timedelta(minutes=index),
            duration_ms=duration,
            degraded=index == 0,
        )
    add_run(tmp_path, "failed", status="failed", duration_ms=None)
    add_run(tmp_path, "cancelled", status="cancelled", duration_ms=None)
    add_run(tmp_path, "running", status="running", duration_ms=None)
    add_run(
        tmp_path,
        "old",
        created_at=now - timedelta(hours=25),
        duration_ms=10_000,
    )

    summary = service.summary(24)
    assert summary == {
        "hours": 24,
        "total": 7,
        "running": 1,
        "succeeded": 4,
        "failed": 1,
        "cancelled": 1,
        "degraded": 1,
        "average_duration_ms": 250,
        "p95_duration_ms": 400,
    }


def test_list_filters_by_status_and_ids_and_get_run_contains_details(
    tmp_path: Path,
):
    service = make_service(tmp_path)
    base = datetime.now(UTC)
    add_run(
        tmp_path,
        "request-1",
        created_at=base,
        conversation_id="conversation-1",
        message_id="message-1",
    )
    add_run(
        tmp_path,
        "request-2",
        status="failed",
        created_at=base + timedelta(seconds=1),
        conversation_id="conversation-2",
        message_id="message-2",
    )
    factory = create_session_factory(tmp_path / "index.sqlite3")
    with factory.begin() as session:
        session.add(
            ChatRunEventRecord(
                run_id="request-1",
                stage="terminal",
                status="succeeded",
                started_at=base,
                ended_at=base,
                duration_ms=0,
                details_json={"local": True},
            )
        )

    assert [run["id"] for run in service.list_runs(50, None, None)] == [
        "request-2",
        "request-1",
    ]
    assert [run["id"] for run in service.list_runs(50, "failed", None)] == ["request-2"]
    for query in ("request-1", "conversation-1", "message-1"):
        assert [run["id"] for run in service.list_runs(50, None, query)] == [
            "request-1"
        ]
    assert service.list_runs(50, None, "not-found") == []
    assert len(service.list_runs(1, None, None)) == 1
    assert service.get_run("request-1")["events"][0]["details"] == {"local": True}
    assert service.get_run("missing") is None
    for invalid_limit in (0, -1, 101):
        with pytest.raises(ValueError, match="limit"):
            service.list_runs(invalid_limit, None, None)


def test_sql_failures_are_fail_open_and_do_not_log_secrets_or_tracebacks(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    class BrokenFactory:
        def __call__(self):
            raise RuntimeError("password=do-not-log")

        def begin(self):
            raise RuntimeError("password=do-not-log")

    service = MonitoringService(BrokenFactory(), tmp_path / "state")
    caplog.set_level(logging.WARNING)
    run_id = service.start_run("question", None)
    assert isinstance(run_id, str)
    service.record_event(run_id, "dense", "succeeded", datetime.now(UTC), 1)
    service.update_preparation(run_id, make_preparation(), [])
    service.finish_run(run_id, "failed", "partial", "sanitized error")

    assert service.summary(24)["total"] == 0
    assert service.list_runs(50, None, None) == []
    assert service.get_run(run_id) is None
    assert service.purge(datetime.now(UTC)) == 0
    assert "do-not-log" not in caplog.text
    assert "Traceback" not in caplog.text


def test_jsonl_failure_does_not_undo_terminal_database_update(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    state_root = tmp_path / "not-a-directory"
    state_root.write_text("file")
    service = MonitoringService(
        create_session_factory(tmp_path / "index.sqlite3"), state_root
    )
    run_id = service.start_run("question", "conversation-1")
    caplog.set_level(logging.WARNING)

    service.finish_run(run_id, "succeeded", "answer")

    assert service.get_run(run_id)["status"] == "succeeded"
    assert "Traceback" not in caplog.text
    assert "not-a-directory" not in caplog.text


def test_purge_deletes_old_db_events_and_jsonl_and_is_idempotent(tmp_path: Path):
    service = make_service(tmp_path)
    cutoff = datetime.now(UTC)
    old_id = service.start_run("old", "conversation-old")
    new_id = service.start_run("new", "conversation-new")
    factory = create_session_factory(tmp_path / "index.sqlite3")
    with factory.begin() as session:
        old = session.get(ChatRunRecord, old_id)
        old.created_at = cutoff - timedelta(days=31)
        old_event = ChatRunEventRecord(
            run_id=old_id,
            stage="prepare",
            status="succeeded",
            started_at=old.created_at,
            ended_at=old.created_at,
            duration_ms=0,
            details_json={},
        )
        session.add(old_event)
    service.finish_run(old_id, "succeeded", "old answer")
    service.finish_run(new_id, "succeeded", "new answer")

    assert service.purge(cutoff) == 1
    assert service.get_run(old_id) is None
    assert service.get_run(new_id) is not None
    with factory() as session:
        assert (
            session.scalars(
                select(ChatRunEventRecord).where(ChatRunEventRecord.run_id == old_id)
            ).all()
            == []
        )
    trace_path = tmp_path / "state" / "monitoring" / "chat-trace.jsonl"
    remaining = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert [snapshot["id"] for snapshot in remaining] == [new_id]

    assert service.purge(cutoff) == 0
    assert [
        json.loads(line) for line in trace_path.read_text().splitlines()
    ] == remaining
