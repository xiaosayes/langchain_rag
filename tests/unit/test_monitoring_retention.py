import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.exc import DatabaseError
from typer.testing import CliRunner

from app.cli import cli_app
from app.config import (
    AppSettings,
    ConfigurationError,
    MonitoringSettings,
    load_settings,
)
from app.db.session import create_session_factory
from app.monitoring.service import MonitoringService
from app.web.records import (
    ChatRunEventRecord,
    ChatRunRecord,
    ConversationRecord,
    FeedbackRecord,
    MessageRecord,
)

runner = CliRunner()


def test_monitoring_purge_removes_only_old_traces_and_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    state_root = tmp_path / "state"
    state_root.mkdir()
    monkeypatch.setenv("RAG_STATE_DIR", str(state_root))
    service = MonitoringService(
        create_session_factory(state_root / "index.sqlite3"), state_root
    )
    factory = service.session_factory
    cutoff = datetime.now(UTC) - timedelta(days=30)
    old_id = service.start_run("old question", "conversation-1")
    new_id = service.start_run("new question", "conversation-1")
    with factory.begin() as session:
        old_run = session.get(ChatRunRecord, old_id)
        old_run.created_at = cutoff - timedelta(minutes=1)
        for run_id, created_at in (
            (old_id, old_run.created_at),
            (new_id, cutoff + timedelta(minutes=1)),
        ):
            session.add(
                ChatRunEventRecord(
                    run_id=run_id,
                    stage="terminal",
                    status="succeeded",
                    started_at=created_at,
                    ended_at=created_at,
                    duration_ms=1,
                    details_json={},
                )
            )
        session.add(ConversationRecord(id="conversation-1", title="Keep me"))
        session.add(
            MessageRecord(
                id="message-1",
                conversation_id="conversation-1",
                role="assistant",
                content="Keep this answer",
            )
        )
        session.add(
            FeedbackRecord(
                id="feedback-1",
                message_id="message-1",
                score=-1,
                comment="Keep this feedback",
            )
        )
    service.finish_run(old_id, "succeeded", "old answer")
    service.finish_run(new_id, "succeeded", "new answer")

    result = runner.invoke(cli_app, ["monitoring-purge", "--days", "30"])

    assert result.exit_code == 0, result.output
    assert "Deleted monitoring runs: 1" in result.output
    assert "Cutoff:" in result.output
    assert service.get_run(old_id) is None
    assert service.get_run(new_id) is not None
    with factory() as session:
        assert session.scalars(select(ChatRunEventRecord)).one().run_id == new_id
        assert session.get(ConversationRecord, "conversation-1") is not None
        assert session.get(MessageRecord, "message-1") is not None
        assert session.get(FeedbackRecord, "feedback-1") is not None
    remaining_snapshots = [
        json.loads(line)
        for line in service.trace_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [snapshot["id"] for snapshot in remaining_snapshots] == [new_id]

    second_result = runner.invoke(cli_app, ["monitoring-purge", "--days", "30"])

    assert second_result.exit_code == 0, second_result.output
    assert "Deleted monitoring runs: 0" in second_result.output
    assert service.get_run(new_id) is not None
    assert [
        json.loads(line)
        for line in service.trace_path.read_text(encoding="utf-8").splitlines()
    ] == remaining_snapshots
    with factory() as session:
        assert session.get(ConversationRecord, "conversation-1") is not None
        assert session.get(MessageRecord, "message-1") is not None
        assert session.get(FeedbackRecord, "feedback-1") is not None


def test_monitoring_retention_is_typed_and_loaded_from_yaml(tmp_path: Path):
    config = tmp_path / "config.yaml"
    config.write_text("monitoring:\n  retention_days: 45\n", encoding="utf-8")

    settings = load_settings(config)

    assert settings.monitoring == MonitoringSettings(retention_days=45)
    assert AppSettings().monitoring.retention_days == 30
    config.write_text("monitoring:\n  retention_days: 0\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="retention_days"):
        load_settings(config)


@pytest.mark.parametrize("days", ["0", "-1", "not-an-integer"])
def test_monitoring_purge_rejects_invalid_days(days: str):
    result = runner.invoke(cli_app, ["monitoring-purge", "--days", days])

    assert result.exit_code == 2


def test_monitoring_purge_database_failure_is_nonzero_and_sanitized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("RAG_STATE_DIR", str(tmp_path / "state"))

    class BrokenFactory:
        def __call__(self):
            raise self.error()

        def begin(self):
            raise self.error()

        @staticmethod
        def error():
            return DatabaseError(
                "DELETE FROM private_table WHERE token = 'CLI_DB_SECRET'",
                None,
                Exception(r"C:\private\CLI_DB_PATH\index.sqlite3"),
            )

    monkeypatch.setattr(
        "app.cli.create_session_factory", lambda _path: BrokenFactory()
    )

    result = runner.invoke(cli_app, ["monitoring-purge", "--days", "30"])

    assert result.exit_code == 1
    assert "Database operation failed" in result.output
    assert "Deleted monitoring runs" not in result.output
    for sentinel in (
        "DELETE FROM",
        "private_table",
        "CLI_DB_SECRET",
        "CLI_DB_PATH",
        "index.sqlite3",
    ):
        assert sentinel not in result.output


def test_monitoring_purge_jsonl_failure_is_nonzero_and_sanitized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("RAG_STATE_DIR", str(tmp_path / "state"))

    def fail_compaction(_self, _run_ids):
        raise OSError(
            r"C:\private\CLI_JSONL_PATH\chat-trace.jsonl password=CLI_JSONL_SECRET"
        )

    monkeypatch.setattr(MonitoringService, "_purge_jsonl", fail_compaction)

    result = runner.invoke(cli_app, ["monitoring-purge", "--days", "30"])

    assert result.exit_code == 1
    assert "Local file or network operation failed" in result.output
    assert "Deleted monitoring runs" not in result.output
    for sentinel in ("CLI_JSONL_PATH", "CLI_JSONL_SECRET", "password=", "chat-trace"):
        assert sentinel not in result.output
