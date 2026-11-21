#!/usr/bin/env python3
"""Write and validate the isolated Qdrant configuration for this project."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QDRANT_ROOT = PROJECT_ROOT / "var" / "qdrant"
CONFIG_PATH = QDRANT_ROOT / "dev-config.yaml"
REPORT_PATH = QDRANT_ROOT / "config-check.txt"


def main() -> None:
    QDRANT_ROOT.mkdir(parents=True, exist_ok=True)
    (QDRANT_ROOT / "storage").mkdir(parents=True, exist_ok=True)
    (QDRANT_ROOT / "snapshots").mkdir(parents=True, exist_ok=True)
    config = {
        "storage": {
            "storage_path": str(QDRANT_ROOT / "storage"),
            "snapshots_path": str(QDRANT_ROOT / "snapshots"),
        },
        "service": {
            "host": "127.0.0.1",
            "http_port": 16333,
            "grpc_port": 16334,
        },
        "telemetry_disabled": True,
    }
    serialized = json.dumps(config, ensure_ascii=False, separators=(",", ":")) + "\n"
    CONFIG_PATH.write_text(serialized, encoding="utf-8")

    loaded = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if loaded != config:
        raise RuntimeError(f"Configuration mismatch: {loaded!r}")

    report = (
        "Qdrant config OK\n"
        f"Python: {sys.executable}\n"
        f"Config: {CONFIG_PATH}\n"
        f"Size: {CONFIG_PATH.stat().st_size} bytes\n"
        f"Content: {serialized}"
    )
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report, flush=True)


if __name__ == "__main__":
    main()
