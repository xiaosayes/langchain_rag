#!/usr/bin/env python3
"""Read and verify the Qdrant config without modifying any files."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "var" / "qdrant" / "dev-config.yaml"


def main() -> None:
    raw = CONFIG_PATH.read_bytes()
    loaded = yaml.safe_load(raw.decode("utf-8"))
    assert loaded["storage"]["storage_path"] == str(
        PROJECT_ROOT / "var" / "qdrant" / "storage"
    )
    assert loaded["storage"]["snapshots_path"] == str(
        PROJECT_ROOT / "var" / "qdrant" / "snapshots"
    )
    assert loaded["service"] == {
        "host": "127.0.0.1",
        "http_port": 16333,
        "grpc_port": 16334,
    }
    assert loaded["telemetry_disabled"] is True
    result = {
        "status": "OK",
        "python": sys.executable,
        "uid": os.getuid(),
        "config": str(CONFIG_PATH),
        "bytes": len(raw),
        "content_sha256": __import__("hashlib").sha256(raw).hexdigest(),
        "service": loaded["service"],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
