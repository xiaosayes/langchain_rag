#!/usr/bin/env python3
"""Write the Qdrant config as root, then hand it to the ragapp runtime user."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QDRANT_ROOT = PROJECT_ROOT / "var" / "qdrant"
CONFIG_PATH = QDRANT_ROOT / "dev-config.yaml"


def main() -> None:
    if os.name != "posix" or os.geteuid() != 0:
        raise SystemExit("Run this config writer as root on the Ubuntu server")

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
    temporary = CONFIG_PATH.with_suffix(".yaml.new")
    temporary.write_text(serialized, encoding="utf-8")
    os.replace(temporary, CONFIG_PATH)
    shutil.chown(CONFIG_PATH, user="ragapp", group="ragapp")
    CONFIG_PATH.chmod(0o644)

    result = subprocess.run(
        [
            "runuser",
            "-u",
            "ragapp",
            "--",
            str(PROJECT_ROOT / "var" / "qdrant" / "1.15.4" / "qdrant"),
            "--version",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0 or "1.15.4" not in result.stdout + result.stderr:
        raise RuntimeError(result.stdout + result.stderr)

    report = (
        "Qdrant config written: " + str(CONFIG_PATH) + "\n"
        "Config bytes: " + str(CONFIG_PATH.stat().st_size) + "\n"
        "Binary: " + result.stdout.strip() + "\n"
        "Next: start Qdrant as ragapp\n"
    )
    (QDRANT_ROOT / "config-write-report.txt").write_text(report, encoding="utf-8")
    print(report, flush=True)


if __name__ == "__main__":
    main()
