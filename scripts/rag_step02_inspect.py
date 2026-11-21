#!/usr/bin/env python3
"""Read-only inspection for the manually uploaded Enterprise RAG project."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess


PROJECT_DIR = Path(__file__).resolve().parent.parent
REQUIRED_FILES = (
    "pyproject.toml",
    "requirements.lock",
    "configs/default.yaml",
    "app/main.py",
    "frontend/package.json",
    "scripts/install_qdrant_binary.py",
)
REQUIRED_DIRECTORIES = (
    "app",
    "configs",
    "frontend",
    "mock_data",
    "scripts",
    "tests",
    "docs",
)


def run(command: list[str]) -> None:
    subprocess.run(command, check=False)


def main() -> None:
    print("\n--- 当前目录 ---")
    print(PROJECT_DIR)
    if not PROJECT_DIR.is_dir():
        print(f"ERROR: 目录不存在: {PROJECT_DIR}")
        raise SystemExit(1)
    os.chdir(PROJECT_DIR)
    run(["ls", "-la"])

    print("\n--- 项目文件检查 ---")
    for name in REQUIRED_FILES:
        status = "OK      " if (PROJECT_DIR / name).is_file() else "MISSING "
        print(f"{status}{name}")

    print("\n--- 项目文件夹检查 ---")
    for name in REQUIRED_DIRECTORIES:
        status = "OK      " if (PROJECT_DIR / name).is_dir() else "MISSING "
        print(f"{status}{name}/")

    print("\n--- Git 状态 ---")
    if (PROJECT_DIR / ".git").exists():
        run(["git", "rev-parse", "--is-inside-work-tree"])
        run(["git", "rev-parse", "HEAD"])
        run(["git", "status", "--short"])
    else:
        print("未上传 .git 目录；手动上传文件时这是正常情况，跳过 Git 检查。")

    print("\n--- Conda 路径配置 ---")
    conda_executable = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if conda_executable and Path(conda_executable).is_file():
        run([conda_executable, "info", "--base"])
        run([conda_executable, "config", "--show", "envs_dirs"])
    else:
        print("未从环境变量定位 Conda，尝试通过登录 Shell 查询。")
        run(["bash", "-ic", "conda info --base && conda config --show envs_dirs"])

    print("\n检查完成；本脚本未修改任何项目、Conda 或系统文件。")


if __name__ == "__main__":
    main()
