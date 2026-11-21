#!/usr/bin/env bash
# Create only the dedicated Enterprise RAG Conda environment.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_EXECUTABLE="${CONDA_EXE:-/root/miniconda3/bin/conda}"
ENV_PREFIX="/data/conda_envs/langchain_test"

printf 'Project root: %s\n' "$PROJECT_ROOT"
printf 'Conda executable: %s\n' "$CONDA_EXECUTABLE"
printf 'Dedicated environment: %s\n' "$ENV_PREFIX"

if [[ ! -x "$CONDA_EXECUTABLE" ]]; then
  printf 'ERROR: Conda executable not found: %s\n' "$CONDA_EXECUTABLE" >&2
  exit 1
fi

if [[ -x "$ENV_PREFIX/bin/python" ]]; then
  printf 'Environment already exists; skipping creation.\n'
else
  if [[ -e "$ENV_PREFIX" ]]; then
    printf 'ERROR: Path exists but is not a usable environment: %s\n' "$ENV_PREFIX" >&2
    printf 'No files were removed. Inspect the path before retrying.\n' >&2
    exit 1
  fi

  "$CONDA_EXECUTABLE" create \
    --prefix "$ENV_PREFIX" \
    --override-channels \
    --channel conda-forge \
    python=3.11 \
    nodejs=22 \
    pip \
    --yes
fi

printf '\n--- Dedicated environment versions ---\n'
"$ENV_PREFIX/bin/python" --version
"$ENV_PREFIX/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 11), sys.version; print(sys.executable)'
"$ENV_PREFIX/bin/node" --version
"$ENV_PREFIX/bin/npm" --version
"$ENV_PREFIX/bin/python" -m pip --version

printf '\nEnvironment creation/verification completed.\n'
printf 'Global Node.js and all other Conda environments were left unchanged.\n'
