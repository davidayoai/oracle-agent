#!/usr/bin/env bash
set -euo pipefail

# Oracle solution runner — produces all artifacts that instruction.md requires.
# Validated by checks.py the same way agent output is.
#
# Pattern:
#   solution/src/pipeline.py   — reference implementation
#   solution/run.py            — thin CLI that calls pipeline functions

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Use /app inside Docker, fall back to local workdir for development.
DEFAULT_APP_DIR="/app"
if [[ ! -d "${DEFAULT_APP_DIR}" || ! -w "${DEFAULT_APP_DIR}" ]]; then
  DEFAULT_APP_DIR="${TASK_DIR}/workdir"
fi
APP_DIR="${OTTER_APP_DIR:-${DEFAULT_APP_DIR}}"
PYTHON_BIN="${PYTHON:-python3}"

# Create the directory structure the agent is expected to produce.
mkdir -p "${APP_DIR}/src" \
         "${APP_DIR}/artifacts" \
         "${APP_DIR}/reports" \
         "${APP_DIR}/models"

# Copy oracle source into /app/src/ — mirrors what the agent should create.
cp "${SCRIPT_DIR}/src/"*.py "${APP_DIR}/src/"

if [[ -f "${SCRIPT_DIR}/run.py" ]]; then
  cp "${SCRIPT_DIR}/run.py" "${APP_DIR}/run.py"
fi

# Resolve data path: Docker location first, then local task_inputs/ fallback.
# EDIT: Change the filename to match your task's input data.
DATA_PATH="${OTTER_DATA_PATH:-/app/data/input_data.npz}"
if [[ ! -f "${DATA_PATH}" ]]; then
  LOCAL="${TASK_DIR}/environment/task_inputs/input_data.npz"
  if [[ -f "${LOCAL}" ]]; then
    DATA_PATH="${LOCAL}"
  fi
fi

# Run the oracle pipeline.
OTTER_APP_DIR="${APP_DIR}" \
OTTER_DATA_PATH="${DATA_PATH}" \
PYTHONPATH="${APP_DIR}/src:${APP_DIR}" \
  "${PYTHON_BIN}" "${APP_DIR}/run.py"
