#!/usr/bin/env bash
# Launch run_train_scene.sh under debugpy for VS Code/Cursor breakpoints.
#
# Usage (same args as run_train_scene.sh):
#   killarney_gpu_interactive
#   source killarney-dev.sh && had_dev
#   ./run_train_scene_debug.sh <scene_id> [sparse_view] [max_steps]
#
# Then on the login node (second terminal):
#   ssh -N -L 5678:127.0.0.1:5678 <compute-node>
#
# Attach from VS Code: Run and Debug → "Attach to GPU training (Killarney)"
#
# Optional env:
#   HAD_DEBUGPY_PORT=5678   (default)
#   HAD_DEBUGPY_WAIT=1      (default; set 0 to run without waiting for attach)

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export HAD_DEBUGPY=1
export HAD_DEBUGPY_PORT="${HAD_DEBUGPY_PORT:-5678}"
export HAD_DEBUGPY_WAIT="${HAD_DEBUGPY_WAIT:-1}"

exec "${SCRIPT_DIR}/run_train_scene.sh" "$@"
