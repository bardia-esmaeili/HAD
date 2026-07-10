#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'USAGE'
Usage:
  ./run_uncertainty_threshold_sweep.sh SCENE [SPARSE_VIEW] [MAX_STEPS]

Environment:
  CONF_THRESHOLDS   Space-separated threshold values (default: 0.5 0.6 0.7 0.8 0.9 0.95)

Each threshold creates a new wall-clock timestamped run under the same METHOD_NAME/scene.
The threshold is recorded in that run's cfg.yml.

Examples:
  ./run_uncertainty_threshold_sweep.sh <scene_id> 9 20000
  CONF_THRESHOLDS="0.7 0.8 0.9" ./run_uncertainty_threshold_sweep.sh <scene_id> 9 20000
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCENE="${1:?usage: ./run_uncertainty_threshold_sweep.sh SCENE [SPARSE_VIEW] [MAX_STEPS]}"
SPARSE_VIEW="${2:-${SPARSE_VIEW:-9}}"
MAX_STEPS="${3:-${MAX_STEPS:-20000}}"

THRESHOLDS="${CONF_THRESHOLDS:-0.5 0.6 0.7 0.8 0.9 0.95}"

for threshold in ${THRESHOLDS}; do
  echo "=== threshold=${threshold} (output dir: auto timestamp) ==="
  UNCERTAINTY_MASK_THRESHOLD="${threshold}" \
    "${SCRIPT_DIR}/run_train_scene.sh" "${SCENE}" "${SPARSE_VIEW}" "${MAX_STEPS}"
done
