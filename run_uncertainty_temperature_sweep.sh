#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'USAGE'
Usage:
  ./run_uncertainty_temperature_sweep.sh SCENE [SPARSE_VIEW] [MAX_STEPS]

Environment:
  CONF_TEMPERATURES              Space-separated temperatures (default: 0.05 0.1 0.2)
  UNCERTAINTY_MASK_THRESHOLD     Soft-threshold center (default: 0.9)

Forces UNCERTAINTY_MASK_MODE=sigmoid. Each temperature creates a new wall-clock
timestamped run under the same METHOD_NAME/scene. Mode, threshold, and temperature
are recorded in that run's cfg.yml.

Confidence source is orthogonal — combine with USE_LVSM=1 (default), USE_ORACLE=1,
or USE_DIFIX_DELTA=1.

Examples:
  ./run_uncertainty_temperature_sweep.sh <scene_id> 9 20000
  CONF_TEMPERATURES="0.05 0.1 0.2 0.5" ./run_uncertainty_temperature_sweep.sh <scene_id> 9 20000
  USE_DIFIX_DELTA=1 ./run_uncertainty_temperature_sweep.sh <scene_id> 9 20000
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCENE="${1:?usage: ./run_uncertainty_temperature_sweep.sh SCENE [SPARSE_VIEW] [MAX_STEPS]}"
SPARSE_VIEW="${2:-${SPARSE_VIEW:-9}}"
MAX_STEPS="${3:-${MAX_STEPS:-20000}}"

TEMPERATURES="${CONF_TEMPERATURES:-0.05 0.1 0.2}"

for temperature in ${TEMPERATURES}; do
  echo "=== mode=sigmoid temperature=${temperature} threshold=${UNCERTAINTY_MASK_THRESHOLD:-0.9} (output dir: auto timestamp) ==="
  UNCERTAINTY_MASK_MODE=sigmoid \
  UNCERTAINTY_MASK_TEMPERATURE="${temperature}" \
    "${SCRIPT_DIR}/run_train_scene.sh" "${SCENE}" "${SPARSE_VIEW}" "${MAX_STEPS}"
done
