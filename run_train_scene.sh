#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Killarney: modules, venv, and path defaults (see HAD/killarney-dev.sh, HAD/KILLARNEY.md).
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/killarney-dev.sh"

DATASET="${DATASET:-dl3dv}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRIPT_DIR}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/${USER}/had/outputs}"
LVSM_ROOT="${LVSM_ROOT:-${PROJECT_ROOT}/LVSM}"
LVSM_CKPT_PATH="${LVSM_CKPT_PATH:-${PROJECT_ROOT}/checkpoints/LVSM_decoder_only_conf_Resi_unet_512}"

SCENE="${1:?usage: ./run_train_scene.sh SCENE_ID_OR_DATA_DIR [SPARSE_VIEW] [MAX_STEPS]}"
SPARSE_VIEW="${2:-${SPARSE_VIEW:-9}}"
USE_LVSM="${USE_LVSM:-1}"
USE_ORACLE="${USE_ORACLE:-0}"
SPLIT_JSON="${SPLIT_JSON:-}"
UNCERTAINTY_MASK_THRESHOLD="${UNCERTAINTY_MASK_THRESHOLD:-0.9}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-}"

if [ "${USE_ORACLE}" = "1" ]; then
  USE_LVSM=0
fi

if [ "${DATASET}" = "mipnerf360" ]; then
  DATA_ROOT="${DATA_ROOT:-${MIPNERF_DATA_ROOT:-${HAD_MIPNERF_DATA_ROOT:-}}}"
  MAX_STEPS="${3:-${MAX_STEPS:-20000}}"
  VIEW_FUSION="${VIEW_FUSION:-1}"
  TARGET_SAMPLE_STEP="${TARGET_SAMPLE_STEP:-1}"
else
  DATA_ROOT="${DATA_ROOT:-${HAD_DL3DV_DATA_ROOT:-}}"
  MAX_STEPS="${3:-${MAX_STEPS:-20000}}"
  VIEW_FUSION="${VIEW_FUSION:-3}"
  TARGET_SAMPLE_STEP="${TARGET_SAMPLE_STEP:-2}"
fi
DATA_FACTOR="${DATA_FACTOR:-4}"

if [[ "${SCENE}" == */nerfstudio ]]; then
  DATA_DIR="${SCENE}"
  SCENE_ID="$(basename "$(dirname "${SCENE}")")"
elif [ -d "${SCENE}" ]; then
  DATA_DIR="${SCENE}"
  SCENE_ID="$(basename "${SCENE}")"
elif [ "${DATASET}" = "mipnerf360" ]; then
  SCENE_ID="${SCENE}"
  DATA_DIR="${DATA_ROOT}/${SCENE_ID}"
else
  SCENE_ID="${SCENE}"
  DATA_DIR="${DATA_ROOT}/${SCENE_ID}/nerfstudio"
fi

if [ "${USE_ORACLE}" = "1" ]; then
  CONF_FLAG="--use_conf"
  LVSM_FLAG="--no-use_lvsm"
  PERFECT_CONF_FLAG="--use_pefect_conf"
elif [ "${USE_LVSM}" = "1" ]; then
  CONF_FLAG="--use_conf"
  LVSM_FLAG="--use_lvsm"
  PERFECT_CONF_FLAG="--no-use_pefect_conf"
else
  CONF_FLAG="--no-use_conf"
  LVSM_FLAG="--no-use_lvsm"
  PERFECT_CONF_FLAG="--no-use_pefect_conf"
fi

if [ "${USE_ORACLE}" = "1" ]; then
  if [ "${DATASET}" = "mipnerf360" ]; then
    METHOD_NAME="dreamaware3d_mipnerf360_oracle_view${SPARSE_VIEW}_fusion${VIEW_FUSION}"
  else
    METHOD_NAME="dreamaware3d_oracle_view${SPARSE_VIEW}_fusion${VIEW_FUSION}"
  fi
elif [ "${DATASET}" = "mipnerf360" ] && [ "${USE_LVSM}" = "1" ]; then
  METHOD_NAME="dreamaware3d_mipnerf360_view${SPARSE_VIEW}_fusion${VIEW_FUSION}"
elif [ "${DATASET}" = "mipnerf360" ]; then
  METHOD_NAME="dreamaware3d_mipnerf360_no_lvsm_view${SPARSE_VIEW}"
elif [ "${USE_LVSM}" = "1" ]; then
  METHOD_NAME="dreamaware3d_lvsm_view${SPARSE_VIEW}_fusion${VIEW_FUSION}"
else
  METHOD_NAME="dreamaware3d_no_lvsm_view${SPARSE_VIEW}"
fi

SCENE_OUTPUT_ROOT="${OUTPUT_ROOT}/${METHOD_NAME}/${SCENE_ID}"

if [ -z "${DATA_ROOT}" ]; then
  echo "Missing DATA_ROOT (set DATA_ROOT or HAD_DL3DV_DATA_ROOT / HAD_MIPNERF_DATA_ROOT)" >&2
  exit 2
fi

if [ ! -d "${DATA_DIR}" ]; then
  echo "Missing data dir: ${DATA_DIR}" >&2
  exit 2
fi

if [ "${DATASET}" = "mipnerf360" ] && [ -z "${SPLIT_JSON}" ]; then
  MIPNERF_SPLIT_ROOT="${MIPNERF_SPLIT_ROOT:-${DATA_ROOT}}"
  SPLIT_JSON="${MIPNERF_SPLIT_ROOT}/${SCENE_ID}/train_test_split_${SPARSE_VIEW}.json"
fi

SPLIT_ARGS=()
if [ -n "${SPLIT_JSON}" ]; then
  if [ ! -f "${SPLIT_JSON}" ]; then
    echo "Missing split json: ${SPLIT_JSON}" >&2
    echo "Set SPLIT_JSON or MIPNERF_SPLIT_ROOT for MipNeRF360 splits." >&2
    exit 2
  fi
  SPLIT_ARGS=(--split_json "${SPLIT_JSON}")
fi

if [ "${USE_LVSM}" = "1" ] && [ ! -e "${LVSM_CKPT_PATH}" ]; then
  echo "Missing LVSM checkpoint path: ${LVSM_CKPT_PATH}" >&2
  exit 2
fi

cd "${PROJECT_ROOT}"
mkdir -p "${SCENE_OUTPUT_ROOT}"
had_export_pythonpath

echo "Scene: ${SCENE_ID}"
echo "Data: ${DATA_DIR}"
echo "Output root: ${SCENE_OUTPUT_ROOT}"
echo "DATASET=${DATASET} USE_LVSM=${USE_LVSM} USE_ORACLE=${USE_ORACLE} SPARSE_VIEW=${SPARSE_VIEW} VIEW_FUSION=${VIEW_FUSION} MAX_STEPS=${MAX_STEPS} UNCERTAINTY_MASK_THRESHOLD=${UNCERTAINTY_MASK_THRESHOLD}"
if [ -n "${RUN_TIMESTAMP}" ]; then
  echo "RUN_TIMESTAMP=${RUN_TIMESTAMP}"
fi
if [ -n "${SPLIT_JSON}" ]; then
  echo "Split json: ${SPLIT_JSON}"
fi

TRAIN_SCRIPT="${PROJECT_ROOT}/examples/gsplat/train_dreamaware3d.py"
TRAIN_ARGS=(
  mcmc
  --data_dir "${DATA_DIR}"
  --data_factor "${DATA_FACTOR}"
  --result_dir "${SCENE_OUTPUT_ROOT}"
  --no-use_eval
  ${PERFECT_CONF_FLAG}
  ${CONF_FLAG}
  --no-partial_setting
  ${LVSM_FLAG}
  --no-lvsm_mode
  --no-normalize-world-space
  --num_sparse_view "${SPARSE_VIEW}"
  --target_sample_step "${TARGET_SAMPLE_STEP}"
  --max_steps "${MAX_STEPS}"
  --view_fusion "${VIEW_FUSION}"
  --uncertainty_mask_threshold "${UNCERTAINTY_MASK_THRESHOLD}"
  "${SPLIT_ARGS[@]}"
)

if [ -n "${RUN_TIMESTAMP}" ]; then
  TRAIN_ARGS+=(--run_timestamp "${RUN_TIMESTAMP}")
fi

if [ "${HAD_DEBUGPY:-0}" = "1" ]; then
  DEBUGPY_PORT="${HAD_DEBUGPY_PORT:-5678}"
  DEBUGPY_WAIT="${HAD_DEBUGPY_WAIT:-1}"
  if ! python -c "import debugpy" 2>/dev/null; then
    echo "Missing debugpy — run: pip install debugpy" >&2
    exit 2
  fi
  DEBUGPY_LISTEN_ARGS=(--listen "127.0.0.1:${DEBUGPY_PORT}")
  if [ "${DEBUGPY_WAIT}" = "1" ]; then
    DEBUGPY_LISTEN_ARGS+=(--wait-for-client)
  fi
  echo "debugpy: listen 127.0.0.1:${DEBUGPY_PORT} wait=${DEBUGPY_WAIT}"
  echo "debugpy: VS Code → Run and Debug → Attach to GPU training (Killarney)"
  echo "debugpy: on login node, tunnel: ssh -N -L ${DEBUGPY_PORT}:127.0.0.1:${DEBUGPY_PORT} $(hostname)"
  python -m debugpy "${DEBUGPY_LISTEN_ARGS[@]}" "${TRAIN_SCRIPT}" "${TRAIN_ARGS[@]}"
else
  python "${TRAIN_SCRIPT}" "${TRAIN_ARGS[@]}"
fi
