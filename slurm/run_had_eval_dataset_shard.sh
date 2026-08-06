#!/bin/bash
#SBATCH --job-name=had-dreamaware
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=64gb
#SBATCH --time=12:00:00
#SBATCH --gpus=a100:1
#SBATCH --output=/home/xi9/code/DreamAware3D_open_source/logs/%x-%j.out
#SBATCH --error=/home/xi9/code/DreamAware3D_open_source/logs/%x-%j.err

set -eo pipefail

RANK="${1:?usage: sbatch run_had_eval_dataset_shard.sh RANK WORLD_SIZE}"
WORLD_SIZE="${2:?usage: sbatch run_had_eval_dataset_shard.sh RANK WORLD_SIZE}"

PROJECT_ROOT="${PROJECT_ROOT:-/home/xi9/code/DreamAware3D_open_source}"
LVSM_ROOT="${LVSM_ROOT:-${PROJECT_ROOT}/LVSM}"
DEFAULT_LVSM_CKPT_PATH="/home/xi9/code/LVSM/experiments/checkpoints/LVSM_decoder_only_conf_Resi_unet_512"
LVSM_CKPT_PATH="${LVSM_CKPT_PATH:-${LVSM_CKPT_DIR:-${DEFAULT_LVSM_CKPT_PATH}}}"
DATASET="${DATASET:-dl3dv}"
if [ "${DATASET}" = "mipnerf360" ]; then
  DEFAULT_DATA_ROOT="${MIPNERF_DATA_ROOT:-/project/siyuh/common/xiliu/MipNeRF360}"
  DEFAULT_SCENES_FILE="${PROJECT_ROOT}/configs/mipnerf360_scenes.txt"
  DEFAULT_VIEW_FUSION=1
  DEFAULT_MAX_STEPS=20000
  DEFAULT_TARGET_SAMPLE_STEP=1
else
  DEFAULT_DATA_ROOT="/project/siyuh/common/xiliu/DL3DV-10K-Benchmark"
  DEFAULT_SCENES_FILE="${PROJECT_ROOT}/configs/dl3dv_eval_scenes.txt"
  DEFAULT_VIEW_FUSION=3
  DEFAULT_MAX_STEPS=20000
  DEFAULT_TARGET_SAMPLE_STEP=2
fi
DATA_ROOT="${DATA_ROOT:-${DEFAULT_DATA_ROOT}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/project/siyuh/common/xiliu/HAD_CVPR2026_V2/outputs}"
SCENES_FILE="${SCENES_FILE:-${DEFAULT_SCENES_FILE}}"

SPARSE_VIEW="${SPARSE_VIEW:-9}"
VIEW_FUSION="${VIEW_FUSION:-${DEFAULT_VIEW_FUSION}}"
USE_LVSM="${USE_LVSM:-1}"
USE_ORACLE="${USE_ORACLE:-0}"
USE_DIFIX_DELTA="${USE_DIFIX_DELTA:-0}"
DATA_FACTOR="${DATA_FACTOR:-4}"
MAX_STEPS="${MAX_STEPS:-${DEFAULT_MAX_STEPS}}"
TARGET_SAMPLE_STEP="${TARGET_SAMPLE_STEP:-${DEFAULT_TARGET_SAMPLE_STEP}}"
MIPNERF_SPLIT_ROOT="${MIPNERF_SPLIT_ROOT:-${DATA_ROOT}}"
SPLIT_JSON="${SPLIT_JSON:-}"
UNCERTAINTY_MASK_THRESHOLD="${UNCERTAINTY_MASK_THRESHOLD:-0.9}"
UNCERTAINTY_MASK_MODE="${UNCERTAINTY_MASK_MODE:-binary}"
UNCERTAINTY_MASK_TEMPERATURE="${UNCERTAINTY_MASK_TEMPERATURE:-1.0}"

# Confidence source: mutually exclusive. Gating (MODE/THRESHOLD/TEMPERATURE) is orthogonal.
if [ "${USE_ORACLE}" = "1" ]; then
  USE_LVSM=0
  USE_DIFIX_DELTA=0
elif [ "${USE_DIFIX_DELTA}" = "1" ]; then
  USE_LVSM=0
  USE_ORACLE=0
fi

if [ "${USE_ORACLE}" = "1" ]; then
  if [ "${DATASET}" = "mipnerf360" ]; then
    METHOD_NAME="dreamaware3d_mipnerf360_oracle_view${SPARSE_VIEW}_fusion${VIEW_FUSION}"
  else
    METHOD_NAME="dreamaware3d_oracle_view${SPARSE_VIEW}_fusion${VIEW_FUSION}"
  fi
  CONF_FLAG="--use_conf"
  LVSM_FLAG="--no-use_lvsm"
  PERFECT_CONF_FLAG="--use_pefect_conf"
  DIFIX_DELTA_FLAG="--no-use_difix_delta_conf"
elif [ "${USE_DIFIX_DELTA}" = "1" ]; then
  if [ "${DATASET}" = "mipnerf360" ]; then
    METHOD_NAME="dreamaware3d_mipnerf360_difix_delta_view${SPARSE_VIEW}_fusion${VIEW_FUSION}"
  else
    METHOD_NAME="dreamaware3d_difix_delta_view${SPARSE_VIEW}_fusion${VIEW_FUSION}"
  fi
  CONF_FLAG="--use_conf"
  LVSM_FLAG="--no-use_lvsm"
  PERFECT_CONF_FLAG="--no-use_pefect_conf"
  DIFIX_DELTA_FLAG="--use_difix_delta_conf"
elif [ "${USE_LVSM}" = "1" ]; then
  if [ "${DATASET}" = "mipnerf360" ]; then
    METHOD_NAME="dreamaware3d_mipnerf360_view${SPARSE_VIEW}_fusion${VIEW_FUSION}"
  else
    METHOD_NAME="dreamaware3d_lvsm_view${SPARSE_VIEW}_fusion${VIEW_FUSION}"
  fi
  CONF_FLAG="--use_conf"
  LVSM_FLAG="--use_lvsm"
  PERFECT_CONF_FLAG="--no-use_pefect_conf"
  DIFIX_DELTA_FLAG="--no-use_difix_delta_conf"
else
  if [ "${DATASET}" = "mipnerf360" ]; then
    METHOD_NAME="dreamaware3d_mipnerf360_no_lvsm_view${SPARSE_VIEW}"
  else
    METHOD_NAME="dreamaware3d_no_lvsm_view${SPARSE_VIEW}"
  fi
  CONF_FLAG="--no-use_conf"
  LVSM_FLAG="--no-use_lvsm"
  PERFECT_CONF_FLAG="--no-use_pefect_conf"
  DIFIX_DELTA_FLAG="--no-use_difix_delta_conf"
fi

if [ ! -f "${SCENES_FILE}" ]; then
  echo "Missing scene list: ${SCENES_FILE}" >&2
  exit 2
fi

if [ "${USE_LVSM}" = "1" ]; then
  if [ -d "${LVSM_CKPT_PATH}" ]; then
    if [ -z "$(find "${LVSM_CKPT_PATH}" -maxdepth 1 -name '*.pt' -print -quit)" ]; then
      echo "Missing LVSM checkpoint .pt files in ${LVSM_CKPT_PATH}" >&2
      exit 2
    fi
  elif [ ! -f "${LVSM_CKPT_PATH}" ]; then
    echo "Missing LVSM checkpoint path: ${LVSM_CKPT_PATH}" >&2
    exit 2
  fi
fi

source /etc/profile.d/modules.sh
module add cuda/11.8.0

export PATH="$HOME/miniconda3/bin:$PATH"
source activate difix3D

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_ROOT}/${METHOD_NAME}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/examples/gsplat:${PROJECT_ROOT}/examples/gsplat/pycolmap:$(dirname "${LVSM_ROOT}"):$PYTHONPATH"
export LVSM_ROOT="${LVSM_ROOT}"
export LVSM_CKPT_PATH="${LVSM_CKPT_PATH}"

mapfile -t SCENES < <(grep -vE "^\s*(#|$)" "${SCENES_FILE}")

for IDX in "${!SCENES[@]}"; do
  if (( IDX % WORLD_SIZE != RANK )); then
    continue
  fi

  SCENE_ID="${SCENES[IDX]}"
  if [ "${DATASET}" = "mipnerf360" ]; then
    DATA="${DATA_ROOT}/${SCENE_ID}"
    SCENE_SPLIT_JSON="${SPLIT_JSON:-${MIPNERF_SPLIT_ROOT:+${MIPNERF_SPLIT_ROOT}/${SCENE_ID}/train_test_split_${SPARSE_VIEW}.json}}"
  else
    DATA="${DATA_ROOT}/${SCENE_ID}/nerfstudio"
    SCENE_SPLIT_JSON="${SPLIT_JSON}"
  fi
  SCENE_OUTPUT_ROOT="${OUTPUT_ROOT}/${METHOD_NAME}/${SCENE_ID}"

  if [ ! -d "${DATA}" ]; then
    echo "Skipping missing data directory: ${DATA}" >&2
    continue
  fi

  SPLIT_ARGS=()
  if [ -n "${SCENE_SPLIT_JSON}" ]; then
    if [ ! -f "${SCENE_SPLIT_JSON}" ]; then
      echo "Skipping scene with missing split json: ${SCENE_SPLIT_JSON}" >&2
      continue
    fi
    SPLIT_ARGS=(--split_json "${SCENE_SPLIT_JSON}")
  fi

  mkdir -p "${SCENE_OUTPUT_ROOT}"
  echo "Processing scene: ${SCENE_ID} dataset=${DATASET} rank=${RANK}/${WORLD_SIZE} use_lvsm=${USE_LVSM} use_oracle=${USE_ORACLE} use_difix_delta=${USE_DIFIX_DELTA}"
  echo "Data dir: ${DATA}"
  echo "Output root: ${SCENE_OUTPUT_ROOT}"
  if [ -n "${SCENE_SPLIT_JSON}" ]; then
    echo "Split json: ${SCENE_SPLIT_JSON}"
  fi
  if [ "${USE_LVSM}" = "1" ]; then
    echo "LVSM checkpoint path: ${LVSM_CKPT_PATH}"
  fi

  if python "${PROJECT_ROOT}/examples/gsplat/train_dreamaware3d.py" mcmc \
      --data_dir "${DATA}" \
      --data_factor "${DATA_FACTOR}" \
      --result_dir "${SCENE_OUTPUT_ROOT}" \
      --no-use_eval \
      ${PERFECT_CONF_FLAG} \
      ${CONF_FLAG} \
      --no-partial_setting \
      ${LVSM_FLAG} \
      ${DIFIX_DELTA_FLAG} \
      --no-lvsm_mode \
      --no-normalize-world-space \
      --num_sparse_view "${SPARSE_VIEW}" \
      --target_sample_step "${TARGET_SAMPLE_STEP}" \
      --max_steps "${MAX_STEPS}" \
      --view_fusion "${VIEW_FUSION}" \
      --uncertainty_mask_mode "${UNCERTAINTY_MASK_MODE}" \
      --uncertainty_mask_threshold "${UNCERTAINTY_MASK_THRESHOLD}" \
      --uncertainty_mask_temperature "${UNCERTAINTY_MASK_TEMPERATURE}" \
      "${SPLIT_ARGS[@]}"; then
    echo "Completed scene: ${SCENE_ID}"
  else
    STATUS="$?"
    echo "Failed scene: ${SCENE_ID} status=${STATUS}" >&2
    echo "Continuing with remaining scenes assigned to rank=${RANK}/${WORLD_SIZE}" >&2
  fi
done
