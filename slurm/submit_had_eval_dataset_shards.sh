#!/usr/bin/env bash
set -euo pipefail

WORLD_SIZE="${1:-24}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/xi9/code/DreamAware3D_open_source}"
SCRIPT_DIR="${PROJECT_ROOT}/slurm"

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
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
LVSM_ROOT="${LVSM_ROOT:-${PROJECT_ROOT}/LVSM}"
DEFAULT_LVSM_CKPT_PATH="/home/xi9/code/LVSM/experiments/checkpoints/LVSM_decoder_only_conf_Resi_unet_512"
LVSM_CKPT_PATH="${LVSM_CKPT_PATH:-${LVSM_CKPT_DIR:-${DEFAULT_LVSM_CKPT_PATH}}}"
SPARSE_VIEW="${SPARSE_VIEW:-9}"
VIEW_FUSION="${VIEW_FUSION:-${DEFAULT_VIEW_FUSION}}"
USE_LVSM="${USE_LVSM:-1}"
USE_ORACLE="${USE_ORACLE:-0}"
DATA_FACTOR="${DATA_FACTOR:-4}"
MAX_STEPS="${MAX_STEPS:-${DEFAULT_MAX_STEPS}}"
TARGET_SAMPLE_STEP="${TARGET_SAMPLE_STEP:-${DEFAULT_TARGET_SAMPLE_STEP}}"
MIPNERF_SPLIT_ROOT="${MIPNERF_SPLIT_ROOT:-${DATA_ROOT}}"
SPLIT_JSON="${SPLIT_JSON:-}"
UNCERTAINTY_MASK_THRESHOLD="${UNCERTAINTY_MASK_THRESHOLD:-0.9}"

if [ "${USE_ORACLE}" = "1" ]; then
  USE_LVSM=0
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

mapfile -t SCENES < <(grep -vE "^\s*(#|$)" "${SCENES_FILE}")
mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}"

JOB_IDS=()
for RANK in $(seq 0 "$((WORLD_SIZE - 1))"); do
  JOB_ID="$(sbatch --parsable \
    --chdir="${PROJECT_ROOT}" \
    --output="${LOG_DIR}/%x-%j.out" \
    --error="${LOG_DIR}/%x-%j.err" \
    --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",DATASET="${DATASET}",DATA_ROOT="${DATA_ROOT}",OUTPUT_ROOT="${OUTPUT_ROOT}",SCENES_FILE="${SCENES_FILE}",LVSM_ROOT="${LVSM_ROOT}",LVSM_CKPT_PATH="${LVSM_CKPT_PATH}",SPARSE_VIEW="${SPARSE_VIEW}",VIEW_FUSION="${VIEW_FUSION}",USE_LVSM="${USE_LVSM}",USE_ORACLE="${USE_ORACLE}",DATA_FACTOR="${DATA_FACTOR}",MAX_STEPS="${MAX_STEPS}",TARGET_SAMPLE_STEP="${TARGET_SAMPLE_STEP}",MIPNERF_SPLIT_ROOT="${MIPNERF_SPLIT_ROOT}",SPLIT_JSON="${SPLIT_JSON}",UNCERTAINTY_MASK_THRESHOLD="${UNCERTAINTY_MASK_THRESHOLD}" \
    "${SCRIPT_DIR}/run_had_eval_dataset_shard.sh" "${RANK}" "${WORLD_SIZE}")"
  JOB_IDS+=("${JOB_ID}")
  echo "submitted DreamAware3D ${DATASET} eval rank=${RANK}/${WORLD_SIZE} use_lvsm=${USE_LVSM} use_oracle=${USE_ORACLE} job_id=${JOB_ID}"
done

if (( ${#JOB_IDS[@]} == 0 )); then
  echo "no jobs submitted"
else
  DEPENDENCY="$(IFS=:; echo "${JOB_IDS[*]}")"
  echo "submitted ${#JOB_IDS[@]} single-GPU shard jobs dependency_group=afterany:${DEPENDENCY}"
fi
