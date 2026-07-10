# HAD on Killarney (Alliance) — single entry point for modules, venv, and paths.
#
# Interactive GPU (login node → compute shell):
#   killarney_gpu_interactive
#   source killarney-dev.sh && ./run_train_scene.sh <scene> 9 20000
#
# Batch GPU (login node → sbatch job, no shell):
#   killarney_gpu_sbatch <scene_id> [sparse_view] [max_steps] [time_limit]
#   Wall time defaults to HAD_SBATCH_TIME below (edit there, not a stale shell export).
#
# Threshold sweep (one sbatch job per value):
#   CONF_THRESHOLDS="0.0 0.2 0.4 0.6 0.8 1.0" killarney_gpu_sbatch <scene_id> 9 20000 5:00:00
#   Or: killarney_gpu_sbatch_sweep <scene_id> 9 20000 5:00:00  (uses default CONF_THRESHOLDS)
# Queue-friendly sweep (one job in Slurm at a time; shell must stay open):
#   CONF_THRESHOLDS="0.0 0.2 0.4 0.6 0.8 1.0" CONF_SWEEP_SUBMIT_MODE=sequential killarney_gpu_sbatch <scene_id> 9 20000 5:00:00
#
# Single run with a custom threshold (output dir still uses auto timestamp):
#   UNCERTAINTY_MASK_THRESHOLD=0.7 killarney_gpu_sbatch <scene_id> 9 20000
#
# Override paths before sourcing, or edit the defaults below:
#   export DATA_ROOT=/project/.../DL3DV-10K-Benchmark
#   export OUTPUT_ROOT=/scratch/$USER/had/outputs
#
# Paths default to the directory containing this file (resolved with pwd -P), not $HOME
# or your shell's current working directory. Run sbatch from /project/... (see KILLARNEY.md).

# shellcheck shell=bash

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "killarney-dev.sh: source this file instead of executing it:" >&2
  echo "  source $(cd "$(dirname "$0")" && pwd)/killarney-dev.sh" >&2
  exit 1
fi

HAD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

# Resolve a path to its physical location (no symlinks).
had_realpath() {
  local p="$1"
  if [[ -d "${p}" ]]; then
    (cd "${p}" && pwd -P)
  elif [[ -f "${p}" ]]; then
    local dir base
    dir="$(cd "$(dirname "${p}")" && pwd -P)"
    base="$(basename "${p}")"
    echo "${dir}/${base}"
  else
    echo "${p}"
  fi
}

# Slurm on Alliance rejects submissions that reference /home.
had_assert_not_home() {
  local label="$1"
  local path="$2"
  if [[ "${path}" == /home/* ]]; then
    echo "killarney-dev.sh: ${label} is under /home (${path})" >&2
    echo "killarney-dev.sh: use a checkout under /project or /scratch (source ./killarney-dev.sh from there)." >&2
    return 1
  fi
}

# ---------------------------------------------------------------------------
# Alliance account + interactive GPU (override AIP_ACCOUNT if auto-detect fails).
# Tune via KILLARNEY_GRES, KILLARNEY_MEM, KILLARNEY_CPUS, KILLARNEY_TIME.
# ---------------------------------------------------------------------------
if [[ -z "${AIP_ACCOUNT:-}" ]]; then
  AIP_ACCOUNT="$(ls -d ~/projects/*/ 2>/dev/null | head -n 1 | sed 's|/*$||' | sed 's|.*/||')"
  export AIP_ACCOUNT
  if [[ -z "${AIP_ACCOUNT}" ]]; then
    echo "killarney-dev.sh: AIP_ACCOUNT is empty (no ~/projects/*/?). Export AIP_ACCOUNT manually." >&2
  fi
fi

killarney_gpu_interactive() {
  if [[ -z "${AIP_ACCOUNT:-}" ]]; then
    echo "killarney_gpu_interactive: set AIP_ACCOUNT first." >&2
    return 1
  fi
  local gres="${KILLARNEY_GRES:-gpu:h100:1}"
  local mem="${KILLARNEY_MEM:-12G}"
  local cpus="${KILLARNEY_CPUS:-4}"
  local timelimit="${KILLARNEY_TIME:-2:00:00}"
  echo "killarney_gpu_interactive: srun --account=${AIP_ACCOUNT} --gres=${gres} ..." >&2
  srun --account="${AIP_ACCOUNT}" --gres="${gres}" --mem="${mem}" -c "${cpus}" --time="${timelimit}" --pty bash
}

# ---------------------------------------------------------------------------
# Module stack — must match wheelhouse torch 2.12.0+computecanada (CUDA 13.2).
# OpenCV module must be loaded before venv activate (Alliance dummy opencv wheel).
# ---------------------------------------------------------------------------
HAD_OPENCV_MODULE="${HAD_OPENCV_MODULE:-opencv/4.13.0}"
HAD_PYTHON_MODULE="${HAD_PYTHON_MODULE:-python/3.12}"
HAD_CUDA_MODULE="${HAD_CUDA_MODULE:-cuda/13.2}"
# L40S (8.9) + H100 (9.0) when building CUDA extensions on a login node.
HAD_CUDA_ARCH_NOGPU="${HAD_CUDA_ARCH_NOGPU:-8.9;9.0}"

# HAD paths — anchored to this repo (pwd -P), not $HOME or $PWD.
# Re-source from your /project/.../HAD checkout so HAD_ROOT is on project storage.
# ---------------------------------------------------------------------------
export PROJECT_ROOT="$(had_realpath "${PROJECT_ROOT:-${HAD_ROOT}}")"
export LVSM_ROOT="$(had_realpath "${LVSM_ROOT:-${PROJECT_ROOT}/LVSM}")"
export LVSM_CKPT_PATH="${LVSM_CKPT_PATH:-${PROJECT_ROOT}/checkpoints/LVSM_decoder_only_conf_Resi_unet_512}"
if [[ -e "${LVSM_CKPT_PATH}" ]]; then
  LVSM_CKPT_PATH="$(had_realpath "${LVSM_CKPT_PATH}")"
fi
export LVSM_CKPT_PATH

# Prefer scratch for large outputs when available; otherwise keep under project tree.
if [[ -z "${OUTPUT_ROOT:-}" ]]; then
  if [[ -d "/scratch/${USER}" ]]; then
    export OUTPUT_ROOT="/scratch/${USER}/had/outputs"
  else
    export OUTPUT_ROOT="${PROJECT_ROOT}/outputs"
  fi
else
  export OUTPUT_ROOT="$(had_realpath "${OUTPUT_ROOT}")"
fi

# Dataset roots live next to the repo by default (${PROJECT_ROOT}/data/...).
: "${HAD_DL3DV_DATA_ROOT:=${PROJECT_ROOT}/data/DL3DV-10K-Benchmark}"
: "${HAD_MIPNERF_DATA_ROOT:=${PROJECT_ROOT}/data/MipNeRF360}"
export HAD_DL3DV_DATA_ROOT="$(had_realpath "${HAD_DL3DV_DATA_ROOT}")"
export HAD_MIPNERF_DATA_ROOT="$(had_realpath "${HAD_MIPNERF_DATA_ROOT}")"
export DATA_ROOT="$(had_realpath "${DATA_ROOT:-${HAD_DL3DV_DATA_ROOT}}")"
export MIPNERF_DATA_ROOT="$(had_realpath "${MIPNERF_DATA_ROOT:-${HAD_MIPNERF_DATA_ROOT}}")"

# ---------------------------------------------------------------------------
# Batch sbatch defaults — edit here (not login-shell exports) so submissions stay predictable.
# Full DL3DV runs often need 1-12:00:00 or longer; use 2:00:00 for short smoke tests.
# ---------------------------------------------------------------------------
HAD_SBATCH_TIME="${HAD_SBATCH_TIME:-2:00:00}"
HAD_SBATCH_GRES="${HAD_SBATCH_GRES:-gpu:h100:1}"
HAD_SBATCH_MEM="${HAD_SBATCH_MEM:-64G}"
HAD_SBATCH_CPUS="${HAD_SBATCH_CPUS:-8}"

# Poll until a Slurm job leaves the queue (completed, failed, or cancelled).
_had_wait_for_slurm_job() {
  local job_id="$1"
  echo "killarney_gpu_sbatch: waiting for job ${job_id} ..." >&2
  while squeue -j "${job_id}" -h 2>/dev/null | grep -q .; do
    sleep 30
  done
  local state
  state="$(sacct -j "${job_id}" -n -X -o State 2>/dev/null | head -n 1 | awk '{print $1}')"
  if [[ -n "${state}" && "${state}" != "COMPLETED" ]]; then
    echo "killarney_gpu_sbatch: job ${job_id} finished with state=${state}" >&2
  else
    echo "killarney_gpu_sbatch: job ${job_id} finished" >&2
  fi
}

# Submit one sbatch job (internal). Optional 5th arg: uncertainty_mask_threshold.
# Prints the numeric job ID on stdout; status messages go to stderr.
_killarney_gpu_sbatch_submit_one() {
  local scene="$1"
  local sparse_view="$2"
  local max_steps="$3"
  local timelimit="$4"
  local threshold="${5:-${UNCERTAINTY_MASK_THRESHOLD:-0.9}}"

  local gres="${HAD_SBATCH_GRES}"
  local mem="${HAD_SBATCH_MEM}"
  local cpus="${HAD_SBATCH_CPUS}"
  local log_dir
  log_dir="$(had_realpath "${HAD_LOG_DIR:-${PROJECT_ROOT}/logs}")"

  local job_name="had-$(echo "${scene}" | cut -c1-20)"
  if [[ "${threshold}" != "0.9" ]]; then
    job_name="${job_name}-t${threshold}"
  fi

  local job_cmd="export HAD_SKIP_AUTO_ACTIVATE=1 PYTHONUNBUFFERED=1 UNCERTAINTY_MASK_THRESHOLD=${threshold} RUN_TIMESTAMP=; source \"${PROJECT_ROOT}/killarney-dev.sh\" && had_dev && \"${PROJECT_ROOT}/run_train_scene.sh\" \"${scene}\" \"${sparse_view}\" \"${max_steps}\""

  mkdir -p "${log_dir}"
  echo "killarney_gpu_sbatch: scene=${scene} SPARSE_VIEW=${sparse_view} MAX_STEPS=${max_steps} UNCERTAINTY_MASK_THRESHOLD=${threshold} --time=${timelimit}" >&2

  local job_id
  job_id="$(
    cd "${PROJECT_ROOT}"
    sbatch --parsable \
      --account="${AIP_ACCOUNT}" \
      --gres="${gres}" \
      --mem="${mem}" \
      -c "${cpus}" \
      --time="${timelimit}" \
      --chdir="${PROJECT_ROOT}" \
      --job-name="${job_name}" \
      --output="${log_dir}/${job_name}-%j.out" \
      --error="${log_dir}/${job_name}-%j.err" \
      --export=NONE,PYTHONUNBUFFERED=1,DATASET="${DATASET:-dl3dv}",DATA_ROOT="${DATA_ROOT}",OUTPUT_ROOT="${OUTPUT_ROOT}",LVSM_CKPT_PATH="${LVSM_CKPT_PATH}",SPARSE_VIEW="${sparse_view}",MAX_STEPS="${max_steps}",UNCERTAINTY_MASK_THRESHOLD="${threshold}",RUN_TIMESTAMP="" \
      --wrap "bash -lc $(printf '%q' "${job_cmd}")"
  )"
  echo "killarney_gpu_sbatch: submitted job_id=${job_id}" >&2
  echo "${job_id}"
}

# Submit run_train_scene.sh via sbatch (login node). Same paths/account as interactive.
# Usage: killarney_gpu_sbatch <scene_id> [sparse_view] [max_steps] [time_limit]
# Set CONF_THRESHOLDS to submit one job per threshold; otherwise a single job is submitted.
# CONF_SWEEP_SUBMIT_MODE=parallel (default) submits all thresholds at once;
# CONF_SWEEP_SUBMIT_MODE=sequential waits for each job to finish before submitting the next.
killarney_gpu_sbatch() {
  local scene="${1:?killarney_gpu_sbatch: scene_id required}"
  local sparse_view="${2:-${SPARSE_VIEW:-9}}"
  local max_steps="${3:-${MAX_STEPS:-10}}"
  local timelimit="${4:-${HAD_SBATCH_TIME}}"

  if [[ -z "${AIP_ACCOUNT:-}" ]]; then
    echo "killarney_gpu_sbatch: set AIP_ACCOUNT first." >&2
    return 1
  fi

  had_assert_not_home "PROJECT_ROOT" "${PROJECT_ROOT}" || return 1
  had_assert_not_home "log_dir" "$(had_realpath "${HAD_LOG_DIR:-${PROJECT_ROOT}/logs}")" || return 1
  had_assert_not_home "DATA_ROOT" "${DATA_ROOT}" || return 1
  had_assert_not_home "OUTPUT_ROOT" "${OUTPUT_ROOT}" || return 1

  echo "killarney_gpu_sbatch: PROJECT_ROOT=${PROJECT_ROOT}" >&2

  if [[ -n "${CONF_THRESHOLDS:-}" ]]; then
    local threshold job_id sweep_mode
    sweep_mode="${CONF_SWEEP_SUBMIT_MODE:-parallel}"
    if [[ "${sweep_mode}" != "parallel" && "${sweep_mode}" != "sequential" ]]; then
      echo "killarney_gpu_sbatch: CONF_SWEEP_SUBMIT_MODE must be parallel or sequential (got ${sweep_mode})" >&2
      return 1
    fi
    echo "killarney_gpu_sbatch: sweep CONF_THRESHOLDS=${CONF_THRESHOLDS} CONF_SWEEP_SUBMIT_MODE=${sweep_mode}" >&2
    if [[ "${sweep_mode}" == "sequential" ]]; then
      echo "killarney_gpu_sbatch: sequential mode — keep this shell open until the sweep completes" >&2
    fi
    for threshold in ${CONF_THRESHOLDS}; do
      job_id="$(_killarney_gpu_sbatch_submit_one "${scene}" "${sparse_view}" "${max_steps}" "${timelimit}" "${threshold}")"
      if [[ "${sweep_mode}" == "sequential" ]]; then
        _had_wait_for_slurm_job "${job_id}"
      fi
    done
    return 0
  fi

  echo "killarney_gpu_sbatch: sbatch --account=${AIP_ACCOUNT} --gres=${HAD_SBATCH_GRES} ..." >&2
  _killarney_gpu_sbatch_submit_one "${scene}" "${sparse_view}" "${max_steps}" "${timelimit}"
}

# Convenience alias: sweep with default CONF_THRESHOLDS if unset.
# Usage: killarney_gpu_sbatch_sweep <scene_id> [sparse_view] [max_steps] [time_limit]
killarney_gpu_sbatch_sweep() {
  CONF_THRESHOLDS="${CONF_THRESHOLDS:-0.5 0.6 0.7 0.8 0.9 0.95}" killarney_gpu_sbatch "$@"
}

had_export_pythonpath() {
  export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/examples/gsplat:${PROJECT_ROOT}/examples/gsplat/pycolmap:$(dirname "${LVSM_ROOT}"):${PYTHONPATH:-}"
  export LVSM_ROOT LVSM_CKPT_PATH
}

# Load modules, activate venv, and export PYTHONPATH for HAD.
# Optional first argument: venv directory (default: ${HAD_ROOT}/venv).
had_dev() {
  local venv_path
  venv_path="$(had_realpath "${1:-${HAD_ROOT}/venv}")"

  if [[ "${VIRTUAL_ENV:-}" == "${venv_path}" ]] && command -v python >/dev/null 2>&1; then
    had_export_pythonpath
    echo "had_dev: already active (${venv_path})"
    return 0
  fi

  if ! module load "${HAD_OPENCV_MODULE}" "${HAD_PYTHON_MODULE}" "${HAD_CUDA_MODULE}"; then
    echo "had_dev: module load failed (opencv, python/3.12, cuda/13.2)" >&2
    return 1
  fi

  if [[ ! -f "${venv_path}/bin/activate" ]]; then
    echo "had_dev: no venv at ${venv_path} — see HAD/KILLARNEY.md to create it" >&2
    return 1
  fi

  # shellcheck source=/dev/null
  source "${venv_path}/bin/activate"

  if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-${HAD_CUDA_ARCH_NOGPU}}"
  fi

  had_export_pythonpath

  echo "had_dev: ${HAD_OPENCV_MODULE} + ${HAD_PYTHON_MODULE} + ${HAD_CUDA_MODULE} + venv (${venv_path})"
  echo "had_dev: PROJECT_ROOT=${PROJECT_ROOT}"
  echo "had_dev: DATA_ROOT=${DATA_ROOT}"
  echo "had_dev: OUTPUT_ROOT=${OUTPUT_ROOT}"
  echo "had_dev: LVSM_CKPT_PATH=${LVSM_CKPT_PATH}"
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    echo "had_dev: GPU visible; TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-<unset, torch autodetect>}"
  else
    echo "had_dev: no GPU in this shell; TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} (for nvcc builds on login)"
  fi
}

if [[ "${HAD_SKIP_AUTO_ACTIVATE:-0}" != "1" ]]; then
  had_dev
fi
