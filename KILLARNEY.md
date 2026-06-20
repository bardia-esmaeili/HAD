# HAD on Alliance Killarney

Notes for setting up the HAD environment on Killarney (Compute Canada / DRAC).
General cluster habits live in `../killarney-setup/notes.txt`; this file is HAD-specific.

## Summary

HAD's upstream scripts assume **conda + `cuda/11.8.0`**. On Killarney you must use:

- **`module load` + `virtualenv`** (no conda)
- **Alliance wheelhouse** for torch/gsplat where possible
- **PyPI with exact pins** for the Hugging Face / DiFix stack
- **`opencv` module loaded before venv activate** (Alliance dummy `opencv-python` wheel otherwise)
- **`cuda/13.2`** to match wheelhouse `torch 2.12.0+computecanada` (built with CUDA 13.2)

---

## Prerequisites

- Repo under `~/projects/` (e.g. `~/projects/aip-jelder/bardiaes/HAD`)
- Shared helpers:

```bash
source ~/projects/aip-jelder/bardiaes/killarney-setup/killarney-env.sh
# sets AIP_ACCOUNT=aip-jelder
```

Interactive GPU (optional):

```bash
killarney_gpu_interactive   # uses srun, not salloc
# re-run module load + venv activate on the compute node
```

---

## Module stack (use everywhere)

Load **before** creating or activating the venv:

```bash
module load gcc opencv/4.13.0 python/3.12 cuda/13.2
```

| Module | Why |
|--------|-----|
| `gcc` | Compiler for CUDA extensions |
| `opencv/4.13.0` | Real `cv2`; required before venv (Alliance OpenCV rule) |
| `python/3.12` | Matches wheelhouse cp312 wheels |
| `cuda/13.2` | Must match `torch.version.cuda` (13.2) when building extensions |

**Do not use `cuda/12.2`** with wheelhouse torch 2.12.0 — nvcc 12.2 vs PyTorch cu132 causes:

```text
RuntimeError: detected CUDA version ('12.2') mismatches PyTorch ('13.2')
```

Check alignment anytime:

```bash
python -c "import torch; print(torch.version.cuda)"
nvcc --version
```

---

## Create the virtualenv

```bash
cd ~/projects/aip-jelder/bardiaes/HAD

module load gcc opencv/4.13.0 python/3.12 cuda/13.2
virtualenv --no-download venv --prompt had
source venv/bin/activate
```

If the venv was created **without** the OpenCV module loaded, recreate it or always follow:
**deactivate → module load → activate**.

---

## Modifications to `requirements.txt`

Comment out packages installed separately (avoids wheelhouse upgrading HF pins and
re-triggering git builds):

```text
# peft==0.9.0
# diffusers==0.25.1
# huggingface-hub==0.25.1
# transformers==4.38.0

# git+https://github.com/rahul-goel/fused-ssim@1272e21a282342e89537159e4bad508b19b34157
```

Keep the rest of `requirements.txt` as-is (including `opencv-python` — it resolves via
the OpenCV module once loaded correctly).

---

## Install order

Always have modules loaded and venv active.

### 1. PyTorch (wheelhouse)

```bash
avail_wheels torch
avail_wheels torchvision

pip install torch==2.12.0 torchvision==0.27.0
python -c "import torch; print(torch.__version__, torch.version.cuda)"
# expect: 2.12.0+computecanada  13.2
```

### 2. Hugging Face / DiFix stack (PyPI, exact pins)

Wheelhouse has much newer versions; HAD needs the old stack for `DifixPipeline`:

```bash
pip install \
  diffusers==0.25.1 \
  transformers==4.38.0 \
  peft==0.9.0 \
  huggingface-hub==0.25.1
```

Optional: `pip install nerfview==0.0.2` if not already installed.

### 3. Remaining requirements

```bash
pip install -r requirements.txt \
  --find-links https://pypi.org/simple/ \
  --prefer-binary
```

If OpenCV fails with the dummy-wheel message:

```bash
deactivate
module load gcc opencv/4.13.0 python/3.12 cuda/13.2
source venv/bin/activate
pip install -r requirements.txt --find-links https://pypi.org/simple/ --prefer-binary
```

After success, some packages appear **outside** the venv (module paths):

- `opencv-python` → OpenCV module (`4.13.0`)
- `scipy`, `matplotlib`, parts of `Pillow` → scipy-stack on `PYTHONPATH`

That is normal on Alliance as long as the same modules are loaded at runtime.

### 4. `fused-ssim` (git, CUDA extension)

**Not** via plain `pip install -r` — needs `--no-build-isolation` so build sees venv torch.

On a **login node** (no GPU), set arch flags for L40S + H100:

```bash
export TORCH_CUDA_ARCH_LIST="8.9;9.0"   # quotes required — semicolon splits commands in bash

pip install --no-build-isolation \
  git+https://github.com/rahul-goel/fused-ssim@1272e21a282342e89537159e4bad508b19b34157
```

If build OOMs or is slow, use an interactive GPU node and run the same command.

---

## Issues encountered and fixes

| Error | Cause | Fix |
|-------|--------|-----|
| `ModuleNotFoundError: No module named 'torch'` building fused-ssim | Pip isolated build env has no torch | `pip install --no-build-isolation ...` |
| CUDA 12.2 vs PyTorch 13.2 | `module load cuda/12.2` with cu132 torch | `module load cuda/13.2` |
| `9.0: command not found` | Unquoted `TORCH_CUDA_ARCH_LIST=8.9;9.0` | `export TORCH_CUDA_ARCH_LIST="8.9;9.0"` |
| OpenCV dummy wheel / `opencv-noinstall` | OpenCV module not loaded before venv | deactivate → module load opencv → activate → pip |
| `numpy<2.0.0` vs wheelhouse 2.4.2 | Pin vs available wheels | Resolved as `numpy 1.26.4+computecanada` in this env |

---

## Expected package versions (this env)

| Package | Version |
|---------|---------|
| torch | 2.12.0+computecanada |
| torchvision | 0.27.0+computecanada |
| diffusers | 0.25.1 |
| transformers | 4.38.0 |
| peft | 0.9.0 |
| huggingface-hub | 0.25.1 |
| gsplat | 1.5.3 |
| fused_ssim | 0.0.0 (git install) |
| numpy | 1.26.4+computecanada |
| nerfview | 0.0.2 |

---

## Daily activation snippet

```bash
source ~/projects/aip-jelder/bardiaes/killarney-setup/killarney-env.sh

module load gcc opencv/4.13.0 python/3.12 cuda/13.2
source ~/projects/aip-jelder/bardiaes/HAD/venv/bin/activate
```

Consider wrapping this in `HAD/killarney-dev.sh` (same pattern as `NAS3R/killarney-dev.sh`).

---

## Smoke test

```bash
python - <<'EOF'
import cv2, scipy, matplotlib
import torch, gsplat, fused_ssim, diffusers, transformers, peft
print("cv2", cv2.__version__)
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("diffusers", diffusers.__version__)
print("cuda available:", torch.cuda.is_available())
EOF
```

`torch.cuda.is_available()` may be `False` on login nodes; test on a GPU node for training.

---

## Running HAD (still to adapt)

Upstream `run_train_scene.sh` / slurm scripts use conda and `cuda/11.8.0`. Replace with
the module stack above.

Set paths when launching:

```bash
export PROJECT_ROOT=~/projects/aip-jelder/bardiaes/HAD
export LVSM_CKPT_PATH=$PROJECT_ROOT/checkpoints/LVSM_decoder_only_conf_Resi_unet_512
export DATA_ROOT=/path/to/DL3DV-10K-Benchmark
export OUTPUT_ROOT=/scratch/$USER/had/outputs
```

Download checkpoint per `README.md`:

- `checkpoints/LVSM_decoder_only_conf_Resi_unet_512/ckpt_0000000000010000.pt`

Example single-scene run:

```bash
./run_train_scene.sh <scene_id> 9 20000
```

Slurm jobs need `--account=$AIP_ACCOUNT` and Killarney GPU types (`gpu:l40s:1` or
`gpu:h100:1`), not `a100`.

---

## Slurm / job checklist

1. `--account=aip-jelder` (or `$AIP_ACCOUNT`)
2. `--gres=gpu:l40s:1` (or h100)
3. `--time=...` set realistically
4. In the job script: `module load gcc opencv/4.13.0 python/3.12 cuda/13.2` then `source venv/bin/activate`
5. Run from `~/projects/...`, not `$HOME`

---

## Differences from upstream HAD

| Upstream | Killarney |
|----------|-----------|
| conda `difix3D` | `virtualenv` + `venv/bin/activate` |
| `cuda/11.8.0` | `cuda/13.2` (matches wheelhouse torch) |
| `pip install -r requirements.txt` (one shot) | Split install: torch → HF pins → requirements → fused-ssim |
| PyPI opencv-python | OpenCV module + dummy pip metapackage |
| A100 in slurm | L40S / H100 on Killarney |
