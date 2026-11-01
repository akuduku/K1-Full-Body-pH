#!/bin/bash
# A CUDA machine (Colab, Kaggle, a lab PC) ready to train on the pH plant, from a fresh clone of this repo:
#   1 the Python packages next to the preinstalled CUDA torch (a constraint keeps that torch; tensordict follows it)
#   2 the C++/CUDA plant (ph_k1_cuda), built for this GPU (TORCH_CUDA_ARCH_LIST, default 7.5+PTX = T4 + newer via PTX)
#   3 its tests ON THE GPU (C++ == Python reference, command path, the get-up task): no training on a plant that
#     disagrees with its reference
# Used by the colab-train job colab/k1-getup.yaml (install step).  PYTHON=... picks the interpreter;
# PH_K1_CUDA=0 PH_TEST_DEVICE=cpu: a dry run on a machine without a GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
TORCH=$($PY -c "import torch; print(torch.__version__)")
echo "[setup] torch $TORCH, CUDA available: $($PY -c 'import torch; print(torch.cuda.is_available())'), $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'no nvidia-smi')"
CONSTRAINTS=$(mktemp)
echo "torch==$TORCH" > "$CONSTRAINTS"
$PY -m pip install -q -c "$CONSTRAINTS" "rsl-rl-lib==5.5.2" tensordict mujoco wandb imageio imageio-ffmpeg tensorboard pytest
echo "[setup] packages: $($PY -c 'import rsl_rl, tensordict, mujoco; print("tensordict", tensordict.__version__, "mujoco", mujoco.__version__)')"
BUILD_LOG=$(mktemp)
start=$(date +%s)
if ! (cd ph_k1_cuda && PH_K1_CUDA="${PH_K1_CUDA:-1}" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5+PTX}" $PY setup.py build_ext --inplace > "$BUILD_LOG" 2>&1); then
    tail -60 "$BUILD_LOG"
    echo "[setup] building the CUDA plant failed"
    exit 1
fi
echo "[setup] CUDA plant built in $(( $(date +%s) - start )) s"
# which of torch's linear-algebra kernels this torch build has for this GPU (Colab's torch 2.11+cu130 on a T4 lacks
# some, e.g. for cholesky_solve: the plant uses linalg.solve)
if [ "${PH_TEST_DEVICE:-cuda}" = cuda ]; then
    $PY - <<'PROBE' || true
import torch
A = torch.eye(4, device="cuda", dtype=torch.float64).expand(2, 4, 4) + 0.1
b = torch.ones(2, 4, 1, device="cuda", dtype=torch.float64)
for name, fn in (("linalg.solve", lambda: torch.linalg.solve(A, b)), ("cholesky", lambda: torch.linalg.cholesky(A))):
    try:
        fn(); torch.cuda.synchronize(); print(f"[setup] torch {name} on the GPU: ok", flush=True)
    except Exception as e:
        print(f"[setup] torch {name} on the GPU: missing ({type(e).__name__})", flush=True)
        break
PROBE
fi
PH_TEST_DEVICE="${PH_TEST_DEVICE:-cuda}" $PY -m pytest ph_k1_cuda/tests rl/getup/tests -q -p no:cacheprovider
echo "[setup] GPU tests passed"
