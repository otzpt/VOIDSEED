#!/usr/bin/env bash
# tinyllm2 (VOIDSEED) — Linux environment setup.
# A Python venv is NOT portable across operating systems, so the Windows .venv
# was intentionally left out of the copy. Run this once on the Linux machine.
#
#   bash setup.sh                 # NVIDIA GPU, CUDA 12.6 (matches original)
#   TORCH_INDEX=cpu bash setup.sh # CPU-only machine
#   TORCH_INDEX=cu121 bash setup.sh
#
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"

echo "==> creating venv (.venv) with $PYTHON"
"$PYTHON" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> upgrading pip"
pip install --upgrade pip

TORCH_INDEX="${TORCH_INDEX:-cu126}"
if [ "$TORCH_INDEX" = "cpu" ]; then
  echo "==> installing torch (CPU build)"
  pip install torch
else
  echo "==> installing torch (${TORCH_INDEX} build)"
  pip install torch --index-url "https://download.pytorch.org/whl/${TORCH_INDEX}"
fi

echo "==> installing project dependencies"
pip install -r requirements.txt

echo
echo "done."
echo "  activate:        source .venv/bin/activate"
echo "  verify torch:    python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'"
echo "  data is ready:   data/tinystories/{train,val}.bin  (already copied)"
echo "  regenerate data: python prepare.py --dataset tinystories"
