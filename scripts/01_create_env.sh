#!/usr/bin/env bash
# Create the single `relocate` env: CUDA 11.8 toolkit, PyTorch, detectron2, requirements.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -d /workspace ]]; then
  PREFIX="${MINICONDA_PREFIX:-/workspace/miniconda3}"
else
  PREFIX="${MINICONDA_PREFIX:-${HOME}/miniconda3}"
fi

if [[ "$(id -u)" -eq 0 ]]; then
  apt-get update
  apt-get install -y build-essential gcc-11 g++-11 git wget unzip ninja-build
else
  sudo apt-get update
  sudo apt-get install -y build-essential gcc-11 g++-11 git wget unzip ninja-build
fi

# shellcheck disable=SC1091
source "${PREFIX}/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx relocate; then
  echo "Conda env 'relocate' already exists. Updating packages."
else
  conda create -y -n relocate python=3.10
fi
conda activate relocate

conda install -y -c "nvidia/label/cuda-11.8.0" cuda-toolkit

python -m pip install --upgrade pip
python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r "${ROOT}/requirements.txt"

# detectron2 imports PIL.Image.LINEAR, which Pillow 10 removed. Keep 9.5.
python -m pip install --force-reinstall --no-deps pillow==9.5.0 numpy==1.23.5

export CUDA_HOME="${CONDA_PREFIX}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;8.9;9.0}"
# CUDA 11.8 nvcc rejects GCC newer than 11. Ubuntu 24.04's default is GCC 13.
if [[ ! -x /usr/bin/gcc-11 || ! -x /usr/bin/g++-11 ]]; then
  echo "gcc-11 and g++-11 are required to build detectron2 against CUDA 11.8." >&2
  exit 1
fi
HOST_BIN="$(mktemp -d)"
ln -s /usr/bin/gcc-11 "${HOST_BIN}/gcc"
ln -s /usr/bin/g++-11 "${HOST_BIN}/g++"
ln -s /usr/bin/gcc-11 "${HOST_BIN}/cc"
ln -s /usr/bin/g++-11 "${HOST_BIN}/c++"
export PATH="${HOST_BIN}:${PATH}"
export CC="${HOST_BIN}/gcc"
export CXX="${HOST_BIN}/g++"
export CUDAHOSTCXX="${HOST_BIN}/g++"
python -m pip install --no-build-isolation "git+https://github.com/MaureenZOU/detectron2-xyz.git"
rm -rf "${HOST_BIN}"

python - <<'PY'
import torch
import PIL
print("torch", torch.__version__, "cuda", torch.version.cuda, "pillow", PIL.__version__)
PY
echo "Env 'relocate' is ready."
