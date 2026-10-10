#!/usr/bin/env bash
# Create the single `relocate` env: CUDA 12.8 toolkit, PyTorch, detectron2, requirements.
#
# Blackwell GPUs (RTX PRO 4500, RTX 50-series, sm_120) have no kernels in the
# CUDA 11.8 wheels. The first CUDA allocation then fails with
# "no kernel image is available for execution on the device".
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

# conda's gcc deactivate hook reads _CONDA_PYTHON_SYSCONFIGDATA_NAME_USED
# even when it was never set. set -u above turns that into a fatal error
# after a successful `conda install`.
run_conda() {
  set +u
  conda "$@"
  local status=$?
  set -u
  return "${status}"
}

if conda env list | awk '{print $1}' | grep -qx relocate; then
  echo "Conda env 'relocate' already exists. Updating packages."
else
  run_conda create -y -n relocate python=3.10
fi
run_conda activate relocate

# nvcc 12.8 compiles detectron2 for sm_120. The PyTorch wheel ships its own runtime.
run_conda install -y -c "nvidia/label/cuda-12.8.0" cuda-toolkit

python -m pip install --upgrade pip
python -m pip install --upgrade \
  torch==2.7.1 torchvision==0.22.1 xformers==0.0.31 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r "${ROOT}/requirements.txt"

# detectron2 imports PIL.Image.LINEAR, which Pillow 10 removed. Keep 9.5.
python -m pip install --force-reinstall --no-deps pillow==9.5.0 numpy==1.23.5

export CUDA_HOME="${CONDA_PREFIX}"
export PATH="${CUDA_HOME}/bin:${PATH}"
hash -r

if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc is not on PATH after installing the CUDA 12.8 toolkit." >&2
  exit 1
fi
NVCC_RELEASE="$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n 1)"
echo "nvcc ${NVCC_RELEASE}"
python - "${NVCC_RELEASE}" <<'PY'
import sys
major, minor = (int(part) for part in sys.argv[1].split(".")[:2])
if (major, minor) < (12, 8):
    raise SystemExit(
        f"nvcc {sys.argv[1]} cannot compile sm_120. Install CUDA 12.8 or newer."
    )
PY

# 8.x/9.0 keep Ampere, Ada, and Hopper. 12.0 is Blackwell (sm_120).
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;8.9;9.0;12.0}"
# CUDA 12.8 accepts GCC 13. Pin GCC 11 so Ubuntu 22.04 and 24.04 use one compiler.
if [[ ! -x /usr/bin/gcc-11 || ! -x /usr/bin/g++-11 ]]; then
  echo "gcc-11 and g++-11 are required to build detectron2." >&2
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

# Pip would reuse the old extension: the git revision did not change, but the
# binary was compiled for CUDA 11.8 and does not contain sm_120.
python -m pip uninstall -y detectron2 || true
python -m pip install --no-build-isolation --no-cache-dir --force-reinstall \
  "git+https://github.com/MaureenZOU/detectron2-xyz.git"
rm -rf "${HOST_BIN}"

python -m pip install --force-reinstall --no-deps pillow==9.5.0 numpy==1.23.5

python - <<'PY'
import torch
import PIL
import detectron2
print(
    "torch", torch.__version__,
    "cuda", torch.version.cuda,
    "pillow", PIL.__version__,
    "detectron2", getattr(detectron2, "__version__", "imported"),
)
print("archs", torch.cuda.get_arch_list())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available. Check the NVIDIA driver on this pod.")
name = torch.cuda.get_device_name(0)
major, minor = torch.cuda.get_device_capability(0)
needed = f"sm_{major}{minor}"
archs = torch.cuda.get_arch_list()
print("device", name, needed)
if needed not in archs:
    raise SystemExit(
        f"{name} ({needed}) is not in this PyTorch build. Architectures: {archs}"
    )
torch.ones(1, device="cuda")
torch.cuda.synchronize()
print("cuda kernel ok")
PY
echo "Env 'relocate' is ready."
