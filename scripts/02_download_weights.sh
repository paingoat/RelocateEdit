#!/usr/bin/env bash
# Download SEEM, Depth Anything V2, big-lama, and AnyDoor into weights/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -d /workspace ]]; then
  PREFIX="${MINICONDA_PREFIX:-/workspace/miniconda3}"
else
  PREFIX="${MINICONDA_PREFIX:-${HOME}/miniconda3}"
fi

# shellcheck disable=SC1091
source "${PREFIX}/etc/profile.d/conda.sh"
conda activate relocate

cd "${ROOT}"
# shellcheck disable=SC1091
source "${ROOT}/scripts/env.sh"
python "${ROOT}/scripts/download_weights.py" "$@"
