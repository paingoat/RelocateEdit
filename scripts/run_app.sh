#!/usr/bin/env bash
# Start Gradio. --share prints a public gradio.live URL.
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

# Pass --offload on a 24 GB GPU. Omit it on 40 GB or larger.
exec python app.py --share --server-name 0.0.0.0 --server-port 7860 "$@"
