#!/usr/bin/env bash
# Full machine setup: Miniconda, conda env, model weights.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bash "${ROOT}/scripts/00_install_miniconda.sh"
bash "${ROOT}/scripts/01_create_env.sh"
bash "${ROOT}/scripts/02_download_weights.sh" "$@"
echo "Setup finished. Start the app with: bash scripts/run_app.sh"
