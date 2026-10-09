#!/usr/bin/env bash
# Install Miniconda into a persistent directory. On RunPod that is /workspace.
set -euo pipefail

if [[ -d /workspace ]]; then
  PREFIX="${MINICONDA_PREFIX:-/workspace/miniconda3}"
else
  PREFIX="${MINICONDA_PREFIX:-${HOME}/miniconda3}"
fi

if [[ -x "${PREFIX}/bin/conda" ]]; then
  echo "Miniconda already installed at ${PREFIX}"
  exit 0
fi

INSTALLER="${TMPDIR:-/tmp}/miniconda.sh"
URL="https://repo.anaconda.com/miniconda/Miniconda3-py310_24.9.2-0-Linux-x86_64.sh"
echo "Downloading Miniconda to ${PREFIX}"
wget -O "${INSTALLER}" "${URL}"
bash "${INSTALLER}" -b -p "${PREFIX}"
rm -f "${INSTALLER}"
"${PREFIX}/bin/conda" config --set auto_activate_base false
echo "Miniconda installed. Open a new shell or: source ${PREFIX}/etc/profile.d/conda.sh"
