# Shared paths and Hugging Face settings. Source this; do not execute it.
# Creates /workspace/data on RunPod when that directory is missing.

if [[ -z "${ROOT:-}" ]]; then
  echo "scripts/env.sh must be sourced after ROOT is set" >&2
  return 1
fi

if [[ -d /workspace ]]; then
  DATA_ROOT="${RELOCATE_DATA_ROOT:-/workspace/data}"
else
  DATA_ROOT="${RELOCATE_DATA_ROOT:-${ROOT}/.cache}"
fi
mkdir -p "${DATA_ROOT}/weights" "${DATA_ROOT}/nltk_data"

if [[ -f "${ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/.env"
  set +a
fi

if [[ -z "${HF_TOKEN:-}" || "${HF_TOKEN}" == hf_your_token_here ]]; then
  unset HF_TOKEN
fi
if [[ -z "${HUGGING_FACE_HUB_TOKEN:-}" || "${HUGGING_FACE_HUB_TOKEN}" == hf_your_token_here ]]; then
  unset HUGGING_FACE_HUB_TOKEN
fi

export HF_HOME="${HF_HOME:-${DATA_ROOT}/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export NLTK_DATA="${NLTK_DATA:-${DATA_ROOT}/nltk_data}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${NLTK_DATA}"

# Final checkpoints stay at RelocateEdit/weights. On RunPod that path is a
# symlink onto the persistent disk, next to the Hugging Face cache.
if [[ -d /workspace ]]; then
  if [[ -L "${ROOT}/weights" || ! -e "${ROOT}/weights" ]]; then
    ln -sfn "${DATA_ROOT}/weights" "${ROOT}/weights"
  fi
fi
