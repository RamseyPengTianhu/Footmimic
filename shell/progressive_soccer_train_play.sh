#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
EXPERIMENT_DIR="${REPO_ROOT}/logs/rsl_rl/g1_flat"

RUN_NAME="${1:-test}"
DEVICE="${2:-0}"

# Strip "cuda:" if user passed it
CUDA_ID="${DEVICE#cuda:}"
export CUDA_VISIBLE_DEVICES="${CUDA_ID}"

cd "${REPO_ROOT}"

python scripts/rsl_rl/train_multi.py --task Anchor-Kick-G1-RNN-v0 \
    --motion_path motions/Video \
    --run_name "${RUN_NAME}" \
    --num_envs 2048 \
    --max_iterations 7000 \
    --device "cuda:0" \
    --headless

LOAD_RUN="$(find "${EXPERIMENT_DIR}" -maxdepth 1 -mindepth 1 -type d -name "*_${RUN_NAME}" | sort | tail -n 1 | xargs -r basename)"

if [[ -z "${LOAD_RUN}" ]]; then
    echo "Failed to resolve load_run from ${EXPERIMENT_DIR}"
    exit 1
fi

echo "Resolved load_run=${LOAD_RUN}"

python scripts/rsl_rl/train_multi.py --task Anchor-CG-Kick-G1-Soccer-RNN-v0 \
    --motion_path motions/Video \
    --load_run "${LOAD_RUN}" \
    --run_name "${RUN_NAME}_resume" \
    --num_envs 2048 \
    --resume True \
    --device "cuda:0" \
    --headless
