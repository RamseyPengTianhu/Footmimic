#!/usr/bin/env bash
#
# DEPRECATED (retired unified-s1/s2/s3-local tasks). Use shell/clean_v0_dribbling_train.sh.
# Previously: the only supported dribbling curriculum:
#   S1 local strict imitation -> S2 reference contact -> S3 local task dribbling
#
# Usage:
#   DRIBBLE_MOTION_PATH=motions/master-aligned \
#     bash shell/progressive_dribbling_train.sh [RUN_NAME]

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
EXPERIMENT_DIR="${REPO_ROOT}/logs/rsl_rl/g1_dribbling"
# Regenerated against the current contact-anchor geometry.  The first ball
# anchor is held stationary from frame 0 until the first labelled contact.
MOTION_PATH="${DRIBBLE_MOTION_PATH:-motions/master-aligned}"
RUN_NAME="${1:-dribbling}"

if [[ "$#" -gt 1 || "${RUN_NAME}" == --* ]]; then
    echo "Usage: bash shell/progressive_dribbling_train.sh [RUN_NAME]" >&2
    exit 2
fi

STAGE1_TASK="Tracking-CG-G1-Motion-RNN-unified-s1-local-strict"
STAGE2_TASK="Tracking-CG-G1-Dribbling-RNN-unified-s2-local-reference"
STAGE3_TASK="Tracking-CG-G1-Dribbling-RNN-unified-s3-local-task"
S1_RUN_NAME="${RUN_NAME}_s1"
S2_RUN_NAME="${RUN_NAME}_s2"
S3_RUN_NAME="${RUN_NAME}_s3"

cd "${REPO_ROOT}"

echo "Stage 1: ${STAGE1_TASK}"
echo "motion_path: ${MOTION_PATH}"
echo "run_name: ${S1_RUN_NAME}"
python scripts/rsl_rl/train_multi.py --task "${STAGE1_TASK}" \
    --motion_path "${MOTION_PATH}" \
    --run_name "${S1_RUN_NAME}" \
    --experiment_name g1_dribbling \
    --num_envs 2000 \
    --max_iterations 4000 \
    --headless

LOAD_S1="$(find "${EXPERIMENT_DIR}" -maxdepth 1 -mindepth 1 -type d -name "*_${S1_RUN_NAME}" | sort | tail -n 1 | xargs -r basename)"
if [[ -z "${LOAD_S1}" ]]; then
    echo "Failed to resolve Stage 1 checkpoint from ${EXPERIMENT_DIR}" >&2
    exit 1
fi

echo "Stage 2: ${STAGE2_TASK}"
echo "resume: ${LOAD_S1}"
echo "run_name: ${S2_RUN_NAME}"
python scripts/rsl_rl/train_multi.py --task "${STAGE2_TASK}" \
    --motion_path "${MOTION_PATH}" \
    --load_run "${LOAD_S1}" \
    --run_name "${S2_RUN_NAME}" \
    --experiment_name g1_dribbling \
    --num_envs 2000 \
    --max_iterations 4000 \
    --resume True \
    --headless

LOAD_S2="$(find "${EXPERIMENT_DIR}" -maxdepth 1 -mindepth 1 -type d -name "*_${S2_RUN_NAME}" | sort | tail -n 1 | xargs -r basename)"
if [[ -z "${LOAD_S2}" ]]; then
    echo "Failed to resolve Stage 2 checkpoint from ${EXPERIMENT_DIR}" >&2
    exit 1
fi

echo "Stage 3: ${STAGE3_TASK}"
echo "resume: ${LOAD_S2}"
echo "run_name: ${S3_RUN_NAME}"
python scripts/rsl_rl/train_multi.py --task "${STAGE3_TASK}" \
    --motion_path "${MOTION_PATH}" \
    --load_run "${LOAD_S2}" \
    --run_name "${S3_RUN_NAME}" \
    --experiment_name g1_dribbling \
    --num_envs 2000 \
    --max_iterations 4000 \
    --resume True \
    --headless

echo "Play Stage 3:"
echo "  python scripts/rsl_rl/play_multi.py --task ${STAGE3_TASK} \\"
echo "    --motion_path \"${MOTION_PATH}\" --load_run \"<RUN_DIR>_s3\" --checkpoint model_XXXX.pt"
