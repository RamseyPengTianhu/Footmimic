#!/usr/bin/env bash
#
# Clean-v0 dribbling launcher (supersedes progressive_dribbling_train.sh, which
# targets the retired unified-s1/s2/s3-local tasks).
#
# Usage:
#   bash shell/clean_v0_dribbling_train.sh stage1 RUN_NAME
#   S1_LOAD_RUN=<run dir> S1_CHECKPOINT=model_6999.pt \
#     bash shell/clean_v0_dribbling_train.sh stage2 RUN_NAME
#
# Environment overrides:
#   DRIBBLE_MOTION_PATH  motion file or directory      (default motions/master-aligned)
#   S2_TASK              Stage-2 task                   (default ...CleanV0-Stage2-RNN)
#   NUM_ENVS, MAX_ITERATIONS
#
# Stage 2 must start from a clean-v0 160-D Stage-1 checkpoint.
# NOTE: this launcher has not been executed yet; verify the first run.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
STAGE="${1:-}"
RUN_NAME="${2:-}"
MOTION_PATH="${DRIBBLE_MOTION_PATH:-motions/master-aligned}"
S1_TASK="Tracking-Flat-G1-CleanV0-Stage1-RNN"
S2_TASK="${S2_TASK:-Tracking-Flat-G1-CleanV0-Stage2-RNN}"

if [[ -z "${STAGE}" || -z "${RUN_NAME}" || "$#" -ne 2 ]]; then
    sed -n 2,17p "${BASH_SOURCE[0]}" >&2
    exit 2
fi

cd "${REPO_ROOT}"

case "${STAGE}" in
stage1)
    python scripts/rsl_rl/train_multi.py --task "${S1_TASK}" \
        --motion_path "${MOTION_PATH}" \
        --run_name "${RUN_NAME}" \
        --experiment_name g1_dribbling \
        --num_envs "${NUM_ENVS:-1024}" \
        --max_iterations "${MAX_ITERATIONS:-7000}" \
        --headless
    ;;
stage2)
    : "${S1_LOAD_RUN:?set S1_LOAD_RUN to the Stage-1 run directory name}"
    : "${S1_CHECKPOINT:?set S1_CHECKPOINT, e.g. model_6999.pt}"
    python scripts/rsl_rl/train_multi.py --task "${S2_TASK}" \
        --motion_path "${MOTION_PATH}" \
        --load_run "${S1_LOAD_RUN}" \
        --checkpoint "${S1_CHECKPOINT}" \
        --resume True \
        --run_name "${RUN_NAME}" \
        --experiment_name g1_dribbling \
        --num_envs "${NUM_ENVS:-512}" \
        --max_iterations "${MAX_ITERATIONS:-4000}" \
        --headless
    ;;
*)
    echo "Unknown stage '${STAGE}' (expected stage1 or stage2)" >&2
    exit 2
    ;;
esac
