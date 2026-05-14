#!/usr/bin/env bash
#
# Progressive Dribbling (2-Stage) — dribble motion + flat motion pretrain
#
# Stage 1: Motion tracking on **flat** ground (same ball/target obs as dribble MDP)
#   - Default:           Tracking-Flat-G1-Motion-RNN-v0
#   - --cg:              Tracking-CG-G1-Motion-RNN-v0
#                        (adds anchor_ball_polar so its checkpoint is
#                         obs-compatible with the CG dribble Stage 2)
#   - --ankle-disturb:   Tracking-Flat-G1-Dribbling-AnkleDisturb-RNN-v0
#                        (ignored when --cg is also set)
#
# Stage 2: Dribbling stage-2 task (baseline or CG variant, resume from Stage 1 run)
#   - Default: Tracking-Flat-G1-Dribbling-RNN-v0
#   - --cg:    Tracking-CG-G1-Dribbling-RNN-v0
#              (annotated CG + demo ball_pos_w stitching)
#   - Heuristic-only CG (no labels): pass task explicitly, e.g.
#       --task Tracking-CG-Heuristic-G1-Dribbling-RNN-v0
#
# Motion directory: set DRIBBLE_MOTION_PATH to your folder of dribble .npz files
# (defaults to motions/dribble). CG training expects ``ball_pos_w`` in each .npz
# (or merged from a sidecar) plus ``dribble_cg_contact`` / ``dribble_cg_foot``
# from dribble_label_tool apply (or kick_frame/kick_end/kick_leg fallback).
#
# Usage:
#   DRIBBLE_MOTION_PATH=motions/my_dribble bash shell/progressive_dribbling_train.sh [RUN_NAME] [--ankle-disturb] [--cg]
#

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
EXPERIMENT_DIR="${REPO_ROOT}/logs/rsl_rl/g1_dribbling"

MOTION_PATH="${DRIBBLE_MOTION_PATH:-motions/dribble}"

RUN_NAME="${1:-dribbling}"
ANKLE_DISTURB=false
USE_CG=false
for arg in "$@"; do
    if [[ "${arg}" == "--ankle-disturb" ]]; then
        ANKLE_DISTURB=true
    elif [[ "${arg}" == "--cg" ]]; then
        USE_CG=true
    fi
done

if [[ "${USE_CG}" == "true" ]]; then
    # CG dribble (Stage 2) adds `anchor_ball_polar` to the policy/critic obs,
    # so Stage 1 must use the obs-compatible CG-pretrain motion env.
    STAGE1_TASK="Tracking-CG-G1-Motion-RNN-v0"
    STAGE2_TASK="Tracking-CG-G1-Dribbling-RNN-v0"
    if [[ "${ANKLE_DISTURB}" == "true" ]]; then
        echo ">>> Warning: --ankle-disturb is ignored under --cg (no CG-compatible ankle-disturb Stage 1 env). <<<"
    fi
    echo ">>> CG motion pretrain Stage 1 (obs-compatible with CG dribble) <<<"
elif [[ "${ANKLE_DISTURB}" == "true" ]]; then
    STAGE1_TASK="Tracking-Flat-G1-Dribbling-AnkleDisturb-RNN-v0"
    STAGE2_TASK="Tracking-Flat-G1-Dribbling-RNN-v0"
    echo ">>> Ankle disturbance Stage 1 <<<"
else
    STAGE1_TASK="Tracking-Flat-G1-Motion-RNN-v0"
    STAGE2_TASK="Tracking-Flat-G1-Dribbling-RNN-v0"
    echo ">>> Flat motion tracking Stage 1 <<<"
fi

cd "${REPO_ROOT}"

echo "════════════════════════════════════════════════════════════════"
echo " Stage 1: ${STAGE1_TASK}"
echo " motion_path: ${MOTION_PATH}"
echo " run_name:    ${RUN_NAME}"
echo "════════════════════════════════════════════════════════════════"

python scripts/rsl_rl/train_multi.py --task "${STAGE1_TASK}" \
    --motion_path "${MOTION_PATH}" \
    --run_name "${RUN_NAME}" \
    --experiment_name g1_dribbling \
    --num_envs 2000 \
    --max_iterations 4000 \
    --headless

LOAD_RUN="$(find "${EXPERIMENT_DIR}" -maxdepth 1 -mindepth 1 -type d -name "*_${RUN_NAME}" | sort | tail -n 1 | xargs -r basename)"

if [[ -z "${LOAD_RUN}" ]]; then
    echo "Failed to resolve Stage 1 checkpoint from ${EXPERIMENT_DIR}"
    exit 1
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo " Stage 2: ${STAGE2_TASK}"
echo " resume: ${LOAD_RUN}"
echo " motion_path: ${MOTION_PATH}"
echo "════════════════════════════════════════════════════════════════"

python scripts/rsl_rl/train_multi.py --task "${STAGE2_TASK}" \
    --motion_path "${MOTION_PATH}" \
    --load_run "${LOAD_RUN}" \
    --run_name "${RUN_NAME}_dribble" \
    --experiment_name g1_dribbling \
    --num_envs 2000 \
    --resume True \
    --headless

echo ""
echo "Play checkpoints (logs live under logs/rsl_rl/g1_dribbling/):"
echo "  Stage 2 dribbling policy:"
echo "    python scripts/rsl_rl/play_multi.py --task ${STAGE2_TASK} \\"
echo "      --motion_path \"${MOTION_PATH}\" --load_run \"<RUN_DIR>_dribble\" --checkpoint model_XXXX.pt ..."
echo "  Stage 1 motion policy — add --experiment_name g1_dribbling:"
echo "    python scripts/rsl_rl/play_multi.py --task ${STAGE1_TASK} \\"
echo "      --experiment_name g1_dribbling --motion_path \"${MOTION_PATH}\" \\"
echo "      --load_run \"${LOAD_RUN}\" --checkpoint model_XXXX.pt ..."
