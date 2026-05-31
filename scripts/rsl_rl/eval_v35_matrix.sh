#!/bin/bash
# V3 vs V3.5-FT Comparison Evaluation Matrix
#
# Runs eval_kick_diagnostic.py for both models across perturbation conditions.
# Output: logs/<run>/eval/kick_diagnostic.json for each condition
#
# Usage: bash scripts/rsl_rl/eval_v35_matrix.sh

set -e

PYTHON="/home/tianhup/anaconda3/envs/HumanoidSoccer/bin/python"
SCRIPT="scripts/rsl_rl/eval_kick_diagnostic.py"
MOTION="motions/Video"
ENVS=32
EPISODES=50

# Model configs: task, load_run, checkpoint, label
V3_TASK="Anchor-CG-Kick-G1-Soccer-RNN-v0"
V3_RUN="2026-04-28_12-15-12_cg_v3_softmask"
V3_CKPT="model_12000.pt"

V35_TASK="Strike-Gated-Kick-G1-Soccer-RNN-v0"
V35_RUN="2026-05-15_15-10-20_v35_strike_ft"
V35_CKPT="model_14000.pt"

# Ball offsets (X direction = forward along kick direction)
OFFSETS=("0.00" "0.05" "0.10" "0.15" "0.20")

echo "============================================="
echo "  V3 vs V3.5-FT Evaluation Matrix"
echo "============================================="
echo ""

for MODEL_LABEL in "v3" "v35_ft"; do
    if [ "$MODEL_LABEL" = "v3" ]; then
        TASK=$V3_TASK
        RUN=$V3_RUN
        CKPT=$V3_CKPT
    else
        TASK=$V35_TASK
        RUN=$V35_RUN
        CKPT=$V35_CKPT
    fi

    echo ""
    echo "===== Model: $MODEL_LABEL ====="
    echo ""

    # 1. Nominal (no perturbation)
    echo "[EVAL] $MODEL_LABEL: nominal (offset=0.0)"
    CUDA_VISIBLE_DEVICES=0 $PYTHON $SCRIPT \
        --task $TASK --motion_path $MOTION \
        --load_run $RUN --checkpoint $CKPT \
        --num_envs $ENVS --eval_episodes $EPISODES \
        --ball_x_offset 0.0 --ball_y_offset 0.0 \
        --headless 2>&1 | tail -80

    # 2. X offsets
    for OFF in "${OFFSETS[@]}"; do
        if [ "$OFF" = "0.00" ]; then
            continue
        fi
        echo ""
        echo "[EVAL] $MODEL_LABEL: x_offset=+$OFF"
        CUDA_VISIBLE_DEVICES=0 $PYTHON $SCRIPT \
            --task $TASK --motion_path $MOTION \
            --load_run $RUN --checkpoint $CKPT \
            --num_envs $ENVS --eval_episodes $EPISODES \
            --ball_x_offset $OFF --ball_y_offset 0.0 \
            --headless 2>&1 | tail -80
    done

    # 3. Y offset (lateral)
    for OFF in "0.05" "0.10"; do
        echo ""
        echo "[EVAL] $MODEL_LABEL: y_offset=+$OFF"
        CUDA_VISIBLE_DEVICES=0 $PYTHON $SCRIPT \
            --task $TASK --motion_path $MOTION \
            --load_run $RUN --checkpoint $CKPT \
            --num_envs $ENVS --eval_episodes $EPISODES \
            --ball_x_offset 0.0 --ball_y_offset $OFF \
            --headless 2>&1 | tail -80
    done

    # 4. Random XY perturbation
    for PERTURB in "0.05" "0.10" "0.15"; do
        echo ""
        echo "[EVAL] $MODEL_LABEL: xy_perturb=±$PERTURB"
        CUDA_VISIBLE_DEVICES=0 $PYTHON $SCRIPT \
            --task $TASK --motion_path $MOTION \
            --load_run $RUN --checkpoint $CKPT \
            --num_envs $ENVS --eval_episodes $EPISODES \
            --ball_xy_perturb $PERTURB \
            --headless 2>&1 | tail -80
    done
done

echo ""
echo "============================================="
echo "  Evaluation Complete"
echo "============================================="
