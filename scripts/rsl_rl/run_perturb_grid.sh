#!/bin/bash
# v3 generalization grid with output saved to file
# Pipes all output to a log file for analysis

OUTFILE="logs/rsl_rl/g1_flat/2026-04-28_12-15-12_cg_v3_softmask/eval/perturb_grid.log"
mkdir -p $(dirname $OUTFILE)

COMMON="--task Anchor-CG-Kick-G1-Soccer-RNN-v0 --motion_path motions/Video \
    --load_run 2026-04-28_12-15-12_cg_v3_softmask --checkpoint model_12000.pt \
    --num_envs 16 --eval_episodes 10 --headless"

echo "Grid started at $(date)" | tee $OUTFILE

for x in -0.10 -0.05 0.00 0.05 0.10; do
  for y in -0.10 0.00 0.10; do
    echo "" | tee -a $OUTFILE
    echo "========================================================" | tee -a $OUTFILE
    echo "  GRID CONDITION: x=${x} y=${y}" | tee -a $OUTFILE
    echo "========================================================" | tee -a $OUTFILE
    CUDA_VISIBLE_DEVICES=0 python scripts/rsl_rl/eval_kick_diagnostic.py \
        $COMMON --ball_x_offset $x --ball_y_offset $y 2>&1 | tee -a $OUTFILE
  done
done

echo "===== GRID COMPLETE at $(date) =====" | tee -a $OUTFILE
echo "Results saved to: $OUTFILE"
