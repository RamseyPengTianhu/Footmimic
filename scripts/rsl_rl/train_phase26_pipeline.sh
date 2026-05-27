#!/bin/bash
# ============================================================================
# Stage2B-v2: Phase-26D + Prior Recon Loss Pipeline
# ============================================================================
# Stage 1: Online DAgger distillation (V10 26D features + prior_recon loss)
# Stage 2: Latent PPO fine-tuning (frozen decoder, zero tracking rewards)
# ============================================================================

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────
GPU_ID="${GPU_ID:-0}"
TASK="Anchor-CG-Kick-G1-Soccer-RNN-v0"
MOTION_PATH="motions/Video_hmr4d_seed"

# Teacher checkpoint (v3 CG teacher)
TEACHER_RUN="2026-04-28_12-15-12_cg_v3_softmask"
TEACHER_CKPT="model_12000.pt"

# DAgger params
DAGGER_ENVS=64
DAGGER_ITERS=200
DAGGER_STEPS_PER_ITER=500
DAGGER_UPDATES_PER_ITER=50
DAGGER_Z_DIM=16
DAGGER_BETA=1e-3
DAGGER_ALPHA_PRIOR=0.5
DAGGER_OUTPUT="models/latent_v2/online_distill_v10_phase26.pt"

# PPO params
PPO_ENVS=4096
PPO_ITERS=6000
PPO_LAB_SCALE=2.0
PPO_RUN_NAME="latent_v2_ppo_v10_phase26_notrack"

echo "============================================================"
echo "  Stage2B-v2: V10 26D + Prior Recon Loss Pipeline"
echo "============================================================"
echo "  GPU:            ${GPU_ID}"
echo "  Teacher:        ${TEACHER_RUN} / ${TEACHER_CKPT}"
echo "  Motion:         ${MOTION_PATH}"
echo "  DAgger output:  ${DAGGER_OUTPUT}"
echo "  alpha_prior:    ${DAGGER_ALPHA_PRIOR}"
echo "  PPO run:        ${PPO_RUN_NAME}"
echo "============================================================"

# ── Stage 1: Online DAgger ─────────────────────────────────────────────────
echo ""
echo "▶ Stage 1/2: Online DAgger (V10 26D + prior_recon loss)"
echo "  feature_version=v10_phase_26d"
echo "  decoder_obs_mode=task_features → proprio(99D) + task_features(26D) = 125D"
echo "  alpha_prior=${DAGGER_ALPHA_PRIOR} → directly trains deployment path"
echo ""

CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/rsl_rl/train_latent_v2_online.py \
    --task ${TASK} \
    --motion_path ${MOTION_PATH} \
    --load_run "${TEACHER_RUN}" \
    --checkpoint ${TEACHER_CKPT} \
    --num_envs ${DAGGER_ENVS} \
    --num_iters ${DAGGER_ITERS} \
    --steps_per_iter ${DAGGER_STEPS_PER_ITER} \
    --updates_per_iter ${DAGGER_UPDATES_PER_ITER} \
    --z_dim ${DAGGER_Z_DIM} \
    --beta ${DAGGER_BETA} \
    --alpha_prior ${DAGGER_ALPHA_PRIOR} \
    --decoder_obs_mode task_features \
    --output_path ${DAGGER_OUTPUT} \
    --device cuda:0 \
    --headless

echo ""
echo "✓ DAgger complete: ${DAGGER_OUTPUT}"
echo ""

# ── Stage 2: Latent PPO (zero tracking rewards) ──────────────────────────
echo "▶ Stage 2/2: Latent PPO (zero tracking rewards, frozen decoder)"
echo ""

CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/rsl_rl/train_latent_v2_ppo.py \
    --task ${TASK} \
    --motion_path ${MOTION_PATH} \
    --latent_model ${DAGGER_OUTPUT} \
    --lab_scale ${PPO_LAB_SCALE} \
    --num_envs ${PPO_ENVS} \
    --max_iterations ${PPO_ITERS} \
    --policy_obs_mode task_features \
    --zero_tracking_rewards \
    --run_name ${PPO_RUN_NAME} \
    --headless

echo ""
echo "============================================================"
echo "  ✓ Pipeline complete!"
echo "  DAgger model:  ${DAGGER_OUTPUT}"
echo "  PPO logs:      logs/rsl_rl/g1_flat/*_${PPO_RUN_NAME}"
echo "============================================================"
