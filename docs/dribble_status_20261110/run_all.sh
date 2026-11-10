#!/bin/bash
# All capability rollouts (no learning). 4 in parallel.
cd /home/teambruce/users/syf/projects/Footmimic
PY=/home/teambruce/users/syf/conda_envs/isaaclab45_211/bin/python
OUT=logs/dribble_status_20261110/raw
R1=motions/master-aligned/master_001675_001884_unitree_g1.npz
R2=motions/master-annotated-003178-grounded/master_003178_003326_unitree_g1.npz
TEACH="--load_run 2026-10-19_12-53-51_teacher_mix_2048 --checkpoint model_10998.pt"
S1="--load_run 2026-08-21_16-27-03_s1_clean_v0_random_bounded_ghost_1024 --checkpoint model_6999.pt"
jobs_list=()
for C in teacher s1; do
  [ $C = teacher ] && CK="$TEACH" || CK="$S1"
  for cond in ghost_follow ghost_static real real_no_resist hold_ahead reset_after_touch; do
    jobs_list+=("$CK --condition $cond --motion_path $R1 --output $OUT/${C}_ref1_${cond}.npz")
  done
  for off in 0.05 0.1 0.2; do
    jobs_list+=("$CK --condition offset --offset $off --motion_path $R1 --output $OUT/${C}_ref1_offset${off}.npz")
  done
done
for cond in ghost_follow real hold_ahead reset_after_touch; do
  jobs_list+=("$TEACH --condition $cond --motion_path $R2 --output $OUT/teacher_ref2_${cond}.npz")
done
i=0
for j in "${jobs_list[@]}"; do
  name=$(echo $j | sed 's/.*--output //;s/.npz//;s#.*/##')
  ( timeout 1500 $PY scripts/dribble/capability_rollout.py $j --num_envs 128 --headless > $OUT/$name.log 2>&1; echo "DONE $name exit=$?" ) &
  i=$((i+1)); if [ $((i % 4)) -eq 0 ]; then wait; fi
done
wait
echo ALL_DONE
