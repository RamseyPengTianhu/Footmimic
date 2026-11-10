#!/bin/bash
# 12 s videos of env 0 (teacher model_10998, Reference 1), overlay: ref frame, foot-touch count, ball speed, contacting link.
cd /home/teambruce/users/syf/projects/Footmimic
PY=/home/teambruce/users/syf/conda_envs/isaaclab45_211/bin/python
V=logs/dribble_status_20261110/videos
R1=motions/master-aligned/master_001675_001884_unitree_g1.npz
TEACH="--load_run 2026-10-19_12-53-51_teacher_mix_2048 --checkpoint model_10998.pt"
S1="--load_run 2026-08-21_16-27-03_s1_clean_v0_random_bounded_ghost_1024 --checkpoint model_6999.pt"
run() { name=$1; shift; timeout 1500 $PY scripts/dribble/capability_rollout.py "$@" --motion_path $R1 --num_envs 1 --video_steps 600 \
  --output $V/$name.npz --video $V/$name.mp4 --headless --enable_cameras > $V/$name.log 2>&1; echo "VIDEO $name exit=$?"; }
run teacher_ref1_ghost_follow $TEACH --condition ghost_follow &
run teacher_ref1_real $TEACH --condition real &
wait
run teacher_ref1_hold_ahead $TEACH --condition hold_ahead &
run s1_ref1_real $S1 --condition real &
wait
echo VIDEOS_DONE
