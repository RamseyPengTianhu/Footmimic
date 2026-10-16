"""Synthetic check of eval_d3_acceptance.py: builds an npz with a known number of touches and a carried ball."""
import os
import subprocess
import sys
import tempfile

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
T, n = 400, 6
z = np.zeros
ts_alive = np.ones((T, n), bool)
speed = z((T, n)); dist = np.full((T, n), 0.5)
# envs 0-2: 22 kicks (every 15 steps; ball near the ankle only for the first 2 steps), speed jumps 1.0 then decays; env 3: only 5 kicks; env 4: ball carried 1.5 s; env 5: lost at step 100
for i in range(3):
    for k in range(22):
        t0 = 5 + 15 * k
        speed[t0:t0 + 15, i] = np.linspace(1.0, 0.4, 15); dist[t0:t0 + 2, i] = 0.12
for k in range(5):
    t0 = 5 + 15 * k; speed[t0:t0 + 15, 3] = np.linspace(1.0, 0.4, 15); dist[t0:t0 + 2, 3] = 0.12
speed[50:130, 4] = 1.0; dist[50:130, 4] = 0.12
ts_alive[100:, 5] = False
reasons = np.array(["motion_clip_end"] * 5 + ["ball_lost"])
root = np.cumsum(np.tile([0.025, 0.0], (T, 1)), axis=0)[:, None, :].repeat(n, 1)
ball = root + np.array([0.4, 0.0])
d = dict(ts_alive=ts_alive, ts_ball_speed=speed, ts_dist_r=dist, ts_root_xy=root, ts_ball_xy=ball,
         ts_frame=np.tile(np.arange(T)[:, None], (1, n)), done_reason=reasons, done_frame=np.full(n, -1))
with tempfile.TemporaryDirectory() as tmp:
    p = os.path.join(tmp, "syn.npz"); np.savez(p, **d)
    out = subprocess.run([sys.executable, os.path.join(here, "eval_d3_acceptance.py"), p, "--n_touch", "20", "--devs", "8", "16"],
                         capture_output=True, text=True).stdout
print(out)
ok = ("gate (>= 20 touches, no loss/fall): 0.50" in out) and ("max 22" in out) and ("carry_suspect (heuristic): 0.17" in out)
print("PASS" if ok else "FAIL: expected gate 0.50 (3 of 6 envs), max 22 touches, carry_suspect 0.17 (1 of 6)")
raise SystemExit(0 if ok else 1)
