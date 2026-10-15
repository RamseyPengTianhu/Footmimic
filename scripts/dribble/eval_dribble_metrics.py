"""Evaluate the dribbling pass criteria from a probe_policy_foot_ball.py npz (first episode per env).

Metrics (agreed before any V1 training):
  * clip_complete : fraction of envs whose episode reached the clip end (not a fall / lost ball)
  * touches       : mean number of ball touches per episode.  A touch = a step where the ball's
                    XY speed jumps by > 0.3 m/s while the right ankle is within 0.30 m of it
                    (touches closer than 5 steps are merged).
  * ball_lost     : fraction of episodes ended by the ball_lost termination
Usage: python scripts/dribble/eval_dribble_metrics.py probe.npz [more.npz ...]
"""
import collections
import sys

import numpy as np

for path in sys.argv[1:]:
    z = np.load(path, allow_pickle=True)
    al, bs, dr = z["ts_alive"], z["ts_ball_speed"], z["ts_dist_r"]
    n = al.shape[1]
    reasons = z["done_reason"].tolist()
    cnt = collections.Counter(reasons)
    complete = np.mean([("motion_clip_end" in r) or (r == "" and z["done_frame"][i] < 0) for i, r in enumerate(reasons)])
    lost = np.mean(["ball_lost" in r for r in reasons])
    fell = np.mean([("anchor_pos_z" in r) or ("anchor_ori" in r) for r in reasons])
    touches = []
    for i in range(n):
        a = al[:, i]
        T = int(a.sum())
        dv = np.diff(bs[:T, i], prepend=bs[0, i])
        near = dr[:T, i] < 0.30
        hit = np.flatnonzero((dv > 0.3) & near)
        merged = []
        for h in hit:
            if not merged or h - merged[-1] >= 5:
                merged.append(h)
        touches.append(len(merged))
    touches = np.array(touches)
    done = z["done_frame"]
    print(f"{path.split('/')[-1]}: envs={n} clip_complete={complete:.2f} fell={fell:.2f} ball_lost={lost:.2f} "
          f"touches/ep mean={touches.mean():.2f} median={np.median(touches):.0f} max={touches.max()} "
          f"| episode end frame median={int(np.median(done[done >= 0])) if (done >= 0).any() else 'n/a'} | reasons={dict(cnt)}")
