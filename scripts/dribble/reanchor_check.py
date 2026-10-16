"""Offline check: episode-fixed vs per-cycle re-anchored world targets, on real rollouts (no sim).

Targets are fixed in the world by mapping the reference into the robot's world with a rigid transform A.  If A is fitted
once per episode (A0, 'fixed'), the robot's tracking drift accumulates in the target-vs-robot offset.  If A is refitted at
every contact-surface cycle boundary (A_j, 're-anchored', NOT every step: within a cycle the targets stay immutable),
the offset only contains the drift accumulated since the last boundary.

For touch k the reported error is |A(ref_pelvis[ev_k]) - robot_root[ev_k]|, i.e. how far the robot is from where the mapped
reference says it should be (the target-relative error: the same vector displaces every target of that touch).
  fixed       : A0 fitted on frames 5..60 of the episode (yaw from the displacement direction, translation at frame 5)
  re-anchored : A_j keeps A0's yaw and is translated so the reference pelvis at the last boundary j (event index multiple
                of L, <= k) lands on the robot root there.  Yaw is NOT refitted per cycle (the probe does not log the yaw);
                re-anchoring yaw as well can only help.

Usage: python scripts/dribble/reanchor_check.py --ref <clip.npz> --cycle 4 probe1.npz [probe2.npz ...]
"""
import argparse

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--ref", required=True)
ap.add_argument("--cycle", type=int, required=True)
args = ap.parse_args()

ref = np.load(args.ref, allow_pickle=True)
refp = ref["body_pos_w"][:, 0, :2]
c = ref["dribble_cg_contact"] > 0
ev = np.flatnonzero(c & ~np.r_[False, c[:-1]])


def rot(th):
    cc, ss = np.cos(th), np.sin(th)
    return np.array([[cc, -ss], [ss, cc]])


for path in args.files:
    z = np.load(path, allow_pickle=True)
    al, rxy, fr = z["ts_alive"], z["ts_root_xy"], z["ts_frame"]
    n = al.shape[1]
    fixed = {k: [] for k in range(len(ev))}
    anch = {k: [] for k in range(len(ev))}
    for i in range(n):
        T = int(al[:, i].sum())
        if T < 70:
            continue
        r = rxy[:T, i]; f = np.clip(fr[:T, i].astype(int), 0, len(refp) - 1)
        a = r[60] - r[5]; b = refp[f[60]] - refp[f[5]]
        th = np.arctan2(a[1], a[0]) - np.arctan2(b[1], b[0]); R = rot(th)
        A0 = lambda p: (p - refp[f[5]]) @ R.T + r[5]
        at_frame = {int(f[t]): t for t in range(T)}
        for k, e in enumerate(ev):
            if e not in at_frame:
                continue
            t = at_frame[e]
            fixed[k].append(np.linalg.norm(r[t] - A0(refp[e])))
            j = (k // args.cycle) * args.cycle
            if j == k or ev[j] not in at_frame:
                anch[k].append(0.0 if j == k else np.nan)
                continue
            tj = at_frame[int(ev[j])]
            Aj = lambda p: (p - refp[ev[j]]) @ R.T + r[tj]
            anch[k].append(np.linalg.norm(r[t] - Aj(refp[e])))
    print(f"== {path.split('/')[-1]} (cycle {args.cycle}); error in m, median over envs [n envs]")
    print("   touch k : fixed | re-anchored (since last cycle boundary)")
    for k in range(len(ev)):
        if len(fixed[k]) >= 3:
            print(f"   {k+1:3d}    : {np.median(fixed[k]):.2f} [{len(fixed[k]):2d}] | {np.nanmedian(anch[k]):.2f}")
