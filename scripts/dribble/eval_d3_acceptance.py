"""D3 acceptance report from a probe_policy_foot_ball.py npz (first episode per env).

Main-line D3 gate: with an unchanged command, >= 80 % of episodes keep touching >= 20 times without losing the ball,
and the deviation shrinks touch by touch.  Reported here:

  gate_rate        : fraction of episodes with >= N_TOUCH touches and no ball_lost / fall   (gate: >= 0.80, N=20)
  touches          : per-episode touch-count distribution
  front_zone       : fraction of alive steps with the ball in the front zone (pelvis-yaw frame; x 0.15..0.70, |y|<=0.40)
  deviation@k      : at touch k (default 8, 16): robot XY position error vs the aligned reference clip, and the ball's
                     pelvis-frame XY error vs the reference ball.  The clip is aligned per env by a rigid yaw+translation
                     fit on frames 5..60 (first 1.2 s), so deviation grows only through drift after that.
  carry_suspect    : heuristic hack flag.  Fraction of episodes in which the ball stays within 0.18 m of an ankle with ball speed
                     > 0.3 m/s for > 0.5 s in a row, i.e. the ball is being pushed along rather than touched.  The probe's
                     contact-sensor reading is unreliable, so this is geometric and can false-positive.

A touch = a step where the ball's XY speed jumps by > 0.3 m/s while an ankle is within 0.30 m; touches closer than 5 steps
merge.  The probe only logs the RIGHT ankle, so left-foot touches are not seen (both references are right-footed).
NOTE the reference clips are 5-7 s with 8-16 touches, so N_TOUCH=20 cannot be met without a looped clip; use --n_touch.

Usage: python scripts/dribble/eval_d3_acceptance.py probe.npz [more.npz] [--n_touch 20] [--ref motions/...npz]
"""
import argparse
import collections

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--n_touch", type=int, default=20)
ap.add_argument("--ref", default="motions/master-aligned/master_001675_001884_unitree_g1.npz")
ap.add_argument("--devs", type=int, nargs="+", default=[8, 16])
args = ap.parse_args()

REF = np.load(args.ref, allow_pickle=True)
REF_P = REF["body_pos_w"][:, 0, :2]
REF_B = REF["ball_pos_w"][:, :2]
_w, _x, _y, _z = REF["body_quat_w"][:, 0].T
REF_YAW = np.arctan2(2 * (_w * _z + _x * _y), 1 - 2 * (_y * _y + _z * _z))


def rot(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]])


def touches_of(speed, dist, alive_n):
    dv = np.diff(speed[:alive_n], prepend=speed[0])
    hit = np.flatnonzero((dv > 0.3) & (dist[:alive_n] < 0.30))
    out = []
    for h in hit:
        if not out or h - out[-1] >= 5:
            out.append(int(h))
    return out


def front_zone(local):
    dx = np.clip(0.15 - local[:, 0], 0, None) + np.clip(local[:, 0] - 0.70, 0, None)
    dy = np.clip(np.abs(local[:, 1]) - 0.40, 0, None)
    return (dx == 0) & (dy == 0)


def longest_run(mask):
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0
        best = max(best, cur)
    return best


for path in args.files:
    z = np.load(path, allow_pickle=True)
    al, bs, dr = z["ts_alive"], z["ts_ball_speed"], z["ts_dist_r"]
    rxy, bxy, fr = z["ts_root_xy"], z["ts_ball_xy"], z["ts_frame"]
    reasons = z["done_reason"].tolist()
    n = al.shape[1]
    counts, gate, zone, carry = [], [], [], []
    devs = {k: ([], []) for k in args.devs}
    for i in range(n):
        T = int(al[:, i].sum())
        if T < 10:
            continue
        r, b = rxy[:T, i], bxy[:T, i]
        tl = touches_of(bs[:, i], dr[:, i], T)
        counts.append(len(tl))
        bad = ("ball_lost" in reasons[i]) or ("anchor_pos_z" in reasons[i]) or ("anchor_ori" in reasons[i])
        gate.append(len(tl) >= args.n_touch and not bad)
        if T > 60:
            a = r[60] - r[5]; bb = REF_P[60] - REF_P[5]
            th = np.arctan2(a[1], a[0]) - np.arctan2(bb[1], bb[0]); R = rot(th)
            fi = np.clip(fr[:T, i].astype(int), 0, len(REF_P) - 1)
            yaw = th + REF_YAW[fi]
            c, s = np.cos(yaw), np.sin(yaw)
            rel = b - r
            local = np.stack([c * rel[:, 0] + s * rel[:, 1], -s * rel[:, 0] + c * rel[:, 1]], 1)
            zone.append(front_zone(local).mean())
            near = (dr[:T, i] < 0.18) & (bs[:T, i] > 0.3)
            carry.append(longest_run(near) * 0.02 > 0.5)
            for k in args.devs:
                if len(tl) >= k:
                    t = tl[k - 1]
                    pred = (REF_P[fi[t]] - REF_P[0]) @ R.T + r[0]
                    devs[k][0].append(np.linalg.norm(r[t] - pred))
                    rb = (REF_B[fi[t]] - REF_P[fi[t]])
                    cr, sr = np.cos(REF_YAW[fi[t]]), np.sin(REF_YAW[fi[t]])
                    ref_local = np.array([cr * rb[0] + sr * rb[1], -sr * rb[0] + cr * rb[1]])
                    devs[k][1].append(np.linalg.norm(local[t] - ref_local))
    counts = np.array(counts)
    print(f"== {path.split('/')[-1]}: {len(counts)} envs, reasons {dict(collections.Counter(reasons))}")
    print(f"   gate (>= {args.n_touch} touches, no loss/fall): {np.mean(gate):.2f}   [need >= 0.80]")
    print(f"   touches/ep: mean {counts.mean():.2f} median {np.median(counts):.0f} p90 {np.percentile(counts, 90):.0f} max {counts.max()}")
    print(f"   ball in front zone: {np.mean(zone):.2f} of alive steps   carry_suspect (heuristic): {np.mean(carry):.2f}")
    for k in args.devs:
        d_r, d_b = devs[k]
        if d_r:
            print(f"   deviation @ touch {k}: n={len(d_r)}  robot pos error median {np.median(d_r):.2f} m | ball pelvis-frame error median {np.median(d_b):.2f} m")
        else:
            print(f"   deviation @ touch {k}: n/a (no episode reached {k} touches)")
