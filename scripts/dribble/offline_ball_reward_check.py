"""Score the command-defined ball rewards on saved probe rollouts (no sim).

Reconstructs, per env/frame, the pelvis-yaw frame (yaw = fitted reference->robot rotation +
reference yaw), the command (reference pelvis velocity in the reference yaw frame) and the ball
state, then evaluates front-zone and velocity-follow rewards.  The 'reference upper bound' row
feeds the reference's own pelvis/ball, i.e. the best any policy could score.

Usage: python scripts/dribble/offline_ball_reward_check.py <probe.npz> [...]
"""
import importlib.util
import os
import sys

import numpy as np
import torch

here = os.path.dirname(__file__)
spec = importlib.util.spec_from_file_location("brm", os.path.join(here, "..", "..", "source", "whole_body_tracking", "soccer", "tasks", "tracking", "mdp", "ball_reward_math.py"))
brm = importlib.util.module_from_spec(spec); spec.loader.exec_module(brm)

DT = 0.02
m = np.load("motions/master-aligned/master_001675_001884_unitree_g1.npz", allow_pickle=True)
refp = m["body_pos_w"][:, 0, :2]; refb = m["ball_pos_w"][:, :2]
w, x, y, z = m["body_quat_w"][:, 0].T
refyaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
refv_raw = m["body_lin_vel_w"][:, 0, :2].astype(np.float64)   # what the env feeds the actor (reference mode)
MIN_CMD = float(os.environ.get("MIN_CMD", "0"))                    # only score frames with |command| above this
SMOOTH = int(os.environ.get("CMD_SMOOTH_FRAMES", "0"))              # half-window in frames; 0 = raw per-frame command


def _smooth(v, k):
    if k <= 0:
        return v
    pad = np.pad(v, ((k, k), (0, 0)), mode="edge")
    c = np.cumsum(np.vstack([np.zeros((1, 2)), pad]), axis=0)
    return (c[2 * k + 1:] - c[:-(2 * k + 1)]) / (2 * k + 1)


refv = _smooth(refv_raw, SMOOTH)


def rot(th):  # world->local rotation matrices for yaw th (shape [...,2,2])
    c, s = np.cos(th), np.sin(th)
    return np.stack([np.stack([c, s], -1), np.stack([-s, c], -1)], -2)


def score(ball_local, ballv_local, cmd_local):
    keep = np.linalg.norm(cmd_local, axis=1) > MIN_CMD
    ball_local, ballv_local, cmd_local = ball_local[keep], ballv_local[keep], cmd_local[keep]
    zr = brm.front_zone_reward(torch.as_tensor(ball_local, dtype=torch.float32)).numpy()
    vr = brm.velocity_follow_reward(torch.as_tensor(ballv_local, dtype=torch.float32), torch.as_tensor(cmd_local, dtype=torch.float32)).numpy()
    return zr, vr


# reference upper bound (robot == reference, ball == reference ball)
f = np.arange(5, len(refp))
bv = np.vstack([np.zeros((1, 2)), np.diff(refb, axis=0) / DT])
R = rot(refyaw)
bl = np.einsum("fij,fj->fi", R, refb - refp); bvl = np.einsum("fij,fj->fi", R, bv); cl = np.einsum("fij,fj->fi", R, refv)
zr, vr = score(bl[f], bvl[f], cl[f])
print(f"[command smoothing half-window = {SMOOTH} frames]")
print(f"{'reference upper bound':34s} zone {zr.mean():.2f}  vel-follow {vr.mean():.2f}  product {np.mean(zr*vr):.2f}   (ball speed {np.linalg.norm(bv[f],axis=1).mean():.2f} m/s)")

for path in sys.argv[1:]:
    z_ = np.load(path, allow_pickle=True)
    al, rxy, bxy, fr = z_["ts_alive"], z_["ts_root_xy"], z_["ts_ball_xy"], z_["ts_frame"]
    n = al.shape[1]; Z = []; V = []; P = []; bs = []
    for i in range(n):
        k = int(al[:, i].sum())
        if k < 70: continue
        r = rxy[:k, i]; b = bxy[:k, i]
        a = r[60] - r[5]; bb = refp[60] - refp[5]; th = np.arctan2(a[1], a[0]) - np.arctan2(bb[1], bb[0])
        fi = np.clip(fr[:k, i].astype(int), 0, len(refp) - 1)
        yaw = th + refyaw[fi]; Rm = rot(yaw)
        bvw = np.vstack([np.zeros((1, 2)), np.diff(b, axis=0) / DT])
        blc = np.einsum("fij,fj->fi", Rm, b - r); bvc = np.einsum("fij,fj->fi", Rm, bvw); cc = np.einsum("fij,fj->fi", rot(refyaw[fi]), refv[fi])
        sl = slice(5, min(k, 120))
        zr, vr = score(blc[sl], bvc[sl], cc[sl]); Z.append(zr.mean()); V.append(vr.mean()); P.append((zr * vr).mean()); bs.append(np.linalg.norm(bvw[sl], axis=1).mean())
    print(f"{os.path.basename(path):34s} zone {np.mean(Z):.2f}  vel-follow {np.mean(V):.2f}  product {np.mean(P):.2f}   (ball speed {np.mean(bs):.2f} m/s, envs {len(Z)})")
