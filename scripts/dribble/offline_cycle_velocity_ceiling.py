"""Ceiling of the *cycle-averaged* velocity-follow score on the references (no sim).

Per window k -> k+L the ball's AVERAGE velocity (anchor displacement / time) is compared with the AVERAGE commanded
body velocity over the same window, both in the pelvis-yaw frame at touch k.  L is the number of touches in one full
contact-surface cycle (Reference 1: out-out-in-in = 4, Reference 2: in-out = 2); L=1 reproduces the single-interval
version.  The score is also reported split into the component along the command (strict) and the lateral one (loose).

  score_full  = exp(-|v_ball_avg - v_cmd_avg| / 0.5)
  score_along = exp(-|along error| / 0.5),   score_lat = exp(-|lateral error| / 1.0)

NOTE: the reference ball path is synthetic (constant speed between anchors) and has no rolling resistance, so these
numbers must be re-measured once the main line's resistance is in the loop.
Usage: python scripts/dribble/offline_cycle_velocity_ceiling.py
"""
import numpy as np

REFS = (("Ref 1 (out-out-in-in)", "motions/master-aligned/master_001675_001884_unitree_g1.npz", 4),
        ("Ref 2 grounded (in-out)", "motions/master-annotated-003178-grounded/master_003178_003326_unitree_g1.npz", 2))


def windows(path, L):
    z = np.load(path, allow_pickle=True)
    fps = int(np.asarray(z["fps"]).reshape(-1)[0]); dt = 1.0 / fps
    c = z["dribble_cg_contact"] > 0
    ev = np.flatnonzero(c & ~np.r_[False, c[:-1]])
    ball = z["ball_pos_w"][:, :2]; vel = z["body_lin_vel_w"][:, 0, :2]
    w, x, y, zz = z["body_quat_w"][:, 0].T
    yaw = np.arctan2(2 * (w * zz + x * y), 1 - 2 * (y * y + zz * zz))
    out = []
    for k in range(len(ev) - L):
        a, b = ev[k], ev[k + L]
        c_, s_ = np.cos(yaw[a]), np.sin(yaw[a]); R = np.array([[c_, s_], [-s_, c_]])
        T = (b - a) * dt
        vb = R @ (ball[b] - ball[a]) / T; vc = R @ vel[a:b].mean(axis=0)
        u = vc / np.linalg.norm(vc); n = np.array([-u[1], u[0]]); e = vb - vc
        out.append((np.linalg.norm(e), abs(e @ u), abs(e @ n), np.linalg.norm(vb), np.linalg.norm(vc)))
    return np.array(out)


print(f"{'reference / window':34s} {'n':>3s} | {'|v_ball|':>8s} {'|v_cmd|':>8s} | {'err full':>8s} {'along':>6s} {'lateral':>7s} | {'score full':>10s} {'along(0.5)':>10s} {'lat(1.0)':>8s}")
for name, path, L in REFS:
    for LL in (1, L):
        r = windows(path, LL)
        label = f"{name}, L={LL}" + (" (single interval)" if LL == 1 else " (full cycle)")
        print(f"{label:34s} {len(r):3d} | {r[:,3].mean():8.2f} {r[:,4].mean():8.2f} | {r[:,0].mean():8.2f} {r[:,1].mean():6.2f} {r[:,2].mean():7.2f} | "
              f"{np.exp(-r[:,0]/0.5).mean():10.2f} {np.exp(-r[:,1]/0.5).mean():10.2f} {np.exp(-r[:,2]/1.0).mean():8.2f}")
