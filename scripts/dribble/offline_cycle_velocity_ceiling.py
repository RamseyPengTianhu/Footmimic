"""Ceiling of the *cycle-averaged* velocity-follow score on the references (no sim).

Main-line suggestion: instead of instantaneous ball speed vs command, compare, per touch interval k -> k+1,
the ball's AVERAGE velocity (displacement / time between the two anchors) with the AVERAGE commanded body
velocity over the same interval.  Both are expressed in the pelvis-yaw frame at touch k.  This removes the
within-interval speed fluctuation that capped the instantaneous score at ~0.46.

  score = exp(-|v_ball_avg - v_cmd_avg| / std),  std = 0.5 m/s
NOTE: the reference ball path is synthetic (constant speed between anchors) and has no rolling resistance, so
these numbers must be re-measured once the main line's resistance is in the loop.
"""
import numpy as np

STD = 0.5
for name, path in (("Ref 1", "motions/master-aligned/master_001675_001884_unitree_g1.npz"),
                   ("Ref 2 (grounded)", "motions/master-annotated-003178-grounded/master_003178_003326_unitree_g1.npz")):
    z = np.load(path, allow_pickle=True)
    fps = int(np.asarray(z["fps"]).reshape(-1)[0]); dt = 1.0 / fps
    c = z["dribble_cg_contact"] > 0
    ev = np.flatnonzero(c & ~np.r_[False, c[:-1]])
    pel = z["body_pos_w"][:, 0, :2]; ball = z["ball_pos_w"][:, :2]; vel = z["body_lin_vel_w"][:, 0, :2]
    q = z["body_quat_w"][:, 0]; w, x, y, zz = q.T
    yaw = np.arctan2(2 * (w * zz + x * y), 1 - 2 * (y * y + zz * zz))
    rows = []
    for a, b in zip(ev[:-1], ev[1:]):
        c_, s_ = np.cos(yaw[a]), np.sin(yaw[a])
        R = np.array([[c_, s_], [-s_, c_]])                      # world -> pelvis-yaw frame at touch a
        T = (b - a) * dt
        v_ball = R @ (ball[b] - ball[a]) / T
        v_cmd = R @ vel[a:b].mean(axis=0)
        rows.append((np.linalg.norm(v_ball - v_cmd), np.linalg.norm(v_ball), np.linalg.norm(v_cmd)))
    r = np.array(rows)
    sc = np.exp(-r[:, 0] / STD)
    print(f"{name}: {len(r)} intervals | |v_ball_avg| {r[:,1].mean():.2f} m/s, |v_cmd_avg| {r[:,2].mean():.2f} m/s | velocity error mean {r[:,0].mean():.2f} m/s "
          f"| cycle-averaged score mean {sc.mean():.2f} (min {sc.min():.2f}, max {sc.max():.2f})")
