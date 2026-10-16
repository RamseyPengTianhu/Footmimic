"""Build a looped dribbling clip from one or more whole contact-surface cycles of an annotated clip.

The segment [a, b) runs from touch event `i` to touch event `i+L` (L = touches per full surface cycle), so frame b is
the same phase as frame a.  Repeat r is the segment placed in the world by G^r, where G is the rigid yaw+XY transform
taking the pelvis pose at frame a onto the pelvis pose at frame b (heading and position continuous at every seam).
Limbs are only approximately periodic, so over the first `blend` frames of every repeat after the first the output
cross-fades (cosine) from the clip's true continuation (original frames b.., placed by G^(r-1)) into the repeat's own start.

Per-frame arrays are copied from the source frame; body poses/velocities, the ball and the flow anchor-frame index are
transformed.  Contact labels come from the repeat, so the looped clip has K*L touches.

  python scripts/dribble/loop_clip.py --input <npz> --start_event 6 --cycle 4 --repeats 6 --output <npz>
"""
import argparse
import os

import numpy as np


def yaw_of(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def qmul(a, b):
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], axis=-1)


def rot2(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True); ap.add_argument("--output", required=True)
    ap.add_argument("--start_event", type=int, required=True); ap.add_argument("--cycle", type=int, required=True)
    ap.add_argument("--repeats", type=int, default=6); ap.add_argument("--blend", type=int, default=10)
    a_ = ap.parse_args()
    z = np.load(a_.input, allow_pickle=True)
    T = len(z["body_pos_w"])
    c = z["dribble_cg_contact"] > 0
    ev = np.flatnonzero(c & ~np.r_[False, c[:-1]])
    a, b = int(ev[a_.start_event]), int(ev[a_.start_event + a_.cycle])
    n, M = b - a, a_.blend
    assert b + M <= T, "not enough frames after the segment for the cross-fade"
    pel = z["body_pos_w"][:, 0]; yaw = yaw_of(z["body_quat_w"][:, 0])
    dpsi = float(np.arctan2(np.sin(yaw[b] - yaw[a]), np.cos(yaw[b] - yaw[a])))
    R = rot2(dpsi); pa, pb = pel[a, :2], pel[b, :2]

    def G(xy, r):                      # apply the seam transform r times to world XY points
        for _ in range(r):
            xy = (xy - pa) @ R.T + pb
        return xy

    def place(frame_idx, r):           # transformed per-frame body data for source frames `frame_idx` under G^r
        P = z["body_pos_w"][frame_idx].copy(); Q = z["body_quat_w"][frame_idx].copy()
        V = z["body_lin_vel_w"][frame_idx].copy(); W = z["body_ang_vel_w"][frame_idx].copy()
        B = z["ball_pos_w"][frame_idx].copy()
        if r:
            Rr = rot2(dpsi * r); qz = np.array([np.cos(dpsi * r / 2), 0, 0, np.sin(dpsi * r / 2)])
            P[..., :2] = G(P[..., :2].reshape(-1, 2), r).reshape(P[..., :2].shape)
            B[..., :2] = G(B[..., :2].reshape(-1, 2), r).reshape(B[..., :2].shape)
            Q = qmul(qz, Q)
            V[..., :2] = V[..., :2] @ Rr.T; W[..., :2] = W[..., :2] @ Rr.T
        return dict(body_pos_w=P, body_quat_w=Q, body_lin_vel_w=V, body_ang_vel_w=W, ball_pos_w=B)

    out = {k: [] for k in z.files if z[k].ndim >= 1 and z[k].shape[0] == T}
    for r in range(a_.repeats):
        src = np.arange(a, b)
        placed = place(src, r)
        joints = {k: z[k][src].astype(np.float64) for k in ("joint_pos", "joint_vel")}
        if r > 0:                      # cross-fade from the true continuation of repeat r-1
            cont = place(np.arange(b, b + M), r - 1)
            cj = {k: z[k][np.arange(b, b + M)].astype(np.float64) for k in ("joint_pos", "joint_vel")}
            w = 0.5 * (1 - np.cos(np.pi * (np.arange(M) + 1) / (M + 1)))
            for k in placed:
                shape = (M,) + (1,) * (placed[k].ndim - 1)
                if k == "body_quat_w":
                    q = (1 - w).reshape(shape) * cont[k] + w.reshape(shape) * placed[k][:M]
                    placed[k][:M] = q / np.linalg.norm(q, axis=-1, keepdims=True)
                else:
                    placed[k][:M] = (1 - w).reshape(shape) * cont[k] + w.reshape(shape) * placed[k][:M]
            for k in joints:
                joints[k][:M] = (1 - w)[:, None] * cj[k] + w[:, None] * joints[k][:M]
        for k in out:
            if k in placed:
                out[k].append(placed[k])
            elif k in joints:
                out[k].append(joints[k].astype(z[k].dtype))
            elif k == "dribble_cg_flow_anchor_frame":
                v = z[k][src].copy(); v[v >= 0] = v[v >= 0] - a + r * n; out[k].append(v)
            else:
                out[k].append(z[k][src])
    res = {k: np.concatenate(v, axis=0).astype(z[k].dtype) for k, v in out.items()}
    for k in z.files:
        if k not in res:
            res[k] = z[k]
    # The loader requires the final contact to have no outgoing flow segment (same convention as the source clips).
    cl = res["dribble_cg_contact"] > 0
    last = int(np.flatnonzero(cl & ~np.r_[False, cl[:-1]])[-1])
    if "dribble_cg_flow_valid" in res:
        res["dribble_cg_flow_valid"][last:] = 0
        res["dribble_cg_flow_dir_local"][last:] = 0.0
        res["dribble_cg_flow_distance"][last:] = -1.0
        res["dribble_cg_flow_duration"][last:] = -1.0
        res["dribble_cg_flow_anchor_frame"][last:] = -1
    os.makedirs(os.path.dirname(os.path.abspath(a_.output)), exist_ok=True)
    np.savez(a_.output, **res)
    cc = res["dribble_cg_contact"] > 0
    nev = int((cc & ~np.r_[False, cc[:-1]]).sum())
    print(f"segment frames {a}->{b} ({n}), yaw gain {np.degrees(dpsi):.1f} deg, pelvis step {np.linalg.norm(pb - pa):.2f} m; "
          f"looped clip {len(res['body_pos_w'])} frames ({len(res['body_pos_w'])/50:.1f} s), {nev} touches -> {a_.output}")


if __name__ == "__main__":
    main()
