"""Inventory motion clips: length, speed, turning, vertical drift, label status.

Usage: python scripts/dribble/inventory_motions.py motions/<dir> [motions/<dir2> ...]
Flags clips whose pelvis height / lowest foot drift (the Reference-2 failure) or whose
sole never reaches the ground.  Feet are body indices 18 (L) / 19 (R) in the 30-body npz.
"""
import glob
import os
import sys

import numpy as np

print(f"{'clip':58s} {'fr':>4s} {'fps':>3s} {'spd':>5s} {'turn°':>5s} {'pelvZ(cm) first/last':>20s} {'footZ(cm) first/last':>20s} labels")
for d in sys.argv[1:]:
    for p in sorted(glob.glob(os.path.join(d, "**", "*.npz"), recursive=True)):
        try:
            z = np.load(p, allow_pickle=True)
            bp = z["body_pos_w"]; bq = z["body_quat_w"]
        except Exception as e:  # noqa: BLE001
            print(f"{p[-58:]:58s} unreadable ({type(e).__name__})"); continue
        fps = int(np.asarray(z["fps"]).reshape(-1)[0]) if "fps" in z.files else -1
        n = len(bp); k = max(1, n // 10)
        pel = bp[:, 0]
        spd = np.linalg.norm(np.diff(pel[:, :2], axis=0), axis=1).mean() * (fps if fps > 0 else 50)
        w, x, y, zz = bq[:, 0].T; yaw = np.unwrap(np.arctan2(2 * (w * zz + x * y), 1 - 2 * (y * y + zz * zz)))
        turn = np.degrees(abs(yaw[-1] - yaw[0]))
        foot = np.minimum(bp[:, 18, 2], bp[:, 19, 2]) if bp.shape[1] > 19 else np.full(n, np.nan)
        lab = "contact" if ("dribble_cg_contact" in z.files and (z["dribble_cg_contact"] > 0).any()) else "-"
        if "dribble_cg_contact" in z.files and lab == "contact":
            c = z["dribble_cg_contact"] > 0; lab = f"{int((c & ~np.r_[False, c[:-1]]).sum())} events"
        drift = (pel[-k:, 2].mean() - pel[:k, 2].mean()) * 100
        fdrift = (np.nanmin(foot[-k:]) - np.nanmin(foot[:k])) * 100
        flag = " DRIFT" if abs(drift) > 8 or abs(fdrift) > 6 else ""
        print(f"{p[-58:]:58s} {n:4d} {fps:3d} {spd:5.2f} {turn:5.0f} {pel[:k,2].mean()*100:9.0f}/{pel[-k:,2].mean()*100:<9.0f} {np.nanmin(foot[:k])*100:9.0f}/{np.nanmin(foot[-k:])*100:<9.0f} {lab}{flag}")
