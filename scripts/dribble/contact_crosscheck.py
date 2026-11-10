"""Cross-check of the ball-contact sensor against ball velocity jumps in capability_rollout.py logs.

A sensor event (any link force > 1 N, rising edge) is matched to a ball speed jump (|dv| > 0.3 m/s in one control
step) within 2 steps.  Unmatched jumps are touches the sensor missed (or ball-ground bounces); unmatched sensor
events are gentle touches that do not change the ball velocity much.
  python scripts/dribble/contact_crosscheck.py --out out.json a.npz b.npz ...
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import capability_analysis as ca  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--out", required=True)
a = ap.parse_args()
res = {}
for f in a.files:
    d = ca.load(f)
    agree = sensor_only = vel_only = 0
    for i in range(d["frame"].shape[0]):
        e = ca.episode(d, i)
        n = e["steps"]
        dv = np.linalg.norm(np.diff(d["ball_vel"][i, :n, :2], axis=0), axis=1)
        jev = ca.events(dv > 0.3)
        sens = [s for s, _, _ in e["_fev"]] + [s for s, _, _ in e["_nev"]]
        used = set()
        for s, _ in jev:
            m = [k for k, x in enumerate(sens) if abs(x - s) <= 2 and k not in used]
            if m:
                agree += 1
                used.add(m[0])
            else:
                vel_only += 1
        sensor_only += len(sens) - len(used)
    res[os.path.basename(f)] = {"matched": agree, "sensor_only": sensor_only, "velocity_jump_only": vel_only}
json.dump({"rule": __doc__.strip().splitlines()[2], "result": res}, open(a.out, "w"), indent=1)
print(json.dumps(res, indent=1))
