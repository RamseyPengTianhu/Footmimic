"""Extract the contact clock (when / which foot / which surface) from annotated dribble clips.

The clock is pure human-motion information: it says *when* a touch happens in clip time and with
which foot and instep surface, never *where the ball goes*.  Output per clip (JSON):

  events[]        : id, start/end frame, window_start/window_end (+/- tolerance), t_start (s), foot, surface,
                    interval_to_next (frames / s), pelvis_speed (m/s) at the touch
  sequence        : the (foot, surface) pattern, e.g. "R-out R-out R-in R-in ..."
  period          : mean / std / min / max inter-touch interval, touches per second, detected repeating unit
  per_frame       : next_event_id, frames_to_next, next_foot, next_surface (-1 = none) for every frame

Usage: python scripts/dribble/extract_contact_clock.py --input <npz|dir> --output-dir <dir> [--tolerance 2]
(The main line's scripts/motion_gen/contact_anchors.py was not available on this branch; the schema here is
deliberately small so it can be mapped onto it.)
"""
import argparse
import glob
import json
import os

import numpy as np

FOOT = {0: "L", 1: "R"}
SURF = {0: "in", 1: "out"}


def segments(mask):
    mask = np.asarray(mask).reshape(-1).astype(bool)
    st = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    en = np.flatnonzero(mask & ~np.r_[mask[1:], False])
    return list(zip(st.tolist(), en.tolist()))


def repeating_unit(seq, max_len=8):
    for L in range(1, min(max_len, len(seq) // 2) + 1):
        if len(seq) % L == 0 and all(seq[i] == seq[i % L] for i in range(len(seq))):
            return seq[:L]
    return None


def extract(path, tol):
    z = np.load(path, allow_pickle=True)
    fps = int(np.asarray(z["fps"]).reshape(-1)[0])
    n = len(z["body_pos_w"])
    segs = segments(z["dribble_cg_contact"] > 0)
    if not segs:
        return None
    foot, surf = z["dribble_cg_foot"], z["dribble_cg_surface"]
    spd = np.linalg.norm(z["body_lin_vel_w"][:, 0, :2], axis=1)
    ev = []
    for i, (a, b) in enumerate(segs):
        nxt = segs[i + 1][0] if i + 1 < len(segs) else None
        ev.append({
            "id": i, "start": a, "end": b, "window_start": max(0, a - tol), "window_end": min(n - 1, b + tol),
            "t_start": round(a / fps, 4), "foot": FOOT.get(int(foot[a]), "?"), "surface": SURF.get(int(surf[a]), "?"),
            "interval_to_next_frames": None if nxt is None else nxt - a,
            "interval_to_next_s": None if nxt is None else round((nxt - a) / fps, 4),
            "pelvis_speed": round(float(spd[a:b + 1].mean()), 3),
        })
    seq = [f"{e['foot']}-{e['surface']}" for e in ev]
    iv = np.array([e["interval_to_next_frames"] for e in ev if e["interval_to_next_frames"] is not None], dtype=float)
    unit = repeating_unit(seq)
    nid = np.full(n, -1); ttn = np.full(n, -1); nf = np.full(n, -1); ns = np.full(n, -1)
    for e in ev[::-1]:
        nid[:e["start"] + 1] = e["id"]; ttn[:e["start"] + 1] = e["start"] - np.arange(e["start"] + 1)
        nf[:e["start"] + 1] = 1 if e["foot"] == "R" else 0; ns[:e["start"] + 1] = 0 if e["surface"] == "in" else 1
    return {
        "source": path, "fps": fps, "num_frames": n, "tolerance_frames": tol, "events": ev, "sequence": seq,
        "period": {
            "num_events": len(ev),
            "interval_frames": {"mean": float(iv.mean()), "std": float(iv.std()), "min": float(iv.min()), "max": float(iv.max())} if len(iv) else None,
            "touches_per_s": float(len(ev) / ((ev[-1]["start"] - ev[0]["start"]) / fps)) if len(ev) > 1 else None,
            "repeating_unit": unit, "feet": sorted(set(e["foot"] for e in ev)),
        },
        "per_frame": {"next_event_id": nid.tolist(), "frames_to_next": ttn.tolist(), "next_foot": nf.tolist(), "next_surface": ns.tolist()},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True); ap.add_argument("--output-dir", required=True); ap.add_argument("--tolerance", type=int, default=2)
    a = ap.parse_args()
    files = [a.input] if os.path.isfile(a.input) else sorted(glob.glob(os.path.join(a.input, "**", "*.npz"), recursive=True))
    os.makedirs(a.output_dir, exist_ok=True)
    for p in files:
        c = extract(p, a.tolerance)
        if c is None:
            print(f"skip (no contact labels): {p}"); continue
        out = os.path.join(a.output_dir, os.path.basename(p).replace(".npz", ".contact_clock.json"))
        json.dump(c, open(out, "w"), indent=1)
        pr = c["period"]; iv = pr["interval_frames"]
        print(f"{os.path.basename(p)}: {pr['num_events']} events, feet {pr['feet']}, interval {iv['mean']:.1f}±{iv['std']:.1f} fr "
              f"({iv['mean']/c['fps']:.2f}s), {pr['touches_per_s']:.2f}/s, unit {pr['repeating_unit']}  -> {out}")


if __name__ == "__main__":
    main()
