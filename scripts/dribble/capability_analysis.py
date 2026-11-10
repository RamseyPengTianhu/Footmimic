"""Offline analysis of capability_rollout.py logs (numpy only; no Isaac).

  python scripts/dribble/capability_analysis.py summary  --out summary.json --csv summary.csv  a.npz b.npz ...
  python scripts/dribble/capability_analysis.py failures --out failures.json --csv failures.csv --episodes 20 real.npz

Definitions (all per env, first episode only, from frame 0):
  foot touch event   ball-sensor force from a foot link > 1 N; rising edge, gaps <= 2 steps merged
  non-foot touch     same, any other link
  ball lost          first step with ball-pelvis XY distance > 1.1 m (same threshold as the training termination)
  consecutive        foot touch events that start before the ball is lost (or the episode ends)
  labelled hit       labelled foot touches inside the labelled contact segment widened by 2 frames
  fall               end reason anchor_pos_z / anchor_ori;  ee_body_pos (foot/hand height error > 0.25 m) separately
  ball-command dev   |v_ball_xy - v_cmd_xy|, v_cmd = local [vx, vy] rotated by the robot pelvis yaw; per step and over
                     1 s windows (ball displacement / 1 s vs mean command); N/A when the ball is written every step
"""

import argparse
import csv
import json

import numpy as np

LOST_M = 1.1
MERGE_GAP = 2
TOL = 2
REACH_XY = 0.30  # reachable ball-ankle XY distance (reference at labelled contacts: 0.13-0.20 m; see SUMMARY item 4)
FALL = ("anchor_pos_z", "anchor_ori")


def yaw_of(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def to_local(vec_xy, yaw):
    c, s = np.cos(-yaw), np.sin(-yaw)
    return np.stack([c * vec_xy[..., 0] - s * vec_xy[..., 1], s * vec_xy[..., 0] + c * vec_xy[..., 1]], -1)


def to_world(vec_xy, yaw):
    return to_local(vec_xy, -yaw)


def events(mask):
    """Rising edges of a bool series with small gaps merged -> list of (start, end) steps (inclusive)."""
    ev, start, last = [], None, -10
    for t, v in enumerate(mask):
        if v:
            if start is None:
                start = t
            elif t - last > MERGE_GAP + 1:
                ev.append((start, last))
                start = t
            last = t
    if start is not None:
        ev.append((start, last))
    return ev


def segments(contact):
    on = contact > 0
    segs, start = [], None
    for f, v in enumerate(on):
        if v and start is None:
            start = f
        if not v and start is not None:
            segs.append((start, f - 1))
            start = None
    if start is not None:
        segs.append((start, len(on) - 1))
    return segs


def load(path):
    d = dict(np.load(path, allow_pickle=True))
    d["meta"] = json.loads(str(d["meta"]))
    return d


def episode(d, i):
    """Per-env first-episode quantities.  Force rows describe the state after step t (reference frame frame[t]+1)."""
    meta = d["meta"]
    T = d["frame"].shape[1]
    end = int(d["end_step"][i])
    n = T if end < 0 else end + 1
    reason = str(d["end_reason"][i]) if end >= 0 else "steps_exhausted"
    dt = meta["step_dt"]
    frame = d["frame"][i, :n]
    pel = d["pelvis_pos"][i, :n]
    yaw = yaw_of(d["pelvis_quat"][i, :n])
    ball = d["ball_pos"][i, :n]
    bvel = d["ball_vel"][i, :n]
    force = d["ball_force_link"][i, :n]
    thr = meta["contact_threshold_N"]
    foot_f = force[:, :2]
    foot_touch = (foot_f > thr).any(-1)
    nonfoot = (force[:, 2:] > thr).any(-1)
    fev = [(s, e, int(np.argmax(foot_f[s:e + 1].max(0)))) for s, e in events(foot_touch)]  # foot 0 left 1 right
    nev = [(s, e, d["link_names"][2 + int(np.argmax(force[s:e + 1, 2:].max(0)))]) for s, e in events(nonfoot)]
    dist = np.linalg.norm(ball[:, :2] - pel[:, :2], axis=-1)
    lost = np.nonzero(dist > LOST_M)[0]
    lost_step = int(lost[0]) if lost.size else None
    consecutive = sum(1 for s, _, _ in fev if lost_step is None or s < lost_step)
    segs = segments(d["ref_contact"])
    ref_foot = d["ref_foot"]
    hits = []
    for s, e in segs:
        lo, hi = s - TOL, e + TOL
        want = int(ref_foot[s])
        hits.append(any(lo <= frame[t] + 1 <= hi and foot_f[t, want] > thr for t in range(n)))
    cmd = d["cmd_local"][i, :n, :2]
    cmd_w = to_world(cmd, yaw)
    written = meta["condition"] in ("ghost_follow", "hold_ahead") or meta["condition"].startswith("ghost")
    dev_step = None if written else float(np.linalg.norm(bvel[:, :2] - cmd_w, axis=-1).mean())
    w = int(round(1.0 / dt))
    dev_win = None
    if not written and n > w:
        disp = (ball[w:, :2] - ball[:-w, :2]) / (w * dt)
        cw = np.stack([np.convolve(cmd_w[:, k], np.ones(w) / w, mode="valid")[: n - w] for k in range(2)], -1)
        dev_win = float(np.linalg.norm(disp - cw, axis=-1).mean())
    drift = np.linalg.norm(pel[:, :2] - d["ref_pelvis_pos_ep"][i, :n], axis=-1)
    ref_pel = d["ref_pelvis_pos"][i, :n]
    return {
        "env": i, "steps": n, "duration_s": n * dt, "end_reason": reason,
        "fell": reason in FALL, "ee_fail": reason == "ee_body_pos",
        "finished": reason in ("motion_clip_end", "time_out"),
        "foot_touches": len(fev), "consecutive_touches": consecutive, "nonfoot_touches": len(nev),
        "nonfoot_links": [str(x[2]) for x in nev],
        "labelled_hits": int(sum(hits)), "labelled_total": len(segs), "labelled_hit_mask": [bool(h) for h in hits],
        "ball_lost_step": lost_step,
        "pelvis_speed": float(np.linalg.norm(d["pelvis_vel"][i, :n, :2], axis=-1).mean()),
        "ref_pelvis_speed": float(np.linalg.norm(np.diff(ref_pel[:, :2], axis=0), axis=-1).mean() / dt) if n > 1 else 0.0,
        "ball_cmd_dev_step": dev_step, "ball_cmd_dev_1s": dev_win,
        "ball_travel_m": float(np.linalg.norm(ball[-1, :2] - ball[0, :2])),
        "path_drift_max": float(drift.max()),
        "offset_angle_deg": float(np.degrees(d["offset_angle"][i])),
        "_fev": fev, "_nev": nev, "_frame": frame, "_pel": pel, "_yaw": yaw, "_ball": ball, "_bvel": bvel,
        "_dist": dist, "_segs": segs,
    }


def stat(v):
    v = [x for x in v if x is not None]
    if not v:
        return None
    a = np.asarray(v, dtype=float)
    return {"mean": float(a.mean()), "median": float(np.median(a)), "max": float(a.max()), "min": float(a.min())}


def summarize(path):
    d = load(path)
    eps = [episode(d, i) for i in range(d["frame"].shape[0])]
    m = d["meta"]
    links = {}
    for e in eps:
        for name in e["nonfoot_links"]:
            links[name] = links.get(name, 0) + 1
    out = {
        "file": path, "condition": m["condition"], "offset": m["offset"], "no_resist": m["no_resist"],
        "checkpoint": m["checkpoint"], "episodes": len(eps),
        "consecutive_touches": stat([e["consecutive_touches"] for e in eps]),
        "foot_touches": stat([e["foot_touches"] for e in eps]),
        "labelled_hit_fraction": float(np.mean([e["labelled_hits"] / e["labelled_total"] for e in eps])),
        "nonfoot_touches_per_episode": float(np.mean([e["nonfoot_touches"] for e in eps])),
        "nonfoot_touch_links": links,
        "fall_rate": float(np.mean([e["fell"] for e in eps])),
        "ee_body_pos_rate": float(np.mean([e["ee_fail"] for e in eps])),
        "finished_clip_rate": float(np.mean([e["finished"] for e in eps])),
        "duration_s": stat([e["duration_s"] for e in eps]),
        "pelvis_speed": stat([e["pelvis_speed"] for e in eps]),
        "ref_pelvis_speed": stat([e["ref_pelvis_speed"] for e in eps]),
        "ball_cmd_dev_step": stat([e["ball_cmd_dev_step"] for e in eps]),
        "ball_cmd_dev_1s": stat([e["ball_cmd_dev_1s"] for e in eps]),
        "ball_lost_rate": float(np.mean([e["ball_lost_step"] is not None for e in eps])),
        "ball_travel_m": stat([e["ball_travel_m"] for e in eps]),
        "path_drift_max": stat([e["path_drift_max"] for e in eps]),
        "end_reasons": {r: sum(e["end_reason"] == r for e in eps) for r in {e["end_reason"] for e in eps}},
    }
    if m["condition"] == "offset":
        q = {}
        for e in eps:
            a = e["offset_angle_deg"] % 360
            key = "front" if (a < 45 or a >= 315) else "left" if a < 135 else "back" if a < 225 else "right"
            q.setdefault(key, []).append(e)
        out["by_direction"] = {k: {"episodes": len(v),
                                   "consecutive_mean": float(np.mean([e["consecutive_touches"] for e in v])),
                                   "labelled_hit_fraction": float(np.mean([e["labelled_hits"] / e["labelled_total"] for e in v])),
                                   "fall_rate": float(np.mean([e["fell"] for e in v]))} for k, v in sorted(q.items())}
    per_env = [{k: v for k, v in e.items() if not k.startswith("_")} for e in eps]
    return out, per_env


def failure(d, i):
    e = episode(d, i)
    dt = d["meta"]["step_dt"]
    n = e["steps"]
    frame, pel, yaw, ball, bvel = e["_frame"], e["_pel"], e["_yaw"], e["_ball"], e["_bvel"]
    feet = d["foot_pos"][i, :n]
    fyaw = yaw_of(d["foot_quat"][i, :n])
    fvec = d["ball_force_feet_vec"][i, :n]
    segs = e["_segs"]
    onsets = np.array([s for s, _ in segs])
    ref_foot = d["ref_foot"]
    ref_surface = d["ref_surface"]
    rec = {k: e[k] for k in ("env", "duration_s", "end_reason", "foot_touches", "consecutive_touches",
                             "nonfoot_touches", "nonfoot_links", "labelled_hits", "ball_lost_step")}
    rec["labelled_hit_mask"] = "".join("1" if h else "0" for h in e["labelled_hit_mask"])
    first_nonfoot = e["_nev"][0][0] if e["_nev"] else None
    rec["first_nonfoot_step"] = first_nonfoot
    rec["first_nonfoot_link"] = str(e["_nev"][0][2]) if e["_nev"] else None

    ref = np.load(d["meta"]["motion_path"])
    ref_yaw = yaw_of(ref["body_quat_w"][:, 0])
    ref_ball = ref["ball_pos_w"]
    fps = float(ref["fps"][0])

    def ref_out(k):
        """Reference ball velocity after labelled touch k (to the next onset), in the reference pelvis-yaw frame."""
        a = int(onsets[k])
        b = int(onsets[k + 1]) if k + 1 < len(onsets) else len(ref_ball) - 1
        v = (ref_ball[b, :2] - ref_ball[a, :2]) * fps / max(b - a, 1)
        vl = to_local(v, ref_yaw[a])
        return round(float(np.linalg.norm(v)), 3), round(float(np.degrees(np.arctan2(vl[1], vl[0]))), 1)

    def touch_info(s, end, foot):
        fr = int(frame[s]) + 1
        k = int(np.argmin(np.abs(onsets - fr)))
        rel = ball[s, :2] - feet[s, foot, :2]
        loc = to_local(rel, fyaw[s, foot])
        dz = float(ball[s, 2] - feet[s, foot, 2])
        f = fvec[s:end + 1, foot]
        fsum = f.sum(0)
        down = float(-fsum[2] / (np.linalg.norm(fsum) + 1e-9))  # >0: foot pushes the ball down
        lateral_in = (loc[1] > 0) if foot == 1 else (loc[1] < 0)
        # ankle-roll origin is at the ankle; the sole reaches ~0.12 m forward, ~0.05 m back
        if down > 0.5:
            region = "on_top(step_on)"
        else:
            side = "inside" if lateral_in else "outside"
            lon = "front" if loc[0] > 0.12 else ("back" if loc[0] < -0.05 else "mid")
            region = f"{side}-{lon}"
        w = slice(s, min(n, end + 6))
        vmax_i = s + int(np.argmax(np.linalg.norm(bvel[w, :2], axis=-1)))
        v = bvel[vmax_i, :2]
        v_loc = to_local(v, yaw[vmax_i])
        return {
            "step": int(s), "ref_frame": fr, "nearest_labelled_onset": int(onsets[k]), "frame_diff": fr - int(onsets[k]),
            "nearest_labelled_index": k, "labelled_foot": "right" if ref_foot[onsets[k]] == 1 else "left",
            "labelled_surface": {0: "inside", 1: "outside"}.get(int(ref_surface[onsets[k]]), "?"),
            "foot": "right" if foot == 1 else "left", "region": region,
            "ball_in_ankle_yaw_frame_cm": [round(float(loc[0]) * 100, 1), round(float(loc[1]) * 100, 1), round(dz * 100, 1)],
            "force_down_fraction": round(down, 2),
            "ball_speed_after": round(float(np.linalg.norm(v)), 3),
            "ball_dir_after_deg_pelvis": round(float(np.degrees(np.arctan2(v_loc[1], v_loc[0]))), 1),
            "ref_ball_speed_after": ref_out(k)[0], "ref_ball_dir_after_deg_pelvis": ref_out(k)[1],
        }

    # first contact by ANY link (foot or not): what the ball did right after it
    anyev = sorted([(a, b, "foot_" + ("right" if c == 1 else "left")) for a, b, c in e["_fev"]] + list(e["_nev"]))
    if anyev:
        a0, b0, link0 = anyev[0]
        w = slice(a0, min(n, b0 + 6))
        vi = a0 + int(np.argmax(np.linalg.norm(bvel[w, :2], axis=-1)))
        v = bvel[vi, :2]
        vl = to_local(v, yaw[vi])
        fr0 = int(frame[a0]) + 1
        k0 = int(np.argmin(np.abs(onsets - fr0)))
        rec["first_contact_any"] = {
            "link": str(link0), "step": int(a0), "ref_frame": fr0, "nearest_labelled_index": k0,
            "frame_diff": fr0 - int(onsets[k0]),
            "ball_speed_after": round(float(np.linalg.norm(v)), 3),
            "ball_dir_after_deg_pelvis": round(float(np.degrees(np.arctan2(vl[1], vl[0]))), 1),
            "ref_ball_speed_after": ref_out(k0)[0], "ref_ball_dir_after_deg_pelvis": ref_out(k0)[1],
        }
    else:
        rec["first_contact_any"] = None
    fev = e["_fev"]
    if fev:
        s, end, foot = fev[0]
        ft = touch_info(s, end, foot)
        rec["first_touch"] = ft
        traj = []
        for t in range(s, min(n, s + 26), 5):
            loc = to_local(ball[t, :2] - pel[t, :2], yaw[t])
            traj.append([round(float((t - s) * dt), 2), round(float(loc[0]), 3), round(float(loc[1]), 3)])
        rec["ball_rel_pelvis_0p5s"] = traj  # [t, x fwd, y left] m
        k = ft["nearest_labelled_index"] + 1  # the labelled touch after the one the first touch belongs to
        if k < len(onsets):
            idx = np.nonzero(frame + 1 >= onsets[k])[0]
            if idx.size:
                t = int(idx[0])
                lf = int(ref_foot[onsets[k]])
                rec["next_labelled"] = {
                    "index": k, "frame": int(onsets[k]), "step": t,
                    "ball_to_labelled_ankle_xy_m": round(float(np.linalg.norm(ball[t, :2] - feet[t, lf, :2])), 3),
                    "ball_rel_pelvis_m": [round(float(x), 3) for x in to_local(ball[t, :2] - pel[t, :2], yaw[t])],
                    "hit": bool(e["labelled_hit_mask"][k]),
                }
            else:
                rec["next_labelled"] = {"index": k, "frame": int(onsets[k]), "step": None, "note": "episode ended before"}
        rec["touches"] = [touch_info(a, b, c) for a, b, c in fev]
    else:
        rec["first_touch"] = None
    # chain break: first labelled touch (from the first one) that is missed while the episode is still running
    rec["chain_break"] = None
    for k, (s0, e0) in enumerate(segs):
        if e["labelled_hit_mask"][k]:
            continue
        idx = np.nonzero(frame + 1 >= s0)[0]
        if not idx.size:
            rec["chain_break"] = {"index": k, "frame": int(s0), "reason": "episode_ended_before",
                                  "end_reason": e["end_reason"]}
            break
        t = int(idx[0])
        lf = int(ref_foot[s0])
        dxy = float(np.linalg.norm(ball[t, :2] - feet[t, lf, :2]))
        rel = to_local(ball[t, :2] - pel[t, :2], yaw[t])
        rec["chain_break"] = {"index": k, "frame": int(s0), "step": t, "ball_to_labelled_ankle_xy_m": round(dxy, 3),
                              "ball_rel_pelvis_m": [round(float(x), 3) for x in rel],
                              "ball_reachable": dxy <= REACH_XY}
        break
    # fall
    rec["fall"] = None
    if e["fell"] or e["ee_fail"]:
        t = n - 1
        fr = int(frame[t]) + 1
        before = int(np.sum(onsets <= fr))
        w0 = max(0, t - 25)
        down_any = False
        for foot in (0, 1):
            fsum = fvec[w0:t + 1, foot]
            mag = np.linalg.norm(fsum, axis=-1)
            down_any |= bool(np.any((mag > 1.0) & (-fsum[:, 2] > 0.5 * mag)))
        dball = float(e["_dist"][t])
        speed_err = float(np.linalg.norm(d["pelvis_vel"][i, t, :2] - to_world(d["cmd_local"][i, t, :2], yaw[t])))
        if down_any:
            cause = "stepped_on_ball"
        elif dball > 1.0:
            cause = "ball_out_of_reach"
        else:
            cause = "imbalance_ball_nearby"
        rec["fall"] = {"step": t, "ref_frame": fr, "labelled_onsets_before": before,
                       "foot_touches_before": sum(1 for s, _, _ in fev if s <= t),
                       "ball_pelvis_dist_m": round(dball, 3), "pelvis_speed_err_mps": round(speed_err, 3),
                       "foot_pressing_ball_last_0p5s": down_any, "cause": cause}
    # primary category (chronological)
    first_foot = fev[0][0] if fev else None
    first_window_hi = segs[0][1] + 10
    if first_nonfoot is not None and (first_foot is None or first_nonfoot < first_foot):
        cat = "A_nonfoot_contact_first"
    elif first_foot is None or (int(frame[first_foot]) + 1) > first_window_hi:
        cat = "B_first_touch_missed"
    else:
        nl = rec.get("next_labelled")
        if nl is None or nl.get("step") is None:
            cat = "F_episode_ended_before_second_touch"
        elif nl["hit"]:
            cat = "E_second_touch_connected"
        elif nl["ball_to_labelled_ankle_xy_m"] > REACH_XY:
            cat = "C_first_touch_ball_out_of_reach"
        else:
            cat = "D_ball_reachable_but_second_touch_missed"
    rec["category"] = cat
    rec["ended_by_fall_or_tracking"] = rec["fall"] is not None
    return rec


def write_csv(path, rows):
    keys = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in r.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("summary", "failures"))
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--csv", default="")
    ap.add_argument("--per_env_dir", default="")
    ap.add_argument("--episodes", type=int, default=20)
    a = ap.parse_args()
    if a.mode == "summary":
        rows = []
        for f in a.files:
            s, per_env = summarize(f)
            rows.append(s)
            if a.per_env_dir:
                import os

                os.makedirs(a.per_env_dir, exist_ok=True)
                name = os.path.basename(f).replace(".npz", "")
                write_csv(os.path.join(a.per_env_dir, name + "_per_episode.csv"), per_env)
        json.dump(rows, open(a.out, "w"), indent=1)
        if a.csv:
            flat = []
            for r in rows:
                fr = {}
                for k, v in r.items():
                    if isinstance(v, dict) and v and all(isinstance(x, (int, float)) for x in v.values()) and k in (
                            "consecutive_touches", "foot_touches", "duration_s", "pelvis_speed", "ref_pelvis_speed",
                            "ball_cmd_dev_step", "ball_cmd_dev_1s", "ball_travel_m", "path_drift_max"):
                        for kk, vv in v.items():
                            fr[f"{k}_{kk}"] = vv
                    else:
                        fr[k] = v
                flat.append(fr)
            write_csv(a.csv, flat)
    else:
        d = load(a.files[0])
        recs = [failure(d, i) for i in range(min(a.episodes, d["frame"].shape[0]))]
        cats = {}
        for r in recs:
            cats[r["category"]] = cats.get(r["category"], 0) + 1
        fall_causes = {}
        for r in recs:
            if r["fall"]:
                fall_causes[r["fall"]["cause"]] = fall_causes.get(r["fall"]["cause"], 0) + 1
        json.dump({"file": a.files[0], "meta": d["meta"], "reach_xy_m": REACH_XY, "category_counts": cats,
                   "fall_cause_counts": fall_causes, "episodes": recs}, open(a.out, "w"), indent=1)
        if a.csv:
            flat = []
            for r in recs:
                ft = r.get("first_touch") or {}
                nl = r.get("next_labelled") or {}
                fl = r.get("fall") or {}
                cb = r.get("chain_break") or {}
                flat.append({"env": r["env"], "category": r["category"], "end_reason": r["end_reason"],
                             "ended_by_fall_or_tracking": r["ended_by_fall_or_tracking"],
                             "duration_s": r["duration_s"], "foot_touches": r["foot_touches"],
                             "consecutive_touches": r["consecutive_touches"], "labelled_hits": r["labelled_hits"],
                             "labelled_hit_mask": r["labelled_hit_mask"], "nonfoot_links": r["nonfoot_links"],
                             "first_nonfoot_step": r["first_nonfoot_step"],
                             "c1_link": (r.get("first_contact_any") or {}).get("link"),
                             "c1_ref_frame": (r.get("first_contact_any") or {}).get("ref_frame"),
                             "c1_frame_diff": (r.get("first_contact_any") or {}).get("frame_diff"),
                             "c1_ball_speed_after": (r.get("first_contact_any") or {}).get("ball_speed_after"),
                             "c1_ball_dir_after_deg": (r.get("first_contact_any") or {}).get("ball_dir_after_deg_pelvis"),
                             "c1_ref_ball_dir_after_deg": (r.get("first_contact_any") or {}).get("ref_ball_dir_after_deg_pelvis"),
                             "t1_ref_frame": ft.get("ref_frame"), "t1_frame_diff": ft.get("frame_diff"),
                             "t1_nearest_labelled_index": ft.get("nearest_labelled_index"),
                             "t1_foot": ft.get("foot"), "t1_region": ft.get("region"),
                             "t1_labelled_surface": ft.get("labelled_surface"),
                             "t1_ball_in_ankle_frame_cm": ft.get("ball_in_ankle_yaw_frame_cm"),
                             "t1_ball_speed_after": ft.get("ball_speed_after"),
                             "t1_ball_dir_after_deg": ft.get("ball_dir_after_deg_pelvis"),
                             "t1_ref_ball_speed_after": ft.get("ref_ball_speed_after"),
                             "t1_ref_ball_dir_after_deg": ft.get("ref_ball_dir_after_deg_pelvis"),
                             "ball_rel_pelvis_0p5s": r.get("ball_rel_pelvis_0p5s"),
                             "next_labelled_frame": nl.get("frame"),
                             "next_ball_to_ankle_xy_m": nl.get("ball_to_labelled_ankle_xy_m"),
                             "next_ball_rel_pelvis_m": nl.get("ball_rel_pelvis_m"), "next_hit": nl.get("hit"),
                             "chain_break_index": cb.get("index"), "chain_break_ball_to_ankle_xy_m": cb.get("ball_to_labelled_ankle_xy_m"),
                             "fall_ref_frame": fl.get("ref_frame"), "fall_labelled_onsets_before": fl.get("labelled_onsets_before"),
                             "fall_cause": fl.get("cause"), "fall_ball_pelvis_dist_m": fl.get("ball_pelvis_dist_m")})
            write_csv(a.csv, flat)
        print(json.dumps({"category_counts": cats, "fall_cause_counts": fall_causes}))


if __name__ == "__main__":
    main()
