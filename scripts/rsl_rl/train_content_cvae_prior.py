"""Train an offline content-conditioned CVAE motion prior from NPZ motions.

This prototype is a bridge between hard tracking and a later CVAE prior reward.
It learns short motion windows conditioned on:

  - semantic phase: approach / prestrike / strike / follow
  - kick leg
  - time relative to kick_frame
  - ball-relative pelvis, swing-foot, support-foot geometry
  - swing/support foot velocity in the kick-local frame

It does not control the policy and is not used by RL yet.
"""

from __future__ import annotations

import argparse
import fnmatch
import glob
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Direct import to avoid pulling in IsaacLab for offline training.
import importlib.util

_model_path = os.path.join(
    os.path.dirname(__file__),
    "content_cvae_prior.py",
)
_spec = importlib.util.spec_from_file_location("content_cvae_prior", os.path.abspath(_model_path))
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
ContentConditionedVAE = _mod.ContentConditionedVAE
vae_loss = _mod.vae_loss


PHASE_APPROACH = 0
PHASE_PRESTRIKE = 1
PHASE_STRIKE = 2
PHASE_FOLLOW = 3
NUM_PHASES = 4

PHASE_NAMES = {
    PHASE_APPROACH: "approach",
    PHASE_PRESTRIKE: "prestrike",
    PHASE_STRIKE: "strike",
    PHASE_FOLLOW: "follow",
}

# Legacy G1 NPZ files do not store body_names. These indices match the observed
# raw body_pos_w order for the current Video motions.
BODY_INDEX = {
    "pelvis": 0,
    "left_ankle_roll_link": 14,
    "right_ankle_roll_link": 18,
    "torso_link": 13,
}

COND_NAMES = [
    "phase_approach",
    "phase_prestrike",
    "phase_strike",
    "phase_follow",
    "leg_left",
    "leg_right",
    "leg_unknown",
    "time_to_kick",
    "ball_from_pelvis_long",
    "ball_from_pelvis_lat",
    "kick_from_ball_long",
    "kick_from_ball_lat",
    "support_from_ball_long",
    "support_from_ball_lat",
    "kick_height_rel_ball",
    "support_height_rel_ball",
    "kick_vel_long",
    "kick_vel_lat",
    "support_vel_long",
    "support_vel_lat",
    "kick_speed_xy",
    "support_speed_xy",
]


def get_motion_files(path: str | list[str] | tuple[str, ...]) -> list[str]:
    paths = [path] if isinstance(path, str) else list(path)
    files: list[str] = []
    for p in paths:
        if os.path.isfile(p):
            files.append(p)
        else:
            files.extend(glob.glob(os.path.join(p, "**", "*.npz"), recursive=True))
    files = sorted(set(files))
    if not files:
        raise ValueError(f"No .npz files found under {path}")
    return files


def normalize_path_key(path: str) -> str:
    return os.path.normpath(str(path)).replace("\\", "/")


def motion_match_keys(path: str) -> set[str]:
    abs_path = os.path.abspath(path)
    return {
        normalize_path_key(path),
        normalize_path_key(abs_path),
        normalize_path_key(os.path.relpath(abs_path)),
        normalize_path_key(os.path.basename(path)),
        normalize_path_key(Path(path).stem),
    }


def motion_pattern_matches(pattern: str, motion_file: str) -> bool:
    pattern = normalize_path_key(pattern)
    keys = motion_match_keys(motion_file)
    if pattern in keys:
        return True
    return any(fnmatch.fnmatch(key, pattern) for key in keys)


def load_filter_manifest(path: str | None) -> dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    if not isinstance(manifest, dict):
        raise ValueError(f"Filter manifest must be a JSON object: {path}")
    return manifest


def _motion_entry_pattern(entry) -> str | None:
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        return entry.get("motion") or entry.get("file") or entry.get("path") or entry.get("name")
    return None


def _motion_list_matches(entries, motion_file: str) -> bool:
    for entry in entries or []:
        pattern = _motion_entry_pattern(entry)
        if pattern and motion_pattern_matches(pattern, motion_file):
            return True
    return False


def _phase_matches(entry: dict, phase_name: str) -> bool:
    phases = entry.get("phases", entry.get("phase"))
    if phases is None:
        return True
    if isinstance(phases, str):
        phases = [phases]
    return phase_name in {str(phase) for phase in phases}


def _range_matches_window(
    entry: dict,
    motion_file: str,
    start: int,
    end: int,
    center: int,
    phase_name: str,
    default_match: str,
) -> bool:
    pattern = _motion_entry_pattern(entry)
    if not pattern or not motion_pattern_matches(pattern, motion_file):
        return False
    if not _phase_matches(entry, phase_name):
        return False

    range_start = int(entry.get("start", entry.get("frame_start", 0)))
    range_end = int(entry.get("end", entry.get("frame_end", range_start + 1)))
    match_mode = str(entry.get("match", default_match))
    if match_mode == "center":
        return range_start <= center < range_end
    if match_mode == "overlap":
        return start < range_end and end > range_start
    raise ValueError(f"Unsupported filter range match mode: {match_mode!r}")


def motion_is_excluded(motion_file: str, manifest: dict) -> tuple[bool, str]:
    include_motions = manifest.get("include_motions", [])
    if include_motions and not _motion_list_matches(include_motions, motion_file):
        return True, "not_in_include_motions"
    if _motion_list_matches(manifest.get("exclude_motions", []), motion_file):
        return True, "exclude_motions"
    return False, ""


def window_is_excluded(
    motion_file: str,
    start: int,
    end: int,
    center: int,
    phase_name: str,
    manifest: dict,
    default_match: str,
) -> bool:
    for entry in manifest.get("exclude_ranges", []):
        if not isinstance(entry, dict):
            raise ValueError("exclude_ranges entries must be JSON objects.")
        if _range_matches_window(entry, motion_file, start, end, center, phase_name, default_match):
            return True
    return False


def _scalar(data: np.lib.npyio.NpzFile, key: str, default):
    if key not in data.files:
        return default
    try:
        return np.asarray(data[key]).flat[0].item()
    except Exception:
        return default


def infer_kick_leg(path: str, data: np.lib.npyio.NpzFile) -> str:
    raw = _scalar(data, "kick_leg", None)
    if raw is not None:
        label = str(raw).strip().lower()
        if label in {"left", "right"}:
            return label
    stem = Path(path).stem.lower()
    if "_left" in stem:
        return "left"
    if "_right" in stem:
        return "right"
    return "unknown"


def foot_names_for_leg(kick_leg: str) -> tuple[str, str]:
    if kick_leg == "left":
        return "left_ankle_roll_link", "right_ankle_roll_link"
    return "right_ankle_roll_link", "left_ankle_roll_link"


def phase_at_frame(t: int, kick_frame: int, kick_end_frame: int, motion_len: int, prestrike_len: int) -> int:
    if kick_frame < 0:
        return PHASE_APPROACH
    if kick_end_frame < 0:
        kick_end_frame = min(kick_frame + 5, motion_len)
    approach_end = max(0, kick_frame - prestrike_len)
    if t < approach_end:
        return PHASE_APPROACH
    if t < kick_frame:
        return PHASE_PRESTRIKE
    if t <= kick_end_frame:
        return PHASE_STRIKE
    return PHASE_FOLLOW


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors by quaternions in wxyz format."""
    q = q.astype(np.float32)
    v = v.astype(np.float32)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True).clip(min=1.0e-8)
    q_vec = q[..., 1:4]
    q_w = q[..., 0:1]
    dot_uv = np.sum(q_vec * v, axis=-1, keepdims=True)
    dot_uu = np.sum(q_vec * q_vec, axis=-1, keepdims=True)
    return (
        2.0 * dot_uv * q_vec
        + (q_w * q_w - dot_uu) * v
        + 2.0 * q_w * np.cross(q_vec, v)
    ).astype(np.float32)


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate world vectors into a quaternion-local frame."""
    q_inv = q.copy().astype(np.float32)
    q_inv[..., 1:4] *= -1.0
    return quat_rotate(q_inv, v)


def extract_frame_features(
    data: np.lib.npyio.NpzFile,
    lower_body_only: bool = True,
    feature_frame: str = "local",
) -> np.ndarray:
    """Extract compact motion-style features from one reference motion."""
    joint_pos = data["joint_pos"].astype(np.float32)
    joint_vel = data["joint_vel"].astype(np.float32)
    body_pos = data["body_pos_w"].astype(np.float32)
    body_quat = data["body_quat_w"].astype(np.float32)
    body_lin_vel = data["body_lin_vel_w"].astype(np.float32)
    body_ang_vel = data["body_ang_vel_w"].astype(np.float32)

    if lower_body_only:
        # Joint order from the conversion scripts: 12 leg joints + 3 waist.
        joint_slice = slice(0, 15)
    else:
        joint_slice = slice(0, joint_pos.shape[1])

    pelvis_height = body_pos[:, BODY_INDEX["pelvis"], 2:3]
    pelvis_lin_vel = body_lin_vel[:, BODY_INDEX["pelvis"], :]
    pelvis_ang_vel = body_ang_vel[:, BODY_INDEX["pelvis"], :]
    if feature_frame == "local":
        pelvis_quat = body_quat[:, BODY_INDEX["pelvis"], :]
        pelvis_lin_vel = quat_rotate_inverse(pelvis_quat, pelvis_lin_vel)
        pelvis_ang_vel = quat_rotate_inverse(pelvis_quat, pelvis_ang_vel)
    elif feature_frame != "world":
        raise ValueError(f"Unsupported feature_frame={feature_frame!r}; expected 'local' or 'world'.")

    return np.concatenate(
        [
            joint_pos[:, joint_slice],
            joint_vel[:, joint_slice],
            pelvis_height,
            pelvis_lin_vel,
            pelvis_ang_vel,
        ],
        axis=-1,
    ).astype(np.float32)


def infer_ball_candidates(
    body_pos: np.ndarray,
    radius_min: float,
    radius_max: float,
    arc_angle: float,
    ball_height: float,
    radius_samples: int,
    angle_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match MotionCommand._compute_soccer_ball_positions, but offline."""
    first_anchor = body_pos[0, BODY_INDEX["torso_link"]]
    last_anchor = body_pos[-1, BODY_INDEX["torso_link"]]
    radius_vec = last_anchor[:2] - first_anchor[:2]
    radius = float(np.linalg.norm(radius_vec))
    base_angle = math.atan2(float(radius_vec[1]), float(radius_vec[0])) if radius > 1.0e-8 else 0.0

    if radius_samples <= 1:
        radius_offsets = np.array([(radius_min + radius_max) * 0.5], dtype=np.float32)
    else:
        radius_offsets = np.linspace(radius_min, radius_max, radius_samples, dtype=np.float32)

    if angle_samples <= 1 or abs(arc_angle) < 1.0e-8:
        angle_offsets = np.array([0.0], dtype=np.float32)
    else:
        angle_offsets = np.linspace(-arc_angle, arc_angle, angle_samples, dtype=np.float32)

    candidates = []
    candidate_radius_offsets = []
    candidate_angle_offsets = []
    for ro in radius_offsets:
        r = max(0.0, radius + float(ro))
        for ao in angle_offsets:
            theta = base_angle + float(ao)
            xy = first_anchor[:2] + r * np.array([math.cos(theta), math.sin(theta)], dtype=np.float32)
            candidates.append([float(xy[0]), float(xy[1]), float(ball_height)])
            candidate_radius_offsets.append(float(ro))
            candidate_angle_offsets.append(float(ao))
    return (
        np.asarray(candidates, dtype=np.float32),
        np.asarray(candidate_radius_offsets, dtype=np.float32),
        np.asarray(candidate_angle_offsets, dtype=np.float32),
    )


def choose_motion_ball(
    body_pos: np.ndarray,
    body_vel: np.ndarray,
    kick_idx: int,
    kick_frame: int,
    args: argparse.Namespace,
) -> np.ndarray:
    """Return one estimated ball position for the whole motion."""
    if args.ball_mode == "nominal":
        return infer_ball_candidates(
            body_pos,
            radius_min=(args.radius_offset_min + args.radius_offset_max) * 0.5,
            radius_max=(args.radius_offset_min + args.radius_offset_max) * 0.5,
            arc_angle=0.0,
            ball_height=args.ball_height,
            radius_samples=1,
            angle_samples=1,
        )[0][0]

    candidates, _, _ = infer_ball_candidates(
        body_pos,
        radius_min=args.radius_offset_min,
        radius_max=args.radius_offset_max,
        arc_angle=args.arc_angle,
        ball_height=args.ball_height,
        radius_samples=args.radius_samples,
        angle_samples=args.angle_samples,
    )
    kf = min(max(kick_frame, 0), body_pos.shape[0] - 1)
    start = max(0, kf - args.ball_search_before)
    end = min(body_pos.shape[0], kf + args.ball_search_after + 1)
    foot_pos = body_pos[start:end, kick_idx]
    foot_vel = body_vel[start:end, kick_idx]

    best_score = float("inf")
    best_ball = candidates[0]
    for ball in candidates:
        rel = foot_pos - ball[None, :]
        dist_xy = np.linalg.norm(rel[:, :2], axis=-1)
        idx = int(np.argmin(dist_xy))
        ball_to_foot = ball[:2] - foot_pos[idx, :2]
        norm = float(np.linalg.norm(ball_to_foot))
        closing = float(np.dot(foot_vel[idx, :2], ball_to_foot / norm)) if norm > 1.0e-8 else 0.0
        # Prefer close candidates near the annotated kick_frame with positive closing speed.
        frame_delta = abs(start + idx - kf)
        score = float(dist_xy[idx]) + 0.003 * frame_delta - 0.02 * max(closing, 0.0)
        if score < best_score:
            best_score = score
            best_ball = ball
    return best_ball.astype(np.float32)


def motion_basis(body_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    first_anchor = body_pos[0, BODY_INDEX["torso_link"]]
    last_anchor = body_pos[-1, BODY_INDEX["torso_link"]]
    forward = last_anchor[:2] - first_anchor[:2]
    norm = float(np.linalg.norm(forward))
    if norm < 1.0e-8:
        forward = np.array([1.0, 0.0], dtype=np.float32)
    else:
        forward = (forward / norm).astype(np.float32)
    side = np.array([-forward[1], forward[0]], dtype=np.float32)
    return forward, side


def local_xy(vec_xy: np.ndarray, forward: np.ndarray, side: np.ndarray, kick_leg: str) -> np.ndarray:
    """Project XY vector to kick-local coordinates, mirrored by kick leg."""
    side_sign = -1.0 if kick_leg == "left" else 1.0
    return np.array(
        [
            float(np.dot(vec_xy, forward)),
            float(np.dot(vec_xy, side) * side_sign),
        ],
        dtype=np.float32,
    )


def condition_at_frame(
    body_pos: np.ndarray,
    body_vel: np.ndarray,
    ball: np.ndarray,
    kick_idx: int,
    support_idx: int,
    kick_leg: str,
    kick_frame: int,
    phase_id: int,
    frame: int,
    time_scale: float,
    forward: np.ndarray,
    side: np.ndarray,
) -> np.ndarray:
    phase = np.zeros(NUM_PHASES, dtype=np.float32)
    phase[phase_id] = 1.0

    leg = np.zeros(3, dtype=np.float32)
    if kick_leg == "left":
        leg[0] = 1.0
    elif kick_leg == "right":
        leg[1] = 1.0
    else:
        leg[2] = 1.0

    time_to_kick = 0.0 if kick_frame < 0 else float(np.clip((frame - kick_frame) / max(time_scale, 1.0), -2.0, 2.0))

    pelvis = body_pos[frame, BODY_INDEX["pelvis"]]
    kick_foot = body_pos[frame, kick_idx]
    support_foot = body_pos[frame, support_idx]
    kick_vel = body_vel[frame, kick_idx]
    support_vel = body_vel[frame, support_idx]

    ball_from_pelvis = local_xy(ball[:2] - pelvis[:2], forward, side, kick_leg)
    kick_from_ball = local_xy(kick_foot[:2] - ball[:2], forward, side, kick_leg)
    support_from_ball = local_xy(support_foot[:2] - ball[:2], forward, side, kick_leg)
    kick_vel_local = local_xy(kick_vel[:2], forward, side, kick_leg)
    support_vel_local = local_xy(support_vel[:2], forward, side, kick_leg)

    return np.concatenate(
        [
            phase,
            leg,
            np.array([time_to_kick], dtype=np.float32),
            ball_from_pelvis,
            kick_from_ball,
            support_from_ball,
            np.array(
                [
                    float(kick_foot[2] - ball[2]),
                    float(support_foot[2] - ball[2]),
                ],
                dtype=np.float32,
            ),
            kick_vel_local,
            support_vel_local,
            np.array(
                [
                    float(np.linalg.norm(kick_vel[:2])),
                    float(np.linalg.norm(support_vel[:2])),
                ],
                dtype=np.float32,
            ),
        ],
        axis=0,
    ).astype(np.float32)


def build_dataset(
    motion_files: list[str],
    window_len: int,
    stride: int,
    prestrike_len: int,
    lower_body_only: bool,
    args: argparse.Namespace,
    return_metadata: bool = False,
):
    windows: list[np.ndarray] = []
    conds: list[np.ndarray] = []
    phases: list[int] = []
    sources: list[str] = []
    metadata: list[dict] = []
    filter_manifest = getattr(args, "filter_manifest_data", None) or {}
    filter_range_match = getattr(args, "filter_range_match", None) or filter_manifest.get("range_match", "overlap")
    filter_stats = {
        "active": bool(filter_manifest),
        "input_motion_files": len(motion_files),
        "included_motion_files": 0,
        "excluded_motion_files": 0,
        "included_windows": 0,
        "excluded_windows": 0,
        "excluded_motion_reasons": {},
    }

    for motion_file in motion_files:
        excluded, reason = motion_is_excluded(motion_file, filter_manifest)
        if excluded:
            filter_stats["excluded_motion_files"] += 1
            filter_stats["excluded_motion_reasons"][motion_file] = reason
            continue
        filter_stats["included_motion_files"] += 1

        data = np.load(motion_file, allow_pickle=True)
        feat = extract_frame_features(
            data,
            lower_body_only=lower_body_only,
            feature_frame=args.feature_frame,
        )
        body_pos = data["body_pos_w"].astype(np.float32)
        body_vel = data["body_lin_vel_w"].astype(np.float32)
        motion_len = feat.shape[0]
        kick_frame = int(_scalar(data, "kick_frame", -1))
        kick_end_frame = int(_scalar(data, "kick_end_frame", -1))
        kick_leg = infer_kick_leg(motion_file, data)
        kick_foot, support_foot = foot_names_for_leg(kick_leg)
        kick_idx = BODY_INDEX[kick_foot]
        support_idx = BODY_INDEX[support_foot]
        ball = choose_motion_ball(body_pos, body_vel, kick_idx, kick_frame, args)
        forward, side = motion_basis(body_pos)

        for start in range(0, max(motion_len - window_len + 1, 0), stride):
            end = start + window_len
            center = start + window_len // 2
            phase = phase_at_frame(center, kick_frame, kick_end_frame, motion_len, prestrike_len)
            phase_name = PHASE_NAMES[phase]
            if window_is_excluded(
                motion_file=motion_file,
                start=start,
                end=end,
                center=center,
                phase_name=phase_name,
                manifest=filter_manifest,
                default_match=filter_range_match,
            ):
                filter_stats["excluded_windows"] += 1
                continue
            windows.append(feat[start:end].reshape(-1))
            conds.append(
                condition_at_frame(
                    body_pos=body_pos,
                    body_vel=body_vel,
                    ball=ball,
                    kick_idx=kick_idx,
                    support_idx=support_idx,
                    kick_leg=kick_leg,
                    kick_frame=kick_frame,
                    phase_id=phase,
                    frame=center,
                    time_scale=args.time_scale,
                    forward=forward,
                    side=side,
                )
            )
            phases.append(phase)
            sources.append(motion_file)
            filter_stats["included_windows"] += 1
            if return_metadata:
                metadata.append(
                    {
                        "source": motion_file,
                        "start": int(start),
                        "center": int(center),
                        "end": int(end),
                        "phase": int(phase),
                        "phase_name": PHASE_NAMES[phase],
                        "kick_frame": int(kick_frame),
                        "kick_end_frame": int(kick_end_frame),
                        "kick_leg": kick_leg,
                        "motion_len": int(motion_len),
                    }
                )

    if not windows:
        raise ValueError("No windows generated. Try a smaller --window_len or a less restrictive filter manifest.")

    x = torch.tensor(np.stack(windows), dtype=torch.float32)
    cond = torch.tensor(np.stack(conds), dtype=torch.float32)
    phase = torch.tensor(phases, dtype=torch.long)
    setattr(args, "_filter_stats", filter_stats)
    if return_metadata:
        return x, cond, phase, sources, metadata
    return x, cond, phase, sources


def split_by_source(sources: list[str], val_split: float, seed: int) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    unique = sorted(set(sources))
    if len(unique) <= 1 or val_split <= 0.0:
        mask = torch.ones(len(sources), dtype=torch.bool)
        return mask, mask, []

    rng = np.random.default_rng(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    n_val = min(len(unique) - 1, max(1, int(round(len(unique) * val_split))))
    val_files = set(shuffled[:n_val])
    train_mask = torch.tensor([src not in val_files for src in sources], dtype=torch.bool)
    val_mask = torch.tensor([src in val_files for src in sources], dtype=torch.bool)
    return train_mask, val_mask, sorted(val_files)


def main():
    parser = argparse.ArgumentParser(description="Train content-conditioned CVAE motion prior")
    parser.add_argument("--motion_path", type=str, nargs="+", required=True)
    parser.add_argument("--output", type=str, default="models/content_cvae_prior.pt")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--beta", type=float, default=1.0e-3)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--window_len", type=int, default=8)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--prestrike_len", type=int, default=20)
    parser.add_argument("--time_scale", type=float, default=40.0)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--full_body", action="store_true", help="Use all joints instead of lower-body + waist")
    parser.add_argument(
        "--filter_manifest",
        type=str,
        default=None,
        help="Optional JSON manifest with include_motions, exclude_motions, and exclude_ranges.",
    )
    parser.add_argument(
        "--filter_range_match",
        choices=["overlap", "center"],
        default=None,
        help="How exclude_ranges match windows. Defaults to manifest range_match or overlap.",
    )
    parser.add_argument(
        "--feature_frame",
        choices=["local", "world"],
        default="local",
        help="Frame for pelvis velocity features. local is less sensitive to global heading drift.",
    )
    parser.add_argument(
        "--world_frame_features",
        action="store_true",
        help="Compatibility alias for --feature_frame world.",
    )
    parser.add_argument("--ball_mode", choices=["best_swing", "nominal"], default="best_swing")
    parser.add_argument("--radius_offset_min", type=float, default=0.0)
    parser.add_argument("--radius_offset_max", type=float, default=0.4)
    parser.add_argument("--arc_angle", type=float, default=math.pi / 9.0)
    parser.add_argument("--ball_height", type=float, default=0.11)
    parser.add_argument("--radius_samples", type=int, default=9)
    parser.add_argument("--angle_samples", type=int, default=11)
    parser.add_argument("--ball_search_before", type=int, default=25)
    parser.add_argument("--ball_search_after", type=int, default=35)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.world_frame_features:
        args.feature_frame = "world"
    args.filter_manifest_data = load_filter_manifest(args.filter_manifest)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    motion_files = get_motion_files(args.motion_path)
    x, cond, phase, sources = build_dataset(
        motion_files,
        window_len=args.window_len,
        stride=args.stride,
        prestrike_len=args.prestrike_len,
        lower_body_only=not args.full_body,
        args=args,
    )
    filter_stats = getattr(args, "_filter_stats", {})

    train_mask, val_mask, val_files = split_by_source(sources, args.val_split, args.seed)
    feature_mean = x[train_mask].mean(dim=0, keepdim=True)
    feature_std = x[train_mask].std(dim=0, keepdim=True).clamp(min=1.0e-4)
    cond_mean = cond[train_mask].mean(dim=0, keepdim=True)
    cond_std = cond[train_mask].std(dim=0, keepdim=True).clamp(min=1.0e-4)

    x_norm = (x - feature_mean) / feature_std
    cond_norm = (cond - cond_mean) / cond_std

    train_ds = TensorDataset(x_norm[train_mask], cond_norm[train_mask])
    val_ds = TensorDataset(x_norm[val_mask], cond_norm[val_mask])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = ContentConditionedVAE(
        input_dim=x.shape[1],
        cond_dim=cond.shape[1],
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
    ).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1.0e-5)

    print(
        f"Loaded {len(motion_files)} files, {len(sources)} windows, "
        f"input_dim={x.shape[1]}, cond_dim={cond.shape[1]}, "
        f"ball_mode={args.ball_mode}, feature_frame={args.feature_frame}"
    )
    if filter_stats.get("active"):
        print(
            "Filter: "
            f"included_motions={filter_stats.get('included_motion_files', 0)}, "
            f"excluded_motions={filter_stats.get('excluded_motion_files', 0)}, "
            f"included_windows={filter_stats.get('included_windows', 0)}, "
            f"excluded_windows={filter_stats.get('excluded_windows', 0)}"
        )
    counts = torch.bincount(phase, minlength=NUM_PHASES)
    print(
        "Phase counts: "
        + ", ".join(f"{PHASE_NAMES[i]}={counts[i].item()}" for i in range(NUM_PHASES))
    )
    if val_files:
        print("Validation files:")
        for path in val_files:
            print(f"  {os.path.relpath(path)}")

    best_val = float("inf")
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        train_total = train_recon = train_kl = 0.0
        n_batches = 0
        for xb, cb in train_loader:
            xb = xb.to(args.device)
            cb = cb.to(args.device)
            recon, mu, logvar = model(xb, cb)
            loss, recon_loss, kl = vae_loss(recon, xb, mu, logvar, beta=args.beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_total += loss.item()
            train_recon += recon_loss.item()
            train_kl += kl.item()
            n_batches += 1

        model.eval()
        val_total = val_recon = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for xb, cb in val_loader:
                xb = xb.to(args.device)
                cb = cb.to(args.device)
                recon, mu, logvar = model(xb, cb)
                loss, recon_loss, _ = vae_loss(recon, xb, mu, logvar, beta=args.beta)
                val_total += loss.item()
                val_recon += recon_loss.item()
                n_val_batches += 1

        train_total /= max(n_batches, 1)
        train_recon /= max(n_batches, 1)
        train_kl /= max(n_batches, 1)
        val_total /= max(n_val_batches, 1)
        val_recon /= max(n_val_batches, 1)

        if val_total < best_val:
            best_val = val_total
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 0 or (epoch + 1) % 25 == 0:
            print(
                f"epoch {epoch + 1:04d}/{args.epochs}: "
                f"train={train_total:.5f} recon={train_recon:.5f} kl={train_kl:.5f} "
                f"val={val_total:.5f} val_recon={val_recon:.5f}"
            )

    assert best_state is not None
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state,
            "input_dim": x.shape[1],
            "cond_dim": cond.shape[1],
            "cond_names": COND_NAMES,
            "num_phases": NUM_PHASES,
            "latent_dim": args.latent_dim,
            "hidden_dim": args.hidden_dim,
            "window_len": args.window_len,
            "prestrike_len": args.prestrike_len,
            "time_scale": args.time_scale,
            "lower_body_only": not args.full_body,
            "feature_frame": args.feature_frame,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "cond_mean": cond_mean,
            "cond_std": cond_std,
            "motion_files": motion_files,
            "sources": sources,
            "val_files": val_files,
            "best_val_loss": best_val,
            "filter_manifest_path": args.filter_manifest,
            "filter_manifest": args.filter_manifest_data,
            "filter_stats": filter_stats,
            "ball_mode": args.ball_mode,
            "ball_args": {
                "radius_offset_min": args.radius_offset_min,
                "radius_offset_max": args.radius_offset_max,
                "arc_angle": args.arc_angle,
                "ball_height": args.ball_height,
                "radius_samples": args.radius_samples,
                "angle_samples": args.angle_samples,
                "ball_search_before": args.ball_search_before,
                "ball_search_after": args.ball_search_after,
            },
        },
        args.output,
    )
    print(f"Saved content CVAE prior to {args.output} (best_val={best_val:.5f})")


if __name__ == "__main__":
    main()
