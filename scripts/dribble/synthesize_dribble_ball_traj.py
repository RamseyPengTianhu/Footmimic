#!/usr/bin/env python3
"""Synthesize contact geometry and contact-to-contact ball-flow labels from CG data.

Pipeline (XGen-style, football simplification):

1. **Contact segments** (``dribble_cg_contact==1``, foot from ``dribble_cg_foot``):
   At a surface-labelled contact instant, transform a calibrated shoe-side
   collision-capsule centre-line point by the ankle-roll link's full quaternion,
   then offset the ball centre along the transformed side normal by the sum of
   the shoe collision radius and ball radius. Inside/outside is defined entirely
   in the ankle-roll link's local frame (right inside = +Y; left inside = -Y),
   so the complete foot yaw/pitch/roll determines the world-space contact
   geometry.

2. **``traj_mode=hybrid``** (recommended): for surface-labelled data, the rising
   edge of each contact segment is the hand-labelled **contact instant**.  Only
   that anchor is constrained to the shoe surface; the remaining ``+5`` CG
   frames stay labelled for training but do not drag the ball with the foot.
   Before the first contact, the ball is held at that first anchor's world
   position; gaps between later contacts linearly interpolate through the
   anchors and the tail releases with the incoming ball velocity.  Missing
   surface labels preserve the same stationary pre-contact rule before their
   legacy contact-adhere segments.

   ``foot_follow``: ball tracks a stepping foot every frame. ``lerp``: legacy gaps +
   anchor spawn (avoid for master data).

4. **Contact-to-contact flow**: for adjacent contact anchors ``k`` and ``k+1``,
   store the outgoing XY direction in the pelvis-yaw frame at anchor ``k``, plus
   the anchor distance and duration. The label is held on ``[k, k+1)``; the
   final contact has no outgoing target. Stage 2 uses these event labels instead
   of treating the interpolated ball path as ground truth.

5. **``dribble_cg_foot_ball_dist[t]``** = XY distance from the reference foot to the
   synthesized ball at frame ``t`` (meters). Computed on **every** frame where a
   stitched ball trajectory exists (contact + approach gaps + tail), not only on
   annotated contact frames.

6. **``dribble_cg_dist_foot[t]``** = which foot the ball was placed against (0 left,
   1 right). ``dribble_cg_foot`` (contact annotation) is left unchanged.

**Foot body indices:** ``pkl_to_npz`` stores all ~30 G1 bodies. Indices 3/6 match the
14-link *tracking subset*, not ground feet — on full clips index 3 often equals
pelvis. Use ``--auto_foot_indices`` (default) to pick the lowest, most mobile foot
links (``LL_FOOT`` / ``LR_FOOT`` class).

Writes ``ball_pos_w`` and the legacy distance arrays for replay/debugging, the
rising-edge mask ``dribble_contact_anchor``, and ``dribble_cg_flow_*`` arrays
for Stage-2 reward supervision into each ``.npz``.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np


def _collect_npz_files(motion_path: Path) -> list[Path]:
    if motion_path.is_file() and motion_path.suffix == ".npz":
        return [motion_path]
    if motion_path.is_dir():
        files = sorted(motion_path.glob("*.npz"))
        if files:
            return files
    raise ValueError(f"No .npz files found at: {motion_path}")


def _yaw_from_quat_wxyz(q: np.ndarray) -> np.ndarray:
    """Extract yaw (rad) from quaternions ``[..., 4]`` in wxyz order."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    return np.arctan2(siny, cosy)


def _yaw_rotate_xy(yaw: np.ndarray, vec_xy: np.ndarray) -> np.ndarray:
    """Rotate 2D offsets by per-frame yaw. ``yaw`` (T,), ``vec_xy`` (2,) -> (T, 2)."""
    c = np.cos(yaw)
    s = np.sin(yaw)
    x, y = vec_xy[0], vec_xy[1]
    return np.stack([c * x - s * y, s * x + c * y], axis=-1)


def _quat_rotate_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a 3-vector by a quaternion in wxyz order."""
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    w = float(q[0])
    qv = q[1:4]
    return v + 2.0 * (w * np.cross(qv, v) + np.cross(qv, np.cross(qv, v)))


def _contact_segments(contact: np.ndarray, foot: np.ndarray) -> list[tuple[int, int, int]]:
    """Return list of (start, end, foot_id) for contiguous contact runs."""
    n = int(contact.size)
    segs: list[tuple[int, int, int]] = []
    i = 0
    while i < n:
        if contact[i] <= 0:
            i += 1
            continue
        j = i + 1
        while j < n and contact[j] > 0:
            j += 1
        votes = foot[i:j]
        votes = votes[votes >= 0]
        if votes.size == 0:
            fid = 1
        else:
            fid = int(np.bincount(votes.astype(np.int64)).argmax())
        segs.append((i, j - 1, fid))
        i = j
    return segs


def _resolve_anchor_body_index(num_bodies: int, anchor_body_index: int) -> int:
    if anchor_body_index >= 0:
        return anchor_body_index
    # Motion exports use body 0 as the floating-base pelvis. Keep heading and
    # fallback translation/velocity pelvis-based for every body layout.
    return 0


def resolve_foot_body_indices(
    body_pos_w: np.ndarray,
    *,
    left_foot_index: int = -1,
    right_foot_index: int = -1,
) -> tuple[int, int]:
    """Return ``(left_body_idx, right_body_idx)`` into ``body_pos_w``.

    For the 14-body tracking subset (from some exporters), ankles are at 3/6.
    For full G1 clips (~30 bodies from ``pkl_to_npz``), pick the two ground foot
    links by median height + horizontal travel (``LL_FOOT`` / ``LR_FOOT``).
    """
    num_bodies = int(body_pos_w.shape[1])
    if left_foot_index >= 0 and right_foot_index >= 0:
        return left_foot_index, right_foot_index

    if num_bodies == 14:
        return 3, 6

    # Full G1 from pkl_to_npz: 30 rigid bodies; ground feet are LL_FOOT/LR_FOOT (~18/19).
    if num_bodies >= 20 and 18 < num_bodies and 19 < num_bodies:
        mean_y_18 = float(np.mean(body_pos_w[:, 18, 1]))
        mean_y_19 = float(np.mean(body_pos_w[:, 19, 1]))
        if mean_y_18 <= mean_y_19:
            return 18, 19
        return 19, 18

    med_z = np.median(body_pos_w[:, :, 2], axis=0)
    travel = np.ptp(body_pos_w[:, :, :2], axis=0)
    travel = np.hypot(travel[:, 0], travel[:, 1])

    candidates = [
        i
        for i in range(num_bodies)
        if med_z[i] < 0.42 and travel[i] > 1.5
    ]
    if len(candidates) < 2:
        order = np.argsort(med_z)
        candidates = order[: min(6, num_bodies)].tolist()

    candidates = sorted(candidates, key=lambda i: (-travel[i], med_z[i]))
    shortlist = candidates[:4]
    if len(shortlist) < 2:
        shortlist = candidates[:2]

    mean_y = {i: float(np.mean(body_pos_w[:, i, 1])) for i in shortlist}
    left_idx = min(shortlist, key=lambda i: mean_y[i])
    right_idx = max(shortlist, key=lambda i: mean_y[i])
    if left_idx == right_idx and len(shortlist) >= 2:
        left_idx, right_idx = shortlist[0], shortlist[1]
    return left_idx, right_idx


def _forward_foot_id(
    t: int,
    body_pos_w: np.ndarray,
    body_quat_w: np.ndarray,
    anchor_body_index: int,
    left_foot_index: int,
    right_foot_index: int,
) -> int:
    """Pick the foot most ahead in anchor yaw (for frame 0 / cold start)."""
    yaw = float(_yaw_from_quat_wxyz(body_quat_w[t, anchor_body_index]))
    c, s = np.cos(-yaw), np.sin(-yaw)

    def forward_x(i: int) -> float:
        x, y = body_pos_w[t, i, 0], body_pos_w[t, i, 1]
        return c * x - s * y

    lf = forward_x(left_foot_index)
    rf = forward_x(right_foot_index)
    return 0 if lf >= rf else 1


def _foot_id_for_frame(
    t: int,
    cg_contact: np.ndarray,
    cg_foot: np.ndarray,
    body_pos_w: np.ndarray,
    body_quat_w: np.ndarray,
    ball: np.ndarray,
    *,
    left_foot_index: int,
    right_foot_index: int,
    anchor_body_index: int,
) -> int:
    """Choose left (0) or right (1) foot from actual body kinematics."""
    lp = body_pos_w[t, left_foot_index, :2]
    rp = body_pos_w[t, right_foot_index, :2]

    if t > 0:
        bprev = ball[t - 1, :2]
        dl = float(np.linalg.norm(lp - bprev))
        dr = float(np.linalg.norm(rp - bprev))
        closest = 0 if dl <= dr else 1
    else:
        closest = _forward_foot_id(
            t, body_pos_w, body_quat_w, anchor_body_index, left_foot_index, right_foot_index
        )

    if cg_contact[t] > 0 and cg_foot[t] >= 0:
        labeled = int(cg_foot[t])
        if t > 0:
            d_lab = dl if labeled == 0 else dr
            d_oth = dr if labeled == 0 else dl
            if d_oth + 0.04 < d_lab:
                return closest
        return labeled
    return closest


def _inward_xy(
    t: int,
    fid: int,
    body_pos_w: np.ndarray,
    left_foot_index: int,
    right_foot_index: int,
    magnitude: float,
) -> np.ndarray:
    """Signed medial offset in XY; a negative magnitude points lateral/outward."""
    if abs(magnitude) <= 1.0e-12:
        return np.zeros(2, dtype=np.float64)
    lp = body_pos_w[t, left_foot_index, :2]
    rp = body_pos_w[t, right_foot_index, :2]
    foot = lp if fid == 0 else rp
    other = rp if fid == 0 else lp
    inward = other - foot
    norm = float(np.linalg.norm(inward))
    if norm < 1e-6:
        return np.zeros(2, dtype=np.float64)
    return magnitude * (inward / norm)


def _ref_foot_per_frame(
    T: int,
    segs: list[tuple[int, int, int]],
) -> np.ndarray:
    """Per-frame foot id for foot–ball distance (separate from contact labels).

    Contact frames use the segment foot. Gaps between touches use the *next*
    segment foot (approach). Before the first / after the last touch use the
    first / last segment foot.
    """
    ref = np.full(T, -1, dtype=np.int8)
    if not segs:
        return ref

    for s, e, fid in segs:
        ref[s : e + 1] = fid

    s0, _, fid0 = segs[0]
    if s0 > 0:
        ref[:s0] = fid0

    for k in range(len(segs) - 1):
        _, e0, _ = segs[k]
        s1, _, fid1 = segs[k + 1]
        for t in range(e0 + 1, s1):
            ref[t] = fid1

    _, e_last, fid_last = segs[-1]
    if e_last < T - 1:
        ref[e_last + 1 :] = fid_last

    return ref


def synthesize_ball_trajectory(
    body_pos_w: np.ndarray,
    body_quat_w: np.ndarray,
    cg_contact: np.ndarray,
    cg_foot: np.ndarray,
    cg_surface: np.ndarray | None = None,
    *,
    left_foot_index: int,
    right_foot_index: int,
    anchor_body_index: int,
    ball_radius: float,
    foot_offset_x: float,
    foot_offset_y: float,
    foot_inner_offset: float,
    foot_surface_offset: float | None,
    foot_collision_radius: float,
    foot_contact_x: float,
    outside_foot_contact_x: float,
    outside_lateral_inset: float,
    foot_half_width: float,
    foot_contact_z: float,
    terminal_forward_limit: float,
    init_forward_dist: float,
    use_foot_yaw: bool,
    traj_mode: str = "hybrid",
    auto_foot_indices: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Return ``(ball_pos_w [T,3], foot_ball_dist [T], dist_foot [T], L_idx, R_idx)``."""
    if ball_radius <= 0.0:
        raise ValueError(f"ball_radius must be positive, got {ball_radius}.")
    if foot_collision_radius < 0.0:
        raise ValueError(
            f"foot_collision_radius must be non-negative, got {foot_collision_radius}."
        )
    if foot_surface_offset is not None and foot_surface_offset < 0.0:
        raise ValueError(
            f"foot_surface_offset must be non-negative when provided, got {foot_surface_offset}."
        )
    if outside_lateral_inset < 0.0:
        raise ValueError(
            f"outside_lateral_inset must be non-negative, got {outside_lateral_inset}."
        )

    T = int(body_pos_w.shape[0])
    num_bodies = int(body_pos_w.shape[1])
    ball = np.zeros((T, 3), dtype=np.float32)
    dist = np.full(T, -1.0, dtype=np.float32)
    dist_foot = np.full(T, -1, dtype=np.int8)

    if auto_foot_indices and (left_foot_index < 0 or right_foot_index < 0):
        left_foot_index, right_foot_index = resolve_foot_body_indices(
            body_pos_w,
            left_foot_index=left_foot_index,
            right_foot_index=right_foot_index,
        )
    anchor_body_index = _resolve_anchor_body_index(num_bodies, anchor_body_index)

    segs = _contact_segments(cg_contact, cg_foot)
    ref_foot = _ref_foot_per_frame(T, segs)
    mode = str(traj_mode).lower().strip()
    placement_foot = np.full(T, -1, dtype=np.int8)

    def _foot_idx(fid: int) -> int:
        return left_foot_index if fid == 0 else right_foot_index

    def _surface_id_at(t: int) -> int:
        if cg_surface is None or t < 0 or t >= int(cg_surface.size):
            return -1
        return int(cg_surface[t])

    def _segment_surface_id(s: int, e: int) -> int:
        if cg_surface is None:
            return -1
        values = np.asarray(cg_surface[s : e + 1], dtype=np.int8)
        values = values[(values == 0) | (values == 1)]
        if values.size == 0:
            return -1
        return int(np.bincount(values.astype(np.int64), minlength=2).argmax())

    def _guard_truncated_terminal_contact() -> None:
        """Cap forward drift caused by a short contact cut off at clip end.

        A terminal segment is considered truncated when it reaches ``T - 1``
        and is shorter than the median of the preceding contact segments.  The
        guard is limited to surface-labelled data so legacy motions retain their
        exact historical synthesis.  Only anchor-local forward is clamped;
        anchor-local lateral (and therefore the annotated side) is preserved.
        """
        if terminal_forward_limit < 0.0 or len(segs) < 2:
            return
        s_last, e_last, _ = segs[-1]
        if e_last != T - 1 or _segment_surface_id(s_last, e_last) not in (0, 1):
            return
        prior_lengths = np.asarray([e - s + 1 for s, e, _ in segs[:-1]], dtype=np.float64)
        expected_length = float(np.median(prior_lengths))
        if (e_last - s_last + 1) >= expected_length:
            return

        for t in range(s_last, e_last + 1):
            anchor_xy = body_pos_w[t, anchor_body_index, :2]
            anchor_yaw = float(_yaw_from_quat_wxyz(body_quat_w[t, anchor_body_index]))
            c, s = np.cos(anchor_yaw), np.sin(anchor_yaw)
            delta = ball[t, :2] - anchor_xy
            local_forward = c * delta[0] + s * delta[1]
            if local_forward <= terminal_forward_limit:
                continue
            local_lateral = -s * delta[0] + c * delta[1]
            ball[t, 0] = anchor_xy[0] + c * terminal_forward_limit - s * local_lateral
            ball[t, 1] = anchor_xy[1] + s * terminal_forward_limit + c * local_lateral

    def _place_contact_frame(t: int, fid: int, surface_id: int | None = None) -> None:
        """Adhere ball to the requested side of the labeled foot."""
        fi = _foot_idx(fid)
        foot_p = body_pos_w[t, fi]
        sid = _surface_id_at(t) if surface_id is None else int(surface_id)
        # Surface semantics and geometry both live in the ankle-roll frame.
        # G1 local +Y points to the left: it is medial for the right foot and
        # lateral for the left foot. The full foot quaternion then carries this
        # fixed inside/outside axis (including yaw, pitch and roll) into world.
        if sid in (0, 1):
            medial_sign = 1.0 if fid == 1 else -1.0
            side_local_sign = medial_sign if sid == 0 else -medial_sign

            foot_q = body_quat_w[t, fi]
            side_normal_w = _quat_rotate_wxyz(
                foot_q,
                np.array([0.0, side_local_sign, 0.0], dtype=np.float64),
            )
            side_normal_xy = side_normal_w[:2]
            norm_xy = float(np.linalg.norm(side_normal_xy))
            if norm_xy < 1.0e-6:
                # A near-vertical local Y axis has no stable horizontal contact
                # normal. Retain the foot-local sign without consulting pelvis.
                side_normal_xy = np.array([0.0, side_local_sign], dtype=np.float64)
            else:
                side_normal_xy /= norm_xy

            # Outside-of-foot dribbling uses the forefoot/toe region, while
            # inside contacts retain the mid-forefoot anchor.  A shared X
            # coordinate incorrectly reduced the distinction to lateral sign.
            surface_contact_x = outside_foot_contact_x if sid == 1 else foot_contact_x
            contact_local = np.array(
                [surface_contact_x, side_local_sign * foot_half_width, foot_contact_z],
                dtype=np.float64,
            )
            contact_w = foot_p.astype(np.float64) + _quat_rotate_wxyz(foot_q, contact_local)
            # ``foot_half_width`` locates the centre line of the G1's outer
            # collision capsule (local Y=+/-0.026 m), not its physical surface.
            # The ball centre therefore has to clear both collision radii.  The
            # old independent 0.11 m default happened to equal the current ball
            # radius, but omitted the 0.01 m shoe capsule and silently became
            # wrong whenever ``--ball_radius`` changed.
            centre_clearance = (
                ball_radius + foot_collision_radius
                if foot_surface_offset is None
                else foot_surface_offset
            )
            if sid == 1:
                # The manually inferred outside anchor is approximate.  A
                # positive inset moves the ball centre toward the shoe while
                # preserving the labelled side and forefoot X template.
                centre_clearance = max(0.0, centre_clearance - outside_lateral_inset)
            ball_xy = contact_w[:2] + side_normal_xy * centre_clearance
        else:
            # Preserve the historical inter-foot medial shift for datasets that
            # do not carry dribble_cg_surface.
            if use_foot_yaw and body_quat_w is not None:
                yaw = _yaw_from_quat_wxyz(body_quat_w[t, fi])
            else:
                yaw = _yaw_from_quat_wxyz(body_quat_w[t, anchor_body_index])
            phi_fwd = np.array([foot_offset_x, foot_offset_y], dtype=np.float64)
            off_fwd = _yaw_rotate_xy(np.asarray([yaw]), phi_fwd)[0]
            off_side = _inward_xy(
                t,
                fid,
                body_pos_w,
                left_foot_index,
                right_foot_index,
                foot_inner_offset,
            )
            ball_xy = foot_p[:2] + off_fwd + off_side
        ball[t, 0] = ball_xy[0]
        ball[t, 1] = ball_xy[1]
        ball[t, 2] = ball_radius
        placement_foot[t] = fid

    def _lerp_gap(t0: int, t1: int, p0: np.ndarray, p1: np.ndarray, fid: int) -> None:
        """Linear XY blend for frames ``t0+1 .. t1-1`` (exclusive endpoints)."""
        if t1 <= t0 + 1:
            return
        gap = t1 - t0
        for j, t in enumerate(range(t0 + 1, t1)):
            alpha = float(j + 1) / float(gap)
            ball[t, :2] = (1.0 - alpha) * p0 + alpha * p1
            ball[t, 2] = ball_radius
            placement_foot[t] = fid

    def _fill_hybrid_gaps() -> None:
        if not segs:
            return

        # Surface labels originate from manually clicked contact instants.  The
        # label tool expands each instant to +5 frames for CG training; ball
        # synthesis must not interpret that window as six frames of rigid
        # foot-ball adhesion.  Constrain only the segment starts and interpolate
        # an independent ball trajectory through those anchors.
        surface_anchors = [
            (s, fid, _segment_surface_id(s, e))
            for s, e, fid in segs
            if _segment_surface_id(s, e) in (0, 1)
        ]
        if len(surface_anchors) == len(segs) and surface_anchors:
            for t, fid, sid in surface_anchors:
                _place_contact_frame(t, fid, sid)

            for k in range(len(surface_anchors) - 1):
                t0, _, _ = surface_anchors[k]
                t1, fid1, _ = surface_anchors[k + 1]
                _lerp_gap(t0, t1, ball[t0, :2], ball[t1, :2], fid1)

            t_first, fid_first, sid_first = surface_anchors[0]
            if t_first > 0:
                # The physical episode starts with a stationary ball at the
                # first contact's world-space target.  Placing frame 0 against
                # the future contact foot and interpolating to this anchor made
                # the ball move without contact and could start it inside the
                # robot's shoe collision geometry.
                ball[:t_first, :2] = ball[t_first, :2]
                ball[:t_first, 2] = ball_radius
                placement_foot[:t_first] = fid_first

            # Release after the final anchor with the incoming anchor-to-anchor
            # velocity.  This avoids both freezing in world space and following
            # the final +5 CG frames with a rapidly rotating foot.
            t_last, fid_last, _ = surface_anchors[-1]
            if t_last < T - 1:
                if len(surface_anchors) >= 2:
                    t_prev = surface_anchors[-2][0]
                    step_xy = (ball[t_last, :2] - ball[t_prev, :2]) / float(t_last - t_prev)
                else:
                    step_xy = body_pos_w[t_last, anchor_body_index, :2] * 0.0
                    if t_last > 0:
                        step_xy = body_pos_w[t_last, anchor_body_index, :2] - body_pos_w[t_last - 1, anchor_body_index, :2]
                    elif T > 1:
                        step_xy = body_pos_w[1, anchor_body_index, :2] - body_pos_w[0, anchor_body_index, :2]
                for t in range(t_last + 1, T):
                    ball[t, :2] = ball[t_last, :2] + float(t - t_last) * step_xy
                    ball[t, 2] = ball_radius
                    placement_foot[t] = fid_last
            return

        for s, e, fid in segs:
            for t in range(s, e + 1):
                seg_fid = int(cg_foot[t]) if cg_foot[t] >= 0 else fid
                _place_contact_frame(t, seg_fid)

        # Do this before gap interpolation: the preceding gap then approaches
        # the guarded terminal endpoint smoothly instead of snapping at contact.
        _guard_truncated_terminal_contact()

        for k in range(len(segs) - 1):
            s0, e0, _ = segs[k]
            s1, _, fid1 = segs[k + 1]
            _lerp_gap(e0, s1, ball[e0, :2], ball[s1, :2], fid1)

        s0, _, fid0 = segs[0]
        if s0 > 0:
            # Legacy contacts without surface labels use the same physical
            # pre-contact invariant as the surface-anchor path above.
            ball[:s0, :2] = ball[s0, :2]
            ball[:s0, 2] = ball_radius
            placement_foot[:s0] = fid0

        _, e_last, fid_last = segs[-1]
        if e_last < T - 1:
            ball[e_last + 1 :, :2] = ball[e_last, :2]
            ball[e_last + 1 :, 2] = ball_radius
            placement_foot[e_last + 1 :] = fid_last

    def _spawn_no_contact() -> None:
        for t in range(T):
            fid = _forward_foot_id(
                t, body_pos_w, body_quat_w, anchor_body_index, left_foot_index, right_foot_index
            )
            _place_contact_frame(t, fid)

    if mode == "hybrid":
        if segs:
            _fill_hybrid_gaps()
        else:
            _spawn_no_contact()
    elif mode == "foot_follow":
        if segs:
            for t in range(T):
                fid = _foot_id_for_frame(
                    t,
                    cg_contact,
                    cg_foot,
                    body_pos_w,
                    body_quat_w,
                    ball,
                    left_foot_index=left_foot_index,
                    right_foot_index=right_foot_index,
                    anchor_body_index=anchor_body_index,
                )
                _place_contact_frame(t, fid)
        else:
            _spawn_no_contact()
    elif mode == "lerp":
        for s, e, fid in segs:
            for t in range(s, e + 1):
                _place_contact_frame(t, fid)

        _guard_truncated_terminal_contact()

        for k in range(len(segs) - 1):
            s0, e0, _ = segs[k]
            s1, _, fid1 = segs[k + 1]
            _lerp_gap(e0, s1, ball[e0, :2], ball[s1, :2], fid1)

        if segs:
            s0, _, fid0 = segs[0]
            anchor0 = body_pos_w[0, anchor_body_index]
            yaw0 = _yaw_from_quat_wxyz(body_quat_w[0, anchor_body_index])
            spawn_xy = anchor0[:2] + _yaw_rotate_xy(np.asarray([yaw0]), np.array([init_forward_dist, 0.0]))[0]
            if s0 > 0:
                _lerp_gap(-1, s0, spawn_xy, ball[s0, :2], fid0)
                ball[0, :2] = spawn_xy
                placement_foot[0] = fid0
        else:
            _spawn_no_contact()

        if segs:
            _, e_last, fid_last = segs[-1]
            if e_last < T - 1:
                ball[e_last + 1 :, :2] = ball[e_last, :2]
                ball[e_last + 1 :, 2] = ball_radius
                placement_foot[e_last + 1 :] = fid_last
    else:
        raise ValueError(f"Unsupported traj_mode={traj_mode!r}; use 'hybrid', 'foot_follow', or 'lerp'.")

    for t in range(T):
        fid = int(placement_foot[t])
        if fid < 0:
            fid = int(ref_foot[t])
        if fid < 0:
            continue
        fi = _foot_idx(fid)
        dxy = body_pos_w[t, fi, :2] - ball[t, :2]
        dist[t] = float(np.linalg.norm(dxy))
        dist_foot[t] = fid

    return ball, dist, dist_foot, left_foot_index, right_foot_index


def synthesize_contact_flow_labels(
    ball_pos_w: np.ndarray,
    body_quat_w: np.ndarray,
    contact_anchor: np.ndarray,
    *,
    anchor_body_index: int,
    fps: float,
    min_anchor_distance: float = 1.0e-4,
) -> dict[str, np.ndarray]:
    """Build causal outgoing-ball labels between adjacent contact anchors.

    Absolute world heading is intentionally not stored. Each outgoing direction
    is expressed in the reference pelvis-yaw frame at the source touch. At
    training time it is rotated once by the simulated pelvis yaw at the actual
    touch and then latched for that segment. This makes the target invariant to
    episode spawn heading and prevents it from drifting as the pelvis turns.
    """
    T = int(ball_pos_w.shape[0])
    direction_local = np.zeros((T, 2), dtype=np.float32)
    distance = np.full(T, -1.0, dtype=np.float32)
    duration = np.full(T, -1.0, dtype=np.float32)
    anchor_frame = np.full(T, -1, dtype=np.int32)
    valid = np.zeros(T, dtype=np.int8)

    def _result() -> dict[str, np.ndarray]:
        return {
            "dribble_cg_flow_dir_local": direction_local,
            "dribble_cg_flow_distance": distance,
            "dribble_cg_flow_duration": duration,
            "dribble_cg_flow_anchor_frame": anchor_frame,
            "dribble_cg_flow_valid": valid,
        }

    if T == 0 or not np.isfinite(fps) or fps <= 0.0:
        return _result()

    anchor_idx = np.flatnonzero(np.asarray(contact_anchor).reshape(-1)[:T] > 0)
    pelvis_idx = _resolve_anchor_body_index(int(body_quat_w.shape[1]), anchor_body_index)
    for t0, t1 in zip(anchor_idx[:-1], anchor_idx[1:]):
        delta_world = np.asarray(ball_pos_w[t1, :2] - ball_pos_w[t0, :2], dtype=np.float64)
        segment_distance = float(np.linalg.norm(delta_world))
        segment_duration = float(t1 - t0) / float(fps)
        if segment_distance < min_anchor_distance or segment_duration <= 0.0:
            continue

        pelvis_yaw = float(_yaw_from_quat_wxyz(body_quat_w[t0, pelvis_idx]))
        delta_local = _yaw_rotate_xy(np.asarray([-pelvis_yaw]), delta_world)[0]
        unit_local = delta_local / segment_distance
        segment_slice = slice(int(t0), int(t1))
        direction_local[segment_slice] = unit_local.astype(np.float32)
        distance[segment_slice] = np.float32(segment_distance)
        duration[segment_slice] = np.float32(segment_duration)
        anchor_frame[segment_slice] = np.int32(t0)
        valid[segment_slice] = np.int8(1)

    return _result()


def _npz_replace(npz_path: Path, updates: dict[str, np.ndarray]) -> None:
    with np.load(npz_path, allow_pickle=True) as old:
        payload = {k: old[k] for k in old.files}
    payload.update(updates)
    np.savez(npz_path, **payload)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthesize contact geometry and contact-to-contact ball-flow labels"
    )
    parser.add_argument("--motion_path", type=str, required=True, help="Directory or single .npz")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Write updated npz copies here (source files are left unchanged).",
    )
    parser.add_argument("--ball_radius", type=float, default=0.11)
    parser.add_argument("--foot_offset_x", type=float, default=0.06, help="Ball ahead of foot (m) in foot yaw frame")
    parser.add_argument("--foot_offset_y", type=float, default=0.0, help="Extra lateral offset (m) in foot yaw frame")
    parser.add_argument(
        "--foot_inner_offset",
        type=float,
        default=0.035,
        help=(
            "Legacy medial shift (m) for motions without surface labels (0=disable)."
        ),
    )
    parser.add_argument(
        "--foot_surface_offset",
        type=float,
        default=None,
        help=(
            "Legacy absolute ball-centre clearance override (m), measured from the outer "
            "shoe collision-capsule centre line. By default the clearance is derived as "
            "--ball_radius + --foot_collision_radius."
        ),
    )
    parser.add_argument(
        "--foot_collision_radius",
        type=float,
        default=0.01,
        help=(
            "Radius (m) of the outer G1 shoe collision capsule whose centre line is at "
            "local Y=+/-foot_half_width (default: 0.01)."
        ),
    )
    parser.add_argument(
        "--foot_contact_x",
        type=float,
        default=0.10,
        help=(
            "Forefoot contact point X in ankle-roll local coordinates (m; default: 0.10). "
            "Inside/outside labels select its lateral side, not a mirrored midfoot anchor."
        ),
    )
    parser.add_argument(
        "--outside_foot_contact_x",
        type=float,
        default=0.16,
        help=(
            "Forefoot/toe contact X used only for outside-foot labels "
            "(m in ankle-roll local coordinates; default: 0.16)."
        ),
    )
    parser.add_argument(
        "--outside_lateral_inset",
        type=float,
        default=0.0,
        help=(
            "Move outside-foot ball centres toward the shoe by this distance while "
            "retaining the outside-side sign (m; default: 0.0)."
        ),
    )
    parser.add_argument(
        "--foot_half_width",
        type=float,
        default=0.026,
        help="Shoe half-width used for inside/outside surface points (m; default: 0.026).",
    )
    parser.add_argument(
        "--foot_contact_z",
        type=float,
        default=-0.025,
        help="Shoe-side contact point Z in ankle-roll local coordinates (m; default: -0.025).",
    )
    parser.add_argument(
        "--terminal_forward_limit",
        type=float,
        default=0.12,
        help=(
            "Maximum ball forward offset (m) from the body anchor for a surface-labelled "
            "contact truncated at clip end; negative disables the guard (default: 0.12)."
        ),
    )
    parser.add_argument("--init_forward_dist", type=float, default=0.45, help="Pre-contact spawn distance (m, lerp mode only)")
    parser.add_argument("--left_foot_index", type=int, default=-1, help="Left foot body index (-1 = auto)")
    parser.add_argument("--right_foot_index", type=int, default=-1, help="Right foot body index (-1 = auto)")
    parser.add_argument(
        "--anchor_body_index",
        type=int,
        default=-1,
        help="Anchor body for heading fallback (-1 = pelvis, body index 0).",
    )
    parser.add_argument(
        "--no_auto_foot_indices",
        action="store_true",
        help="Use fixed --left_foot_index/--right_foot_index (default 3/6) instead of auto-detect.",
    )
    parser.add_argument(
        "--traj_mode",
        type=str,
        default="hybrid",
        choices=("hybrid", "foot_follow", "lerp"),
        help=(
            "hybrid: surface contact-instant anchors + gap lerp (legacy adhere without surface); "
            "foot_follow: always track foot; lerp: legacy."
        ),
    )
    parser.add_argument(
        "--no_foot_yaw",
        action="store_true",
        help=(
            "Use pelvis/anchor yaw for legacy contacts without surface labels "
            "(surface-labelled contacts always use full foot yaw/pitch/roll)."
        ),
    )
    args = parser.parse_args()

    src_root = Path(args.motion_path)
    out_root = Path(args.output_dir) if args.output_dir else None
    if out_root is not None:
        out_root.mkdir(parents=True, exist_ok=True)

    files = _collect_npz_files(src_root)
    updated = 0
    for f in files:
        target = (out_root / f.name) if out_root is not None else f
        if out_root is not None:
            shutil.copy2(f, target)

        with np.load(target, allow_pickle=True) as d:
            if "body_pos_w" not in d.files or "body_quat_w" not in d.files:
                print(f"[SKIP] {f.name}: missing body_pos_w / body_quat_w")
                continue
            body_pos = np.asarray(d["body_pos_w"], dtype=np.float32)
            body_quat = np.asarray(d["body_quat_w"], dtype=np.float32)
            T = int(body_pos.shape[0])
            if "dribble_cg_contact" in d.files:
                cg_contact = np.asarray(d["dribble_cg_contact"], dtype=np.int8).reshape(-1)[:T]
            else:
                print(f"[SKIP] {f.name}: no dribble_cg_contact")
                continue
            if "dribble_cg_foot" in d.files:
                cg_foot = np.asarray(d["dribble_cg_foot"], dtype=np.int8).reshape(-1)[:T]
            else:
                cg_foot = np.full(T, -1, dtype=np.int8)
            if "dribble_cg_surface" in d.files:
                cg_surface = np.asarray(d["dribble_cg_surface"], dtype=np.int8).reshape(-1)[:T]
            else:
                cg_surface = None
            fps = float(np.asarray(d["fps"] if "fps" in d.files else 50.0).reshape(-1)[0])

        li = 3 if args.no_auto_foot_indices and args.left_foot_index < 0 else args.left_foot_index
        ri = 6 if args.no_auto_foot_indices and args.right_foot_index < 0 else args.right_foot_index

        ball, dist, dist_foot, li, ri = synthesize_ball_trajectory(
            body_pos,
            body_quat,
            cg_contact,
            cg_foot,
            cg_surface,
            left_foot_index=li,
            right_foot_index=ri,
            anchor_body_index=args.anchor_body_index,
            ball_radius=args.ball_radius,
            foot_offset_x=args.foot_offset_x,
            foot_offset_y=args.foot_offset_y,
            foot_inner_offset=args.foot_inner_offset,
            foot_surface_offset=args.foot_surface_offset,
            foot_collision_radius=args.foot_collision_radius,
            foot_contact_x=args.foot_contact_x,
            outside_foot_contact_x=args.outside_foot_contact_x,
            outside_lateral_inset=args.outside_lateral_inset,
            foot_half_width=args.foot_half_width,
            foot_contact_z=args.foot_contact_z,
            terminal_forward_limit=args.terminal_forward_limit,
            init_forward_dist=args.init_forward_dist,
            use_foot_yaw=not args.no_foot_yaw,
            traj_mode=args.traj_mode,
            auto_foot_indices=not args.no_auto_foot_indices,
        )

        contact_anchor = np.zeros(T, dtype=np.int8)
        if T > 0:
            contact_anchor[0] = np.int8(cg_contact[0] > 0)
        if T > 1:
            contact_anchor[1:] = ((cg_contact[1:] > 0) & (cg_contact[:-1] <= 0)).astype(np.int8)

        flow_labels = synthesize_contact_flow_labels(
            ball,
            body_quat,
            contact_anchor,
            anchor_body_index=args.anchor_body_index,
            fps=fps,
        )

        _npz_replace(
            target,
            {
                "ball_pos_w": ball.astype(np.float32),
                "dribble_cg_foot_ball_dist": dist.astype(np.float32),
                "dribble_cg_dist_foot": dist_foot.astype(np.int8),
                "dribble_contact_anchor": contact_anchor,
                **flow_labels,
            },
        )
        n_labeled = int(np.sum(dist >= 0))
        n_contact = int(np.sum(cg_contact > 0))
        d_med = float(np.median(dist[dist >= 0])) if n_labeled > 0 else float("nan")
        valid_flow = flow_labels["dribble_cg_flow_valid"] > 0
        n_flow_segments = int(
            np.unique(flow_labels["dribble_cg_flow_anchor_frame"][valid_flow]).size
        )
        print(
            f"[OK] {target.name}: feet L/R={li}/{ri} ball {ball.shape}, "
            f"dist_frames={n_labeled} (contact={n_contact}), median={d_med:.3f}m, "
            f"flow_segments={n_flow_segments}"
        )
        updated += 1

    print(f"[DONE] Updated {updated}/{len(files)} files.")


if __name__ == "__main__":
    main()
