"""Interface-v1 observation terms for the dribble line (D-a).

Builds the student observation groups of docs/milestones/SKILL_INTERFACE_V1_PLAN_20261016_CN.md section 2.5 on
top of the shared, torch-only module ``soccer.latent.skill_event`` (which is not modified here):

  ball (10)      gated ball observation, possession flag fixed to 1 for D-a
  event (13)     current touch event from the clip's ``dribble_cg_*`` labels
  phase (4)      event-driven soft phase (shared ``SoftPhase``)
  next (11)      preview of the following event
  ref window     encoder input (training only): future reference joint positions + current reference
                 joint velocity + reference pelvis angular velocity

Event construction from labels (one event per contiguous ``dribble_cg_contact`` segment):
  foot     ``dribble_cg_foot`` (1 = right)
  region   ``dribble_surface_to_region(dribble_cg_surface)`` (inner / outer)
  target   outgoing flow ``dir * distance`` at the first ``dribble_cg_flow_valid`` frame of the segment (reference
           pelvis-yaw frame at the touch), rotated into the current reference heading frame
  height 0, mode ARRIVE, lifted 0
The "current" event is the first one whose window has not ended yet.  Clips without labels (stand / walk) have no
event: event, phase and next are all zero (see DRIBBLE_HANDOFF.md, interface requests).

Soft phase (D-a, ghost ball): the shared ``SoftPhase`` is driven by the *reference* geometry -- distance between the
reference ball and the event region on the reference foot -- and ``touched`` = reference frame inside the labelled
window, because a ghost ball cannot trigger a real touch.  Stage 2' replaces both with the simulated ball and contact
sensor.  ``SoftPhase.update`` is called once per observation compute, so the phase term must only appear in one group.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat

from soccer.latent import skill_event as se

DRIBBLE_REACH = 0.18  # m; touches happen at 0.12-0.16 m ball-centre to region-centre in both references

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _yaw(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _rot2(v: torch.Tensor, ang: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(ang), torch.sin(ang)
    return torch.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1]], dim=-1)


class _DribbleEventCache:
    """Per-motion event tables and per-env soft-phase state, attached to the motion command."""

    def __init__(self, env: ManagerBasedRLEnv, command):
        m = command.motion
        dev = command.device
        self.device = dev
        names = list(command.cfg.body_names)
        self.foot_ref_idx = (names.index("left_ankle_roll_link"), names.index("right_ankle_roll_link"))
        robot = env.scene[command.cfg.asset_name]
        self.foot_robot_idx = (robot.find_bodies("left_ankle_roll_link")[0][0], robot.find_bodies("right_ankle_roll_link")[0][0])
        self.anchor = command.motion_anchor_body_index

        contact = m.dribble_cg_contact > 0  # [M, T]
        num_m, T = contact.shape
        lengths = m.file_lengths
        events = []
        for i in range(num_m):
            c = contact[i, : int(lengths[i])]
            prev = torch.cat([torch.zeros(1, dtype=torch.bool, device=dev), c[:-1]])
            nxt = torch.cat([c[1:], torch.zeros(1, dtype=torch.bool, device=dev)])
            starts = torch.nonzero(c & ~prev).flatten().tolist()
            ends = torch.nonzero(c & ~nxt).flatten().tolist()
            rows = []
            for a, b in zip(starts, ends):
                valid = torch.nonzero(m.dribble_cg_flow_valid[i, a : b + 1] > 0).flatten()
                if valid.numel():
                    j = a + int(valid[0])
                    target = m.dribble_cg_flow_dir_local[i, j] * m.dribble_cg_flow_distance[i, j].clamp(min=0.0)
                else:
                    target = torch.zeros(2, device=dev)
                rows.append((a, b, int(m.dribble_cg_foot[i, a]) == 1, int(m.dribble_cg_surface[i, a]), target))
            events.append(rows)
        K = max(1, max(len(r) for r in events))
        self.ev_start = torch.full((num_m, K), -1, dtype=torch.long, device=dev)
        self.ev_end = torch.full_like(self.ev_start, -1)
        self.ev_right = torch.zeros((num_m, K), dtype=torch.bool, device=dev)
        self.ev_region = torch.full((num_m, K), se.UNSPECIFIED, dtype=torch.long, device=dev)
        self.ev_target = torch.zeros((num_m, K, 2), device=dev)
        self.ev_count = torch.tensor([len(r) for r in events], dtype=torch.long, device=dev)
        for i, rows in enumerate(events):
            for k, (a, b, right, surf, target) in enumerate(rows):
                self.ev_start[i, k], self.ev_end[i, k], self.ev_right[i, k] = a, b, right
                self.ev_region[i, k] = int(se.dribble_surface_to_region(torch.tensor([surf]))[0])
                self.ev_target[i, k] = target
        # current event per frame: first event whose window end >= frame
        self.cur = torch.full((num_m, T), -1, dtype=torch.long, device=dev)
        frames = torch.arange(T, device=dev)
        for i in range(num_m):
            for k in range(int(self.ev_count[i]) - 1, -1, -1):
                self.cur[i, frames <= self.ev_end[i, k]] = k
        n = env.num_envs
        # Dribble-line soft-phase settings (env side; the shared SoftPhase class is used unchanged):
        # reach 0.18 m instead of the shoot/pass default 0.25 m, plus "re-arm" and "no miss rule", implemented by
        # gating the distance fed to SoftPhase.update (see latent_soft_phase and DRIBBLE_HANDOFF.md section 17).
        self.phase = se.SoftPhase(n, dev, reach=DRIBBLE_REACH)
        self.armed = torch.ones(n, dtype=torch.bool, device=dev)
        self.last_key = torch.full((n,), -2, dtype=torch.long, device=dev)
        self.num_events_total = int(self.ev_count.sum())

    # --- per-step lookups --------------------------------------------------------------------------
    def current(self, command):
        mi = command.motion_idx.long()
        f = command.time_steps.long().clamp(min=0, max=self.cur.shape[1] - 1)
        k = self.cur[mi, f]
        has = k >= 0
        kk = k.clamp(min=0)
        return mi, f, k, has, kk

    def event_fields(self, mi, kk, has):
        right = self.ev_right[mi, kk] & has
        region = torch.where(has, self.ev_region[mi, kk], torch.full_like(kk, se.UNSPECIFIED))
        target = self.ev_target[mi, kk] * has.unsqueeze(-1)
        return right, region, target

    def to_heading(self, command, mi, f, kk, target_anchor):
        """Rotate a target given in the reference pelvis-yaw frame at the touch into the current heading frame."""
        q = command.motion.body_quat_w[:, :, self.anchor]
        a = self.ev_start[mi, kk].clamp(min=0)
        return _rot2(target_anchor, _yaw(q[mi, a]) - _yaw(q[mi, f]))


def _cache(env: ManagerBasedRLEnv, command_name: str) -> tuple[object, _DribbleEventCache]:
    command = env.command_manager.get_term(command_name)
    cache = getattr(command, "_latent_v1_cache", None)
    if cache is None:
        cache = _DribbleEventCache(env, command)
        command._latent_v1_cache = cache
    return command, cache


def _region_world(robot_or_ref_pos, quat, region, right):
    return robot_or_ref_pos + quat_apply(quat, se.region_center(region, right))


# --- observation terms -------------------------------------------------------------------------------


def latent_ball(env: ManagerBasedRLEnv, command_name: str = "motion", possession: float = 1.0) -> torch.Tensor:
    """Gated ball observation (10): pelvis-frame ball position and velocity, ball offset from the current event's
    foot region (pelvis frame), possession flag."""
    command, c = _cache(env, command_name)
    robot = env.scene[command.cfg.asset_name]
    ball = env.scene["soccer_ball"]
    pel_p, pel_q = command.robot_anchor_pos_w, command.robot_anchor_quat_w
    pos_b = quat_apply_inverse(pel_q, ball.data.root_pos_w - pel_p)
    vel_b = quat_apply_inverse(pel_q, ball.data.root_lin_vel_w)
    mi, f, k, has, kk = c.current(command)
    right, region, _ = c.event_fields(mi, kk, has)
    fid = torch.where(right, c.foot_robot_idx[1], c.foot_robot_idx[0])
    ar = torch.arange(env.num_envs, device=env.device)
    region_w = _region_world(robot.data.body_pos_w[ar, fid], robot.data.body_quat_w[ar, fid], region, right)
    from_region_b = quat_apply_inverse(pel_q, ball.data.root_pos_w - region_w) * has.unsqueeze(-1)
    p = torch.full((env.num_envs,), float(possession), device=env.device)
    return se.ball_obs(pos_b, vel_b, from_region_b, p)


def latent_current_event(env: ManagerBasedRLEnv, command_name: str = "motion") -> torch.Tensor:
    """Current touch event (13); all zero for clips without labels."""
    command, c = _cache(env, command_name)
    mi, f, k, has, kk = c.current(command)
    right, region, target = c.event_fields(mi, kk, has)
    target_h = c.to_heading(command, mi, f, kk, target) * has.unsqueeze(-1)
    z = torch.zeros(env.num_envs, device=env.device)
    ev = se.encode_event(right, region, z, target_h, z + se.MODE_ARRIVE, z)
    return ev * has.unsqueeze(-1)


def latent_next_event(env: ManagerBasedRLEnv, command_name: str = "motion") -> torch.Tensor:
    """Preview of the event after the current one (11); zero when there is none."""
    command, c = _cache(env, command_name)
    mi, f, k, has, kk = c.current(command)
    k2 = (k + 1).clamp(min=0, max=c.ev_start.shape[1] - 1)
    has_next = has & ((k + 1) < c.ev_count[mi])
    right, region, target = c.event_fields(mi, k2, has_next)
    target_h = c.to_heading(command, mi, f, k2, target) * has_next.unsqueeze(-1)
    return se.encode_next(has_next, right, region, target_h)


def latent_soft_phase(env: ManagerBasedRLEnv, command_name: str = "motion") -> torch.Tensor:
    """Event-driven soft phase (4), driven by the reference ball / reference foot geometry for D-a."""
    command, c = _cache(env, command_name)
    m = command.motion
    mi, f, k, has, kk = c.current(command)
    right, region, _ = c.event_fields(mi, kk, has)
    fidx = torch.where(right, c.foot_ref_idx[1], c.foot_ref_idx[0])
    foot_p = m.body_pos_w[mi, f, fidx]
    foot_q = m.body_quat_w[mi, f, fidx]
    region_w = _region_world(foot_p, foot_q, region, right)
    dist = torch.linalg.vector_norm(m.ball_pos_w[mi, f] - region_w, dim=-1)
    touched = has & (f >= c.ev_start[mi, kk]) & (f <= c.ev_end[mi, kk])
    key = torch.where(has, mi * 10000 + kk, torch.full_like(kk, -1))
    new = key != c.last_key
    c.phase.start_event(new & has, dist)
    c.last_key = key
    # Re-arm: a new event that starts with the ball already inside reach (normal in dribbling, the ball stays near
    # the foot) may only enter "contact" after the ball has left reach once.
    reach = c.phase.reach
    c.armed = torch.where(new & has, dist > reach, c.armed | (dist > reach))
    in_approach = c.phase.stage == se.APPROACH
    in_contact = c.phase.stage == se.CONTACT
    d = torch.where(in_approach & ~c.armed, torch.maximum(dist, reach + 1e-3), dist)
    # No miss rule for dribbling: leave "contact" only on the touch, never because the ball drifted away.
    d = torch.where(in_contact, torch.minimum(d, 1.5 * reach), d)
    out = c.phase.update(d, touched)
    return out * has.unsqueeze(-1)


def latent_ref_window(env: ManagerBasedRLEnv, command_name: str = "motion") -> torch.Tensor:
    """Encoder input (177): reference joint positions at future offsets, current reference joint velocity and
    reference pelvis angular velocity in the pelvis frame.  Training only."""
    command, c = _cache(env, command_name)
    m = command.motion
    mi = command.motion_idx.long()
    f = command.time_steps.long()
    last = (m.file_lengths[mi] - 1).clamp(min=0)
    jp = [m.joint_pos[mi, torch.minimum(f + o, last)] for o in se.REF_OFFSETS]
    jv = m.joint_vel[mi, torch.minimum(f, last)]
    q = m.body_quat_w[mi, torch.minimum(f, last), c.anchor]
    w = quat_apply_inverse(q, m.body_ang_vel_w[mi, torch.minimum(f, last), c.anchor])
    return torch.cat(jp + [jv, w], dim=-1)
