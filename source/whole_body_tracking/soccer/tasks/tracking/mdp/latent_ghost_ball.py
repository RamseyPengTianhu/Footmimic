"""Wide-distribution ghost ball for Stage 1' (dribble line, D-b).

Stage 1' never touches the ball, but its ball observation must already cover what the real ball will do in Stage 2':
the clean-v0 probe showed that a ball left 2-3 m behind the robot is out of the observation range Stage 1 was trained
on (ball always 0.25-0.8 m in front) and made the policy fall.  Here the collision-free ghost lives in the pelvis-yaw
frame with its own relative velocity:

  reset  half the episodes start in the old "in front" box, half anywhere in a wide box (behind, sideways, far);
         one of three motion modes (equal probability):
           body-still   at rest relative to the body (moves with it)
           world-still  at rest on the ground: the robot walks past it, so it slides backward in the pelvis frame --
                        exactly the "ball left behind" case; when it leaves the box it is placed in front again
           random       a random relative velocity up to ``max_speed``, reflected at the box walls, occasionally
                        re-drawn
The ball velocity observation is the world velocity expressed in the pelvis frame, so world-still gives 0 and
body-still gives the body velocity.

The world pose and velocity are written every control step after the command update, so the observation of the same
step sees them.  Nothing here touches the robot, the rewards or the shared interface modules.
"""

from __future__ import annotations

import math

import torch
from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat

BALL_RADIUS = 0.11


BODY_STILL, WORLD_STILL, RANDOM = 0, 1, 2


class _GhostState:
    def __init__(self, n: int, device):
        self.off = torch.zeros(n, 2, device=device)
        self.vel = torch.zeros(n, 2, device=device)
        self.mode = torch.zeros(n, dtype=torch.long, device=device)


def _state(env) -> _GhostState:
    st = getattr(env, "_latent_ghost_state", None)
    if st is None:
        st = _GhostState(env.num_envs, env.device)
        env._latent_ghost_state = st
    return st


def _uniform(lo: float, hi: float, n: int, device) -> torch.Tensor:
    return lo + (hi - lo) * torch.rand(n, device=device)


def _random_velocity(n: int, max_speed: float, device) -> torch.Tensor:
    ang = _uniform(-math.pi, math.pi, n, device)
    spd = _uniform(0.0, max_speed, n, device)
    return torch.stack([spd * torch.cos(ang), spd * torch.sin(ang)], dim=-1)


def ghost_ball_reset(
    env,
    env_ids: torch.Tensor,
    near_box: tuple = ((0.15, 0.70), (-0.40, 0.40)),
    wide_box: tuple = ((-2.0, 2.5), (-2.0, 2.0)),
    near_prob: float = 0.5,
    max_speed: float = 2.0,
) -> None:
    """Reset event: sample the ghost's pelvis-yaw-frame offset and relative velocity."""
    st = _state(env)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    n, dev = len(env_ids), env.device
    near = torch.rand(n, device=dev) < near_prob
    box_x = [torch.where(near, torch.full((n,), a, device=dev), torch.full((n,), b, device=dev)) for a, b in zip(near_box[0], wide_box[0])]
    box_y = [torch.where(near, torch.full((n,), a, device=dev), torch.full((n,), b, device=dev)) for a, b in zip(near_box[1], wide_box[1])]
    st.off[env_ids, 0] = box_x[0] + (box_x[1] - box_x[0]) * torch.rand(n, device=dev)
    st.off[env_ids, 1] = box_y[0] + (box_y[1] - box_y[0]) * torch.rand(n, device=dev)
    mode = torch.randint(0, 3, (n,), device=dev)
    st.mode[env_ids] = mode
    st.vel[env_ids] = torch.where((mode == RANDOM).unsqueeze(-1), _random_velocity(n, max_speed, dev), torch.zeros(n, 2, device=dev))


def ghost_ball_step(
    env,
    env_ids,
    wide_box: tuple = ((-2.0, 2.5), (-2.0, 2.0)),
    max_speed: float = 2.0,
    change_prob: float = 0.01,
    command_name: str = "motion",
    asset_name: str = "soccer_ball",
) -> None:
    """Interval event (every control step, global time): advance the ghost and write its pose and velocity."""
    st = _state(env)
    dt = float(env.step_dt)
    command = env.command_manager.get_term(command_name)
    robot = env.scene[command.cfg.asset_name]
    ball = env.scene[asset_name]
    q = yaw_quat(command.robot_anchor_quat_w)
    pad = torch.zeros(env.num_envs, 1, device=env.device)
    body_vel_h = quat_apply_inverse(q, robot.data.root_lin_vel_w)[:, :2]  # body velocity in the pelvis-yaw frame
    world = st.mode == WORLD_STILL
    rand = st.mode == RANDOM
    # relative velocity: random mode uses its own; world-still moves backward with the body; body-still is 0
    rel = torch.where(world.unsqueeze(-1), -body_vel_h, st.vel)
    st.off += rel * dt
    for axis in (0, 1):
        lo, hi = wide_box[axis]
        out = (st.off[:, axis] < lo) | (st.off[:, axis] > hi)
        st.vel[:, axis] = torch.where(out & rand, -st.vel[:, axis], st.vel[:, axis])
        st.off[:, axis] = st.off[:, axis].clamp(lo, hi)
    # a world-still ball that left the box is placed in front again (a new ball to dribble)
    left = world & ((st.off[:, 0] <= wide_box[0][0]) | (st.off[:, 1].abs() >= wide_box[1][1]))
    if bool(left.any()):
        k = int(left.sum())
        st.off[left, 0] = _uniform(0.15, 0.70, k, env.device)
        st.off[left, 1] = _uniform(-0.40, 0.40, k, env.device)
    change = rand & (torch.rand(env.num_envs, device=env.device) < change_prob)
    if bool(change.any()):
        st.vel[change] = _random_velocity(int(change.sum()), max_speed, env.device)

    pos = command.robot_anchor_pos_w + quat_apply(q, torch.cat([st.off, pad], dim=-1))
    pos[:, 2] = env.scene.env_origins[:, 2] + BALL_RADIUS
    # world velocity: body + relative (world-still -> 0)
    vel = robot.data.root_lin_vel_w.clone()
    vel[:, :2] += quat_apply(q, torch.cat([rel, pad], dim=-1))[:, :2]
    vel[:, 2] = 0.0
    quat = torch.zeros(env.num_envs, 4, device=env.device)
    quat[:, 0] = 1.0
    ball.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1))
    ball.write_root_velocity_to_sim(torch.cat([vel, torch.zeros_like(vel)], dim=-1))
