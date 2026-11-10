"""Pure-torch math for the command-defined ball rewards (no Isaac Lab imports).

Both terms are defined only from the *current* ball state and the *current* locomotion
command, so they carry no reference ball trajectory and no event/contact gate:

* front zone   : the ball sits in a region in front of the pelvis (pelvis-yaw frame).
* velocity follow : the ball's velocity (pelvis-yaw frame) follows the commanded body
  velocity ``[vx, vy]`` in direction and magnitude.

Together they are hard to satisfy without actually moving the ball: a ball left behind
leaves the zone, and a ball standing still has zero velocity against a non-zero command.
"""

from __future__ import annotations

import torch


def front_zone_reward(
    ball_xy_local: torch.Tensor,
    x_range: tuple[float, float] = (0.15, 0.70),
    y_half_width: float = 0.40,
    softness: float = 0.25,
) -> torch.Tensor:
    """Smooth box reward: 1 inside the zone, Gaussian fall-off ``softness`` outside it."""
    x, y = ball_xy_local[..., 0], ball_xy_local[..., 1]
    dx = torch.clamp(x_range[0] - x, min=0.0) + torch.clamp(x - x_range[1], min=0.0)
    dy = torch.clamp(y.abs() - y_half_width, min=0.0)
    return torch.exp(-(dx * dx + dy * dy) / (softness * softness))


def velocity_follow_reward(
    ball_vel_xy_local: torch.Tensor,
    command_xy: torch.Tensor,
    std: float = 0.5,
    shape: str = "exponential",
) -> torch.Tensor:
    """Reward the ball velocity matching the commanded body velocity (vector error).

    ``exponential`` keeps a long-range gradient (error 1.3 m/s -> 0.07 at std 0.5);
    ``gaussian`` is available but is ~0 for large errors.
    """
    err = torch.linalg.vector_norm(ball_vel_xy_local - command_xy, dim=-1)
    if shape == "exponential":
        return torch.exp(-err / std)
    if shape == "gaussian":
        return torch.exp(-(err * err) / (std * std))
    raise ValueError(f"unknown shape {shape!r}")


class CycleVelocityFollow:
    """Cycle-averaged velocity follow (stateful, pure torch).

    A *cycle* is one full contact-surface cycle of the contact clock (Reference 1: out-out-in-in = 4 touches,
    Reference 2: in-out = 2).  Between two cycle boundaries the object accumulates the commanded world velocity
    and, at the next boundary, compares

        v_ball_avg = (ball_xy - ball_xy_at_cycle_start) / elapsed
        v_cmd_avg  = (integral of commanded world velocity) / elapsed

    returning ``exp(-|v_ball_avg - v_cmd_avg| / std)`` as a pulse on the boundary step (zero elsewhere).  Within a
    full cycle the left/right (inside/outside) lateral swings cancel, which is why the single-interval version
    topped out at ~0.45 and this one at 0.78-0.89 on the references.

    The boundary mask comes from the reference contact clock, never from detected physical contact, so a missed
    touch cannot suppress the signal.  The first boundary after ``reset`` only starts a cycle (no pulse).
    """

    def __init__(self, num_envs: int, device, std: float = 0.5):
        self.std = float(std)
        self.start_pos = torch.zeros(num_envs, 2, device=device)
        self.cmd_sum = torch.zeros(num_envs, 2, device=device)
        self.elapsed = torch.zeros(num_envs, device=device)
        self.active = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.last_score = torch.zeros(num_envs, device=device)

    def reset(self, env_ids: torch.Tensor) -> None:
        self.cmd_sum[env_ids] = 0.0
        self.elapsed[env_ids] = 0.0
        self.active[env_ids] = False
        self.last_score[env_ids] = 0.0

    def step(self, ball_xy_w: torch.Tensor, cmd_xy_w: torch.Tensor, boundary: torch.Tensor, dt: float) -> torch.Tensor:
        self.cmd_sum += cmd_xy_w * dt * self.active.unsqueeze(-1)
        self.elapsed += dt * self.active
        emit = boundary & self.active & (self.elapsed > 1e-6)
        t = self.elapsed.clamp(min=1e-6).unsqueeze(-1)
        err = torch.linalg.vector_norm((ball_xy_w - self.start_pos) / t - self.cmd_sum / t, dim=-1)
        score = torch.where(emit, torch.exp(-err / self.std), torch.zeros_like(err))
        self.last_score = torch.where(emit, score, self.last_score)
        restart = boundary
        self.start_pos = torch.where(restart.unsqueeze(-1), ball_xy_w, self.start_pos)
        self.cmd_sum = torch.where(restart.unsqueeze(-1), torch.zeros_like(self.cmd_sum), self.cmd_sum)
        self.elapsed = torch.where(restart, torch.zeros_like(self.elapsed), self.elapsed)
        self.active = self.active | restart
        return score
