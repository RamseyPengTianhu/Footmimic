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
