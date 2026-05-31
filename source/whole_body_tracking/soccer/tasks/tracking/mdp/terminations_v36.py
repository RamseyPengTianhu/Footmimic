"""V3.6 Phase-free interaction termination.

Replaces the kick_end_frame-based termination with a simple check at the end
of the episode to ensure the ball was actually struck with sufficient force.
"""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from soccer.tasks.tracking.mdp.commands_multi_motion_soccer import MotionCommand


def phase_free_interaction_fail(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_speed_threshold: float = 1.0,
    margin: int = 5,
) -> torch.Tensor:
    """Terminate with failure if the ball is not moving at the end of the episode.
    
    Evaluates only in the final `margin` frames of the motion. If the ball speed
    is below the threshold, the episode is marked as terminated (failure), which
    provides a negative value penalty.
    
    This replaces `interaction_termination` which used `kick_end_frame`.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    t = command.time_steps
    end_step = (command.motion_length - margin).clamp(min=0)
    
    at_end = t >= end_step
    
    if not torch.any(at_end):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        
    soccer_ball = env.scene["soccer_ball"]
    ball_speed_xy = torch.norm(soccer_ball.data.root_lin_vel_w[:, :2], dim=-1)
    
    # If the episode is ending and the ball is barely moving, they failed to kick it.
    failed = at_end & (ball_speed_xy < ball_speed_threshold)
    
    return failed
