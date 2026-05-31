"""V3.6b phase tracking schedule.

Compared with v36a, this keeps a small swing-leg prior in strike and makes the
support foot prior stronger.  Exact foot placement is still relaxed; the goal
is to preserve kicking morphology while ball geometry is handled by rewards.
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from soccer.tasks.tracking.mdp.commands_multi_motion_soccer import MotionCommand


def compute_tracking_multipliers(env: ManagerBasedRLEnv, command_name: str = "motion") -> dict[str, torch.Tensor]:
    command: MotionCommand = env.command_manager.get_term(command_name)
    kf = command.kick_frame
    kef = command.kick_end_frame
    t = command.time_steps
    has_annotation = kf >= 0

    num_envs = env.num_envs
    device = env.device

    strike_vals = {
        "body_pos": 0.45,
        "body_ori": 0.65,
        "swing_foot": 0.15,
        "support_foot": 0.85,
        "joint": 0.60,
        "anchor_ori": 0.65,
    }

    ft_vals = {
        "body_pos": 0.55,
        "body_ori": 0.75,
        "swing_foot": 0.35,
        "support_foot": 0.60,
        "joint": 0.55,
        "anchor_ori": 0.75,
    }

    out = {key: torch.ones(num_envs, device=device) for key in strike_vals}
    if not torch.any(has_annotation):
        return out

    prestrike_start = kf - 18
    strike_start = kf - 3
    prestrike_mask = has_annotation & (t >= prestrike_start) & (t < strike_start)
    prestrike_w = ((t - prestrike_start).float() / 15.0).clamp(0.0, 1.0)

    effective_kef = torch.where(kef >= 0, kef, kf + 8)
    strike_end = effective_kef + 8
    strike_mask = has_annotation & (t >= strike_start) & (t <= strike_end)

    ft_end = effective_kef + 24
    ft_mask = has_annotation & (t > strike_end) & (t <= ft_end)
    ft_w = ((t - strike_end).float() / 16.0).clamp(0.0, 1.0)

    recovery_mask = has_annotation & (t > ft_end)

    for key in out:
        s_val = strike_vals[key]
        f_val = ft_vals[key]
        out[key][prestrike_mask] = 1.0 - prestrike_w[prestrike_mask] * (1.0 - s_val)
        out[key][strike_mask] = s_val
        out[key][ft_mask] = s_val + ft_w[ft_mask] * (f_val - s_val)
        out[key][recovery_mask] = f_val

    return out
