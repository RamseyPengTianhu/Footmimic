"""V3.6 Phase-dependent tracking multiplier logic.

Implements a smooth decay schedule based on reference kick_frame.
"""

import torch
from isaaclab.envs import ManagerBasedRLEnv
from soccer.tasks.tracking.mdp.commands_multi_motion_soccer import MotionCommand

def compute_tracking_multipliers(env: ManagerBasedRLEnv, command_name: str = "motion") -> dict[str, torch.Tensor]:
    """Computes a dictionary of per-env tracking multipliers based on current phase.
    
    Channels:
        - body_pos
        - body_ori
        - swing_foot
        - support_foot
        - joint
        - anchor_ori
    
    Phases:
        - Approach: t < kf - 15  (Multiplier = 1.0)
        - Pre-strike: kf - 15 <= t < kf - 3 (Linear decay to Strike values)
        - Strike: kf - 3 <= t <= kef + 5 (Strike values)
        - Follow-through: kef + 5 < t <= kef + 20 (Linear interp to FT values)
        - Recovery: t > kef + 20 (FT values)
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    kf = command.kick_frame
    kef = command.kick_end_frame
    t = command.time_steps
    has_annotation = kf >= 0

    N = env.num_envs
    device = env.device

    # Define base values
    strike_vals = {
        "body_pos": 0.4,
        "body_ori": 0.6,
        "swing_foot": 0.0,
        "support_foot": 0.7,
        "joint": 0.5,
        "anchor_ori": 0.6,
    }
    
    ft_vals = {
        "body_pos": 0.5,
        "body_ori": 0.7,
        "swing_foot": 0.3,
        "support_foot": 0.5,
        "joint": 0.5,
        "anchor_ori": 0.7,
    }

    # Initialize all to 1.0 (Approach)
    out = {k: torch.ones(N, device=device) for k in strike_vals.keys()}

    # Helper for unannotated
    if not torch.any(has_annotation):
        return out

    # Calculate interpolation weights
    # 1. Pre-strike decay (1.0 -> Strike)
    prestrike_start = kf - 15
    strike_start = kf - 3
    prestrike_mask = has_annotation & (t >= prestrike_start) & (t < strike_start)
    prestrike_w = ((t - prestrike_start).float() / 12.0).clamp(0.0, 1.0)

    # 2. Strike hold (Strike)
    effective_kef = torch.where(kef >= 0, kef, kf + 5)
    strike_end = effective_kef + 5
    strike_mask = has_annotation & (t >= strike_start) & (t <= strike_end)

    # 3. Follow-through recovery (Strike -> FT)
    ft_end = effective_kef + 20
    ft_mask = has_annotation & (t > strike_end) & (t <= ft_end)
    ft_w = ((t - strike_end).float() / 15.0).clamp(0.0, 1.0)

    # 4. Final recovery (FT)
    recov_mask = has_annotation & (t > ft_end)

    # Apply values
    for k in out.keys():
        s_val = strike_vals[k]
        f_val = ft_vals[k]

        # Pre-strike (1.0 to s_val)
        out[k][prestrike_mask] = 1.0 - prestrike_w[prestrike_mask] * (1.0 - s_val)
        
        # Strike (s_val)
        out[k][strike_mask] = s_val
        
        # Follow-through (s_val to f_val)
        out[k][ft_mask] = s_val + ft_w[ft_mask] * (f_val - s_val)
        
        # Recovery (f_val)
        out[k][recov_mask] = f_val

    return out
