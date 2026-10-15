"""Constant-deceleration rolling resistance for the soccer ball.

Main-line spec (2026-10-15): every physics step apply a horizontal force of magnitude ``mu_r * m * g``
(``mu_r = 0.05``) opposite to the ball's horizontal velocity, and none when the horizontal speed is below
``v_min = 0.05 m/s``.  Unlike linear damping this gives a speed-independent deceleration, so slow dribbling
balls and fast shots both behave sensibly.

NOTE ``force_scale``: for a ball that *rolls*, a force at the centre of mass is shared with the spin
(``a = F / (m (1 + I/(m R^2)))``, 0.71 for a solid sphere), so the measured deceleration is below ``mu_r * g``.
``scripts/dribble/measure_ball_roll.py --const_mu`` measures the effective value; ``force_scale`` corrects it.

The force buffer persists between calls, so with a policy-rate call (``ball_rolling_resistance_event``)
the force is held for one control step; its magnitude is capped at ``m*speed/hold_dt`` so it cannot reverse
the ball inside that interval.
"""

from __future__ import annotations

import torch


def apply_ball_rolling_resistance(
    ball,
    mu_r: float = 0.05,
    gravity: float = 9.81,
    v_min: float = 0.05,
    hold_dt: float = 0.005,
    force_scale: float = 1.0,
) -> torch.Tensor:
    """Set the ball's external force buffer; returns the applied force magnitude [N] per env."""
    vel = ball.data.root_lin_vel_w[:, :2]
    speed = torch.linalg.vector_norm(vel, dim=-1)
    mass = ball.data.default_mass.to(vel.device).view(-1)
    magnitude = torch.minimum(mu_r * gravity * force_scale * mass, mass * speed / max(hold_dt, 1e-6))
    magnitude = torch.where(speed > v_min, magnitude, torch.zeros_like(magnitude))
    direction = -vel / speed.clamp(min=1e-6).unsqueeze(-1)
    forces = torch.zeros(vel.shape[0], 1, 3, device=vel.device)
    forces[:, 0, :2] = magnitude.unsqueeze(-1) * direction
    ball.set_external_force_and_torque(forces, torch.zeros_like(forces), is_global=True)
    return magnitude


def ball_rolling_resistance_event(
    env,
    env_ids,
    mu_r: float = 0.05,
    gravity: float = 9.81,
    v_min: float = 0.05,
    force_scale: float = 1.0,
    asset_name: str = "soccer_ball",
) -> None:
    """Interval event (every control step) wrapper around :func:`apply_ball_rolling_resistance`."""
    apply_ball_rolling_resistance(
        env.scene[asset_name], mu_r, gravity, v_min, hold_dt=float(env.step_dt), force_scale=force_scale
    )


def apply_ball_velocity_level_deceleration(
    ball,
    decel: float = 0.05 * 9.81,
    v_min: float = 0.05,
    dt: float = 0.005,
) -> torch.Tensor:
    """Engine-independent constant deceleration applied directly to the ball's velocity.

    Reduces the horizontal speed by ``decel * dt`` per call and scales the horizontal angular velocity by the same
    ratio, so a rolling ball stays rolling and the *effective* deceleration is exactly ``decel`` regardless of the
    ball's moment of inertia (a centre-of-mass force is shared with the spin, and the share differs between
    engines).  Call once per physics step.  Returns the new horizontal speed.
    """
    lin = ball.data.root_lin_vel_w.clone()
    ang = ball.data.root_ang_vel_w.clone()
    speed = torch.linalg.vector_norm(lin[:, :2], dim=-1)
    new_speed = torch.where(speed > v_min, (speed - decel * dt).clamp(min=0.0), speed)
    ratio = new_speed / speed.clamp(min=1e-6)
    ratio = torch.where(speed > v_min, ratio, torch.ones_like(ratio))
    lin[:, :2] *= ratio.unsqueeze(-1)
    ang[:, :2] *= ratio.unsqueeze(-1)
    ball.write_root_velocity_to_sim(torch.cat([lin, ang], dim=-1))
    return new_speed


def ball_constant_deceleration_event(
    env,
    env_ids,
    decel: float = 0.05 * 9.81,
    v_min: float = 0.05,
    asset_name: str = "soccer_ball",
) -> None:
    """Interval event (every control step) wrapper around :func:`apply_ball_velocity_level_deceleration`.

    Pair it with ``apply_ball_damping(scene_cfg, 0.0, 0.0)``: the asset's own residual damping otherwise adds a
    speed-dependent deceleration on top (measured +0.06 m/s^2 at 5 m/s, +0.12 at 9 m/s).
    """
    apply_ball_velocity_level_deceleration(env.scene[asset_name], decel, v_min, dt=float(env.step_dt))
