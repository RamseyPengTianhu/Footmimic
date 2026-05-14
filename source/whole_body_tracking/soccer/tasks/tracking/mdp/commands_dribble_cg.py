"""Dribbling motion command with XGen-style demo ball stitching.

Uses per-frame ``ball_pos_w`` from motion (when present) and the same yaw-only
anchor transform as body tracking targets so the ball follows the stitched
interaction trajectory. Optional ``dribble_cg_snap_mode``:

- ``full`` (default): every step writes the demo ball pose into simulation.
- ``non_contact_only``: only overwrite the ball when the CG label says
  non-contact, leaving physics during annotated contact segments.

Contact / foot masks come from ``dribble_cg_contact`` / ``dribble_cg_foot`` in
``.npz``, or from ``kick_frame`` / ``kick_end_frame`` / ``kick_leg`` fallback
(see :class:`MultiMotionLoader`).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_inv, quat_mul, yaw_quat

from .commands_multi_motion_soccer import MotionCommand, MotionCommandCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class DribbleCGMotionCommand(MotionCommand):
    """Soccer motion command + demo ball sync for dribbling CG."""

    def _use_demo_ball(self) -> bool:
        return bool(getattr(self.cfg, "dribble_cg_use_demo_ball", True))

    def get_dribble_demo_ball_goal_world(self) -> tuple[torch.Tensor, torch.Tensor]:
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if not self._use_demo_ball():
            mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        else:
            mask = self.motion.motion_has_ball_demo[self.motion_idx]
        goal = self._demo_ball_world(env_ids)
        return goal, mask

    def _demo_ball_world(self, env_ids: torch.Tensor) -> torch.Tensor:
        """World-frame demo ball positions for env_ids (aligned to anchor tracking)."""
        mi = self.motion_idx[env_ids]
        ts = self.time_steps[env_ids]
        env_origins = self._env.scene.env_origins[env_ids]

        mb = self.motion.ball_pos_w[mi, ts] + env_origins
        ma = self.anchor_pos_w[env_ids]
        ra = self.robot_anchor_pos_w[env_ids]
        delta = ra.clone()
        delta[:, 2] = ma[:, 2]

        anchor_quat = self.anchor_quat_w[env_ids]
        robot_quat = self.robot_anchor_quat_w[env_ids]
        dq = yaw_quat(quat_mul(robot_quat, quat_inv(anchor_quat)))
        rel = mb - ma
        return delta + quat_apply(dq, rel)

    def _demo_ball_lin_vel_w(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Finite-difference demo ball linear velocity in world frame."""
        mi = self.motion_idx[env_ids]
        ts = self.time_steps[env_ids]
        ts_prev = torch.clamp(ts - 1, min=0)
        env_origins = self._env.scene.env_origins[env_ids]

        mb0 = self.motion.ball_pos_w[mi, ts_prev] + env_origins
        mb1 = self.motion.ball_pos_w[mi, ts] + env_origins
        fps = float(self.motion.fps)
        if isinstance(fps, torch.Tensor):
            fps = float(fps.item())
        return (mb1 - mb0) * fps

    def _should_snap_demo_ball(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Per-env bool: write sim ball from demo this step."""
        if not self._use_demo_ball():
            return torch.zeros(env_ids.numel(), device=self.device, dtype=torch.bool)
        mi = self.motion_idx[env_ids]
        has_demo = self.motion.motion_has_ball_demo[mi]
        mode = str(getattr(self.cfg, "dribble_cg_snap_mode", "full")).lower().strip()
        if mode == "non_contact_only":
            in_ref_contact = self.motion.dribble_cg_contact[mi, self.time_steps[env_ids]] > 0
            return has_demo & ~in_ref_contact
        return has_demo

    def _front_ball_positions(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Env-local ball positions in front of the motion's starting anchor."""
        mi = self.motion_idx[env_ids]
        first_anchor = self.motion._body_pos_w[mi, 0, self.motion_anchor_body_index]
        first_quat = self.motion._body_quat_w[mi, 0, self.motion_anchor_body_index]

        forward = quat_apply(
            yaw_quat(first_quat),
            torch.tensor([1.0, 0.0, 0.0], device=self.device, dtype=first_anchor.dtype).expand(env_ids.numel(), -1),
        )
        lateral = quat_apply(
            yaw_quat(first_quat),
            torch.tensor([0.0, 1.0, 0.0], device=self.device, dtype=first_anchor.dtype).expand(env_ids.numel(), -1),
        )

        distance = float(getattr(self.cfg, "dribble_cg_front_ball_distance", 0.45))
        lateral_offset = float(getattr(self.cfg, "dribble_cg_front_ball_lateral_offset", 0.0))
        height = float(getattr(self.cfg, "dribble_cg_front_ball_height", self._target_height))

        ball_pos = first_anchor + distance * forward + lateral_offset * lateral
        ball_pos[:, 2] = height
        return ball_pos

    def _compute_soccer_ball_positions(self, env_ids: Sequence[int] | torch.Tensor):
        ids = self._to_env_id_tensor(env_ids)
        if ids.numel() == 0:
            return

        has_demo = self._should_snap_demo_ball(ids)
        demo_ids = ids[has_demo]
        fallback_ids = ids[~has_demo]

        if fallback_ids.numel() > 0:
            fallback_mode = str(getattr(self.cfg, "dribble_cg_fallback_ball_mode", "arc_endpoint")).lower().strip()
            if fallback_mode == "front":
                self.soccer_ball_pos[fallback_ids] = self._front_ball_positions(fallback_ids)
            else:
                super()._compute_soccer_ball_positions(fallback_ids)
        if demo_ids.numel() > 0:
            env_origins = self._env.scene.env_origins[demo_ids]
            self.soccer_ball_pos[demo_ids] = self._demo_ball_world(demo_ids) - env_origins

    def _sync_demo_ball_after_step(self):
        """Kinematic sync of sim ball to demo trajectory (subset of envs)."""
        if self.soccer_ball is None:
            return
        if hasattr(self.soccer_ball, "is_initialized") and not self.soccer_ball.is_initialized:
            return
        env_origins = getattr(self._env.scene, "env_origins", None)
        if env_origins is None:
            return

        all_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        snap = self._should_snap_demo_ball(all_ids)
        if not torch.any(snap):
            return

        ids = all_ids[snap]
        ball_pos_w = self._demo_ball_world(ids)
        ball_quat = ball_pos_w.new_zeros((ids.numel(), 4))
        ball_quat[:, 0] = 1.0
        ball_lin_vel = self._demo_ball_lin_vel_w(ids)
        ball_ang_vel = ball_pos_w.new_zeros((ids.numel(), 3))
        ball_state = torch.cat([ball_pos_w, ball_quat, ball_lin_vel, ball_ang_vel], dim=-1)
        self.soccer_ball.write_root_state_to_sim(ball_state, env_ids=ids)

        self.soccer_ball_pos[ids] = ball_pos_w - env_origins[ids]
        self.target_point_pos[ids] = self.soccer_ball_pos[ids]
        if self.target_point_marker is not None:
            self.target_point_marker.visualize(ball_pos_w)

    def _update_command(self):
        super()._update_command()
        self._sync_demo_ball_after_step()


@configclass
class DribbleCGMotionCommandCfg(MotionCommandCfg):
    """Config for :class:`DribbleCGMotionCommand`.

    Extra fields (read via ``getattr`` on older cfg objects default in command):

    dribble_cg_snap_mode:
        ``full`` — always snap to demo when ``ball_pos_w`` exists.
        ``non_contact_only`` — snap only outside annotated contact frames.
    """

    class_type: type = DribbleCGMotionCommand

    dribble_cg_snap_mode: str = "full"
    dribble_cg_use_demo_ball: bool = True
    dribble_cg_fallback_ball_mode: str = "arc_endpoint"
    dribble_cg_front_ball_distance: float = 0.45
    dribble_cg_front_ball_lateral_offset: float = 0.0
    dribble_cg_front_ball_height: float = 0.11
