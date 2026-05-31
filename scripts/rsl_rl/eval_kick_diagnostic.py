"""Phase-based kick diagnostic evaluation script.

Breaks each kick episode into APPROACH / PRE_STRIKE / STRIKE phases and
logs detailed biomechanics metrics per phase.
"""
import argparse, sys, os, glob, json, math
from enum import IntEnum
from typing import Optional

from isaaclab.app import AppLauncher
import cli_args

parser = argparse.ArgumentParser(description="Kick diagnostic eval.")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--motion_file", type=str, default=None)
parser.add_argument("--motion_path", type=str, default=None)
parser.add_argument("--eval_episodes", type=int, default=20)
parser.add_argument("--cg_margin", type=int, default=5)
parser.add_argument("--ball_x_offset", type=float, default=0.0,
                    help="Deterministic X offset to ball position (m, in kick direction)")
parser.add_argument("--ball_y_offset", type=float, default=0.0,
                    help="Deterministic Y offset to ball position (m, lateral)")
parser.add_argument("--ball_xy_perturb", type=float, default=0.0,
                    help="Random uniform XY perturbation to ball position (m)")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import numpy as np
from collections import defaultdict
from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import (DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg,
                           ManagerBasedRLEnvCfg, multi_agent_to_single_agent)
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
import soccer.tasks  # noqa: F401


class Phase(IntEnum):
    APPROACH = 0
    PRE_STRIKE = 1
    STRIKE = 2


def get_motion_files(path):
    if os.path.isfile(path):
        return [path]
    return sorted(glob.glob(os.path.join(path, "*.npz")))


def _quat_to_yaw(q):
    """wxyz quaternion -> yaw angle."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class KickDiagnosticEvaluator:
    """Collects per-phase metrics for kick diagnostic."""

    def __init__(self, env, num_motions, motion_names, target_eps, device, cg_margin=5,
                 ball_x_offset=0.0, ball_y_offset=0.0, ball_xy_perturb=0.0):
        self.env = env
        self.base_env = env.unwrapped if hasattr(env, "unwrapped") else env
        self.N = env.num_envs
        self.num_motions = num_motions
        self.motion_names = motion_names
        self.target_eps = target_eps
        self.device = device
        self.cg_margin = cg_margin
        self.ball_x_offset = ball_x_offset
        self.ball_y_offset = ball_y_offset
        self.ball_xy_perturb = ball_xy_perturb

        # Per-env state
        self.env_motion_idx = torch.zeros(self.N, dtype=torch.long, device=device)
        self.env_step = torch.zeros(self.N, dtype=torch.long, device=device)

        # Per-env per-episode accumulators (reset each episode)
        self._init_accumulators()

        self.results = defaultdict(list)
        self.episodes_done = torch.zeros(num_motions, dtype=torch.long)

        # Cache body indices
        cmd = self.base_env.command_manager.get_term("motion")
        robot = cmd.robot
        self._support_idx = robot.body_names.index("left_ankle_roll_link")
        self._kick_idx = robot.body_names.index("right_ankle_roll_link")
        self._pelvis_idx = robot.body_names.index("pelvis")
        # Ankle joint indices
        ankle_names = ["left_ankle_pitch_joint", "left_ankle_roll_joint",
                       "right_ankle_pitch_joint", "right_ankle_roll_joint"]
        self._ankle_jids = torch.as_tensor(
            robot.find_joints(ankle_names, preserve_order=True)[0],
            dtype=torch.long, device=device)

        # Strike discriminator (optional, for D_at_contact metric)
        self._disc = None
        self._disc_extractor = None
        disc_path = "models/strike_discriminator.pt"
        if os.path.exists(disc_path):
            try:
                from soccer.tasks.tracking.mdp.strike_discriminator import (
                    StrikeDiscriminator, StrikeFeatureExtractor, INPUT_DIM)
                ckpt = torch.load(disc_path, map_location=device, weights_only=False)
                self._disc = StrikeDiscriminator(
                    input_dim=ckpt.get("input_dim", INPUT_DIM),
                    hidden=ckpt.get("hidden", 64))
                self._disc.load_state_dict(ckpt["model_state_dict"])
                self._disc.to(device)
                self._disc.eval()
                self._disc_extractor = StrikeFeatureExtractor()
                self._disc_extractor.init_indices(robot)
                print(f"[INFO] Loaded strike discriminator from {disc_path}")
            except Exception as e:
                print(f"[WARN] Could not load discriminator: {e}")
                self._disc = None

    def _init_accumulators(self):
        N, D = self.N, self.device
        z = lambda: torch.zeros(N, device=D)
        zl = lambda: torch.zeros(N, dtype=torch.long, device=D)
        # Phase frame counts
        self.approach_frames = zl()
        self.prestrike_frames = zl()
        self.strike_frames = zl()
        # APPROACH
        self.early_contact_count = zl()
        self.approach_min_ball_dist = torch.full((N,), 999.0, device=D)
        self.approach_body_err_sum = z()
        # PRE_STRIKE - support foot (ball-relative)
        self.ps_support_lateral_sum = z()
        self.ps_support_longit_sum = z()
        self.ps_support_yaw_err_sum = z()
        self.ps_support_vel_sum = z()
        self.ps_support_grf_sum = z()
        self.ps_pelvis_err_sum = z()
        # STRIKE
        self.kick_contacted = torch.zeros(N, dtype=torch.bool, device=D)
        self.kick_foot_vel_at_contact = z()
        self.ankle_vel_at_contact = z()
        self.peak_ball_speed = z()
        self.ball_dir_align = z()
        self.correct_foot = torch.zeros(N, dtype=torch.bool, device=D)
        # IMPACT ACQUISITION — kick foot-ball closest approach
        self.min_kick_ball_dist_xy = torch.full((N,), 999.0, device=D)
        self.min_kick_ball_dist_3d = torch.full((N,), 999.0, device=D)
        self.argmin_frame = zl()
        self.kick_speed_at_argmin = z()
        self.kick_height_at_argmin = z()
        self.kick_long_miss = z()
        self.kick_lat_miss = z()
        # Store kick_dir at argmin for lat/long decomposition
        self._argmin_kick_dir = torch.zeros(N, 2, device=D)
        self._argmin_side_dir = torch.zeros(N, 2, device=D)
        self._argmin_rel_xy = torch.zeros(N, 2, device=D)
        # TRUE CONTACT METRICS — recorded at first ball contact
        self.contact_frame = zl()  # frame when first contact happens
        self.contact_dist_xy = z()  # kick foot-ball XY dist at contact
        self.contact_kick_speed = z()  # kick foot speed at contact
        self.contact_support_lat = z()  # support foot lateral offset at contact
        self.contact_support_long = z()  # support foot longitudinal offset at contact
        self.contact_support_yaw_err = z()  # support foot yaw error at contact
        self.contact_support_height = z()  # support foot Z height at contact (GRF proxy)
        self.contact_d_score = z()  # strike discriminator score at contact
        # POST-STRIKE STABILITY — tracked after contact
        self.post_max_tilt = z()  # max base tilt angle after contact
        self.post_max_angvel_xy = z()  # max angular velocity XY after contact
        self.post_frames = zl()  # frames survived after contact

    def _reset_accumulators(self, ids):
        self.env_step[ids] = 0
        self.approach_frames[ids] = 0
        self.prestrike_frames[ids] = 0
        self.strike_frames[ids] = 0
        self.early_contact_count[ids] = 0
        self.approach_min_ball_dist[ids] = 999.0
        self.approach_body_err_sum[ids] = 0
        self.ps_support_lateral_sum[ids] = 0
        self.ps_support_longit_sum[ids] = 0
        self.ps_support_yaw_err_sum[ids] = 0
        self.ps_support_vel_sum[ids] = 0
        self.ps_support_grf_sum[ids] = 0
        self.ps_pelvis_err_sum[ids] = 0
        self.kick_contacted[ids] = False
        self.kick_foot_vel_at_contact[ids] = 0
        self.ankle_vel_at_contact[ids] = 0
        self.peak_ball_speed[ids] = 0
        self.ball_dir_align[ids] = 0
        self.correct_foot[ids] = False
        self.min_kick_ball_dist_xy[ids] = 999.0
        self.min_kick_ball_dist_3d[ids] = 999.0
        self.argmin_frame[ids] = 0
        self.kick_speed_at_argmin[ids] = 0
        self.kick_height_at_argmin[ids] = 0
        self.kick_long_miss[ids] = 0
        self.kick_lat_miss[ids] = 0
        self._argmin_kick_dir[ids] = 0
        self._argmin_side_dir[ids] = 0
        self._argmin_rel_xy[ids] = 0
        self.contact_frame[ids] = 0
        self.contact_dist_xy[ids] = 0
        self.contact_kick_speed[ids] = 0
        self.contact_support_lat[ids] = 0
        self.contact_support_long[ids] = 0
        self.contact_support_yaw_err[ids] = 0
        self.contact_support_height[ids] = 0
        self.contact_d_score[ids] = 0
        self.post_max_tilt[ids] = 0
        self.post_max_angvel_xy[ids] = 0
        self.post_frames[ids] = 0

    def assign_motions_round_robin(self):
        cmd = self.base_env.command_manager.get_term("motion")
        for i in range(self.N):
            self.env_motion_idx[i] = i % self.num_motions
        cmd.motion_idx[:] = self.env_motion_idx
        cmd.motion_length[:] = cmd.motion.file_lengths[self.env_motion_idx]
        cmd.time_steps[:] = 0

    def _perturb_ball_position(self, env_ids=None):
        """Perturb ball XY position for generalization testing.
        
        Applies deterministic offset (ball_x_offset, ball_y_offset) plus
        optional random uniform perturbation (ball_xy_perturb).
        """
        if self.ball_x_offset == 0.0 and self.ball_y_offset == 0.0 and self.ball_xy_perturb == 0.0:
            return
        
        cmd = self.base_env.command_manager.get_term("motion")
        soccer_ball = cmd.soccer_ball
        if soccer_ball is None:
            return
        
        if env_ids is None:
            ids = torch.arange(self.N, device=self.device)
        else:
            ids = env_ids
        if ids.numel() == 0:
            return
        
        env_origins = getattr(self.base_env.scene, "env_origins", None)
        if env_origins is None:
            return
        
        # Read current ball state
        ball_state = soccer_ball.data.root_state_w[ids].clone()  # [K, 13]
        
        # Apply deterministic offset
        ball_state[:, 0] += self.ball_x_offset
        ball_state[:, 1] += self.ball_y_offset
        
        # Apply random perturbation
        if self.ball_xy_perturb > 0:
            perturb = (torch.rand(ids.numel(), 2, device=self.device) - 0.5) * 2 * self.ball_xy_perturb
            ball_state[:, 0] += perturb[:, 0]
            ball_state[:, 1] += perturb[:, 1]
        
        # Zero out velocity (ball starts stationary)
        ball_state[:, 7:] = 0.0
        
        # Write back and update command state
        # All sim buffers and command tensors are inference tensors
        local_xy = ball_state[:, :2] - env_origins[ids, :2]
        with torch.inference_mode():
            soccer_ball.write_root_state_to_sim(ball_state.clone(), env_ids=ids)
            cmd.soccer_ball_pos[ids, 0] = local_xy[:, 0]
            cmd.soccer_ball_pos[ids, 1] = local_xy[:, 1]
            cmd.target_point_pos[ids] = cmd.soccer_ball_pos[ids].clone()

    def _get_phase(self, cmd):
        """Return per-env phase enum tensor."""
        kf = cmd.kick_frame
        kef = cmd.kick_end_frame
        t = cmd.time_steps
        has_ann = kf >= 0
        is_strike = has_ann & (t >= kf)
        is_prestrike = has_ann & (t >= (kf - self.cg_margin)) & (t < kf)
        is_approach = has_ann & ~is_strike & ~is_prestrike
        # Default unannotated to STRIKE (no penalty)
        phase = torch.full((self.N,), Phase.STRIKE, dtype=torch.long, device=self.device)
        phase[is_approach] = Phase.APPROACH
        phase[is_prestrike] = Phase.PRE_STRIKE
        phase[is_strike] = Phase.STRIKE
        return phase

    def _get_ball_contact(self, cmd):
        """Return (has_contact, force_mag) from ball sensor."""
        try:
            sensor = self.base_env.scene["soccer_ball_contact"]
            forces = sensor.data.net_forces_w_history
            if forces.dim() == 4:
                fv = forces[:, :, 0, :2].sum(dim=1)
            else:
                fv = forces[:, 0, :2]
            fm = torch.norm(fv, dim=-1)
            return fm > 5.0, fm
        except Exception:
            z = torch.zeros(self.N, device=self.device)
            return z.bool(), z

    def _get_kick_direction(self, cmd):
        """Normalized kick direction (ball -> target destination) in XY."""
        env_origins = getattr(self.base_env.scene, "env_origins", None)
        ball_w = self.base_env.scene["soccer_ball"].data.root_pos_w[:, :2]
        dest_w = cmd.target_destination_pos[:, :2]
        if env_origins is not None:
            dest_w = dest_w + env_origins[:, :2]
        d = dest_w - ball_w
        return d / torch.norm(d, dim=-1, keepdim=True).clamp(min=1e-6)

    def step(self, rewards, dones, infos):
        self.env_step += 1
        cmd = self.base_env.command_manager.get_term("motion")
        robot = cmd.robot
        phase = self._get_phase(cmd)
        has_contact, force_mag = self._get_ball_contact(cmd)

        ball_pos_w = self.base_env.scene["soccer_ball"].data.root_pos_w
        ball_vel_w = self.base_env.scene["soccer_ball"].data.root_lin_vel_w
        pelvis_pos = robot.data.body_pos_w[:, self._pelvis_idx]
        support_pos = robot.data.body_pos_w[:, self._support_idx]
        support_quat = robot.data.body_quat_w[:, self._support_idx]
        support_vel = robot.data.body_lin_vel_w[:, self._support_idx]
        kick_vel = robot.data.body_lin_vel_w[:, self._kick_idx]

        kick_dir = self._get_kick_direction(cmd)
        side_dir = torch.stack([-kick_dir[:, 1], kick_dir[:, 0]], dim=-1)

        # Ball-relative support foot position
        rel = support_pos[:, :2] - ball_pos_w[:, :2]
        longit = torch.sum(rel * kick_dir, dim=-1)
        lateral = torch.sum(rel * side_dir, dim=-1)

        # Support foot yaw error vs kick direction
        support_yaw = _quat_to_yaw(support_quat)
        desired_yaw = torch.atan2(kick_dir[:, 1], kick_dir[:, 0])
        yaw_err = torch.atan2(torch.sin(support_yaw - desired_yaw),
                              torch.cos(support_yaw - desired_yaw)).abs()

        pelvis_ball_dist = torch.norm(pelvis_pos[:, :2] - ball_pos_w[:, :2], dim=-1)
        ball_speed = torch.norm(ball_vel_w[:, :2], dim=-1)

        # --- Phase-specific accumulation ---
        app = phase == Phase.APPROACH
        ps = phase == Phase.PRE_STRIKE
        st = phase == Phase.STRIKE

        self.approach_frames[app] += 1
        self.prestrike_frames[ps] += 1
        self.strike_frames[st] += 1

        # APPROACH
        self.early_contact_count[app & has_contact] += 1
        self.approach_min_ball_dist[app] = torch.minimum(
            self.approach_min_ball_dist[app], pelvis_ball_dist[app])
        body_err = cmd.metrics.get("error_body_pos", torch.zeros(self.N, device=self.device))
        self.approach_body_err_sum[app] += body_err[app]

        # PRE_STRIKE
        self.ps_support_lateral_sum[ps] += lateral[ps]
        self.ps_support_longit_sum[ps] += longit[ps]
        self.ps_support_yaw_err_sum[ps] += yaw_err[ps]
        self.ps_support_vel_sum[ps] += torch.norm(support_vel[ps, :2], dim=-1)
        # Support foot ground contact: use Z height as proxy (< 0.05m = grounded)
        support_z = support_pos[:, 2]
        self.ps_support_grf_sum[ps] += (support_z[ps] < 0.05).float()
        self.ps_pelvis_err_sum[ps] += body_err[ps]

        # STRIKE + PRE_STRIKE (match training CG1 window: kick_frame - margin)
        legal_phase = st | ps  # contacts in PRE_STRIKE also count as valid kicks
        self.peak_ball_speed = torch.maximum(self.peak_ball_speed, ball_speed)
        new_contact = legal_phase & has_contact & ~self.kick_contacted
        if torch.any(new_contact):
            self.kick_contacted[new_contact] = True
            self.kick_foot_vel_at_contact[new_contact] = torch.norm(
                kick_vel[new_contact, :2], dim=-1)
            ankle_v = robot.data.joint_vel[:, self._ankle_jids]
            self.ankle_vel_at_contact[new_contact] = torch.sum(
                ankle_v[new_contact, 2:] ** 2, dim=-1).sqrt()
            # Check correct foot (closest to ball)
            kick_pos = robot.data.body_pos_w[new_contact, self._kick_idx, :2]
            sup_pos = robot.data.body_pos_w[new_contact, self._support_idx, :2]
            bp = ball_pos_w[new_contact, :2]
            kick_dist = torch.norm(kick_pos - bp, dim=-1)
            sup_dist = torch.norm(sup_pos - bp, dim=-1)
            self.correct_foot[new_contact] = kick_dist < sup_dist
            # TRUE CONTACT METRICS at first contact
            self.contact_frame[new_contact] = self.env_step[new_contact]
            self.contact_dist_xy[new_contact] = kick_dist
            self.contact_kick_speed[new_contact] = torch.norm(kick_vel[new_contact, :2], dim=-1)
            self.contact_support_lat[new_contact] = lateral[new_contact]
            self.contact_support_long[new_contact] = longit[new_contact]
            self.contact_support_yaw_err[new_contact] = yaw_err[new_contact]
            self.contact_support_height[new_contact] = support_pos[new_contact, 2]
            # D(state) at contact
            if self._disc is not None and self._disc_extractor is not None:
                with torch.no_grad():
                    feats = self._disc_extractor.compute(self.base_env, cmd)
                    d_scores = self._disc(feats)
                    self.contact_d_score[new_contact] = d_scores[new_contact]

        # Ball direction alignment (continuous during strike + pre-strike)
        if torch.any(legal_phase & (ball_speed > 0.5)):
            bdir = ball_vel_w[:, :2] / ball_speed.unsqueeze(-1).clamp(min=1e-6)
            align = torch.sum(bdir * kick_dir, dim=-1).clamp(-1, 1)
            mask = legal_phase & (ball_speed > 0.5)
            self.ball_dir_align[mask] = torch.maximum(self.ball_dir_align[mask], align[mask])

        # --- IMPACT ACQUISITION: kick foot-ball closest approach (all phases) ---
        kick_pos_full = robot.data.body_pos_w[:, self._kick_idx]  # (N, 3)
        kick_ball_xy = torch.norm(kick_pos_full[:, :2] - ball_pos_w[:, :2], dim=-1)  # (N,)
        kick_ball_3d = torch.norm(kick_pos_full[:, :3] - ball_pos_w[:, :3], dim=-1)  # (N,)
        closer = kick_ball_xy < self.min_kick_ball_dist_xy
        if torch.any(closer):
            self.min_kick_ball_dist_xy[closer] = kick_ball_xy[closer]
            self.min_kick_ball_dist_3d[closer] = kick_ball_3d[closer]
            self.argmin_frame[closer] = self.env_step[closer]
            self.kick_speed_at_argmin[closer] = torch.norm(kick_vel[closer, :2], dim=-1)
            self.kick_height_at_argmin[closer] = kick_pos_full[closer, 2]
            # Store directions for lat/long decomposition at record time
            rel_xy = kick_pos_full[closer, :2] - ball_pos_w[closer, :2]
            self._argmin_rel_xy[closer] = rel_xy
            self._argmin_kick_dir[closer] = kick_dir[closer]
            self._argmin_side_dir[closer] = side_dir[closer]

        # --- POST-STRIKE STABILITY tracking ---
        post_contact = self.kick_contacted  # envs where contact already happened
        if torch.any(post_contact):
            from isaaclab.utils.math import quat_apply_inverse as qai
            base_quat = robot.data.root_quat_w  # (N, 4)
            grav = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.N, 3)
            proj_grav = qai(base_quat, grav)
            tilt = 1.0 + proj_grav[:, 2]  # 0=upright, ~2=inverted
            self.post_max_tilt[post_contact] = torch.maximum(
                self.post_max_tilt[post_contact], tilt[post_contact])
            ang_vel = robot.data.root_ang_vel_w  # (N, 3)
            angvel_xy = torch.sqrt(ang_vel[:, 0].square() + ang_vel[:, 1].square())
            self.post_max_angvel_xy[post_contact] = torch.maximum(
                self.post_max_angvel_xy[post_contact], angvel_xy[post_contact])
            self.post_frames[post_contact] += 1

        # --- Handle resets ---
        if isinstance(dones, dict):
            done_m = dones.get("terminated", torch.zeros(self.N, dtype=torch.bool, device=self.device))
            tout_m = dones.get("truncated", torch.zeros(self.N, dtype=torch.bool, device=self.device))
            reset = done_m | tout_m
        else:
            reset = dones.bool() if not isinstance(dones, bool) else torch.full((self.N,), dones, dtype=torch.bool, device=self.device)
            tout_m = infos.get("time_outs", torch.zeros(self.N, dtype=torch.bool, device=self.device)).to(self.device, torch.bool) if isinstance(infos, dict) else torch.zeros(self.N, dtype=torch.bool, device=self.device)
            done_m = reset & ~tout_m

        if torch.any(reset):
            self._record(reset, done_m, tout_m, cmd)
            # Re-perturb ball for the next episode (env already reset ball position)
            reset_ids = torch.where(reset)[0]
            self._perturb_ball_position(env_ids=reset_ids)

    def _safe_div(self, s, c):
        return (s / c.clamp(min=1).float()).item()

    def _record(self, reset, terminated, truncated, cmd):
        for idx in torch.where(reset)[0]:
            i = idx.item()
            mid = int(self.env_motion_idx[i].item())
            if self.episodes_done[mid] >= self.target_eps:
                continue
            steps = int(self.env_step[i].item())
            if steps < 2:
                continue

            af = max(int(self.approach_frames[i].item()), 1)
            pf = max(int(self.prestrike_frames[i].item()), 1)
            sf = max(int(self.strike_frames[i].item()), 1)

            ep = {
                "motion": self.motion_names[mid],
                "steps": steps, "terminated": bool(terminated[i]),
                # APPROACH
                "early_contact_frames": int(self.early_contact_count[i].item()),
                "approach_min_ball_dist": round(self.approach_min_ball_dist[i].item(), 3),
                "approach_body_err": round(self._safe_div(self.approach_body_err_sum[i], self.approach_frames[i]), 4),
                # PRE_STRIKE (ball-relative support foot)
                "support_lateral_mean": round(self._safe_div(self.ps_support_lateral_sum[i], self.prestrike_frames[i]), 3),
                "support_longit_mean": round(self._safe_div(self.ps_support_longit_sum[i], self.prestrike_frames[i]), 3),
                "support_yaw_err_mean": round(self._safe_div(self.ps_support_yaw_err_sum[i], self.prestrike_frames[i]), 3),
                "support_vel_mean": round(self._safe_div(self.ps_support_vel_sum[i], self.prestrike_frames[i]), 3),
                "support_grounded_ratio": round(self._safe_div(self.ps_support_grf_sum[i], self.prestrike_frames[i]), 2),
                "prestrike_pelvis_err": round(self._safe_div(self.ps_pelvis_err_sum[i], self.prestrike_frames[i]), 4),
                # STRIKE
                "kick_contacted": bool(self.kick_contacted[i].item()),
                "kick_foot_vel": round(self.kick_foot_vel_at_contact[i].item(), 2),
                "ankle_vel_at_contact": round(self.ankle_vel_at_contact[i].item(), 3),
                "peak_ball_speed": round(self.peak_ball_speed[i].item(), 2),
                "ball_dir_align": round(self.ball_dir_align[i].item(), 3),
                "correct_foot": bool(self.correct_foot[i].item()),
                # TRUE CONTACT METRICS
                "contact_frame_vs_kf": int(self.contact_frame[i].item()) - int(cmd.kick_frame[i].item()) if cmd.kick_frame[i] >= 0 and self.kick_contacted[i] else None,
                "contact_dist_xy": round(self.contact_dist_xy[i].item(), 3) if self.kick_contacted[i] else None,
                "contact_kick_speed": round(self.contact_kick_speed[i].item(), 2) if self.kick_contacted[i] else None,
                "contact_support_lat": round(self.contact_support_lat[i].item(), 3) if self.kick_contacted[i] else None,
                "contact_support_long": round(self.contact_support_long[i].item(), 3) if self.kick_contacted[i] else None,
                "contact_support_yaw_err": round(self.contact_support_yaw_err[i].item(), 3) if self.kick_contacted[i] else None,
                "contact_support_height": round(self.contact_support_height[i].item(), 3) if self.kick_contacted[i] else None,
                "d_at_contact": round(self.contact_d_score[i].item(), 3) if self.kick_contacted[i] else None,
                # IMPACT ACQUISITION (argmin-based, kept for reference)
                "min_kick_ball_dist_xy": round(self.min_kick_ball_dist_xy[i].item(), 3),
                "min_kick_ball_dist_3d": round(self.min_kick_ball_dist_3d[i].item(), 3),
                "argmin_frame_vs_kf": int(self.argmin_frame[i].item()) - int(cmd.kick_frame[i].item()) if cmd.kick_frame[i] >= 0 else 0,
                "kick_speed_at_argmin": round(self.kick_speed_at_argmin[i].item(), 2),
                "kick_height_at_argmin": round(self.kick_height_at_argmin[i].item(), 3),
                "kick_long_miss": round(torch.sum(self._argmin_rel_xy[i] * self._argmin_kick_dir[i]).item(), 3),
                "kick_lat_miss": round(torch.sum(self._argmin_rel_xy[i] * self._argmin_side_dir[i]).item(), 3),
                # POST-STRIKE STABILITY
                "post_max_tilt": round(self.post_max_tilt[i].item(), 3) if self.kick_contacted[i] else None,
                "post_max_angvel_xy": round(self.post_max_angvel_xy[i].item(), 2) if self.kick_contacted[i] else None,
                "post_frames": int(self.post_frames[i].item()) if self.kick_contacted[i] else None,
            }

            # --- Failure classification ---
            contacted = ep["kick_contacted"]
            fell = ep["terminated"]
            bspd = ep["peak_ball_speed"]
            dira = ep["ball_dir_align"]
            early = ep["early_contact_frames"] > 0

            if fell and not contacted:
                ep["outcome"] = "fall"
            elif early and not contacted:
                ep["outcome"] = "early_collision"
            elif not contacted:
                ep["outcome"] = "miss"
            elif bspd < 2.0:
                ep["outcome"] = "weak_contact"
            elif dira < 0.5:
                ep["outcome"] = "wrong_direction"
            else:
                ep["outcome"] = "success"

            self.results[mid].append(ep)
            self.episodes_done[mid] += 1
            self._reset_accumulators(torch.tensor([i], device=self.device))

            for m in range(self.num_motions):
                if self.episodes_done[m] < self.target_eps:
                    self.env_motion_idx[i] = m
                    cmd.motion_idx[i] = m
                    cmd.motion_length[i] = cmd.motion.file_lengths[m]
                    break

    def is_done(self):
        return all(self.episodes_done[m] >= self.target_eps for m in range(self.num_motions))

    def print_report(self):
        print("\n" + "=" * 130)
        print("  KICK DIAGNOSTIC REPORT (Phase-Based)")
        if self.ball_x_offset != 0.0 or self.ball_y_offset != 0.0 or self.ball_xy_perturb != 0.0:
            print(f"  Ball Perturbation: x_offset={self.ball_x_offset:+.3f}m, "
                  f"y_offset={self.ball_y_offset:+.3f}m, xy_perturb=±{self.ball_xy_perturb:.3f}m")
        print("=" * 130)

        header = (f"{'Motion':<35} {'Kick%':>5} {'Fall%':>5} | "
                  f"{'EarlyCt':>7} {'MinDst':>6} | "
                  f"{'Lat':>5} {'Long':>5} {'YawE':>5} {'Vel':>5} {'Gnd%':>5} | "
                  f"{'KickV':>5} {'AnkV':>5} {'BSpd':>5} {'DirA':>5} {'Foot%':>5}")
        phase_labels = f"{'':35} {'':5} {'':5} | {'--- APPROACH ---':>14} | {'------- PRE-STRIKE (ball-rel) -------':>30} | {'----------- STRIKE -----------':>30}"
        print(f"\n{phase_labels}")
        print(header)
        print("-" * len(header))

        all_eps = []
        for mid in range(self.num_motions):
            eps = self.results[mid]
            if not eps:
                continue
            n = len(eps)
            all_eps.extend(eps)

            kick_r = sum(1 for e in eps if e["kick_contacted"]) / n
            fall_r = sum(1 for e in eps if e["terminated"]) / n
            ec = np.mean([e["early_contact_frames"] for e in eps])
            md = np.mean([e["approach_min_ball_dist"] for e in eps])
            lat = np.mean([e["support_lateral_mean"] for e in eps])
            lon = np.mean([e["support_longit_mean"] for e in eps])
            yaw = np.mean([e["support_yaw_err_mean"] for e in eps])
            vel = np.mean([e["support_vel_mean"] for e in eps])
            grf = np.mean([e["support_grounded_ratio"] for e in eps])
            kv = np.mean([e["kick_foot_vel"] for e in eps])
            av = np.mean([e["ankle_vel_at_contact"] for e in eps])
            bs = np.mean([e["peak_ball_speed"] for e in eps])
            da = np.mean([e["ball_dir_align"] for e in eps])
            cf = sum(1 for e in eps if e["correct_foot"]) / max(1, sum(1 for e in eps if e["kick_contacted"]))

            name = self.motion_names[mid][:34]
            print(f"{name:<35} {kick_r*100:>4.0f}% {fall_r*100:>4.0f}% | "
                  f"{ec:>7.1f} {md:>6.2f} | "
                  f"{lat:>5.2f} {lon:>5.2f} {yaw:>5.2f} {vel:>5.2f} {grf*100:>4.0f}% | "
                  f"{kv:>5.1f} {av:>5.2f} {bs:>5.1f} {da:>5.2f} {cf*100:>4.0f}%")

        if all_eps:
            n = len(all_eps)
            print(f"\n{'AGGREGATE':<35} "
                  f"{sum(1 for e in all_eps if e['kick_contacted'])/n*100:>4.0f}% "
                  f"{sum(1 for e in all_eps if e['terminated'])/n*100:>4.0f}% | "
                  f"{np.mean([e['early_contact_frames'] for e in all_eps]):>7.1f} "
                  f"{np.mean([e['approach_min_ball_dist'] for e in all_eps]):>6.2f} | "
                  f"{np.mean([e['support_lateral_mean'] for e in all_eps]):>5.2f} "
                  f"{np.mean([e['support_longit_mean'] for e in all_eps]):>5.2f} "
                  f"{np.mean([e['support_yaw_err_mean'] for e in all_eps]):>5.2f} "
                  f"{np.mean([e['support_vel_mean'] for e in all_eps]):>5.2f} "
                  f"{np.mean([e['support_grounded_ratio'] for e in all_eps])*100:>4.0f}% | "
                  f"{np.mean([e['kick_foot_vel'] for e in all_eps]):>5.1f} "
                  f"{np.mean([e['ankle_vel_at_contact'] for e in all_eps]):>5.2f} "
                  f"{np.mean([e['peak_ball_speed'] for e in all_eps]):>5.1f} "
                  f"{np.mean([e['ball_dir_align'] for e in all_eps]):>5.2f} "
                  f"{'':>5}")
        print("=" * 130)
        print("\nLegend: Lat/Long = support foot offset from ball (m), YawE = yaw error (rad)")
        print("        Vel = support foot XY speed (m/s), Gnd% = % frames grounded (Z < 0.05m)")
        print("        KickV = kick foot speed at contact (m/s), AnkV = ankle joint vel at contact")
        print("        BSpd = peak ball speed (m/s), DirA = ball direction alignment [-1,1]")
        print("        Foot% = correct foot usage rate\n")

        # --- IMPACT ACQUISITION TABLE ---
        if all_eps:
            print("=" * 100)
            print("  IMPACT ACQUISITION")
            print("=" * 100)
            ia_header = (f"{'Motion':<35} {'MisXY':>6} {'Mis3D':>6} {'ArgΔ':>5} "
                         f"{'KSpd':>5} {'KHgt':>5} {'LnMs':>5} {'LtMs':>5}")
            print(ia_header)
            print("-" * len(ia_header))

            for mid in range(self.num_motions):
                eps = self.results[mid]
                if not eps:
                    continue
                name = self.motion_names[mid][:34]
                mxy = np.mean([e["min_kick_ball_dist_xy"] for e in eps])
                m3d = np.mean([e["min_kick_ball_dist_3d"] for e in eps])
                arg = np.mean([e["argmin_frame_vs_kf"] for e in eps])
                ksp = np.mean([e["kick_speed_at_argmin"] for e in eps])
                kht = np.mean([e["kick_height_at_argmin"] for e in eps])
                kln = np.mean([e["kick_long_miss"] for e in eps])
                klt = np.mean([e["kick_lat_miss"] for e in eps])
                print(f"{name:<35} {mxy:>6.3f} {m3d:>6.3f} {arg:>5.0f} "
                      f"{ksp:>5.2f} {kht:>5.3f} {kln:>+5.2f} {klt:>+5.2f}")

            mxy = np.mean([e["min_kick_ball_dist_xy"] for e in all_eps])
            m3d = np.mean([e["min_kick_ball_dist_3d"] for e in all_eps])
            arg = np.mean([e["argmin_frame_vs_kf"] for e in all_eps])
            ksp = np.mean([e["kick_speed_at_argmin"] for e in all_eps])
            kht = np.mean([e["kick_height_at_argmin"] for e in all_eps])
            kln = np.mean([e["kick_long_miss"] for e in all_eps])
            klt = np.mean([e["kick_lat_miss"] for e in all_eps])
            print(f"\n{'AGGREGATE':<35} {mxy:>6.3f} {m3d:>6.3f} {arg:>5.0f} "
                  f"{ksp:>5.2f} {kht:>5.3f} {kln:>+5.2f} {klt:>+5.2f}")

            print("\nLegend: MisXY/3D = min kick foot-ball dist (m), ArgΔ = argmin_frame - kick_frame")
            print("        KSpd = kick foot speed at closest (m/s), KHgt = kick foot height at closest (m)")
            print("        LnMs = longitudinal miss (+ = past ball, - = short)")
            print("        LtMs = lateral miss (+ = left of kick dir, - = right)")

        # --- OUTCOME BREAKDOWN ---
        if all_eps:
            print("\n" + "=" * 80)
            print("  OUTCOME BREAKDOWN")
            print("=" * 80)
            outcomes = ["success", "miss", "weak_contact", "wrong_direction", "early_collision", "fall"]
            out_header = f"{'Motion':<35} " + " ".join(f"{o[:7]:>7}" for o in outcomes)
            print(out_header)
            print("-" * len(out_header))

            for mid in range(self.num_motions):
                eps = self.results[mid]
                if not eps:
                    continue
                n = len(eps)
                name = self.motion_names[mid][:34]
                counts = " ".join(f"{sum(1 for e in eps if e.get('outcome') == o)/n*100:>6.0f}%" for o in outcomes)
                print(f"{name:<35} {counts}")

            n = len(all_eps)
            counts = " ".join(f"{sum(1 for e in all_eps if e.get('outcome') == o)/n*100:>6.0f}%" for o in outcomes)
            print(f"\n{'AGGREGATE':<35} {counts}")
            print("=" * 80)

        # --- SUPPORT FOOT BY OUTCOME ---
        if all_eps:
            print("\n" + "=" * 140)
            print("  SUPPORT FOOT BY OUTCOME (at contact moment)")
            print("=" * 140)
            outcomes = ["success", "miss", "early_collision", "fall", "weak_contact"]

            # Pre-strike averages
            sf_header = (f"{'Outcome':<18} {'N':>4} | "
                         f"{'Lat':>6} {'Long':>6} {'YawE':>6} {'Vel':>6} {'Gnd%':>5} | "
                         f"{'CΔ':>4} {'cLat':>6} {'cLong':>6} {'cYawE':>6} {'cHgt':>6} | "
                         f"{'BSpd':>6} {'KickV':>6} | "
                         f"{'pTilt':>5} {'pAngV':>5} {'pFrm':>4}")
            print(sf_header)
            print("-" * len(sf_header))

            for outcome in outcomes:
                grp = [e for e in all_eps if e.get("outcome") == outcome]
                if not grp:
                    continue
                n = len(grp)
                # Pre-strike averages
                lat = np.mean([e["support_lateral_mean"] for e in grp])
                lon = np.mean([e["support_longit_mean"] for e in grp])
                yaw = np.mean([e["support_yaw_err_mean"] for e in grp])
                vel = np.mean([e["support_vel_mean"] for e in grp])
                gnd = np.mean([e["support_grounded_ratio"] for e in grp])
                bs = np.mean([e["peak_ball_speed"] for e in grp])
                kv = np.mean([e["kick_foot_vel"] for e in grp])
                # Post-strike metrics
                post_grp = [e for e in grp if e.get("post_max_tilt") is not None]
                pt_str = f"{np.mean([e['post_max_tilt'] for e in post_grp]):>5.2f}" if post_grp else "   --"
                pa_str = f"{np.mean([e['post_max_angvel_xy'] for e in post_grp]):>5.1f}" if post_grp else "   --"
                pf_str = f"{np.mean([e['post_frames'] for e in post_grp]):>4.0f}" if post_grp else "  --"
                # True contact metrics (only for contacted episodes)
                contacted = [e for e in grp if e.get("contact_frame_vs_kf") is not None]
                if contacted:
                    cd = np.mean([e["contact_frame_vs_kf"] for e in contacted])
                    clat = np.mean([e["contact_support_lat"] for e in contacted])
                    clon = np.mean([e["contact_support_long"] for e in contacted])
                    cyaw = np.mean([e["contact_support_yaw_err"] for e in contacted])
                    chgt = np.mean([e["contact_support_height"] for e in contacted])
                    print(f"{outcome:<18} {n:>4} | "
                          f"{lat:>6.3f} {lon:>6.3f} {yaw:>6.3f} {vel:>6.3f} {gnd*100:>4.0f}% | "
                          f"{cd:>4.0f} {clat:>6.3f} {clon:>6.3f} {cyaw:>6.3f} {chgt:>6.3f} | "
                          f"{bs:>6.2f} {kv:>6.2f} | "
                          f"{pt_str} {pa_str} {pf_str}")
                else:
                    print(f"{outcome:<18} {n:>4} | "
                          f"{lat:>6.3f} {lon:>6.3f} {yaw:>6.3f} {vel:>6.3f} {gnd*100:>4.0f}% | "
                          f"{'--':>4} {'--':>6} {'--':>6} {'--':>6} {'--':>6} | "
                          f"{bs:>6.2f} {kv:>6.2f} | "
                          f"{pt_str} {pa_str} {pf_str}")

            print("=" * 140)
            print("\nLegend: Lat/Long/YawE/Vel/Gnd% = pre-strike averages (support foot)")
            print("        CΔ = contact_frame - kick_frame, cLat/cLong/cYawE/cHgt = support foot AT contact moment")
            print("        Gnd% = % of pre-strike frames where support foot Z < 0.05m")
            print("        pTilt = max base tilt after contact, pAngV = max angular vel XY, pFrm = frames survived")

    def save_json(self, path):
        raw = {self.motion_names[k]: v for k, v in self.results.items()}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(raw, f, indent=2)
        print(f"[INFO] Saved to {path}")


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.motion_file:
        mfiles = [args_cli.motion_file]
    elif args_cli.motion_path:
        mfiles = get_motion_files(args_cli.motion_path)
    else:
        raise ValueError("--motion_file or --motion_path required")
    env_cfg.commands.motion.motion_files = mfiles
    if hasattr(env_cfg.commands.motion, "strike_motion_files"):
        env_cfg.commands.motion.strike_motion_files = mfiles

    resume = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] Loading: {resume}")

    import gymnasium as gym
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    base_env = env.unwrapped
    cmd = base_env.command_manager.get_term("motion")

    evaluator = KickDiagnosticEvaluator(
        env, cmd.motion.num_files, cmd.motion.motion_name,
        args_cli.eval_episodes, base_env.device, args_cli.cg_margin,
        ball_x_offset=args_cli.ball_x_offset,
        ball_y_offset=args_cli.ball_y_offset,
        ball_xy_perturb=args_cli.ball_xy_perturb)
    evaluator.assign_motions_round_robin()
    evaluator._perturb_ball_position()

    # Print perturbation info
    if args_cli.ball_x_offset != 0.0 or args_cli.ball_y_offset != 0.0 or args_cli.ball_xy_perturb != 0.0:
        print(f"[INFO] Ball perturbation: x_offset={args_cli.ball_x_offset:.3f}m, "
              f"y_offset={args_cli.ball_y_offset:.3f}m, xy_perturb=±{args_cli.ball_xy_perturb:.3f}m")

    obs, _ = env.get_observations()
    step = 0
    max_steps = args_cli.eval_episodes * 500 * cmd.motion.num_files

    while simulation_app.is_running() and not evaluator.is_done() and step < max_steps:
        with torch.inference_mode():
            actions = policy(obs)
            obs, rew, dones, infos = env.step(actions)
        evaluator.step(rew, dones, infos)
        step += 1
        if step % 500 == 0:
            done_str = ", ".join(f"{evaluator.motion_names[m]}={evaluator.episodes_done[m]}"
                                for m in range(evaluator.num_motions))
            print(f"[EVAL] Step {step} | {done_str}")

    evaluator.print_report()
    out_dir = os.path.join(os.path.dirname(resume), "eval")
    evaluator.save_json(os.path.join(out_dir, "kick_diagnostic.json"))
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
