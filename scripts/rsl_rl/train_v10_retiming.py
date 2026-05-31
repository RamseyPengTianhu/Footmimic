"""v10.2 Event-Retiming PPO Training.

Trains the BC-initialized policy to follow retimed event phases via:
1. Per-env random shift on segment_bounds (curriculum: ±2 → ±5 → ±10)
2. Retimed rewards: phase-gated contact, timing reward, event-warped tracking
3. Disabled fixed-timing rewards from base env
4. Conservative feature dropout on motor_prior / history
"""
import argparse, sys, os, glob, math
from isaaclab.app import AppLauncher
import cli_args

parser = argparse.ArgumentParser(description="v10.2 Retiming PPO")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default="Event-Conditioned-Kick-G1-Soccer-v0")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--motion_file", type=str, required=True)
parser.add_argument("--bc_checkpoint", type=str, required=True)
parser.add_argument("--bc_reg_weight", type=float, default=1.0)
parser.add_argument("--shift_max", type=int, default=2, help="Max shift magnitude for stage")
parser.add_argument("--motor_noise", type=float, default=0.1, help="Motor prior noise scale")
parser.add_argument("--hist_dropout", type=float, default=0.05, help="Action hist dropout rate")
parser.add_argument("--timing_sigma", type=float, default=5.0, help="Contact timing sigma frames")
parser.add_argument("--strike_grace", type=int, default=3, help="Frames to expand STRIKE window backwards")

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from collections import defaultdict

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import soccer.tasks
from soccer.utils.my_on_policy_runner import MotionOnPolicyRunner
from rsl_rl.algorithms import PPO
from soccer.tasks.tracking.mdp.event_conditioned_obs_builder import V10ObsBuilder
from soccer.tasks.tracking.mdp.event_phase import (
    compute_segment_bounds, compute_event_phase, compute_event_obs,
    event_warped_ref_index, PHASE_APPROACH, PHASE_PRESTRIKE, PHASE_STRIKE, PHASE_FOLLOWTHRU,
)


def get_motion_files(p):
    if os.path.isfile(p): return [p]
    if os.path.isdir(p):
        f = sorted(glob.glob(os.path.join(p, "*.npz")))
        if not f: raise ValueError(f"No .npz in {p}")
        return f
    raise ValueError(f"Invalid: {p}")


class V10MLPActor(nn.Module):
    def __init__(self, obs_dim, action_dim=29, hidden_dims=None):
        super().__init__()
        hidden_dims = hidden_dims or [512, 256, 128]
        layers = []
        d = obs_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.ELU()]
            d = h
        layers.append(nn.Linear(d, action_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)


# ============================================================================
# Obs layout (454D)
# [0:64]    current_proprio
# [64:325]  proprio_hist (joint_pos:87 + joint_vel:87 + action_hist:87)
# [325:354] last_action
# [354:384] ball_hist
# [384:392] event_obs
# [392:414] ball_foot_rel
# [414:454] motor_prior
# ============================================================================
SLICE_ACTION_HIST = (238, 325)   # 87D within proprio_hist
SLICE_MOTOR_PRIOR = (414, 454)   # 40D


class RetimingV10Wrapper(RslRlVecEnvWrapper):
    """Wrapper that adds per-env retiming shifts and LegalKick-lite gated rewards.

    LegalKick-lite gate:
      - first_contact in STRIKE  → reward + ball outcome enabled
      - first_contact in APPROACH → heavy penalty + terminate
      - first_contact in PRESTRIKE → graded penalty, no ball outcome
      - first_contact in FOLLOWTHRU → no reward, no ball outcome
    """

    def __init__(self, env, shift_max=2, motor_noise=0.1, hist_dropout=0.05,
                 timing_sigma=5.0, strike_grace=3, training=True, **kwargs):
        super().__init__(env, **kwargs)
        uw = self.unwrapped
        command = uw.command_manager.get_term("motion")
        n = uw.num_envs
        dev = uw.device
        nj = command.robot.data.joint_pos.shape[1]

        self.shift_max = shift_max
        self.motor_noise = motor_noise
        self.hist_dropout = hist_dropout
        self.timing_sigma = timing_sigma
        self.strike_grace = strike_grace
        self.training = training

        # V10 obs builder
        self.v10_builder = V10ObsBuilder(num_envs=n, num_joints=nj, device=dev)
        self.v10_builder.init_segment_bounds(command)
        self.num_obs = 454
        self.num_privileged_obs = 454

        # Per-env state
        self.per_env_shift = torch.zeros(n, dtype=torch.long, device=dev)
        self.original_bounds = self.v10_builder.segment_bounds.clone()
        self.retimed_bounds = self.v10_builder.segment_bounds.clone()
        self.first_contact = torch.zeros(n, dtype=torch.bool, device=dev)
        self.contact_step = torch.full((n,), -1.0, device=dev)
        self.contact_phase = torch.full((n,), -1, dtype=torch.long, device=dev)
        self.legal_strike = torch.zeros(n, dtype=torch.bool, device=dev)
        self.early_contact = torch.zeros(n, dtype=torch.bool, device=dev)

        # Episode-level diagnostic accumulators
        self._diag_legal_count = 0
        self._diag_early_count = 0
        self._diag_total_episodes = 0
        self._diag_swing_vel_sum = 0.0
        self._diag_swing_vel_count = 0

        # Sample initial shifts
        self._resample_shifts(torch.arange(n, device=dev))

    def _resample_shifts(self, env_ids):
        """Sample random integer shifts for given envs."""
        n = env_ids.shape[0]
        dev = env_ids.device
        command = self.unwrapped.command_manager.get_term("motion")

        # Random shift ∈ [-shift_max, +shift_max]
        shifts = torch.randint(-self.shift_max, self.shift_max + 1, (n,), device=dev)
        self.per_env_shift[env_ids] = shifts

        # Store original bounds
        for i, eid in enumerate(env_ids):
            mid = command.motion_idx[eid].item()
            kf = command.motion.kick_frames[mid].item()
            kef = command.motion.kick_end_frames[mid].item()
            ml = command.motion_length[eid].item()
            if kf < 0:
                kf, kef = ml, ml

            # Original bounds
            ob = compute_segment_bounds(kf, kef, ml,
                prestrike_duration=self.v10_builder.PRESTRIKE_DURATION,
                min_strike_duration=self.v10_builder.MIN_STRIKE_DURATION)
            # Apply strike grace: expand STRIKE window backwards
            grace_prestrike_end = max(ob.approach_end, ob.prestrike_end - self.strike_grace)
            self.original_bounds[eid, 0] = ob.approach_end
            self.original_bounds[eid, 1] = grace_prestrike_end
            self.original_bounds[eid, 2] = ob.strike_end
            self.original_bounds[eid, 3] = ob.motion_length

            # Retimed bounds (shifted kick_frame)
            s = shifts[i].item()
            skf = max(1, min(int(kf + s), ml - 2))
            skef = max(skf + 1, min(int(kef + s), ml - 1))
            rb = compute_segment_bounds(skf, skef, ml,
                prestrike_duration=self.v10_builder.PRESTRIKE_DURATION,
                min_strike_duration=self.v10_builder.MIN_STRIKE_DURATION)
            # Apply strike grace to retimed bounds too
            grace_rb_prestrike_end = max(rb.approach_end, rb.prestrike_end - self.strike_grace)
            self.retimed_bounds[eid, 0] = rb.approach_end
            self.retimed_bounds[eid, 1] = grace_rb_prestrike_end
            self.retimed_bounds[eid, 2] = rb.strike_end
            self.retimed_bounds[eid, 3] = rb.motion_length

            # Update the builder's segment_bounds to retimed version
            self.v10_builder.segment_bounds[eid] = self.retimed_bounds[eid]

        # Reset contact tracking
        self.first_contact[env_ids] = False
        self.contact_step[env_ids] = -1.0

    def _apply_feature_noise(self, obs):
        """Apply conservative feature noise during training."""
        if not self.training:
            return obs
        n = obs.shape[0]
        dev = obs.device

        # Motor prior: multiplicative noise
        if self.motor_noise > 0:
            s, e = SLICE_MOTOR_PRIOR
            noise = 1.0 + (torch.rand(n, 1, device=dev) - 0.5) * 2 * self.motor_noise
            obs[:, s:e] = obs[:, s:e] * noise

        # Action history: per-env Bernoulli dropout
        if self.hist_dropout > 0:
            s, e = SLICE_ACTION_HIST
            mask = torch.rand(n, 1, device=dev) > self.hist_dropout
            obs[:, s:e] = obs[:, s:e] * mask.float()

        return obs

    def _compute_retiming_rewards(self, dones):
        """Compute LegalKick-lite gated rewards.

        Returns:
            reward: per-env reward tensor
            early_terminate: per-env bool tensor for APPROACH early contact
        """
        uw = self.unwrapped
        command = uw.command_manager.get_term("motion")
        dev = uw.device
        n = uw.num_envs
        t = command.time_steps.float()

        reward = torch.zeros(n, device=dev)
        early_terminate = torch.zeros(n, dtype=torch.bool, device=dev)

        # --- Detect ball contact via speed ---
        soccer_ball = uw.scene["soccer_ball"]
        ball_vel = soccer_ball.data.root_lin_vel_w[:, :3]
        ball_speed = torch.norm(ball_vel[:, :2], dim=-1)
        new_contact = (~self.first_contact) & (ball_speed > 0.5)

        if new_contact.any():
            self.first_contact[new_contact] = True
            self.contact_step[new_contact] = t[new_contact]

            phase_id, phase_phi = compute_event_phase(t, self.retimed_bounds)

            # ===== LegalKick-lite gate =====
            for idx in new_contact.nonzero(as_tuple=True)[0]:
                p = phase_id[idx].item()
                self.contact_phase[idx] = p

                if p == PHASE_STRIKE:
                    # --- LEGAL STRIKE: reward + enable ball outcome ---
                    self.legal_strike[idx] = True
                    reward[idx] += 50.0

                    # Ball outcome rewards (gated behind legal strike)
                    bs = ball_speed[idx]
                    reward[idx] += 10.0 * (1.0 - torch.exp(-bs / 1.2))  # ball_speed

                    # Direction alignment
                    ball_dir = ball_vel[idx, :2] / (bs + 1e-6)
                    ball_pos = soccer_ball.data.root_pos_w[idx, :2]
                    target_pos = command.target_destination_pos[idx, :2]
                    desired_dir = target_pos - ball_pos
                    desired_dir = desired_dir / (torch.norm(desired_dir) + 1e-6)
                    align = (ball_dir * desired_dir).sum()
                    reward[idx] += 30.0 * torch.clamp(align, 0.0, 1.0)

                elif p == PHASE_APPROACH:
                    # --- ILLEGAL EARLY (approach): heavy penalty + terminate ---
                    self.early_contact[idx] = True
                    reward[idx] -= 30.0
                    early_terminate[idx] = True

                elif p == PHASE_PRESTRIKE:
                    # --- PRESTRIKE: graded penalty, no ball outcome ---
                    self.early_contact[idx] = True
                    # Grace window: last 2 frames before strike
                    retimed_strike_start = self.retimed_bounds[idx, 1]  # prestrike_end
                    time_to_strike = retimed_strike_start - t[idx]
                    if time_to_strike <= 2.0:
                        reward[idx] -= 5.0   # grace window
                    else:
                        reward[idx] -= 20.0

                elif p == PHASE_FOLLOWTHRU:
                    # --- FOLLOWTHRU: no reward, no ball outcome ---
                    reward[idx] += 0.0

            # Contact timing reward (only for legal strikes)
            legal_new = new_contact & self.legal_strike
            if legal_new.any():
                retimed_strike = self.retimed_bounds[legal_new, 1]
                timing_err = self.contact_step[legal_new] - retimed_strike
                r_timing = torch.exp(-(timing_err ** 2) / (self.timing_sigma ** 2))
                reward[legal_new] += 20.0 * r_timing

        # --- Diagnostic: swing foot velocity at contact (log only, not gated) ---
        if new_contact.any():
            try:
                robot = command.robot
                # right_ankle_roll_link body index
                body_names = robot.data.body_names
                if "right_ankle_roll_link" in body_names:
                    swing_idx = body_names.index("right_ankle_roll_link")
                    swing_vel = robot.data.body_lin_vel_w[new_contact, swing_idx, :3]
                    swing_speed = torch.norm(swing_vel, dim=-1)
                    self._diag_swing_vel_sum += swing_speed.sum().item()
                    self._diag_swing_vel_count += swing_speed.shape[0]
            except Exception:
                pass  # Diagnostic only — never crash training

        # --- Event-warped joint tracking (every step, lower body) ---
        phase_id, phase_phi = compute_event_phase(t, self.retimed_bounds)
        ref_idx = event_warped_ref_index(phase_id, phase_phi, self.original_bounds)
        ref_idx_long = ref_idx.long().clamp(0, command.motion.joint_pos.shape[1] - 1)

        ref_joint = command.motion.joint_pos[command.motion_idx, ref_idx_long]
        cur_joint = command.robot.data.joint_pos

        # Lower body only: joints 0-11 (hip, knee, ankle)
        lb = 12
        joint_err = ((cur_joint[:, :lb] - ref_joint[:, :lb]) ** 2).mean(dim=-1)
        r_track = torch.exp(-joint_err / 0.5)
        reward += 2.0 * r_track

        return reward, early_terminate

    def _compute_v10(self):
        command = self.unwrapped.command_manager.get_term("motion")
        obs = self.v10_builder.compute(self.unwrapped, command)
        return self._apply_feature_noise(obs)

    def get_observations(self):
        obs, extras = super().get_observations()
        obs_v10 = self._compute_v10()
        if isinstance(obs, dict):
            obs["policy"] = obs_v10
        else:
            obs = obs_v10
        if "observations" not in extras:
            extras["observations"] = {}
        extras["observations"]["critic"] = obs_v10
        return obs, extras

    def step(self, actions):
        obs, rew, dones, infos = super().step(actions)
        command = self.unwrapped.command_manager.get_term("motion")

        self.v10_builder.update_history(self.unwrapped, command, actions, dones)

        # Compute LegalKick-lite gated rewards
        retiming_rew, early_term = self._compute_retiming_rewards(dones)
        rew = rew + retiming_rew

        # Apply early contact termination (APPROACH only)
        if early_term.any():
            dones = dones | early_term

        # --- Comprehensive logging ---
        if "log" not in infos:
            infos["log"] = {}

        # Track episode completions for diagnostic rates
        newly_done = dones
        if newly_done.any():
            n_done = newly_done.sum().item()
            self._diag_total_episodes += n_done
            self._diag_legal_count += (self.legal_strike & newly_done).sum().item()
            self._diag_early_count += (self.early_contact & newly_done).sum().item()

        # Timing diagnostics
        contacted = self.first_contact
        if contacted.any():
            cd_event = (self.contact_step[contacted] -
                        self.retimed_bounds[contacted, 1]).mean().item()
            cd_orig = (self.contact_step[contacted] -
                        self.original_bounds[contacted, 1]).mean().item()
            infos["log"]["cd_event"] = cd_event
            infos["log"]["cd_original"] = cd_orig

        # LegalKick diagnostic rates
        if self._diag_total_episodes > 0:
            infos["log"]["legal_strike_pct"] = (
                100.0 * self._diag_legal_count / self._diag_total_episodes)
            infos["log"]["early_contact_pct"] = (
                100.0 * self._diag_early_count / self._diag_total_episodes)

        # Swing foot velocity diagnostic
        if self._diag_swing_vel_count > 0:
            infos["log"]["swing_vel_at_contact"] = (
                self._diag_swing_vel_sum / self._diag_swing_vel_count)

        # Contact phase histogram (mean phase at first contact)
        has_phase = self.contact_phase >= 0
        if has_phase.any():
            infos["log"]["contact_phase_mean"] = self.contact_phase[has_phase].float().mean().item()

        infos["log"]["mean_shift"] = self.per_env_shift.float().mean().item()
        infos["log"]["retiming_reward"] = retiming_rew.mean().item()

        # Compute v10 obs
        obs_v10 = self._compute_v10()
        if isinstance(obs, dict):
            obs["policy"] = obs_v10
        else:
            obs = obs_v10
        if "observations" not in infos:
            infos["observations"] = {}
        infos["observations"]["critic"] = obs_v10

        # Handle resets: resample shifts for done envs
        if dones.any():
            done_ids = dones.nonzero(as_tuple=True)[0]
            # Reset per-env contact/legal state for done envs
            self.contact_phase[done_ids] = -1
            self.legal_strike[done_ids] = False
            self.early_contact[done_ids] = False
            self._resample_shifts(done_ids)

        return obs, rew, dones, infos

    def reset(self):
        obs, extras = super().reset()
        command = self.unwrapped.command_manager.get_term("motion")
        all_ids = torch.arange(self.unwrapped.num_envs, device=self.unwrapped.device)
        self._resample_shifts(all_ids)
        self.v10_builder.reset(all_ids)

        obs_v10 = self._compute_v10()
        if isinstance(obs, dict):
            obs["policy"] = obs_v10
        else:
            obs = obs_v10
        if "observations" not in extras:
            extras["observations"] = {}
        extras["observations"]["critic"] = obs_v10
        return obs, extras


class V10RetimingRunner(MotionOnPolicyRunner):
    """Runner with conditional BC reg and shift-aware logging."""

    def __init__(self, env, train_cfg, log_dir, device, bc_model, bc_weight):
        super().__init__(env, train_cfg, log_dir, device, registry_name=None)
        self.bc_model = bc_model
        self.bc_model.eval()
        for p in self.bc_model.parameters():
            p.requires_grad = False
        self.bc_weight_init = bc_weight
        self.bc_weight = bc_weight
        self._original_update = self.alg.update
        self.alg.update = self._update_with_bc
        self.alg.bc_weight = bc_weight
        self.alg.bc_loss = 0.0

    def _update_with_bc(self):
        alg = self.alg
        mean_vl = mean_sl = mean_ent = mean_bc = 0

        gen = (alg.storage.recurrent_mini_batch_generator(alg.num_mini_batches, alg.num_learning_epochs)
               if alg.policy.is_recurrent else
               alg.storage.mini_batch_generator(alg.num_mini_batches, alg.num_learning_epochs))

        for (obs_b, cobs_b, act_b, tv_b, adv_b, ret_b, olp_b, omu_b, osig_b, hid_b, mask_b, rnd_b) in gen:
            bs = obs_b.shape[0]
            alg.policy.act(obs_b, masks=mask_b, hidden_states=hid_b[0])
            alp = alg.policy.get_actions_log_prob(act_b)
            vb = alg.policy.evaluate(cobs_b, masks=mask_b, hidden_states=hid_b[1])
            mu = alg.policy.action_mean[:bs]
            sig = alg.policy.action_std[:bs]
            ent = alg.policy.entropy[:bs]

            if alg.desired_kl is not None and alg.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sig / osig_b + 1e-5) +
                        (osig_b**2 + (omu_b - mu)**2) / (2*sig**2) - 0.5, dim=-1).mean()
                    if kl > alg.desired_kl * 2:
                        alg.learning_rate = max(1e-5, alg.learning_rate / 1.5)
                    elif kl < alg.desired_kl / 2 and kl > 0:
                        alg.learning_rate = min(1e-2, alg.learning_rate * 1.5)
                    for pg in alg.optimizer.param_groups:
                        pg["lr"] = alg.learning_rate

            ratio = torch.exp(alp - olp_b.squeeze())
            surr = -adv_b.squeeze() * ratio
            surr_c = -adv_b.squeeze() * ratio.clamp(1-alg.clip_param, 1+alg.clip_param)
            surr_loss = torch.max(surr, surr_c).mean()

            if alg.use_clipped_value_loss:
                vc = tv_b + (vb - tv_b).clamp(-alg.clip_param, alg.clip_param)
                vl = torch.max((vb - ret_b)**2, (vc - ret_b)**2).mean()
            else:
                vl = (ret_b - vb).pow(2).mean()

            loss = surr_loss + alg.value_loss_coef * vl - alg.entropy_coef * ent.mean()

            # BC reg with decay
            if self.bc_weight > 1e-6:
                with torch.no_grad():
                    expert = self.bc_model(obs_b)
                bc_loss = ((mu - expert)**2).mean()
                loss = loss + self.bc_weight * bc_loss
                mean_bc += bc_loss.item()

            alg.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(alg.policy.parameters(), alg.max_grad_norm)
            alg.optimizer.step()
            mean_vl += vl.item()
            mean_sl += surr_loss.item()
            mean_ent += ent.mean().item()

        nu = alg.num_learning_epochs * alg.num_mini_batches
        mean_vl /= nu; mean_sl /= nu; mean_ent /= nu; mean_bc /= nu
        alg.storage.clear()

        # Decay BC weight
        self.bc_weight = self.bc_weight_init * max(0.0, 1.0 - self.current_learning_iteration / 300)

        return {"value_function": mean_vl, "surrogate": mean_sl, "entropy": mean_ent,
                "bc_loss": mean_bc, "bc_weight": self.bc_weight}

    def load(self, path, load_optimizer=True):
        ld = torch.load(path, map_location=self.device, weights_only=False)
        self.alg.policy.load_state_dict(ld["model_state_dict"])
        if load_optimizer and ld.get("optimizer_state_dict"):
            self.alg.optimizer.load_state_dict(ld["optimizer_state_dict"])
        if self.empirical_normalization:
            if ld.get("obs_norm_state_dict"):
                self.obs_normalizer.load_state_dict(ld["obs_norm_state_dict"])
        self.current_learning_iteration = int(ld.get("iter", 0))
        self.tot_timesteps = int(ld.get("total_steps", 0))
        print(f"[INFO] Loaded checkpoint from iter {self.current_learning_iteration}")


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations

    # Critical PPO overrides for BC init
    agent_cfg.empirical_normalization = False
    agent_cfg.policy.init_noise_std = 0.05
    agent_cfg.algorithm.learning_rate = 1e-5
    agent_cfg.algorithm.schedule = "fixed"
    agent_cfg.algorithm.entropy_coef = 0.0

    env_cfg.scene.num_envs = args_cli.num_envs or env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.commands.motion.motion_files = get_motion_files(args_cli.motion_file)

    # --- Disable ALL fixed-timing and ball outcome rewards BEFORE env creation ---
    # Fixed-timing rewards (replaced by wrapper LegalKick-lite gate)
    env_cfg.rewards.target_point_contact.weight = 0.0
    env_cfg.rewards.early_collision_penalty.weight = 0.0
    env_cfg.rewards.contact_graph_match.weight = 0.0
    # Ball outcome rewards (re-implemented in wrapper, gated by LegalKick)
    env_cfg.rewards.sideways_kick.weight = 0.0
    env_cfg.rewards.ball_velocity_direction_alignment.weight = 0.0
    env_cfg.rewards.ball_speed_reward.weight = 0.0
    print("[v10.2] Disabled fixed-timing rewards: target_point_contact, "
          "early_collision_penalty, contact_graph_match")
    print("[v10.2] Disabled ungated ball outcome rewards: sideways_kick, "
          "ball_velocity_direction_alignment, ball_speed_reward")
    print("[v10.2] Ball outcome rewards are now gated by LegalKick-lite")

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", "v10_retiming"))
    log_dir = os.path.join(log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                           + f"_shift{args_cli.shift_max}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RetimingV10Wrapper(
        env, shift_max=args_cli.shift_max,
        motor_noise=args_cli.motor_noise,
        hist_dropout=args_cli.hist_dropout,
        timing_sigma=args_cli.timing_sigma,
        strike_grace=args_cli.strike_grace,
        training=True,
    )

    # Load BC model
    dev = env_cfg.sim.device
    bc_ckpt = torch.load(args_cli.bc_checkpoint, map_location=dev, weights_only=False)
    bc_model = V10MLPActor(bc_ckpt["obs_dim"], bc_ckpt["action_dim"]).to(dev)
    bc_model.load_state_dict(bc_ckpt["model_state_dict"])

    runner = V10RetimingRunner(
        env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device,
        bc_model=bc_model, bc_weight=args_cli.bc_reg_weight,
    )

    if agent_cfg.resume:
        rp = agent_cfg.load_checkpoint
        if not os.path.exists(rp):
            rp = os.path.join(log_root, rp)
        if not os.path.exists(rp):
            rp = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
        runner.load(rp)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    print(f"\n[v10.2] shift_max={args_cli.shift_max}, motor_noise={args_cli.motor_noise}, "
          f"hist_dropout={args_cli.hist_dropout}, bc_weight={args_cli.bc_reg_weight}, "
          f"strike_grace={args_cli.strike_grace}")

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
