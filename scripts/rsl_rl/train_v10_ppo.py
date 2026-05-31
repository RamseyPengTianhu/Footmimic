"""Launch v10.1b PPO Fine-Tuning with BC Regularization."""

import argparse
import sys
import os

from isaaclab.app import AppLauncher
import cli_args

parser = argparse.ArgumentParser(description="Train v10 PPO with BC Regularization.")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default="Anchor-CG-Kick-G1-Soccer-RNN-v0")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--motion_file", type=str, required=True)
parser.add_argument("--bc_checkpoint", type=str, required=True, help="Path to frozen BC model for regularization.")
parser.add_argument("--bc_reg_weight", type=float, default=1.0, help="Initial BC regularization weight.")

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import torch.nn as nn
from datetime import datetime

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import soccer.tasks
from soccer.utils.my_on_policy_runner import MotionOnPolicyRunner
from rsl_rl.algorithms import PPO

# V10 Actor matching BC Architecture
class V10MLPActor(nn.Module):
    def __init__(self, obs_dim, action_dim=29, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        layers = []
        for h in hidden_dims:
            layers.append(nn.Linear(obs_dim, h))
            layers.append(nn.ELU())
            obs_dim = h
        layers.append(nn.Linear(obs_dim, action_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, obs):
        return self.net(obs)


class V10OnPolicyRunner(MotionOnPolicyRunner):
    """Runner that adds BC regularization to PPO without replacing the algorithm.
    
    Instead of subclassing PPO (which would discard the already-initialized storage),
    we monkey-patch the update method to inject BC loss into the standard PPO loop.
    """
    def __init__(self, env, train_cfg, log_dir, device, bc_model, bc_weight):
        super().__init__(env, train_cfg, log_dir, device, registry_name=None)
        
        # Store BC components
        self.bc_model = bc_model
        self.bc_model.eval()
        for param in self.bc_model.parameters():
            param.requires_grad = False
        self.bc_weight = bc_weight
        
        # Save reference to the original update method
        self._original_update = self.alg.update
        # Replace with our BC-augmented version
        self.alg.update = self._update_with_bc
        # Expose bc_weight/bc_loss on the algorithm for logging
        self.alg.bc_weight = bc_weight
        self.alg.bc_loss = 0.0

    def _update_with_bc(self):
        """PPO update with BC regularization injected into the loss."""
        alg = self.alg
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_bc_loss = 0

        if alg.policy.is_recurrent:
            generator = alg.storage.recurrent_mini_batch_generator(alg.num_mini_batches, alg.num_learning_epochs)
        else:
            generator = alg.storage.mini_batch_generator(alg.num_mini_batches, alg.num_learning_epochs)

        for (
            obs_batch, critic_obs_batch, actions_batch, target_values_batch,
            advantages_batch, returns_batch, old_actions_log_prob_batch,
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch, rnd_state_batch,
        ) in generator:
            original_batch_size = obs_batch.shape[0]

            # Recompute actions
            alg.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = alg.policy.get_actions_log_prob(actions_batch)
            value_batch = alg.policy.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            
            mu_batch = alg.policy.action_mean[:original_batch_size]
            sigma_batch = alg.policy.action_std[:original_batch_size]
            entropy_batch = alg.policy.entropy[:original_batch_size]

            # KL-based LR adaptation
            if alg.desired_kl is not None and alg.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if kl_mean > alg.desired_kl * 2.0:
                        alg.learning_rate = max(1e-5, alg.learning_rate / 1.5)
                    elif kl_mean < alg.desired_kl / 2.0 and kl_mean > 0.0:
                        alg.learning_rate = min(1e-2, alg.learning_rate * 1.5)
                    for param_group in alg.optimizer.param_groups:
                        param_group["lr"] = alg.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - alg.clip_param, 1.0 + alg.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value loss
            if alg.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -alg.clip_param, alg.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + alg.value_loss_coef * value_loss - alg.entropy_coef * entropy_batch.mean()

            # ===== BC REGULARIZATION =====
            with torch.no_grad():
                expert_actions = self.bc_model(obs_batch)
            current_mean = alg.policy.action_mean
            bc_loss = ((current_mean - expert_actions) ** 2).mean()
            loss = loss + self.bc_weight * bc_loss
            # ==============================

            alg.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(alg.policy.parameters(), alg.max_grad_norm)
            alg.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_bc_loss += bc_loss.item()

        num_updates = alg.num_learning_epochs * alg.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_bc_loss /= num_updates
        alg.storage.clear()

        return {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "bc_loss": mean_bc_loss,
            "bc_weight": self.bc_weight,
        }

    def load(self, path, load_optimizer=True):
        """Load checkpoint, tolerating missing keys from hand-crafted checkpoints."""
        loaded_dict = torch.load(path, map_location=self.device, weights_only=False)
        self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        if load_optimizer and loaded_dict.get("optimizer_state_dict") is not None:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        if self.empirical_normalization:
            if loaded_dict.get("obs_norm_state_dict") is not None:
                self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
            if loaded_dict.get("privileged_obs_norm_state_dict") is not None:
                self.privileged_obs_normalizer.load_state_dict(loaded_dict["privileged_obs_norm_state_dict"])
        self.current_learning_iteration = int(loaded_dict.get("iter", 0))
        self.tot_timesteps = int(loaded_dict.get("total_steps", 0))
        print(f"[INFO] Loaded checkpoint from iter {self.current_learning_iteration}")


import glob

def get_motion_files(motion_path):
    if os.path.isfile(motion_path):
        return [motion_path]
    elif os.path.isdir(motion_path):
        files = sorted(glob.glob(os.path.join(motion_path, "*.npz")))
        if not files:
            raise ValueError(f"No .npz files in {motion_path}")
        return files
    else:
        raise ValueError(f"Invalid path: {motion_path}")

from soccer.tasks.tracking.mdp.event_conditioned_obs_builder import V10ObsBuilder

class V10RslRlVecEnvWrapper(RslRlVecEnvWrapper):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        command = self.unwrapped.command_manager.get_term("motion")
        self.v10_builder = V10ObsBuilder(
            num_envs=self.unwrapped.num_envs,
            num_joints=command.robot.data.joint_pos.shape[1],
            device=self.unwrapped.device,
        )
        # Critical: initialize segment bounds from current motion commands
        self.v10_builder.init_segment_bounds(command)
        self.num_obs = 454
        self.num_privileged_obs = 454  # critic also uses 454D v10 obs

    def _compute_v10(self):
        command = self.unwrapped.command_manager.get_term("motion")
        return self.v10_builder.compute(self.unwrapped, command)

    def get_observations(self) -> tuple[torch.Tensor, dict]:
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
        
    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        obs, rew, dones, infos = super().step(actions)
        command = self.unwrapped.command_manager.get_term("motion")
        
        # update_history internally handles dones (resets history + segment_bounds)
        self.v10_builder.update_history(self.unwrapped, command, actions, dones)
            
        obs_v10 = self._compute_v10()
        if isinstance(obs, dict):
            obs["policy"] = obs_v10
        else:
            obs = obs_v10
        if "observations" not in infos:
            infos["observations"] = {}
        infos["observations"]["critic"] = obs_v10
            
        return obs, rew, dones, infos

    def reset(self) -> tuple[torch.Tensor, dict]:
        obs, extras = super().reset()
        command = self.unwrapped.command_manager.get_term("motion")
        self.v10_builder.init_segment_bounds(command)
        self.v10_builder.reset(torch.arange(self.unwrapped.num_envs, device=self.unwrapped.device))
        
        obs_v10 = self._compute_v10()
        if isinstance(obs, dict):
            obs["policy"] = obs_v10
        else:
            obs = obs_v10
        if "observations" not in extras:
            extras["observations"] = {}
        extras["observations"]["critic"] = obs_v10
        return obs, extras

@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    # CLI max_iterations is not handled by update_rsl_rl_cfg — override manually
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations

    # ===== Critical overrides for BC-initialized PPO =====
    # BC model was trained on raw obs — normalization would corrupt the input
    agent_cfg.empirical_normalization = False
    # Conservative exploration — BC policy is already close to good
    agent_cfg.policy.init_noise_std = 0.05
    # Very low fixed LR — adaptive schedule ramps LR when KL is low (BC-constrained)
    agent_cfg.algorithm.learning_rate = 1e-5
    agent_cfg.algorithm.schedule = "fixed"
    # Lower entropy to avoid destabilizing BC-initialized policy
    agent_cfg.algorithm.entropy_coef = 0.0
    # =====================================================

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.commands.motion.motion_files = get_motion_files(args_cli.motion_file)

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    log_dir = os.path.join(log_root_path, datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{agent_cfg.run_name}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = V10RslRlVecEnvWrapper(env)

    # Load BC Model for Regularization
    print(f"[INFO] Loading BC Regularization Model from {args_cli.bc_checkpoint}")
    bc_ckpt = torch.load(args_cli.bc_checkpoint, map_location=env_cfg.sim.device, weights_only=False)
    bc_model = V10MLPActor(bc_ckpt["obs_dim"], bc_ckpt["action_dim"]).to(env_cfg.sim.device)
    bc_model.load_state_dict(bc_ckpt["model_state_dict"])
    
    runner = V10OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device,
        bc_model=bc_model, bc_weight=args_cli.bc_reg_weight
    )

    if agent_cfg.resume:
        if os.path.exists(agent_cfg.load_checkpoint):
            resume_path = agent_cfg.load_checkpoint
        elif os.path.exists(os.path.join(log_root_path, agent_cfg.load_checkpoint)):
            resume_path = os.path.join(log_root_path, agent_cfg.load_checkpoint)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
