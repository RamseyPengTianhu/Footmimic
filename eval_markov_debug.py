import torch, os, sys
from isaaclab.app import AppLauncher
app_launcher = AppLauncher({"headless": True, "num_envs": 16})
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
import soccer.tasks  # noqa: F401
from latent_v2_models import LatentActionModel
from compute_task_features import compute_ball_foot_relation

ckpt = torch.load("models/latent_v2/markov_prior_vq_k16_hold2.pt", map_location="cuda:0", weights_only=False)
model = LatentActionModel(
    obs_dim=int(ckpt["obs_dim"]), action_dim=int(ckpt["action_dim"]), z_dim=int(ckpt["z_dim"]),
    hidden_dims=list(ckpt["hidden_dims"]), decoder_obs_mode=ckpt["decoder_obs_mode"],
    prior_type="vq", num_codes=int(ckpt["num_codes"]),
    commitment_weight=float(ckpt.get("commitment_weight", 0.25)), markov_prior=True,
).to("cuda:0")
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

env = gym.make("Anchor-CG-Kick-G1-Soccer-RNN-v0", render_mode=None)
env = RslRlVecEnvWrapper(env)
base_env = env.unwrapped
obs, _ = env.get_observations()

code_hold = 2
hold_ctr = torch.zeros(16, dtype=torch.long, device="cuda:0")
prev_code = torch.full((16,), model.prior.start_token, dtype=torch.long, device="cuda:0")
held_zq = torch.zeros(16, model.z_dim, device="cuda:0")

codes_chosen = []

for step in range(100):
    with torch.no_grad():
        tf = compute_ball_foot_relation(base_env)
        dec_obs = model.select_decoder_obs(obs, task_features=tf)
        needs = (hold_ctr % code_hold == 0)
        if needs.any():
            logits = model.prior(dec_obs, prev_code=prev_code)
            new_code = logits[needs].argmax(dim=-1)
            held_zq[needs] = model.codebook.lookup(new_code)
            prev_code[needs] = new_code
            codes_chosen.append(new_code[0].item())
        action = model.decoder(dec_obs, held_zq)
    
    obs, _, dones, _ = env.step(action)
    hold_ctr += 1
    
    if dones.any():
        for i in dones.nonzero(as_tuple=True)[0]:
            hold_ctr[i] = 0
            prev_code[i] = model.prior.start_token

print("First env codes:", codes_chosen)
simulation_app.close()
