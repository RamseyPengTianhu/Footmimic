"""Load BC weights into PPO ActorCritic model for v10.1b fine-tuning.

Usage:
    python scripts/rsl_rl/load_bc_into_ppo.py \
        --bc_checkpoint logs/rsl_rl/v10_bc/bc_pretrained_v2.pt \
        --output_path logs/rsl_rl/v10_bc/ppo_initialized_v2.pt
"""
import argparse
import os
import torch
import torch.nn as nn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bc_checkpoint", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--std", type=float, default=0.05, help="Initial std for PPO exploration")
    args = parser.parse_args()

    print(f"[INFO] Loading BC checkpoint from {args.bc_checkpoint}...")
    bc_ckpt = torch.load(args.bc_checkpoint, map_location=args.device, weights_only=False)
    
    obs_dim = bc_ckpt["obs_dim"]
    action_dim = bc_ckpt["action_dim"]
    
    print(f"[INFO] BC Model: obs_dim={obs_dim}, action_dim={action_dim}")
    
    # In RslRl, ActorCritic expects standard config
    from rsl_rl.modules import ActorCritic
    
    # Note: RslRl PPO config for G1FlatPPORunnerCfg uses [512, 256, 128]
    # We instantiate the exact ActorCritic matching the BC model structure
    actor_critic = ActorCritic(
        num_actor_obs=obs_dim,
        num_critic_obs=obs_dim,  # Critic obs can be anything, but let's match for init
        num_actions=action_dim,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=args.std,
    ).to(args.device)
    
    print(f"[INFO] PPO Actor initialized with std={args.std}")
    
    # BC weights are stored as `model_state_dict` which contains keys like:
    # net.0.weight, net.0.bias, net.2.weight, ...
    bc_state_dict = bc_ckpt["model_state_dict"]
    
    # RSL-RL ActorCritic actor network is `actor` which is a sequential with keys:
    # 0.weight, 0.bias, 2.weight, ...
    actor_state_dict = actor_critic.actor.state_dict()
    
    new_actor_dict = {}
    for bc_key, bc_val in bc_state_dict.items():
        if bc_key.startswith("net."):
            ac_key = bc_key.replace("net.", "")
            if ac_key in actor_state_dict:
                new_actor_dict[ac_key] = bc_val
            else:
                print(f"[WARN] BC key {ac_key} not found in PPO actor!")
    
    actor_critic.actor.load_state_dict(new_actor_dict)
    print(f"[INFO] Successfully loaded {len(new_actor_dict)} tensors into PPO actor mean network.")
    
    # Set the std
    actor_critic.std.data.fill_(args.std)
    
    # Create the checkpoint format expected by RSL-RL runner.load()
    ppo_ckpt = {
        "model_state_dict": actor_critic.state_dict(),
        # Add basic dummy values to bypass assertions if needed
        "optimizer_state_dict": None,
        "iter": 0,
    }
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(ppo_ckpt, args.output_path)
    print(f"[INFO] Saved PPO initialized checkpoint to {args.output_path}")
    
    # Verify deterministic output (Sanity Check)
    print("\n[INFO] Running Sanity Check...")
    dummy_obs = torch.randn(2, obs_dim, device=args.device)
    
    # Load BC Model
    from train_v10_bc import V10MLPActor
    bc_model = V10MLPActor(obs_dim, action_dim).to(args.device)
    bc_model.load_state_dict(bc_state_dict)
    bc_model.eval()
    
    with torch.no_grad():
        a_bc = bc_model(dummy_obs)
        a_ppo = actor_critic.actor(dummy_obs)
        diff = (a_bc - a_ppo).abs().max().item()
        
    print(f"Max absolute difference between BC and PPO actor mean: {diff:.6f}")
    if diff < 1e-5:
        print("[INFO] Sanity Check PASSED! The PPO actor perfectly matches the BC model.")
    else:
        print("[ERROR] Sanity Check FAILED! Architecture or weight loading mismatch.")

if __name__ == "__main__":
    main()
