"""VQ Code Semantic Diagnostic.

Runs teacher-driven rollouts through a frozen VQ model, recording for each
timestep: posterior code, prior code, ref_phase, geo_phase, ball/foot geometry.
Then produces cross-tabulation tables answering:

1. Does the codebook have phase semantics?
2. Where do prior and posterior disagree?
3. What code sequences lead to clean vs late vs noattempt?

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/rsl_rl/eval_vq_code_semantics.py \
        --task Anchor-CG-Kick-G1-Soccer-RNN-v0 \
        --motion_path motions/Video_hmr4d_seed \
        --latent_v2_model models/latent_v2/online_distill_vq_k16_hold2_seq.pt \
        --load_run <teacher_run> \
        --num_envs 32 --eval_episodes 100 \
        --device cuda:0 --headless
"""
from __future__ import annotations
import argparse, os, sys, glob, json
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--motion_path", type=str, default=None)
parser.add_argument("--latent_v2_model", type=str, required=True)
parser.add_argument("--eval_episodes", type=int, default=100, help="Episodes per motion.")
parser.add_argument("--num_envs", type=int, default=32)

from isaaclab.app import AppLauncher
import cli_args

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if getattr(args_cli, "headless", False) and "DISPLAY" in os.environ:
    os.environ.pop("DISPLAY", None)
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from collections import defaultdict
from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
import soccer.tasks  # noqa: F401

# Phase constants (matching event_phase.py)
PHASE_APPROACH = 0
PHASE_PRESTRIKE = 1
PHASE_STRIKE = 2
PHASE_FOLLOW = 3
PHASE_NAMES = {0: "approach", 1: "prestrike", 2: "strike", 3: "follow"}

# Geo-phase thresholds
GEO_IDLE = -1
GEO_APPROACH = 0
GEO_PRESTRIKE = 1
GEO_STRIKE = 2
GEO_POST_CONTACT = 3
GEO_PHASE_NAMES = {-1: "idle", 0: "approach", 1: "prestrike", 2: "strike", 3: "post_contact"}


def compute_geo_phase(ball_dist, foot_speed, closing_vel, had_contact):
    """Assign geo_phase per env based on ball-foot geometry.

    Args: all [N] tensors
    Returns: [N] long tensor of geo phase IDs
    """
    N = ball_dist.shape[0]
    phase = torch.full((N,), GEO_IDLE, dtype=torch.long, device=ball_dist.device)
    # Post-contact overrides everything
    phase[had_contact] = GEO_POST_CONTACT
    # Strike: foot moving fast + close to ball (before contact)
    strike_mask = (~had_contact) & (foot_speed > 2.0) & (ball_dist < 0.4)
    phase[strike_mask] = GEO_STRIKE
    # Prestrike: close + closing but not yet swinging hard
    prestrike_mask = (~had_contact) & (~strike_mask) & (ball_dist < 0.5) & (closing_vel > 0.3)
    phase[prestrike_mask] = GEO_PRESTRIKE
    # Approach: further out but closing
    approach_mask = (~had_contact) & (phase == GEO_IDLE) & (closing_vel > 0)
    phase[approach_mask] = GEO_APPROACH
    return phase


def _load_model(path, device):
    from latent_v2_models import LatentActionModel
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = LatentActionModel(
        obs_dim=int(ckpt["obs_dim"]),
        action_dim=int(ckpt["action_dim"]),
        z_dim=int(ckpt["z_dim"]),
        hidden_dims=list(ckpt["hidden_dims"]),
        decoder_obs_mode=ckpt.get("decoder_obs_mode", "full"),
        prior_type=ckpt.get("prior_type", "mlp"),
        num_codes=int(ckpt.get("num_codes", 16)),
        commitment_weight=float(ckpt.get("commitment_weight", 0.25)),
        markov_prior=bool(ckpt.get("markov_prior", False)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ckpt


class EpisodeTracker:
    """Track per-env episode data: code sequences + phase labels."""

    def __init__(self, num_envs, num_motions, motion_names, episodes_per_motion, device):
        self.num_envs = num_envs
        self.num_motions = num_motions
        self.motion_names = motion_names
        self.episodes_per_motion = episodes_per_motion
        self.device = device
        # Per-env running buffers
        self.ep_post_codes = [[] for _ in range(num_envs)]
        self.ep_prior_codes = [[] for _ in range(num_envs)]
        self.ep_ref_phases = [[] for _ in range(num_envs)]
        self.ep_geo_phases = [[] for _ in range(num_envs)]
        self.ep_ball_dist = [[] for _ in range(num_envs)]
        self.ep_foot_speed = [[] for _ in range(num_envs)]
        # Motion assignment
        self.env_motion_idx = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.episodes_done = torch.zeros(num_motions, dtype=torch.long, device=device)
        # Completed episodes
        self.completed = []  # list of dicts

    def assign_motions_round_robin(self, cmd):
        """Assign motions to envs round-robin."""
        for i in range(self.num_envs):
            m = i % self.num_motions
            self.env_motion_idx[i] = m
            cmd.motion_idx[i] = m

    def record_step(self, post_code, prior_code, ref_phase, geo_phase, ball_dist, foot_speed):
        """Record one timestep for all envs."""
        for i in range(self.num_envs):
            self.ep_post_codes[i].append(post_code[i].item())
            self.ep_prior_codes[i].append(prior_code[i].item())
            self.ep_ref_phases[i].append(ref_phase[i].item())
            self.ep_geo_phases[i].append(geo_phase[i].item())
            self.ep_ball_dist[i].append(ball_dist[i].item())
            self.ep_foot_speed[i].append(foot_speed[i].item())

    def on_done(self, env_idx, terminated):
        """Finalize episode for one env."""
        m = self.env_motion_idx[env_idx].item()
        if self.episodes_done[m] >= self.episodes_per_motion:
            return
        ep = {
            "motion_idx": m,
            "motion_name": self.motion_names[m],
            "post_codes": list(self.ep_post_codes[env_idx]),
            "prior_codes": list(self.ep_prior_codes[env_idx]),
            "ref_phases": list(self.ep_ref_phases[env_idx]),
            "geo_phases": list(self.ep_geo_phases[env_idx]),
            "ball_dist": list(self.ep_ball_dist[env_idx]),
            "foot_speed": list(self.ep_foot_speed[env_idx]),
            "terminated": bool(terminated),
            "length": len(self.ep_post_codes[env_idx]),
        }
        self.completed.append(ep)
        self.episodes_done[m] += 1
        # Reset buffers
        self.ep_post_codes[env_idx] = []
        self.ep_prior_codes[env_idx] = []
        self.ep_ref_phases[env_idx] = []
        self.ep_geo_phases[env_idx] = []
        self.ep_ball_dist[env_idx] = []
        self.ep_foot_speed[env_idx] = []
        # Reassign if needed
        if self.episodes_done[m] >= self.episodes_per_motion:
            # Find next motion that needs episodes
            for nm in range(self.num_motions):
                if self.episodes_done[nm] < self.episodes_per_motion:
                    self.env_motion_idx[env_idx] = nm
                    return nm
        return None

    def is_done(self):
        return (self.episodes_done >= self.episodes_per_motion).all()


def print_code_phase_table(title, codes, phases, num_codes, phase_names):
    """Print Code × Phase cross-tabulation."""
    phase_ids = sorted(phase_names.keys())
    counts = np.zeros((num_codes, len(phase_ids)), dtype=int)
    for c, p in zip(codes, phases):
        col = phase_ids.index(p)
        counts[c, col] += 1
    # Normalize per code (row)
    row_sums = counts.sum(axis=1, keepdims=True)
    pcts = np.where(row_sums > 0, counts / row_sums * 100, 0)

    hdr = f"{'Code':>6s}"
    for pid in phase_ids:
        hdr += f" {phase_names[pid]:>10s}"
    hdr += f" {'Total':>8s}"
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    print(hdr)
    print("-" * 70)
    for k in range(num_codes):
        row = f"{k:>6d}"
        for j in range(len(phase_ids)):
            row += f" {pcts[k,j]:>9.1f}%"
        row += f" {row_sums[k,0]:>8d}"
        print(row)
    # Phase totals
    col_sums = counts.sum(axis=0)
    total_row = f"{'Total':>6s}"
    for j in range(len(phase_ids)):
        total_row += f" {col_sums[j]:>10d}"
    total_row += f" {counts.sum():>8d}"
    print("-" * 70)
    print(total_row)


def print_phase_code_table(title, codes, phases, num_codes, phase_names):
    """Print Phase × Code: which codes dominate each phase."""
    phase_ids = sorted(phase_names.keys())
    counts = np.zeros((len(phase_ids), num_codes), dtype=int)
    for c, p in zip(codes, phases):
        row = phase_ids.index(p)
        counts[row, c] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    pcts = np.where(row_sums > 0, counts / row_sums * 100, 0)

    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    for i, pid in enumerate(phase_ids):
        name = phase_names[pid]
        total = row_sums[i, 0]
        # Sort codes by frequency for this phase
        order = np.argsort(-pcts[i])
        top = [(order[j], pcts[i, order[j]]) for j in range(min(5, num_codes)) if pcts[i, order[j]] > 0.5]
        top_str = ", ".join(f"c{c}={p:.1f}%" for c, p in top)
        print(f"  {name:>12s} (N={total:>6d}): {top_str}")


def print_agreement_table(post_codes, prior_codes, phases, phase_names):
    """Print prior/posterior agreement by phase."""
    phase_ids = sorted(phase_names.keys())
    print(f"\n{'='*70}")
    print(f"  Prior vs Posterior Agreement by Phase")
    print(f"{'='*70}")
    print(f"{'Phase':>12s} {'N':>8s} {'Agree%':>8s} {'Top Disagree':>30s}")
    print("-" * 70)
    post_arr = np.array(post_codes)
    prior_arr = np.array(prior_codes)
    phase_arr = np.array(phases)
    for pid in phase_ids:
        mask = phase_arr == pid
        n = mask.sum()
        if n == 0:
            continue
        agree = (post_arr[mask] == prior_arr[mask]).sum()
        pct = agree / n * 100
        # Find most common disagreements
        disagree_mask = mask & (post_arr != prior_arr)
        if disagree_mask.sum() > 0:
            pairs = list(zip(prior_arr[disagree_mask], post_arr[disagree_mask]))
            from collections import Counter
            top_d = Counter(pairs).most_common(3)
            d_str = ", ".join(f"pr{p}→po{q}({c})" for (p, q), c in top_d)
        else:
            d_str = "-"
        print(f"{phase_names[pid]:>12s} {n:>8d} {pct:>7.1f}% {d_str:>30s}")
    # Overall
    n_total = len(post_codes)
    agree_total = (post_arr == prior_arr).sum()
    print(f"{'OVERALL':>12s} {n_total:>8d} {agree_total/n_total*100:>7.1f}%")


def identify_strike_codes(codes, phases, num_codes, phase_names):
    """Identify codes that are strike-dominant."""
    phase_ids = sorted(phase_names.keys())
    strike_idx = phase_ids.index(PHASE_STRIKE) if PHASE_STRIKE in phase_ids else None
    if strike_idx is None:
        print("\n[WARN] No strike phase found")
        return []
    counts = np.zeros((num_codes, len(phase_ids)), dtype=int)
    for c, p in zip(codes, phases):
        col = phase_ids.index(p)
        counts[c, col] += 1
    row_sums = counts.sum(axis=1, keepdims=True).clip(min=1)
    pcts = counts / row_sums * 100
    strike_codes = []
    print(f"\n{'='*70}")
    print(f"  Strike Code Identification (P(strike|code) > 30%)")
    print(f"{'='*70}")
    for k in range(num_codes):
        if pcts[k, strike_idx] > 30:
            strike_codes.append(k)
            print(f"  Code {k:>2d}: P(strike|code)={pcts[k,strike_idx]:.1f}%, "
                  f"N_strike={counts[k, strike_idx]}, N_total={row_sums[k,0]:.0f}")
    if not strike_codes:
        print("  No codes with P(strike|code) > 30%")
    return strike_codes


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)

    if args_cli.motion_path:
        if os.path.isdir(args_cli.motion_path):
            motion_files = sorted(glob.glob(os.path.join(args_cli.motion_path, "*.npz")))
        else:
            motion_files = [args_cli.motion_path]
        env_cfg.commands.motion.motion_files = motion_files
        if hasattr(env_cfg.commands.motion, "strike_motion_files"):
            env_cfg.commands.motion.strike_motion_files = motion_files

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env)
    base_env = env.unwrapped
    device = base_env.device

    # Load VQ model
    model, ckpt = _load_model(args_cli.latent_v2_model, device)
    assert ckpt.get("prior_type") == "vq", "This script requires a VQ model"
    code_hold = int(ckpt.get("code_hold", 1))
    num_codes = int(ckpt.get("num_codes", 16))
    use_tf = ("task_features" in ckpt.get("decoder_obs_mode", "full"))
    print(f"[INFO] VQ model: K={num_codes}, code_hold={code_hold}, obs_mode={ckpt.get('decoder_obs_mode')}")

    # Load teacher policy
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume)
    teacher_policy = runner.get_inference_policy(device=device)
    print(f"[INFO] Teacher loaded from {resume}")

    # Setup
    cmd = base_env.command_manager.get_term("motion")
    robot = base_env.scene["robot"]
    soccer_ball = base_env.scene["soccer_ball"]
    swing_idx = robot.body_names.index("right_ankle_roll_link")

    tracker = EpisodeTracker(
        env.num_envs, cmd.motion.num_files, cmd.motion.motion_name,
        args_cli.eval_episodes, device,
    )
    tracker.assign_motions_round_robin(cmd)

    # Hold state for posterior and prior (independent)
    post_hold_ctr = torch.zeros(env.num_envs, dtype=torch.long, device=device)
    post_held_code = torch.zeros(env.num_envs, dtype=torch.long, device=device)
    post_held_zq = torch.zeros(env.num_envs, model.z_dim, device=device)
    # For Markov prior, init prev_code to START token; otherwise 0
    prior_init_code = model.prior.start_token if model.markov_prior else 0
    prior_held_code = torch.full((env.num_envs,), prior_init_code, dtype=torch.long, device=device)
    had_contact = torch.zeros(env.num_envs, dtype=torch.bool, device=device)

    obs, _ = env.get_observations()
    step = 0
    max_steps = args_cli.eval_episodes * 500 * cmd.motion.num_files

    while simulation_app.is_running() and not tracker.is_done() and step < max_steps:
        with torch.inference_mode():
            obs_v3 = obs
            if use_tf:
                from compute_task_features import compute_ball_foot_relation
                tf = compute_ball_foot_relation(base_env)
            else:
                tf = None
            dec_obs = model.select_decoder_obs(obs_v3, task_features=tf)

            # Teacher action → encoder → posterior code
            teacher_action = teacher_policy(obs_v3)
            z_e = model.encoder(dec_obs, teacher_action)
            needs_update = (post_hold_ctr % code_hold == 0)
            if needs_update.any():
                z_q, new_code, _ = model.codebook.quantize(z_e)
                post_held_code[needs_update] = new_code[needs_update]
                post_held_zq[needs_update] = z_q[needs_update]

            # Prior code (independent, greedy) — pass prev_code for Markov prior
            prev_code = prior_held_code if model.markov_prior else None
            prior_logits = model.prior(dec_obs, prev_code=prev_code)
            prior_needs = (post_hold_ctr % code_hold == 0)  # same cadence
            if prior_needs.any():
                prior_held_code[prior_needs] = prior_logits[prior_needs].argmax(dim=-1)

            # Decode using posterior code for teacher-quality rollout
            actions = model.decoder(dec_obs, post_held_zq)

            # Geometry for phase labeling
            ball_pos = soccer_ball.data.root_pos_w[:, :3]
            swing_pos = robot.data.body_pos_w[:, swing_idx, :3]
            swing_vel = robot.data.body_lin_vel_w[:, swing_idx, :3]
            ball_dist = torch.norm((ball_pos - swing_pos)[:, :2], dim=-1)
            foot_speed = torch.norm(swing_vel[:, :2], dim=-1)
            # Closing velocity (how fast foot approaches ball)
            to_ball = ball_pos[:, :2] - swing_pos[:, :2]
            to_ball_dir = to_ball / to_ball.norm(dim=-1, keepdim=True).clamp(min=1e-4)
            closing_vel = (swing_vel[:, :2] * to_ball_dir).sum(dim=-1)

        # Contact detection & geo_phase outside inference_mode
        ball_vel = soccer_ball.data.root_lin_vel_w[:, :2]
        ball_speed = torch.norm(ball_vel, dim=-1)
        new_contact = (~had_contact) & (ball_speed > 1.0) & (ball_dist < 0.4)
        had_contact = had_contact | new_contact
        # Phases
        ref_phase = cmd.event_phase_id.long()
        geo_phase = compute_geo_phase(ball_dist, foot_speed, closing_vel, had_contact)

        # Record
        tracker.record_step(post_held_code, prior_held_code, ref_phase, geo_phase,
                            ball_dist, foot_speed)

        # Step env with posterior-decoded actions
        actions = actions.clone()
        obs, rew, dones, infos = env.step(actions)
        post_hold_ctr += 1

        # Handle resets
        if isinstance(dones, dict):
            reset_mask = dones.get("terminated", torch.zeros(env.num_envs, dtype=torch.bool, device=device)) | \
                         dones.get("truncated", torch.zeros(env.num_envs, dtype=torch.bool, device=device))
            term_mask = dones.get("terminated", torch.zeros(env.num_envs, dtype=torch.bool, device=device))
        else:
            reset_mask = dones.bool()
            term_mask = reset_mask

        for i in range(env.num_envs):
            if reset_mask[i]:
                new_m = tracker.on_done(i, term_mask[i].item())
                post_hold_ctr[i] = 0
                had_contact[i] = False
                prior_held_code[i] = prior_init_code  # Reset Markov state
                if new_m is not None:
                    cmd.motion_idx[i] = new_m

        step += 1
        if step % 500 == 0:
            done_str = ", ".join(
                f"{tracker.motion_names[m]}={tracker.episodes_done[m]}"
                for m in range(tracker.num_motions))
            print(f"[EVAL] Step {step} | {done_str}")

    # ═══════════════════════════════════════════════════════════════════
    # ANALYSIS
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n[INFO] Collected {len(tracker.completed)} episodes")

    # Flatten all timestep data
    all_post = []
    all_prior = []
    all_ref = []
    all_geo = []
    for ep in tracker.completed:
        all_post.extend(ep["post_codes"])
        all_prior.extend(ep["prior_codes"])
        all_ref.extend(ep["ref_phases"])
        all_geo.extend(ep["geo_phases"])

    # Table 1: Posterior Code × Ref Phase
    print_code_phase_table("POSTERIOR Code × Ref Phase", all_post, all_ref,
                           num_codes, PHASE_NAMES)

    # Table 2: Prior Code × Ref Phase
    print_code_phase_table("PRIOR Code × Ref Phase", all_prior, all_ref,
                           num_codes, PHASE_NAMES)

    # Table 3: Posterior Code × Geo Phase
    print_code_phase_table("POSTERIOR Code × Geo Phase", all_post, all_geo,
                           num_codes, GEO_PHASE_NAMES)

    # Table 4: Phase → Code (which codes dominate each ref phase?)
    print_phase_code_table("Ref Phase → Posterior Code (top 5)", all_post, all_ref,
                           num_codes, PHASE_NAMES)
    print_phase_code_table("Ref Phase → Prior Code (top 5)", all_prior, all_ref,
                           num_codes, PHASE_NAMES)

    # Table 5: Prior vs Posterior agreement by ref phase
    print_agreement_table(all_post, all_prior, all_ref, PHASE_NAMES)

    # Table 6: Strike code identification (from posterior)
    strike_codes = identify_strike_codes(all_post, all_ref, num_codes, PHASE_NAMES)

    # Table 7: Code sequence around strike for sample episodes
    print(f"\n{'='*70}")
    print(f"  Code Sequences Around Strike Window (first 10 eps with strike)")
    print(f"{'='*70}")
    shown = 0
    for ep in tracker.completed:
        if shown >= 10:
            break
        refs = ep["ref_phases"]
        # Find first frame where ref_phase == STRIKE
        strike_start = None
        for t, r in enumerate(refs):
            if r == PHASE_STRIKE:
                strike_start = t
                break
        if strike_start is None:
            continue
        # Show window [-10, +10] around strike start
        lo = max(0, strike_start - 10)
        hi = min(len(refs), strike_start + 10)
        post_seq = ep["post_codes"][lo:hi]
        prior_seq = ep["prior_codes"][lo:hi]
        ref_seq = [PHASE_NAMES.get(r, "?") for r in refs[lo:hi]]
        dist_seq = [f"{d:.2f}" for d in ep["ball_dist"][lo:hi]]
        agree_seq = ["✓" if p == q else "✗" for p, q in zip(post_seq, prior_seq)]
        print(f"\n  Episode {shown+1} ({ep['motion_name']}, len={ep['length']})")
        print(f"    frame:   {list(range(lo, hi))}")
        print(f"    ref_ph:  {ref_seq}")
        print(f"    post_c:  {post_seq}")
        print(f"    prior_c: {prior_seq}")
        print(f"    agree:   {agree_seq}")
        print(f"    b_dist:  {dist_seq}")
        shown += 1

    # Save JSON
    out_dir = os.path.join(os.path.dirname(args_cli.latent_v2_model), "eval")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "vq_code_semantics.json")
    summary = {
        "num_episodes": len(tracker.completed),
        "num_codes": num_codes,
        "code_hold": code_hold,
        "strike_codes": strike_codes,
        "episodes": tracker.completed[:20],  # save first 20 for inspection
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[INFO] Saved to {out_path}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
