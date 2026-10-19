"""Stage 2 (minimal) acceptance on Tracking-Flat-G1-BallDribble: one reference clip, real ball, no learning.

Per completed episode (all start at frame 0):
  touch completion  a labelled touch counts when the labelled foot touches the ball inside its contact segment
                    widened by --tol frames; touches after an early termination count as missed
  front zone        fraction of alive steps with the ball inside x in [0.15, 0.70], |y| <= 0.40 (pelvis-yaw frame)
  terminations      falls, ball lost, non-foot contact, clip finished
  rewards           mean weighted reward per step for each term (hack check: compare with --zero_action)

  python scripts/dribble/eval_ball_dribble.py --motion_path <ref npz> --load_run <run> --checkpoint model_10998.pt \
      --num_envs 64 --steps 800 --output out.json --headless [--zero_action]
"""

import argparse
import collections
import json
import os
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rsl_rl"))
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Tracking-Flat-G1-BallDribble")
parser.add_argument("--motion_path", required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=800)
parser.add_argument("--tol", type=int, default=2, help="frames added on each side of a labelled contact segment")
parser.add_argument("--zero_action", action="store_true", help="hack check: hold the default pose, no policy")
parser.add_argument("--output", required=True)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import quat_apply, quat_inv, yaw_quat  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import soccer.tasks  # noqa: E402,F401
from soccer.tasks.tracking.mdp import ball_contact  # noqa: E402
from soccer.utils.my_on_policy_runner import MotionOnPolicyRunner  # noqa: E402

FALL_TERMS = ("anchor_pos_z", "anchor_ori", "ee_body_pos")
FINISH_TERMS = ("time_out", "motion_clip_end")


def contact_segments(contact: torch.Tensor):
    on = contact > 0
    segs, start = [], None
    for f, v in enumerate(on.tolist()):
        if v and start is None:
            start = f
        if not v and start is not None:
            segs.append((start, f - 1))
            start = None
    if start is not None:
        segs.append((start, len(on) - 1))
    return segs


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.commands.motion.motion_files = [args_cli.motion_path]
    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg))
    base = env.unwrapped
    dev = base.device
    path = "zero_action"
    if args_cli.zero_action:
        act = lambda obs: torch.zeros(base.num_envs, base.action_manager.total_action_dim, device=dev)  # noqa: E731
        policy = None
    else:
        runner = MotionOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.git_status_repos = []
        path = get_checkpoint_path(os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)),
                                   agent_cfg.load_run, agent_cfg.load_checkpoint)
        runner.load(path)
        policy = runner.alg.policy
        policy.eval()
        act = runner.get_inference_policy(device=agent_cfg.device)

    cmd = base.command_manager.get_term("motion")
    tm, rm = base.termination_manager, base.reward_manager
    contact = cmd.motion.dribble_cg_contact[0]
    foot_lab = cmd.motion.dribble_cg_foot[0]
    segs = contact_segments(contact)
    n_ev = len(segs)
    ev_lo = torch.tensor([max(0, s - args_cli.tol) for s, _ in segs], device=dev)
    ev_hi = torch.tensor([e + args_cli.tol for _, e in segs], device=dev)
    ev_foot = torch.tensor([int(foot_lab[s]) for s, _ in segs], device=dev)  # 0 left, 1 right
    print(f"[BALL] {n_ev} labelled touches, feet {ev_foot.tolist()}")

    N = base.num_envs
    hit = torch.zeros(N, n_ev, dtype=torch.bool, device=dev)
    wrong_foot = torch.zeros(N, dtype=torch.long, device=dev)
    zone_steps = torch.zeros(N, device=dev)
    alive_steps = torch.zeros(N, device=dev)
    ep = collections.Counter()
    reasons = collections.Counter()
    rew_sum = torch.zeros(len(rm.active_terms), device=dev)
    rew_n = 0
    per_event_hits = torch.zeros(n_ev, device=dev)
    ball_start = torch.zeros(N, 2, device=dev)
    link_steps = torch.zeros(len(ball_contact.BALL_FILTER_BODY_NAMES), device=dev)
    ball_last = torch.zeros(N, 2, device=dev)
    obs, _ = env.get_observations()
    with torch.inference_mode():
        if policy is not None:
            policy.reset()
        for _ in range(args_cli.steps):
            frame = cmd.time_steps.clone()
            # ball in pelvis-yaw frame, measured before the step (state the policy acted on)
            rel = base.scene["soccer_ball"].data.root_pos_w - cmd.robot_pelvis_pos_w
            loc = quat_apply(quat_inv(yaw_quat(cmd.robot_anchor_quat_w)), rel)
            in_zone = (loc[:, 0] >= 0.15) & (loc[:, 0] <= 0.70) & (loc[:, 1].abs() <= 0.40)
            zone_steps += in_zone.float()
            ball_xy = base.scene["soccer_ball"].data.root_pos_w[:, :2]
            fresh = alive_steps == 0
            ball_start[fresh] = ball_xy[fresh]
            ball_last[:] = ball_xy
            alive_steps += 1.0
            obs, _, dones, _ = env.step(act(obs))
            if policy is not None:
                policy.reset(dones)
            rew_sum += rm._step_reward.sum(dim=0)
            rew_n += N
            touch = ball_contact.ball_foot_contact(base)  # [N, 2] left/right, after the step
            link_steps += (ball_contact.ball_force_by_body(base) > 1.0).float().sum(dim=0)
            f = (frame + 1).unsqueeze(1)
            in_win = (f >= ev_lo.unsqueeze(0)) & (f <= ev_hi.unsqueeze(0))  # [N, E]
            right_foot_touch = touch[:, 1].unsqueeze(1)
            left_foot_touch = touch[:, 0].unsqueeze(1)
            correct = torch.where(ev_foot.unsqueeze(0) == 1, right_foot_touch, left_foot_touch)
            wrong = torch.where(ev_foot.unsqueeze(0) == 1, left_foot_touch, right_foot_touch)
            hit |= in_win & correct
            wrong_foot += (in_win & wrong).any(dim=1).long()
            d = dones.bool()
            if d.any():
                ids = d.nonzero().flatten()
                for t in tm.active_terms:
                    reasons[t] += int((tm.get_term(t).bool() & d).sum())
                ep["episodes"] += ids.numel()
                ep["touch_hits"] += int(hit[ids].sum())
                ep["zone_steps"] += float(zone_steps[ids].sum())
                ep["alive_steps"] += float(alive_steps[ids].sum())
                ep["wrong_foot_steps"] += int(wrong_foot[ids].sum())
                ep["ball_travel"] += float((ball_last[ids] - ball_start[ids]).norm(dim=-1).sum())
                per_event_hits += hit[ids].float().sum(dim=0)
                hit[ids] = False
                zone_steps[ids] = 0.0
                alive_steps[ids] = 0.0
                wrong_foot[ids] = 0

    n_ep = max(ep["episodes"], 1)
    falls = sum(reasons[t] for t in FALL_TERMS)
    finished = sum(reasons[t] for t in FINISH_TERMS)
    out = {
        "checkpoint": path, "motion_path": args_cli.motion_path, "num_envs": N, "steps": args_cli.steps,
        "episodes": ep["episodes"], "touches_per_episode": n_ev,
        "touch_completion": ep["touch_hits"] / (n_ep * n_ev),
        "touch_completion_per_event": (per_event_hits / n_ep).tolist(),
        "front_zone_fraction": ep["zone_steps"] / max(ep["alive_steps"], 1.0),
        "fall_fraction": falls / n_ep,
        "finished_clip_fraction": finished / n_ep,
        "terminations": dict(reasons),
        "wrong_foot_steps_per_episode": ep["wrong_foot_steps"] / n_ep,
        "mean_episode_steps": ep["alive_steps"] / n_ep,
        "ball_travel_m": ep["ball_travel"] / n_ep,
        "ball_contact_steps_by_link": {n: int(c) for n, c in zip(ball_contact.BALL_FILTER_BODY_NAMES, link_steps.tolist()) if c > 0},
        "reward_per_step": {name: float(rew_sum[i] / max(rew_n, 1)) for i, name in enumerate(rm.active_terms)},
    }
    out["reward_per_step_total"] = sum(out["reward_per_step"].values())
    json.dump(out, open(args_cli.output, "w"), indent=1)
    print("[BALL]", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()
                                if k not in ("reward_per_step", "touch_completion_per_event")}))
    print("[BALL] per-event", [round(x, 2) for x in out["touch_completion_per_event"]])
    print("[BALL] reward/step", {k: round(v, 3) for k, v in out["reward_per_step"].items()})
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
