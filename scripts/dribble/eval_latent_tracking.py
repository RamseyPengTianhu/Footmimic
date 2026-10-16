"""D-b acceptance: tracking error and falls of a checkpoint, in posterior or prior mode (no learning).

Works for both the latent tasks (LatentPPO) and the clean-v0 Stage-1 task, so the two can be compared on the same
clips.  Runs ``--steps`` control steps with the deterministic policy, keeps the task's own terminations, and reports
the mean of the command's tracking metrics over alive env-steps plus termination counts.

  python scripts/dribble/eval_latent_tracking.py --task Tracking-Flat-G1-LatentDribble-Db \
      --motion_path <dir or npz> --load_run <run dir> --checkpoint model_5999.pt [--prior] --num_envs 64 --steps 600 \
      --output out.json --headless
"""

import argparse
import collections
import glob
import json
import os
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rsl_rl"))
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", required=True)
parser.add_argument("--motion_path", required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--prior", action="store_true", help="latent tasks: run on the prior (no reference window)")
parser.add_argument("--output", required=True)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import soccer.tasks  # noqa: E402,F401
from soccer.utils.my_on_policy_runner import MotionOnPolicyRunner  # noqa: E402

FALL_TERMS = ("anchor_pos_z", "anchor_ori", "bad_anchor_pos_z", "bad_anchor_ori")


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    p = args_cli.motion_path
    env_cfg.commands.motion.motion_files = [p] if os.path.isfile(p) else sorted(glob.glob(os.path.join(p, "*.npz")))
    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg))
    runner_cls = MotionOnPolicyRunner
    if getattr(agent_cfg.algorithm, "class_name", "") == "LatentPPO":
        from soccer.utils.latent_runner import LatentOnPolicyRunner as runner_cls
    runner = runner_cls(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.git_status_repos = []
    path = get_checkpoint_path(os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)), agent_cfg.load_run, agent_cfg.load_checkpoint)
    runner.load(path)
    policy = runner.alg.policy
    policy.eval()
    act = runner.get_inference_policy(device=agent_cfg.device)  # includes RSL-RL's obs normaliser when enabled
    if args_cli.prior:
        if not hasattr(policy, "use_prior"):
            raise SystemExit("--prior needs a latent task")
        policy.use_prior = True
    base = env.unwrapped
    cmd = base.command_manager.get_term("motion")
    tm = base.termination_manager
    sums, count = collections.defaultdict(float), 0
    reasons = collections.Counter()
    obs, _ = env.get_observations()
    with torch.inference_mode():
        policy.reset()
        for _ in range(args_cli.steps):
            obs, _, dones, extras = env.step(act(obs))
            policy.reset(dones)
            for k, v in cmd.metrics.items():
                if k.startswith("error_"):
                    sums[k] += float(v.mean())
            count += 1
            d = dones.bool()
            if d.any():
                for t in tm.active_terms:
                    reasons[t] += int((tm.get_term(t).bool() & d).sum())
    falls = sum(reasons[t] for t in FALL_TERMS if t in reasons)
    finished = sum(reasons[t] for t in ("time_out", "motion_clip_end") if t in reasons)
    failures = sum(reasons.values()) - finished
    out = {
        "task": args_cli.task, "checkpoint": path, "mode": "prior" if args_cli.prior else "posterior",
        "motion_path": args_cli.motion_path, "num_envs": base.num_envs, "steps": args_cli.steps,
        "metrics_mean": {k: v / max(count, 1) for k, v in sums.items()},
        "terminations": dict(reasons),
        "episodes_ended": sum(reasons.values()), "episodes_finished_clip": finished,
        "failure_fraction_of_episodes": failures / max(sum(reasons.values()), 1),
        "fall_fraction_of_episodes": falls / max(sum(reasons.values()), 1),
    }
    json.dump(out, open(args_cli.output, "w"), indent=1)
    print("[EVAL]", json.dumps({k: out[k] for k in ("mode", "terminations", "episodes_finished_clip", "failure_fraction_of_episodes", "fall_fraction_of_episodes")}))
    print("[EVAL] metrics", {k: round(v, 4) for k, v in out["metrics_mean"].items()})
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
