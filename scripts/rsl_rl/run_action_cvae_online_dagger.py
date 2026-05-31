"""Run lightweight online DAgger rounds for action-CVAE distillation.

This script is an orchestrator.  It does not launch Isaac Sim itself; instead it
sequentially calls:

  1. dagger_action_cvae.py              student rollout + teacher labels
  2. train_action_cvae_distill.py       retrain on base + accumulated DAgger data
  3. eval_action_cvae_decoder_rollout.py closed-loop prior evaluation

The goal is to approximate LATENT-style online distillation without building a
single long-running Isaac/train loop.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


def _bool_flag(cmd: list[str], enabled: bool, flag: str):
    if enabled:
        cmd.append(flag)


def _run(cmd: list[str], *, dry_run: bool):
    print("\n" + "=" * 100)
    print(shlex.join(cmd))
    print("=" * 100)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def _model_for_round(output_dir: Path, round_idx: int, prefix: str) -> Path:
    return output_dir / "models" / f"{prefix}_r{round_idx}.pt"


def _dagger_for_round(output_dir: Path, round_idx: int) -> Path:
    return output_dir / "data" / f"dagger_round{round_idx}.pt"


def main():
    parser = argparse.ArgumentParser(description="Iterated online DAgger for action-CVAE distillation.")
    parser.add_argument("--initial_model", type=str, required=True)
    parser.add_argument("--base_rollout_data", type=str, nargs="+", required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_prefix", type=str, default="action_cvae_online_dagger")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warm_start", action="store_true", default=True)
    parser.add_argument("--no_warm_start", action="store_false", dest="warm_start")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")

    # Isaac / teacher collection args.
    parser.add_argument("--task", type=str, default="Anchor-V3CVAE-Kick-G1-Soccer-RNN-v0")
    parser.add_argument("--motion_path", type=str, required=True)
    parser.add_argument("--load_run", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_envs", type=int, default=128)
    parser.add_argument("--num_episodes", type=int, default=1000)
    parser.add_argument("--teacher_mix", type=float, default=0.35)
    parser.add_argument("--student_mode", choices=["prior_mean", "prior_sample"], default="prior_mean")
    parser.add_argument("--sample_scale", type=float, default=0.5)
    parser.add_argument("--clip_actions", type=float, default=0.0)
    parser.add_argument("--collect_device", "--device", dest="collect_device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--headless", action="store_true", default=True)

    # Training args.
    parser.add_argument("--obs_slices", type=str, default="0:384,392:414")
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--action_horizon", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--beta", type=float, default=3.0e-4)
    parser.add_argument("--prior_recon_weight", type=float, default=0.08)
    parser.add_argument("--val_split", type=float, default=0.05)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--train_device", type=str, default="cuda")
    parser.add_argument("--phase_balance", action="store_true", default=True)
    parser.add_argument("--no_phase_balance", action="store_false", dest="phase_balance")

    # Eval args.
    parser.add_argument("--eval_task", type=str, default=None)
    parser.add_argument("--eval_episodes", type=int, default=100)
    parser.add_argument("--eval_envs", type=int, default=32)
    parser.add_argument("--eval_device", type=str, default="cuda:0")
    parser.add_argument("--eval_mode", choices=["prior_mean", "prior_sample"], default="prior_mean")
    parser.add_argument("--skip_eval", action="store_true")

    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir)
    (output_dir / "data").mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(parents=True, exist_ok=True)
    (output_dir / "eval").mkdir(parents=True, exist_ok=True)

    current_model = Path(args.initial_model)
    dagger_data: list[Path] = []
    eval_task = args.eval_task or args.task

    for round_idx in range(1, args.rounds + 1):
        dagger_path = _dagger_for_round(output_dir, round_idx)
        model_path = _model_for_round(output_dir, round_idx, args.model_prefix)

        if args.skip_existing and dagger_path.exists():
            print(f"[INFO] Skip existing DAgger data: {dagger_path}")
        else:
            collect_cmd = [
                sys.executable,
                str(script_dir / "dagger_action_cvae.py"),
                "--task",
                args.task,
                "--motion_path",
                args.motion_path,
                "--student_model",
                str(current_model),
                "--load_run",
                args.load_run,
                "--checkpoint",
                args.checkpoint,
                "--num_envs",
                str(args.num_envs),
                "--num_episodes",
                str(args.num_episodes),
                "--teacher_mix",
                str(args.teacher_mix),
                "--mode",
                args.student_mode,
                "--sample_scale",
                str(args.sample_scale),
                "--clip_actions",
                str(args.clip_actions),
                "--seed",
                str(args.seed + round_idx - 1),
                "--output_path",
                str(dagger_path),
                "--device",
                args.collect_device,
            ]
            _bool_flag(collect_cmd, args.headless, "--headless")
            _run(collect_cmd, dry_run=args.dry_run)

        dagger_data.append(dagger_path)

        train_inputs = [*args.base_rollout_data, *[str(path) for path in dagger_data]]
        if args.skip_existing and model_path.exists():
            print(f"[INFO] Skip existing model: {model_path}")
        else:
            train_cmd = [
                sys.executable,
                str(script_dir / "train_action_cvae_distill.py"),
                "--rollout_data",
                *train_inputs,
                "--output_path",
                str(model_path),
                "--obs_slices",
                args.obs_slices,
                "--latent_dim",
                str(args.latent_dim),
                "--action_horizon",
                str(args.action_horizon),
                "--epochs",
                str(args.epochs),
                "--batch_size",
                str(args.batch_size),
                "--beta",
                str(args.beta),
                "--prior_recon_weight",
                str(args.prior_recon_weight),
                "--val_split",
                str(args.val_split),
                "--max_samples",
                str(args.max_samples),
                "--device",
                args.train_device,
                "--seed",
                str(args.seed + round_idx - 1),
            ]
            if args.warm_start:
                train_cmd.extend(["--init_model", str(current_model)])
            _bool_flag(train_cmd, args.phase_balance, "--phase_balance")
            _run(train_cmd, dry_run=args.dry_run)

        current_model = model_path

        if not args.skip_eval:
            eval_json = output_dir / "eval" / f"{args.model_prefix}_r{round_idx}_{args.eval_mode}.json"
            eval_cmd = [
                sys.executable,
                str(script_dir / "eval_action_cvae_decoder_rollout.py"),
                "--task",
                eval_task,
                "--motion_path",
                args.motion_path,
                "--model",
                str(current_model),
                "--mode",
                args.eval_mode,
                "--sample_scale",
                str(args.sample_scale),
                "--num_envs",
                str(args.eval_envs),
                "--eval_episodes",
                str(args.eval_episodes),
                "--device",
                args.eval_device,
                "--output_json",
                str(eval_json),
            ]
            _bool_flag(eval_cmd, args.headless, "--headless")
            _run(eval_cmd, dry_run=args.dry_run)

    print("\n[INFO] Online DAgger complete.")
    print(f"[INFO] Final model: {current_model}")


if __name__ == "__main__":
    # Keep child Isaac processes away from stale SSH DISPLAY by default.
    if "DISPLAY" in os.environ:
        os.environ.pop("DISPLAY", None)
    main()
