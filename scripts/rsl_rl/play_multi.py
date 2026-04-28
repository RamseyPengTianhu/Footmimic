"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import datetime

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=600, help="Length of the recorded video (in steps).")
parser.add_argument("--dual_view", action="store_true", default=False, help="Record split-screen video (front + back view).")
parser.add_argument("--path_tracing", action="store_true", default=False, help="Use Path Tracing renderer for higher quality.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to a single motion file. When specified, only this motion is played and exported.")
parser.add_argument("--motion_path", type=str, default=None, help="The path to the directory containing motion files for random sampling (no export).")

parser.add_argument("--export_motion_name", type=str, default=None, help="Select one motion for exporter (required when --motion_file is used).")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video or args_cli.dual_view:
    args_cli.enable_cameras = True
    # Allow headless video recording over SSH.
    if not hasattr(args_cli, 'headless'):
        args_cli.headless = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import glob
import pathlib
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import soccer.tasks  # noqa: F401
from soccer.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx

def get_motion_files(motion_path: str) -> list[str]:
    """
    Get a list of motion files.
    
    Args:
        motion_path: File path or directory path.
        
    Returns:
        List of motion file paths.
    """
    if os.path.isfile(motion_path):
        # Single-file input.
        return [motion_path]
    elif os.path.isdir(motion_path):
        # Directory input: collect all .npz files.
        motion_files = glob.glob(os.path.join(motion_path, "*.npz"))
        if not motion_files:
            raise ValueError(f"No .npz files found in directory: {motion_path}")
        motion_files.sort()
        print(f"Found {len(motion_files)} motion files in {motion_path}")
        for file in motion_files:
            print(f"  - {os.path.basename(file)}")
        return motion_files
    else:
        raise ValueError(f"Invalid path: {motion_path}. Must be a file or directory.")


def _get_cg_overlay(env, timestep: int) -> str:
    """Build a text overlay showing Contact Graph phase, timestep, and ball speed."""
    try:
        base_env = env.unwrapped if hasattr(env, 'unwrapped') else env
        cmd = base_env.command_manager.get_term("motion_command")

        t = int(cmd.time_steps[0].item())
        kf = int(cmd.kick_frame[0].item()) if hasattr(cmd, 'kick_frame') else -1
        kef = int(cmd.kick_end_frame[0].item()) if hasattr(cmd, 'kick_end_frame') else -1

        # Determine CG phase (matches _get_cg_phase in rewards.py).
        margin = 5
        if kf < 0:
            phase = "CG=N/A (no annotation)"
        elif t < kf - margin:
            phase = "CG=0 (Approach)"
        else:
            phase = "CG=1 (Kick)"

        # Ball speed.
        ball_speed = 0.0
        try:
            ball = base_env.scene["ball"]
            ball_vel = ball.data.root_lin_vel_w[0].cpu().numpy()
            ball_speed = float(np.linalg.norm(ball_vel[:2]))
        except Exception:
            pass

        lines = [
            f"Step: {timestep}  |  Frame: {t}/{kf}",
            f"Phase: {phase}",
            f"Ball Speed: {ball_speed:.2f} m/s",
        ]
        return "\n".join(lines)
    except Exception as e:
        return f"Step: {timestep}\nCG: error ({e})"

@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    env_cfg.viewer.origin_type = None
    env_cfg.viewer.asset_name = None

    # For video recording: set a wide-angle camera that follows the robot.
    if args_cli.video:
        env_cfg.viewer.eye = (5.0, 5.0, 3.0)       # 5m back + 5m side + 3m up
        env_cfg.viewer.lookat = (0.0, 0.0, 0.5)     # look at robot's waist height
        env_cfg.viewer.origin_type = "asset_root"    # camera follows the robot
        env_cfg.viewer.asset_name = "robot"           # track the robot asset
        env_cfg.viewer.env_index = 0

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    motion_files: list[str] = []

    if args_cli.wandb_path:
        import wandb

        run_path = args_cli.wandb_path

        api = wandb.Api()
        if "model" in args_cli.wandb_path:
            run_path = "/".join(args_cli.wandb_path.split("/")[:-1])
        wandb_run = api.run(run_path)
        # loop over files in the run
        files = [file.name for file in wandb_run.files() if "model" in file.name]
        # files are all model_xxx.pt find the largest filename
        if "model" in args_cli.wandb_path:
            file = args_cli.wandb_path.split("/")[-1]
        else:
            file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))

        wandb_file = wandb_run.file(str(file))
        wandb_file.download("./logs/rsl_rl/temp", replace=True)

        print(f"[INFO]: Loading model checkpoint from: {run_path}/{file}")
        resume_path = f"./logs/rsl_rl/temp/{file}"

        if args_cli.motion_file is not None:
            print(f"[INFO]: Using motion file from CLI: {args_cli.motion_file}")
            env_cfg.commands.motion.motion_file = args_cli.motion_file
            motion_files = [args_cli.motion_file]

        art = next((a for a in wandb_run.used_artifacts() if a.type == "motions"), None)
        if art is None:
            print("[WARN] No model artifact found in the run.")
        else:
            env_cfg.commands.motion.motion_file = str(pathlib.Path(art.download()) / "motion.npz")

    else:
        # Select single-motion or multi-motion mode from CLI args.
        if args_cli.motion_file is not None:
            # Single-motion mode: play and export.
            motion_files = [args_cli.motion_file]
            print(f"[INFO]: Using single motion file: {args_cli.motion_file}")
        elif args_cli.motion_path is not None:
            # Multi-motion mode: random sampling for playback (no export by default).
            motion_files = get_motion_files(args_cli.motion_path)
        else:
            raise ValueError("Either --motion_file or --motion_path must be specified.")
        
        # For state-machine environments: auto-split approach/strike files.
        approach_files = [f for f in motion_files if f.endswith("_approach.npz")]
        strike_files = [f for f in motion_files if f.endswith("_strike.npz")]

        if approach_files and strike_files:
            env_cfg.commands.motion.motion_files = approach_files
            if hasattr(env_cfg.commands.motion, "strike_motion_files"):
                env_cfg.commands.motion.strike_motion_files = strike_files
                print(f"[INFO] State-machine mode: {len(approach_files)} approach + {len(strike_files)} strike files")
            else:
                env_cfg.commands.motion.motion_files = motion_files
        else:
            env_cfg.commands.motion.motion_files = motion_files
            if hasattr(env_cfg.commands.motion, "strike_motion_files"):
                env_cfg.commands.motion.strike_motion_files = motion_files
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    log_dir = os.path.dirname(resume_path)

    # wrap for video recording
    if args_cli.video:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        video_dir = os.path.join(log_dir, "videos", f"play_{timestamp}")
        video_kwargs = {
            "video_folder": video_dir,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print(f"[INFO] Recording video ({args_cli.video_length} steps) to: {video_dir}")
        print("[INFO] Use --video_length N to control clip duration.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_targets: list[tuple[str, str]] = []

    if args_cli.motion_file is not None:
        # Single-file mode: export directly using the requested name or file name.
        export_name = args_cli.export_motion_name or os.path.basename(args_cli.motion_file)
        export_targets.append((args_cli.motion_file, export_name))
    elif args_cli.motion_path is not None and args_cli.export_motion_name is not None:
        # Directory mode: export by matching names from export_motion_name.
        if args_cli.export_motion_name.strip().lower() == "all":
            export_targets = [(mf, os.path.basename(mf)) for mf in motion_files]
        else:
            requested_names = [n.strip() for n in args_cli.export_motion_name.split(",") if n.strip()]
            for name in requested_names:
                match = next(
                    (
                        mf
                        for mf in motion_files
                        if os.path.splitext(os.path.basename(mf))[0] == os.path.splitext(name)[0]
                        or os.path.basename(mf) == name
                    ),
                    None,
                )
                if match is None:
                    raise ValueError(f"Requested export motion '{name}' not found in {args_cli.motion_path}.")
                export_targets.append((match, name))

    if export_targets:
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        ckpt = args_cli.checkpoint.split('_')[1].split('.')[0]

        for motion_file, export_name in export_targets:
            export_stem = os.path.splitext(export_name)[0]
            filename = f"policy_{ckpt}_{export_stem}.onnx"
            export_motion_policy_as_onnx(
                env.unwrapped,
                ppo_runner.alg.policy,
                normalizer=ppo_runner.obs_normalizer,
                path=export_model_dir,
                filename=filename,
                motion_name=export_name,
            )
            attach_onnx_metadata(
                env.unwrapped,
                args_cli.wandb_path if args_cli.wandb_path else "none",
                export_model_dir,
                filename=filename,
            )
            print(f"[INFO]: Exported policy for {export_name} to: {os.path.join(export_model_dir, filename)}")
    else:
        print("[INFO]: Skipping policy export (set --export_motion_name to enable export).")
    
    # --- Dual-view recorder setup (optional) ---
    dual_recorder = None
    if args_cli.dual_view:
        from dual_view_recorder import DualViewRecorder

        video_dir = os.path.join(log_dir, "videos",
                                 f"dual_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        dual_recorder = DualViewRecorder(
            env=env.unwrapped if hasattr(env, 'unwrapped') else env,
            output_dir=video_dir,
            resolution=(960, 540),
            front_offset=(4.0, 3.0, 2.5),
            back_offset=(-4.0, -3.0, 2.5),
            lookat_offset=0.5,
            fps=30,
            path_tracing=args_cli.path_tracing,
        )
        dual_recorder.setup()
        # Need some warmup frames for the renderer.
        for _ in range(5):
            env.unwrapped.sim.render()
        print(f"[INFO] Dual-view recording: {args_cli.video_length} steps → {video_dir}")

    # reset environment
    # breakpoint()
    obs, _ = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)

        # Capture frame for dual-view recording with CG overlay.
        if dual_recorder is not None:
            overlay = _get_cg_overlay(env, timestep)
            dual_recorder.capture(overlay_text=overlay)

        if args_cli.video or args_cli.dual_view:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # Save dual-view video.
    if dual_recorder is not None:
        dual_recorder.save()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
