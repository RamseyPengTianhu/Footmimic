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
parser.add_argument(
    "--video_length",
    type=int,
    default=None,
    help="Simulation steps to run (default: sum of motion frames for --motion_path, else motion length / 600).",
)
parser.add_argument("--dual_view", action="store_true", default=False, help="Record split-screen video (front + back view).")
parser.add_argument(
    "--cam_layout",
    type=str,
    default="task_front",
    choices=["diagonal", "task_front", "task_front_side"],
    help="Camera preset. task_front: on +X side, looks ~ -X (robot frontal); diagonal: legacy oblique.",
)
parser.add_argument(
    "--record_all_motions",
    action="store_true",
    default=False,
    help="Alias for default multi-file --motion_path behaviour (sequential playback, auto step count).",
)
parser.add_argument("--path_tracing", action="store_true", default=False, help="Use Path Tracing renderer for higher quality.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used by the playback environment.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to a single motion file. When specified, only this motion is played and exported.")
parser.add_argument(
    "--motion_path",
    type=str,
    default=None,
    help="Path to a motion .npz or a directory; directories are played sequentially in sorted order.",
)

parser.add_argument("--export_motion_name", type=str, default=None, help="Select one motion for exporter (required when --motion_file is used).")
parser.add_argument(
    "--disable_training_terminations",
    action="store_true",
    default=False,
    help="Disable ball_lost / ee_body_pos etc. for full-clip playback (debug only).",
)
parser.add_argument(
    "--replay_ball_spawn_mode",
    type=str,
    default=None,
    choices=["reference_first_contact", "legacy"],
    help="Replay-only ball reset mode; does not change training configs.",
)
parser.add_argument(
    "--replay_ball_fallback_mode",
    type=str,
    default=None,
    choices=["front", "arc_endpoint"],
    help="Replay-only fallback ball placement when using legacy spawn mode.",
)
parser.add_argument(
    "--replay_ball_front_distance",
    type=float,
    default=None,
    help="Replay-only pelvis-local forward ball distance in meters.",
)
parser.add_argument(
    "--replay_ball_observation_mode",
    type=str,
    default=None,
    choices=["actual", "fixed_front"],
    help="Replay-only actor ball observation: actual or fixed far-front proxy.",
)
parser.add_argument(
    "--replay_ball_observation_distance",
    type=float,
    default=None,
    help="Distance used by the fixed_front replay actor observation in meters.",
)
parser.add_argument(
    "--locomotion_cmd_vx",
    type=float,
    default=None,
    help="Manual locomotion cmd: forward linear speed (m/s); pelvis-local for local-twist tasks.",
)
parser.add_argument(
    "--locomotion_cmd_vy",
    type=float,
    default=None,
    help="Manual locomotion cmd: lateral speed (m/s); pelvis-local for local-twist tasks.",
)
parser.add_argument(
    "--locomotion_cmd_vz",
    type=float,
    default=None,
    help="Manual locomotion cmd: vertical linear speed (m/s).",
)
parser.add_argument(
    "--locomotion_cmd_wx",
    type=float,
    default=None,
    help="Manual locomotion cmd: roll rate (rad/s).",
)
parser.add_argument(
    "--locomotion_cmd_wy",
    type=float,
    default=None,
    help="Manual locomotion cmd: pitch rate (rad/s).",
)
parser.add_argument(
    "--locomotion_cmd_speed",
    type=float,
    nargs="+",
    default=None,
    help="Polar cmd speed(s) in m/s. Multiple values = multi-segment sequence (with heading + duration).",
)
parser.add_argument(
    "--locomotion_cmd_heading",
    type=float,
    nargs="+",
    default=None,
    help="Polar cmd heading(s) rad from task +X. Must match length of --locomotion_cmd_speed.",
)
parser.add_argument(
    "--locomotion_cmd_duration",
    type=float,
    nargs="+",
    default=None,
    help="Hold duration(s) in seconds per segment. Must match length of --locomotion_cmd_speed.",
)
parser.add_argument(
    "--locomotion_task_state",
    type=str,
    nargs="+",
    choices=["idle", "dribble", "stop"],
    default=None,
    help=(
        "High-level state(s) for polar segments: idle, dribble, or stop. "
        "Must match --locomotion_cmd_speed; omitted states infer dribble for positive speed and stop for zero."
    ),
)
parser.add_argument(
    "--locomotion_cmd_wz",
    type=float,
    nargs="*",
    default=None,
    help="Optional yaw rate (rad/s) per polar segment; one value broadcasts to all segments.",
)
parser.add_argument(
    "--locomotion_cmd_plan",
    type=str,
    default=None,
    help=(
        "JSON local-twist playback plan. Each segment supplies pelvis-local "
        "vx, vy, wz, and duration_s; incompatible with --locomotion_cmd_speed/heading/state."
    ),
)
parser.add_argument(
    "--locomotion_cmd_loop",
    dest="locomotion_cmd_loop",
    action="store_true",
    default=True,
    help="Loop a multi-segment polar command from the final segment back to the first (default).",
)
parser.add_argument(
    "--locomotion_cmd_hold_last",
    dest="locomotion_cmd_loop",
    action="store_false",
    help="Keep the final multi-segment polar command active instead of looping.",
)
parser.add_argument(
    "--locomotion_cmd_reset_on_end",
    action="store_true",
    default=False,
    help=(
        "Reset robot, ball, and manual command sequence after the final polar segment duration. "
        "Takes precedence over --locomotion_cmd_loop/--locomotion_cmd_hold_last."
    ),
)
parser.add_argument(
    "--diagnostic",
    action="store_true",
    default=False,
    help=(
        "Save action-layer, command, pelvis/torso, contact-region, phase, and reward telemetry to a .npz file."
    ),
)
parser.add_argument(
    "--diagnostic_stride",
    type=int,
    default=1,
    help="Record every N simulator steps with --diagnostic (default: 1).",
)
parser.add_argument(
    "--disable_obs_noise",
    action="store_true",
    default=False,
    help=(
        "Disable policy-observation corruption during playback. This leaves the "
        "checkpoint, physics, rewards, and training configuration unchanged and "
        "is intended for strictly deterministic checkpoint diagnostics."
    ),
)
parser.add_argument(
    "--exact_reset_state",
    action="store_true",
    default=False,
    help=(
        "Playback-only diagnostic: disable command pose, velocity, and joint-position reset perturbations. "
        "The task's reset frame convention, checkpoint, rewards, and terminations remain unchanged."
    ),
)
parser.add_argument(
    "--stochastic_policy",
    action="store_true",
    default=False,
    help=(
        "Playback-only diagnostic: sample the checkpoint's Gaussian policy as PPO collection does, "
        "instead of using the deterministic mean action. Never use this flag for deployment acceptance."
    ),
)
parser.add_argument(
    "--counterfactual_ball_event",
    type=int,
    default=None,
    help=(
        "Replay-only intervention: once per episode, place the physical ball at "
        "the live-aligned reference anchor for this zero-based S2 contact event. "
        "The policy/checkpoint, robot state, rewards, and curriculum are unchanged."
    ),
)
parser.add_argument(
    "--counterfactual_ball_lead_frames",
    type=int,
    default=12,
    help=(
        "Inject --counterfactual_ball_event this many motion frames before its "
        "labelled center (default: 12, or 0.24 s for a 50 Hz reference)."
    ),
)
parser.add_argument(
    "--show_s2_contact_regions",
    action="store_true",
    default=False,
    help=(
        "Show the active S2 reference-foot frame, coarse front gate, and "
        "expected inside/outside instep half-space."
    ),
)
parser.add_argument(
    "--s2_curriculum_level",
    type=int,
    choices=range(5),
    default=None,
    help=(
        "Override the S2 playback curriculum level (0=2, 1=4, 2=8, "
        "3=full exact, 4=full disturbed). By default a new-format "
        "checkpoint restores its saved level; legacy checkpoints start at level 0."
    ),
)
parser.add_argument(
    "--upper_body_reference_margin",
    type=float,
    default=None,
    help=(
        "Play-only counterfactual: limit each shoulder/elbow/wrist position target to this many radians "
        "from the current reference. Omit to use the checkpoint actions unchanged."
    ),
)
parser.add_argument(
    "--upper_body_constraint_group",
    type=str,
    choices=["wrists", "wrists_elbows", "upper_body"],
    default="upper_body",
    help=(
        "Joints affected by --upper_body_reference_margin: wrists; wrists_elbows; or all "
        "shoulders, elbows, and wrists (default)."
    ),
)
parser.add_argument(
    "--waist_reference_margin",
    type=float,
    default=None,
    help=(
        "Play-only counterfactual: limit waist yaw/roll/pitch position targets to this many radians "
        "from the current reference. Omit to use the checkpoint waist actions unchanged."
    ),
)
parser.add_argument(
    "--waist_roll_stiffness_scale",
    type=float,
    default=None,
    help=(
        "Play-only counterfactual: multiply waist-roll PD stiffness by this positive factor and "
        "its damping by sqrt(factor). The waist-pitch actuator remains unchanged."
    ),
)

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
import json
import pathlib
import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_inv, yaw_quat

from soccer.tasks.tracking.mdp.rewards_dribbling import (
    dribbling_contact_telemetry,
    soccer_ball_contact_force_magnitude,
    soccer_ball_contact_net_force_w,
)
from soccer.tasks.tracking.mdp.observations_anchor import s2_sparse_ball_reference_context

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.sensors import ContactSensor
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import soccer.tasks  # noqa: F401
from soccer.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx
from soccer.utils.checkpoint_loading import load_checkpoint_with_obs_expand, set_s2_curriculum_level

def motion_frame_count(motion_file: str) -> int:
    """Return the number of frames in a motion .npz file."""
    data = np.load(motion_file)
    if "joint_pos" in data:
        return int(data["joint_pos"].shape[0])
    for key in data.files:
        arr = data[key]
        if hasattr(arr, "shape") and len(arr.shape) >= 1:
            return int(arr.shape[0])
    raise ValueError(f"Could not infer frame count from {motion_file}")


def _env_step_s(env_cfg) -> float:
    """Wall-clock seconds per env step."""
    return float(env_cfg.decimation) * float(env_cfg.sim.dt)


def _create_counterfactual_ball_event_state(env, event_id: int, lead_frames: int) -> dict:
    """Create a frozen-policy, replay-only sparse-contact intervention state."""
    if event_id < 0:
        raise ValueError("--counterfactual_ball_event must be non-negative")
    if lead_frames < 0:
        raise ValueError("--counterfactual_ball_lead_frames must be non-negative")
    base_env = _resolve_base_env(env)
    command = base_env.command_manager.get_term("motion")
    required = (
        "s2_contact_reference_anchor_target_w",
        "_s2_event_frames_padded",
        "motion_idx",
        "time_steps",
    )
    missing = [name for name in required if not hasattr(command, name)]
    if missing:
        raise ValueError(
            "Counterfactual ball-event replay requires an S2-labelled motion "
            f"command; missing: {', '.join(missing)}"
        )
    max_events = int(command._s2_event_frames_padded.shape[1])
    if event_id >= max_events:
        raise ValueError(
            f"--counterfactual_ball_event={event_id} exceeds the available "
            f"zero-based event range [0, {max_events - 1}]"
        )
    return {
        "event_id": int(event_id),
        "lead_frames": int(lead_frames),
        "injected": torch.zeros(base_env.num_envs, dtype=torch.bool, device=base_env.device),
        "count": 0,
    }


def _apply_counterfactual_ball_event(env, state: dict | None) -> bool:
    """Inject the ball at the selected reference anchor exactly once per episode."""
    if state is None:
        return False
    base_env = _resolve_base_env(env)
    command = base_env.command_manager.get_term("motion")
    event_id = int(state["event_id"])
    event_ids = torch.full(
        (base_env.num_envs,), event_id, dtype=torch.long, device=base_env.device
    )
    motion_idx = command.motion_idx.to(torch.long)
    event_frame = command._s2_event_frames_padded[motion_idx, event_id]
    inject_frame = event_frame - int(state["lead_frames"])
    due = (
        ~state["injected"]
        & (event_frame >= 0)
        & (inject_frame >= 0)
        & (command.time_steps.to(torch.long) >= inject_frame)
    )
    if not bool(torch.any(due)):
        return False

    anchor_w, anchor_valid = command.s2_contact_reference_anchor_target_w(event_ids)
    due &= anchor_valid.to(torch.bool)
    if not bool(torch.any(due)):
        return False
    env_ids = torch.nonzero(due, as_tuple=False).squeeze(-1)
    ball = base_env.scene["soccer_ball"]
    ball_state = ball.data.root_state_w[env_ids].clone()
    ball_state[:, :3] = anchor_w[env_ids]
    ball_state[:, 7:13] = 0.0
    ball.write_root_state_to_sim(ball_state, env_ids=env_ids)
    state["injected"][env_ids] = True
    state["count"] += int(env_ids.numel())
    if bool(torch.any(env_ids == 0)):
        frame0 = int(command.time_steps[0].item())
        target0 = anchor_w[0].detach().cpu().tolist()
        print(
            "[INFO] Counterfactual ball injection: "
            f"event={event_id} frame={frame0} target_w="
            f"({target0[0]:+.3f},{target0[1]:+.3f},{target0[2]:+.3f}) m"
        )
    return True


def _disable_play_terminations(env_cfg) -> list[str]:
    """Disable failure termination terms so playback runs full clips (v1.20 behaviour)."""
    if not hasattr(env_cfg, "terminations"):
        return []

    terms = env_cfg.terminations
    disabled: list[str] = []

    for name in getattr(type(terms), "__annotations__", {}):
        if name.startswith("_"):
            continue
        setattr(terms, name, None)
        disabled.append(name)

    for name in terms.__dict__:
        if name.startswith("_") or name in disabled:
            continue
        setattr(terms, name, None)
        disabled.append(name)

    return sorted(set(disabled))


def _setup_play_episode_limit(env_cfg, motion_files: list[str], *, keep_failure_terms: bool) -> int:
    """Size episodes for playback and drop the training 10 s / 500-step cap.

    Returns the recommended total step count (single clip = its frame count;
    multi-clip sequential run = sum of frame counts).
    """
    if not motion_files:
        return 600

    step_s = _env_step_s(env_cfg)
    frame_counts = [motion_frame_count(f) for f in motion_files]
    max_frames = max(frame_counts)

    # Training uses episode_length_s=10 (500 steps). Playback needs at least one full clip.
    env_cfg.episode_length_s = max_frames * step_s + 2.0
    if keep_failure_terms:
        if hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out = None
        print("[INFO] Playback: training failure terminations active (ball_lost, ee_body_pos, …).")
    else:
        disabled_terms = _disable_play_terminations(env_cfg)
        if disabled_terms:
            print(f"[INFO] Playback: disabled terminations: {', '.join(disabled_terms)}")

    if len(motion_files) > 1 and hasattr(env_cfg.commands.motion, "sampling_strategy"):
        env_cfg.commands.motion.sampling_strategy = "sequential"

    if len(motion_files) > 1:
        return sum(frame_counts)
    return max_frames


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


_BALL_SENSOR_NAME = "soccer_ball_contact"
_CONTACT_FORCE_THRESHOLD = 1.0
_LAST_TERM_REASON: str = "-"
_DIAGNOSTIC_SCHEMA_VERSION = "dribble-v10-physx-foot-contact-points"

_ARM_DIAGNOSTIC_JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# The diagnostic archive keeps arm and trunk arrays separate so analysis scripts
# can distinguish their control paths without relying on array slices.
_TRUNK_DIAGNOSTIC_JOINT_NAMES = [
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
]
_WAIST_REFERENCE_CONSTRAINT_GROUP = "waist"
_TRUNK_DIAGNOSTIC_BODY_NAMES = ["pelvis", "torso_link"]
# These match the G1 actuator ``effort_limit_sim`` values in ``soccer/robots/g1.py``.
# They make saturation visible in an offline diagnostic without depending on a
# simulator-internal actuator API.
_TRUNK_EFFORT_LIMITS = [88.0, 50.0, 50.0]

_UPPER_BODY_CONSTRAINT_JOINT_GROUPS = {
    "wrists": [
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ],
    "wrists_elbows": [
        "left_elbow_joint",
        "right_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ],
    "upper_body": _ARM_DIAGNOSTIC_JOINT_NAMES,
}

_FOOT_DIAGNOSTIC_BODY_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]
_PHYSX_CONTACT_POINTS_PER_FOOT = 8


class _DiagnosticDetailedContactSensor(ContactSensor):
    """Playback-only contact sensor whose detailed PhysX view is created before stepping."""

    # Populated by ``main`` before scene construction.  This list belongs only
    # to the independent all-link diagnostic view; the ordinary ContactSensor
    # keeps the task's two logical-foot filters so playback cannot change the
    # S2 contact MDP it is trying to measure.
    diagnostic_filter_prim_paths_expr: tuple[str, ...] = ()

    def _initialize_impl(self):
        super()._initialize_impl()
        leaf_pattern = self.cfg.prim_path.rsplit("/", 1)[-1]
        template_prim_path = self._parent_prims[0].GetPath().pathString
        body_names = [
            prim.GetPath().pathString.rsplit("/", 1)[-1]
            for prim in sim_utils.find_matching_prims(
                template_prim_path + "/" + leaf_pattern
            )
        ]
        body_names_regex = r"(" + "|".join(body_names) + r")"
        body_names_glob = (
            f"{self.cfg.prim_path.rsplit('/', 1)[0]}/{body_names_regex}"
        ).replace(".*", "*")
        detailed_filter_expr = (
            self.diagnostic_filter_prim_paths_expr
            if self.diagnostic_filter_prim_paths_expr
            else tuple(self.cfg.filter_prim_paths_expr)
        )
        filter_paths_glob = [
            expression.replace(".*", "*")
            for expression in detailed_filter_expr
        ]
        max_contact_data_count = max(
            2 * _PHYSX_CONTACT_POINTS_PER_FOOT,
            self._num_envs
            * max(1, len(filter_paths_glob))
            * _PHYSX_CONTACT_POINTS_PER_FOOT,
        )
        self.detailed_contact_physx_view = (
            self._physics_sim_view.create_rigid_contact_view(
                body_names_glob,
                filter_patterns=filter_paths_glob,
                max_contact_data_count=max_contact_data_count,
            )
        )
        self.detailed_contact_filter_patterns = tuple(filter_paths_glob)


def _create_detailed_foot_contact_view(base_env) -> dict:
    """Create a diagnostic-only PhysX view for ball contacts with each foot.

    Isaac Lab's high-level :class:`ContactSensor` deliberately exposes only
    aggregated normal forces.  The underlying PhysX tensor view can also
    expose the individual contact points, normals, separation, and friction,
    but only when it is created with a non-zero contact-data capacity.  Keep
    that detailed view local to playback diagnostics so training observations,
    rewards, physics, and memory use are unchanged.
    """
    ball_sensor = base_env.scene.sensors[_BALL_SENSOR_NAME]
    physics_view = getattr(ball_sensor, "_physics_sim_view", None)
    if physics_view is None:
        raise RuntimeError("The soccer-ball ContactSensor has no initialized PhysX simulation view.")

    sensor_pattern = "/World/envs/env_*/SoccerBall"
    robot = base_env.scene["robot"]
    filter_patterns = [
        f"/World/envs/env_*/Robot/{body_name}" for body_name in robot.body_names
    ]
    # A convex contact manifold normally has at most four points.  Eight per
    # ball-foot pair leaves headroom while keeping the diagnostic buffer small.
    max_contact_data_count = max(
        2 * _PHYSX_CONTACT_POINTS_PER_FOOT,
        int(base_env.num_envs) * len(filter_patterns) * _PHYSX_CONTACT_POINTS_PER_FOOT,
    )
    view = getattr(ball_sensor, "detailed_contact_physx_view", None)
    if view is None:
        view = physics_view.create_rigid_contact_view(
            sensor_pattern,
            filter_patterns=filter_patterns,
            max_contact_data_count=max_contact_data_count,
        )
    else:
        filter_patterns = list(
            getattr(
                ball_sensor,
                "detailed_contact_filter_patterns",
                ball_sensor.cfg.filter_prim_paths_expr,
            )
        )
        max_contact_data_count = max(
            2 * _PHYSX_CONTACT_POINTS_PER_FOOT,
            int(base_env.num_envs)
            * max(1, len(filter_patterns))
            * _PHYSX_CONTACT_POINTS_PER_FOOT,
        )
    if int(view.filter_count) != len(filter_patterns):
        raise RuntimeError(
            "Detailed ball-foot contact view resolved an unexpected number of filters: "
            f"expected {len(filter_patterns)}, got {view.filter_count}."
        )

    sensor_paths = list(view.sensor_paths)
    sensor_index = next(
        (index for index, path in enumerate(sensor_paths) if "/env_0/" in path),
        0,
    )
    foot_body_ids, foot_body_names = robot.find_bodies(
        _FOOT_DIAGNOSTIC_BODY_NAMES, preserve_order=True
    )
    if list(foot_body_names) != _FOOT_DIAGNOSTIC_BODY_NAMES:
        raise RuntimeError(
            "Could not resolve detailed-contact foot bodies in the requested order: "
            f"expected {_FOOT_DIAGNOSTIC_BODY_NAMES}, got {foot_body_names}."
        )

    resolved_filter_paths = np.asarray(view.filter_paths)[sensor_index]
    resolved_filter_names = np.asarray(
        [str(path).rsplit("/", 1)[-1] for path in resolved_filter_paths]
    )
    foot_filter_indices = []
    for foot_name in _FOOT_DIAGNOSTIC_BODY_NAMES:
        matches = np.flatnonzero(resolved_filter_names == foot_name)
        if matches.size != 1:
            raise RuntimeError(
                f"Detailed contact view resolved {matches.size} filters for {foot_name}."
            )
        foot_filter_indices.append(int(matches[0]))

    print(
        "[INFO] Detailed PhysX ball-contact diagnostic enabled: "
        f"sensor={sensor_paths[sensor_index]} tracked_body_subset_filters={view.filter_count} "
        f"capacity={max_contact_data_count}."
    )
    state = {
        "view": view,
        "sensor_index": int(sensor_index),
        "sensor_pattern": sensor_pattern,
        "filter_patterns": np.asarray(filter_patterns),
        "filter_paths": np.asarray(view.filter_paths),
        "filter_names": resolved_filter_names,
        "foot_filter_indices": np.asarray(foot_filter_indices, dtype=np.int64),
        "foot_body_ids": torch.as_tensor(
            foot_body_ids, dtype=torch.long, device=base_env.device
        ),
        "foot_body_names": np.asarray(_FOOT_DIAGNOSTIC_BODY_NAMES),
        "ball_sensor": ball_sensor,
        "points_per_foot": _PHYSX_CONTACT_POINTS_PER_FOOT,
        "substep_samples": [],
    }
    # Contact manifolds can exist for only one physics substep, while policy
    # diagnostics run after several decimated substeps.  Capture every sensor
    # update and consume the accumulated manifolds once per control step.
    import types

    original_update = ball_sensor._update_buffers_impl

    def _update_and_capture(sensor_self, env_ids):
        original_update(env_ids)
        sample = _read_detailed_foot_contact_instant(base_env, state)
        if bool(sample["any_robot_contact"]):
            state["substep_samples"].append(sample)

    ball_sensor._update_buffers_impl = types.MethodType(_update_and_capture, ball_sensor)
    return state


def _empty_detailed_foot_contact_sample(points_per_foot: int) -> dict[str, np.ndarray | int | bool]:
    """Fixed-shape empty sample used when no ball-foot manifold exists."""
    return {
        "count": np.zeros(2, dtype=np.int32),
        "truncated": np.zeros(2, dtype=bool),
        "force": np.zeros((2, points_per_foot), dtype=np.float32),
        "point_w": np.full((2, points_per_foot, 3), np.nan, dtype=np.float32),
        "normal_w": np.full((2, points_per_foot, 3), np.nan, dtype=np.float32),
        "separation": np.full((2, points_per_foot), np.nan, dtype=np.float32),
        "point_foot": np.full((2, points_per_foot, 3), np.nan, dtype=np.float32),
        "normal_foot": np.full((2, points_per_foot, 3), np.nan, dtype=np.float32),
        "relative_point_velocity_w": np.full(
            (2, points_per_foot, 3), np.nan, dtype=np.float32
        ),
        "relative_point_velocity_foot": np.full(
            (2, points_per_foot, 3), np.nan, dtype=np.float32
        ),
        "normal_force_w": np.zeros((2, 3), dtype=np.float32),
        "friction_force_w": np.zeros((2, 3), dtype=np.float32),
        "normal_impulse_w": np.zeros((2, 3), dtype=np.float32),
        "force_weighted_point_w": np.full((2, 3), np.nan, dtype=np.float32),
        "force_weighted_point_foot": np.full((2, 3), np.nan, dtype=np.float32),
        "ball_center_foot": np.full((2, 3), np.nan, dtype=np.float32),
        "filtered_force_w": np.zeros((2, 3), dtype=np.float32),
        "filtered_impulse_w": np.zeros((2, 3), dtype=np.float32),
        "filtered_ball_relative_velocity_foot": np.full(
            (2, 3), np.nan, dtype=np.float32
        ),
        "filtered_has_contact": False,
        "filtered_contact_foot": -1,
        "filtered_contact_substeps": np.zeros(2, dtype=np.int32),
        "has_contact": False,
        "contact_foot": -1,
        "any_robot_contact": False,
        "contact_body_idx": -1,
        "contact_body_force": 0.0,
        "contact_substeps": np.zeros(2, dtype=np.int32),
        "motion_frame": -1,
        "event_id": -1,
        "event_frame": -1,
        "expected_foot": -1,
        "expected_surface": -1,
        "frame_error": np.nan,
    }


def _read_detailed_foot_contact_instant(base_env, state: dict) -> dict[str, np.ndarray | int | bool]:
    """Read env-0's exact ball-foot contact manifolds for one physics substep."""
    view = state["view"]
    sensor_index = state["sensor_index"]
    points_per_foot = state["points_per_foot"]
    sample = _empty_detailed_foot_contact_sample(points_per_foot)

    forces, points, normals, separations, counts, starts = view.get_contact_data(
        dt=float(base_env.physics_dt)
    )
    # Copy before get_friction_data(): PhysX reuses count/start buffers inside
    # the view and may overwrite them on the second query.
    forces = forces.detach().clone()
    points = points.detach().clone()
    normals = normals.detach().clone()
    separations = separations.detach().clone()
    counts = counts.detach().clone()
    starts = starts.detach().clone()
    friction, _friction_points, friction_counts, friction_starts = view.get_friction_data(
        dt=float(base_env.physics_dt)
    )
    friction = friction.detach().clone()
    friction_counts = friction_counts.detach().clone()
    friction_starts = friction_starts.detach().clone()

    robot = base_env.scene["robot"]
    ball = base_env.scene["soccer_ball"]
    foot_ids = state["foot_body_ids"]
    foot_pos = robot.data.body_pos_w[0, foot_ids]
    foot_com_pos = robot.data.body_com_pos_w[0, foot_ids]
    foot_quat = robot.data.body_quat_w[0, foot_ids]
    foot_lin_vel = robot.data.body_lin_vel_w[0, foot_ids]
    foot_ang_vel = robot.data.body_ang_vel_w[0, foot_ids]
    ball_pos = ball.data.root_pos_w[0]
    ball_com_pos = ball.data.root_com_pos_w[0]
    ball_lin_vel = ball.data.root_lin_vel_w[0]
    ball_ang_vel = ball.data.root_ang_vel_w[0]

    # Unlike the unfiltered net force used by the legacy detector, this matrix
    # contains only ball contacts with the two ankle links.  Ground friction
    # therefore cannot be mistaken for a foot touch.
    force_matrix = getattr(state["ball_sensor"]._data, "force_matrix_w", None)
    if isinstance(force_matrix, torch.Tensor) and force_matrix.numel() >= 6:
        # Store detailed diagnostic feet in its declared left/right order,
        # independently of the sensor filter order required by S2 rewards
        # (right/left first, followed by all remaining robot links).
        filtered_force_w = force_matrix[
            0, 0, state["foot_filter_indices"].tolist()
        ].detach().clone()
        filtered_force_magnitude = torch.linalg.vector_norm(filtered_force_w, dim=-1)
        sample["filtered_force_w"] = filtered_force_w.cpu().numpy()
        filtered_has_contact = bool(torch.any(filtered_force_magnitude > 1.0).item())
        sample["filtered_has_contact"] = filtered_has_contact
        sample["filtered_contact_foot"] = (
            int(torch.argmax(filtered_force_magnitude).item())
            if filtered_has_contact
            else -1
        )

    pair_force_magnitudes = torch.zeros(2, device=base_env.device)
    all_filter_force = torch.zeros(int(view.filter_count), device=base_env.device)
    for filter_index in range(int(view.filter_count)):
        filter_count = int(counts[sensor_index, filter_index].item())
        if filter_count <= 0:
            continue
        filter_start = int(starts[sensor_index, filter_index].item())
        # PhysX may report the scalar normal force with either sign depending
        # on the body ordering of the contact pair.  It is a magnitude for all
        # contact-existence and aggregation decisions below.
        all_filter_force[filter_index] = forces[
            filter_start : filter_start + filter_count, 0
        ].abs().sum()
    for foot_index in range(2):
        filter_index = int(state["foot_filter_indices"][foot_index])
        count = int(counts[sensor_index, filter_index].item())
        start = int(starts[sensor_index, filter_index].item())
        kept = min(count, points_per_foot)
        sample["count"][foot_index] = count
        sample["truncated"][foot_index] = count > points_per_foot
        if kept > 0:
            sl = slice(start, start + kept)
            pair_forces = forces[sl, 0]
            pair_points = points[sl]
            pair_normals = normals[sl]
            pair_separations = separations[sl, 0]
            normal_force_w = pair_forces.unsqueeze(-1) * pair_normals
            pair_force_magnitudes[foot_index] = pair_forces.abs().sum()

            expanded_quat = foot_quat[foot_index].unsqueeze(0).expand(kept, -1)
            point_foot = quat_apply_inverse(
                expanded_quat,
                pair_points - foot_pos[foot_index].unsqueeze(0),
            )
            normal_foot = quat_apply_inverse(expanded_quat, pair_normals)
            ball_point_velocity = ball_lin_vel.unsqueeze(0) + torch.cross(
                ball_ang_vel.unsqueeze(0).expand(kept, -1),
                pair_points - ball_com_pos.unsqueeze(0),
                dim=-1,
            )
            foot_point_velocity = foot_lin_vel[foot_index].unsqueeze(0) + torch.cross(
                foot_ang_vel[foot_index].unsqueeze(0).expand(kept, -1),
                pair_points - foot_com_pos[foot_index].unsqueeze(0),
                dim=-1,
            )
            relative_velocity_w = ball_point_velocity - foot_point_velocity
            relative_velocity_foot = quat_apply_inverse(expanded_quat, relative_velocity_w)

            sample["force"][foot_index, :kept] = pair_forces.detach().cpu().numpy()
            sample["point_w"][foot_index, :kept] = pair_points.detach().cpu().numpy()
            sample["normal_w"][foot_index, :kept] = pair_normals.detach().cpu().numpy()
            sample["separation"][foot_index, :kept] = pair_separations.detach().cpu().numpy()
            sample["point_foot"][foot_index, :kept] = point_foot.detach().cpu().numpy()
            sample["normal_foot"][foot_index, :kept] = normal_foot.detach().cpu().numpy()
            sample["relative_point_velocity_w"][foot_index, :kept] = (
                relative_velocity_w.detach().cpu().numpy()
            )
            sample["relative_point_velocity_foot"][foot_index, :kept] = (
                relative_velocity_foot.detach().cpu().numpy()
            )
            aggregate_force_w = normal_force_w.sum(dim=0)
            sample["normal_force_w"][foot_index] = aggregate_force_w.detach().cpu().numpy()
            sample["normal_impulse_w"][foot_index] = (
                aggregate_force_w * float(base_env.physics_dt)
            ).detach().cpu().numpy()
            weights = pair_forces.abs()
            weight_sum = weights.sum()
            if float(weight_sum.item()) > 1.0e-8:
                weighted_point_w = (pair_points * weights.unsqueeze(-1)).sum(dim=0) / weight_sum
                sample["force_weighted_point_w"][foot_index] = (
                    weighted_point_w.detach().cpu().numpy()
                )
                sample["force_weighted_point_foot"][foot_index] = quat_apply_inverse(
                    foot_quat[foot_index].unsqueeze(0),
                    (weighted_point_w - foot_pos[foot_index]).unsqueeze(0),
                )[0].detach().cpu().numpy()

        friction_count = int(friction_counts[sensor_index, filter_index].item())
        friction_start = int(friction_starts[sensor_index, filter_index].item())
        if friction_count > 0:
            sample["friction_force_w"][foot_index] = (
                friction[friction_start : friction_start + friction_count]
                .sum(dim=0)
                .detach()
                .cpu()
                .numpy()
            )

        sample["ball_center_foot"][foot_index] = quat_apply_inverse(
            foot_quat[foot_index].unsqueeze(0),
            (ball_pos - foot_pos[foot_index]).unsqueeze(0),
        )[0].detach().cpu().numpy()
        foot_velocity_at_ball_center = foot_lin_vel[foot_index] + torch.cross(
            foot_ang_vel[foot_index], ball_pos - foot_com_pos[foot_index], dim=-1
        )
        sample["filtered_ball_relative_velocity_foot"][foot_index] = (
            quat_apply_inverse(
                foot_quat[foot_index].unsqueeze(0),
                (ball_lin_vel - foot_velocity_at_ball_center).unsqueeze(0),
            )[0]
            .detach()
            .cpu()
            .numpy()
        )

    has_contact = bool(torch.any(pair_force_magnitudes > 0.0).item())
    any_robot_contact = bool(torch.any(all_filter_force > 0.0).item())
    sample["has_contact"] = has_contact
    sample["any_robot_contact"] = any_robot_contact
    sample["contact_foot"] = (
        int(torch.argmax(pair_force_magnitudes).item()) if has_contact else -1
    )
    if any_robot_contact or bool(sample["filtered_has_contact"]):
        contact_body_idx = int(torch.argmax(all_filter_force).item())
        sample["contact_body_idx"] = contact_body_idx
        sample["contact_body_force"] = float(all_filter_force[contact_body_idx].item())
        command = base_env.command_manager.get_term("motion")
        sample["motion_frame"] = int(command.time_steps[0].item())
        for sample_key, attr_name in (
            ("event_id", "s2_contact_event_id_ref"),
            ("event_frame", "s2_contact_event_frame_ref"),
            ("expected_foot", "s2_contact_event_foot_ref"),
            ("expected_surface", "s2_contact_event_surface_ref"),
        ):
            value = getattr(command, attr_name, None)
            if isinstance(value, torch.Tensor):
                sample[sample_key] = int(value[0].item())
        if int(sample["event_frame"]) >= 0:
            sample["frame_error"] = int(sample["motion_frame"]) - int(sample["event_frame"])
    return sample


def _consume_detailed_foot_contact(base_env, state: dict) -> dict[str, np.ndarray | int | bool]:
    """Collapse all detailed manifolds since the previous policy sample."""
    samples = state["substep_samples"]
    state["substep_samples"] = []
    points_per_foot = state["points_per_foot"]
    if not samples:
        # Keep the ball-center-in-foot-frame telemetry populated even when no
        # contact occurred during the decimation interval.
        return _read_detailed_foot_contact_instant(base_env, state)

    merged = _empty_detailed_foot_contact_sample(points_per_foot)
    merged["any_robot_contact"] = True
    peak_robot_contact = max(samples, key=lambda sample: float(sample["contact_body_force"]))
    merged["contact_body_idx"] = peak_robot_contact["contact_body_idx"]
    merged["contact_body_force"] = peak_robot_contact["contact_body_force"]
    for key in (
        "motion_frame",
        "event_id",
        "event_frame",
        "expected_foot",
        "expected_surface",
        "frame_error",
    ):
        merged[key] = peak_robot_contact[key]
    total_impulse_magnitude = np.zeros(2, dtype=np.float32)
    for foot_index in range(2):
        active = [sample for sample in samples if int(sample["count"][foot_index]) > 0]
        merged["contact_substeps"][foot_index] = len(active)
        if not active:
            # Use the latest pose for ball-center telemetry on the inactive foot.
            merged["ball_center_foot"][foot_index] = samples[-1]["ball_center_foot"][foot_index]
            continue

        merged["count"][foot_index] = max(int(sample["count"][foot_index]) for sample in active)
        merged["truncated"][foot_index] = any(
            bool(sample["truncated"][foot_index]) for sample in active
        )
        merged["normal_impulse_w"][foot_index] = np.sum(
            [sample["normal_impulse_w"][foot_index] for sample in active], axis=0
        )
        total_impulse_magnitude[foot_index] = float(
            np.linalg.norm(merged["normal_impulse_w"][foot_index])
        )

        peak = max(
            active,
            key=lambda sample: float(np.sum(np.abs(sample["force"][foot_index]))),
        )
        merged["normal_force_w"][foot_index] = peak["normal_force_w"][foot_index]
        merged["friction_force_w"][foot_index] = peak["friction_force_w"][foot_index]
        merged["force_weighted_point_w"][foot_index] = peak["force_weighted_point_w"][foot_index]
        merged["force_weighted_point_foot"][foot_index] = peak[
            "force_weighted_point_foot"
        ][foot_index]
        merged["ball_center_foot"][foot_index] = peak["ball_center_foot"][foot_index]

        candidates = []
        for sample in active:
            valid = np.flatnonzero(np.isfinite(sample["force"][foot_index]))
            valid = valid[np.abs(sample["force"][foot_index, valid]) > 0.0]
            for point_index in valid:
                candidates.append(
                    (
                        float(abs(sample["force"][foot_index, point_index])),
                        sample,
                        int(point_index),
                    )
                )
        candidates.sort(key=lambda item: item[0], reverse=True)
        merged["truncated"][foot_index] |= len(candidates) > points_per_foot
        for output_index, (force, sample, point_index) in enumerate(
            candidates[:points_per_foot]
        ):
            merged["force"][foot_index, output_index] = force
            for key in (
                "point_w",
                "normal_w",
                "point_foot",
                "normal_foot",
                "relative_point_velocity_w",
                "relative_point_velocity_foot",
            ):
                merged[key][foot_index, output_index] = sample[key][foot_index, point_index]
            merged["separation"][foot_index, output_index] = sample["separation"][
                foot_index, point_index
            ]

    merged["has_contact"] = bool(np.any(total_impulse_magnitude > 0.0))
    merged["contact_foot"] = (
        int(np.argmax(total_impulse_magnitude)) if bool(merged["has_contact"]) else -1
    )
    return merged


def _update_last_termination_reason(base_env, env_idx: int = 0) -> None:
    """Track the latest failure termination from the env's termination manager."""
    global _LAST_TERM_REASON
    tm = getattr(base_env, "termination_manager", None)
    if tm is None:
        return
    active: list[str] = []
    for name in tm.active_terms:
        try:
            if tm.get_term_cfg(name).time_out:
                continue
            if bool(tm.get_term(name)[env_idx].item()):
                active.append(name)
        except Exception:
            continue
    if active:
        _LAST_TERM_REASON = ", ".join(active)


def _active_failure_termination_reason(base_env, env_idx: int = 0) -> str:
    """Return all currently active non-timeout termination terms for one env."""
    tm = getattr(base_env, "termination_manager", None)
    if tm is None:
        return ""
    active: list[str] = []
    for name in tm.active_terms:
        try:
            if tm.get_term_cfg(name).time_out:
                continue
            if bool(tm.get_term(name)[env_idx].item()):
                active.append(name)
        except Exception:
            continue
    return ", ".join(active)


def _reward_term_values(base_env, env_idx: int = 0) -> np.ndarray:
    """Read the reward manager's per-term contribution for the most recent step."""
    reward_manager = getattr(base_env, "reward_manager", None)
    values = getattr(reward_manager, "_step_reward", None)
    if values is None:
        return np.empty(0, dtype=np.float32)
    return values[env_idx].detach().cpu().numpy().copy()


def _resolve_base_env(env):
    """Unwrap gym / RSL-RL wrappers to the underlying Isaac Lab env."""
    base = env
    while hasattr(base, "env"):
        base = base.env
    if hasattr(base, "unwrapped"):
        base = base.unwrapped
    return base


def _get_joint_position_action_term(base_env):
    """Return the position-action term and its action-index → robot-joint mapping."""
    action_term = base_env.action_manager.get_term("joint_pos")
    action_joint_ids = getattr(action_term, "_joint_ids", None)
    if action_joint_ids is None:
        action_joint_ids = getattr(action_term, "joint_ids", None)
    if action_joint_ids is None:
        raise RuntimeError("The joint_pos action term does not expose its controlled joint ids.")
    if isinstance(action_joint_ids, slice):
        action_joint_ids = torch.arange(base_env.scene["robot"].num_joints, device=base_env.device)
    return action_term, torch.as_tensor(action_joint_ids, dtype=torch.long, device=base_env.device)


def _action_ids_for_robot_joint_ids(action_joint_ids: torch.Tensor, robot_joint_ids) -> list[int]:
    """Map robot joint ids to their positions in the policy action vector."""
    robot_to_action = {int(robot_id): action_id for action_id, robot_id in enumerate(action_joint_ids.tolist())}
    try:
        return [robot_to_action[int(robot_id)] for robot_id in robot_joint_ids]
    except KeyError as exc:
        raise RuntimeError(
            f"joint_pos action does not control robot joint id {exc.args[0]}."
        ) from exc


def _world_quat_to_rpy(quat: torch.Tensor) -> torch.Tensor:
    """Convert Isaac Lab scalar-first quaternions to roll/pitch/yaw in radians."""
    w, x, y, z = quat.unbind(dim=-1)
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x.square() + y.square()))
    pitch = torch.asin(torch.clamp(2.0 * (w * y - z * x), min=-1.0, max=1.0))
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))
    return torch.stack((roll, pitch, yaw), dim=-1)


def _diagnostic_reward_params(base_env, reward_name: str) -> dict:
    """Return a reward's effective params, unwrapping task-state gates."""
    rewards_cfg = getattr(getattr(base_env, "cfg", None), "rewards", None)
    term_cfg = getattr(rewards_cfg, reward_name, None)
    params = getattr(term_cfg, "params", None)
    if not isinstance(params, dict):
        return {}
    nested = params.get("reward_params")
    return dict(nested) if isinstance(nested, dict) else dict(params)


def _diagnostic_contact_settings(base_env) -> dict:
    """Resolve the exact legal-touch geometry configured for this task."""
    params = _diagnostic_reward_params(base_env, "dribbling_legal_foot_touch")
    default_force_threshold = 14.0
    if not params:
        params = _diagnostic_reward_params(base_env, "s2_effective_touch_bonus")
        if not params:
            # The clean S2 task removes the historical transition-closure
            # reward.  Contact diagnostics must still consume the exact same
            # event geometry as training, which remains registered on the
            # physical-candidate term.
            params = _diagnostic_reward_params(
                base_env, "s2_physical_contact_candidate_bonus"
            )
        # S2 deliberately has no force-validity cap. Infinity keeps the generic
        # playback classifier aligned with that contract while raw force is
        # still exported separately.
        default_force_threshold = float("inf")
    all_body_cfg = params.get("all_body_cfg")
    return {
        "command_name": str(params.get("command_name", "motion")),
        "ball_sensor_name": str(params.get("ball_sensor_name", _BALL_SENSOR_NAME)),
        "all_body_cfg": all_body_cfg,
        "num_ankle_links": int(params.get("num_ankle_links", 2)),
        "contact_surface": str(params.get("contact_surface", "any")),
        "medial_y_min": float(params.get("medial_y_min", 0.018)),
        "contact_force_threshold": float(
            params.get("force_threshold", default_force_threshold)
        ),
        "cg_gated": bool(params.get("cg_gated", False)),
        "cg_surface_gated": bool(params.get("cg_surface_gated", False)),
        "body_names": tuple(getattr(all_body_cfg, "body_names", ()) or ()),
    }


def _active_locomotion_command(command) -> dict[str, torch.Tensor | str | bool]:
    """Describe the velocity command actually active for this policy step.

    Reference-mode tasks derive their command from the yaw-aligned motion
    anchor instead of the resampled/manual command buffers. This helper keeps
    diagnostic fields correct for every mode.
    """
    mode = str(getattr(command, "locomotion_command_mode", "reference"))
    if hasattr(command, "locomotion_lin_vel_command_w"):
        lin_vel_w = command.locomotion_lin_vel_command_w()
    else:
        lin_vel_w = command.anchor_lin_vel_w
    if hasattr(command, "locomotion_ang_vel_command_w"):
        ang_vel_w = command.locomotion_ang_vel_command_w()
    else:
        ang_vel_w = command.anchor_ang_vel_w

    is_local_twist = str(getattr(command.cfg, "locomotion_command_frame", "world")) == "pelvis_local"
    if is_local_twist and hasattr(command, "locomotion_twist_command_b"):
        twist_b = command.locomotion_twist_command_b()
        reference_twist_b = command.reference_locomotion_twist_b()
        blend_alpha = float(command.locomotion_twist_blend_alpha())
    else:
        # Legacy-only fallback.  These fields are retained for a uniform
        # archive schema but are not a local-frame ground-truth comparison.
        twist_b = torch.stack((lin_vel_w[:, 0], lin_vel_w[:, 1], ang_vel_w[:, 2]), dim=-1)
        reference_twist_b = twist_b
        blend_alpha = 1.0

    speed = torch.norm(lin_vel_w[:, :2], dim=-1)
    heading = torch.atan2(lin_vel_w[:, 1], lin_vel_w[:, 0])
    heading_valid = speed > 1.0e-6
    if mode in {"manual", "resampled"}:
        target_speed = getattr(command, "locomotion_cmd_target_speed", command.locomotion_cmd_speed)
        target_heading = getattr(command, "locomotion_cmd_target_heading", command.locomotion_cmd_heading)
    else:
        # A reference command has no independent requested endpoint.
        target_speed = speed
        target_heading = heading
    return {
        "mode": mode,
        "lin_vel_w": lin_vel_w,
        "ang_vel_w": ang_vel_w,
        "speed": speed,
        "heading": heading,
        "heading_valid": heading_valid,
        "target_speed": target_speed,
        "target_heading": target_heading,
        "twist_b": twist_b,
        "reference_twist_b": reference_twist_b,
        "blend_alpha": blend_alpha,
    }


def _create_joint_reference_constraint(env, margin: float, group: str, joint_names: list[str]) -> dict:
    """Prepare a play-only joint-target clamp around the current reference."""
    if margin < 0.0:
        raise ValueError(f"{group} reference margin must be non-negative.")

    base_env = _resolve_base_env(env)
    command = base_env.command_manager.get_term("motion")
    robot = base_env.scene[command.cfg.asset_name]
    action_term, action_joint_ids = _get_joint_position_action_term(base_env)
    if getattr(action_term, "uses_direct_upper_body_latent", False):
        raise ValueError(
            "Play-only joint reference clamps are incompatible with the direct upper-body latent interface. "
            "Use the environment's reference envelope and turn-aware trunk limits instead."
        )
    robot_joint_ids, found_names = robot.find_joints(joint_names, preserve_order=True)
    if len(robot_joint_ids) != len(joint_names):
        raise RuntimeError(f"Could not resolve all {group} constraint joints; found {found_names}.")

    action_ids = _action_ids_for_robot_joint_ids(action_joint_ids, robot_joint_ids)

    scale = getattr(action_term, "_scale", None)
    offset = getattr(action_term, "_offset", None)
    if scale is None or offset is None:
        raise RuntimeError("The joint_pos action term does not expose scale/offset required for target clamping.")

    return {
        "group": group,
        "joint_names": np.asarray(joint_names),
        "margin": float(margin),
        "robot_joint_ids": torch.as_tensor(robot_joint_ids, dtype=torch.long, device=base_env.device),
        "action_ids": torch.as_tensor(action_ids, dtype=torch.long, device=base_env.device),
        "scale": scale,
        "offset": offset,
    }


def _create_upper_body_reference_constraint(env, margin: float, group: str) -> dict:
    """Prepare a play-only shoulder/elbow/wrist target clamp."""
    return _create_joint_reference_constraint(
        env, margin, group, _UPPER_BODY_CONSTRAINT_JOINT_GROUPS[group]
    )


def _create_waist_reference_constraint(env, margin: float) -> dict:
    """Prepare a play-only waist yaw/roll/pitch target clamp."""
    return _create_joint_reference_constraint(
        env, margin, _WAIST_REFERENCE_CONSTRAINT_GROUP, _TRUNK_DIAGNOSTIC_JOINT_NAMES
    )


def _constrain_reference_actions(env, actions: torch.Tensor, constraint: dict) -> torch.Tensor:
    """Keep each commanded upper-body target within ``q_ref ± margin``.

    This is a counterfactual evaluation intervention only: it alters actions
    after policy inference and never changes the checkpoint or training MDP.
    """
    base_env = _resolve_base_env(env)
    command = base_env.command_manager.get_term("motion")
    action_ids = constraint["action_ids"]
    robot_joint_ids = constraint["robot_joint_ids"]
    scale = constraint["scale"][:, action_ids]
    offset = constraint["offset"][:, action_ids]
    if torch.any(torch.abs(scale) < 1.0e-8):
        raise RuntimeError(f"Cannot constrain {constraint['group']} actions with a zero action scale.")

    target = actions[:, action_ids] * scale + offset
    reference = command.joint_pos[:, robot_joint_ids]
    constrained_target = torch.clamp(
        target,
        min=reference - constraint["margin"],
        max=reference + constraint["margin"],
    )
    constrained_actions = actions.clone()
    constrained_actions[:, action_ids] = (constrained_target - offset) / scale
    return constrained_actions


def _single_joint_actuator_parameter(value, joint_name: str, scale: float):
    """Return a one-joint actuator parameter, preserving scalar config values."""
    if not isinstance(value, dict):
        return float(value) * scale
    if joint_name in value:
        return {joint_name: float(value[joint_name]) * scale}
    if len(value) == 1:
        return {joint_name: float(next(iter(value.values()))) * scale}
    raise ValueError(f"Cannot select {joint_name} from multi-joint actuator parameter {value}.")


def _apply_play_waist_roll_stiffness_scale(env_cfg, scale: float | None) -> float:
    """Split the waist actuator and strengthen only waist roll for one playback."""
    if scale is not None and scale <= 0.0:
        raise ValueError("--waist_roll_stiffness_scale must be positive.")

    actuators = dict(env_cfg.scene.robot.actuators)
    waist_cfg = actuators.pop("waist", None)
    if waist_cfg is None:
        if "waist_roll_control" in actuators:
            if scale is not None:
                raise ValueError(
                    "The control task already uses the training waist-roll PD scale (2.0); "
                    "do not also pass --waist_roll_stiffness_scale."
                )
            print("[INFO] Control task uses the training waist-roll PD scale: stiffness x2.000, damping x1.414.")
            return 2.0
        raise RuntimeError("Play-only waist-roll override requires the robot 'waist' actuator config.")
    if scale is None:
        return 1.0

    damping_scale = float(scale) ** 0.5
    actuators["waist_roll_play"] = waist_cfg.replace(
        joint_names_expr=["waist_roll_joint"],
        stiffness=_single_joint_actuator_parameter(waist_cfg.stiffness, "waist_roll_joint", scale),
        damping=_single_joint_actuator_parameter(waist_cfg.damping, "waist_roll_joint", damping_scale),
    )
    actuators["waist_pitch_play"] = waist_cfg.replace(
        joint_names_expr=["waist_pitch_joint"],
        stiffness=_single_joint_actuator_parameter(waist_cfg.stiffness, "waist_pitch_joint", 1.0),
        damping=_single_joint_actuator_parameter(waist_cfg.damping, "waist_pitch_joint", 1.0),
    )
    env_cfg.scene.robot.actuators = actuators
    print(
        "[INFO] Play-only waist-roll PD override: "
        f"stiffness x{scale:.3f}, damping x{damping_scale:.3f}; waist pitch unchanged."
    )
    return float(scale)


def _create_diagnostic(
    env,
    log_dir: str,
    stride: int,
    constraints: list[dict] | None = None,
    waist_roll_stiffness_scale: float = 1.0,
) -> dict:
    """Prepare a per-step arm, waist, and trunk-motion trace for one playback env."""
    if stride <= 0:
        raise ValueError("--diagnostic_stride must be positive.")
    constraints = [] if constraints is None else constraints

    base_env = _resolve_base_env(env)
    command = base_env.command_manager.get_term("motion")
    robot = base_env.scene[command.cfg.asset_name]
    joint_ids, found_names = robot.find_joints(_ARM_DIAGNOSTIC_JOINT_NAMES, preserve_order=True)
    if len(joint_ids) != len(_ARM_DIAGNOSTIC_JOINT_NAMES):
        raise RuntimeError(
            "Could not resolve all arm joints for the diagnostic. "
            f"Expected {_ARM_DIAGNOSTIC_JOINT_NAMES}, found {found_names}."
        )
    trunk_joint_ids, trunk_found_names = robot.find_joints(
        _TRUNK_DIAGNOSTIC_JOINT_NAMES, preserve_order=True
    )
    if len(trunk_joint_ids) != len(_TRUNK_DIAGNOSTIC_JOINT_NAMES):
        raise RuntimeError(
            "Could not resolve all waist diagnostic joints. "
            f"Expected {_TRUNK_DIAGNOSTIC_JOINT_NAMES}, found {trunk_found_names}."
        )
    trunk_body_ids = [robot.body_names.index(name) for name in _TRUNK_DIAGNOSTIC_BODY_NAMES]
    try:
        trunk_reference_body_ids = [command.cfg.body_names.index(name) for name in _TRUNK_DIAGNOSTIC_BODY_NAMES]
    except ValueError as exc:
        raise RuntimeError(
            f"Motion command does not expose every trunk diagnostic body: {_TRUNK_DIAGNOSTIC_BODY_NAMES}."
        ) from exc
    action_term, action_joint_ids = _get_joint_position_action_term(base_env)
    if hasattr(action_term, "diagnostic_snapshot_enabled"):
        action_term.diagnostic_snapshot_enabled = True
    arm_action_ids = _action_ids_for_robot_joint_ids(action_joint_ids, joint_ids)
    trunk_full_action_ids = _action_ids_for_robot_joint_ids(action_joint_ids, trunk_joint_ids)
    direct_upper_latent = bool(getattr(action_term, "uses_direct_upper_body_latent", False))
    if direct_upper_latent:
        trunk_action_ids = action_term.policy_action_ids_for_robot_joint_ids(trunk_joint_ids)
    else:
        trunk_action_ids = _action_ids_for_robot_joint_ids(action_joint_ids, trunk_joint_ids)

    output_dir = os.path.join(log_dir, "diagnostics")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir, f"diagnostic_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
    )
    reward_manager = getattr(base_env, "reward_manager", None)
    reward_term_names = np.asarray(getattr(reward_manager, "_term_names", []))
    level_entry_iteration = getattr(
        command, "_s2_curriculum_level_entry_iteration", None
    )
    if isinstance(level_entry_iteration, torch.Tensor):
        level_entry_iteration = level_entry_iteration.detach().cpu().numpy().copy()
    else:
        level_entry_iteration = np.empty(0, dtype=np.int64)
    checkpoint_iteration = getattr(command, "_s2_training_iteration", None)
    checkpoint_iteration = (
        int(checkpoint_iteration.item())
        if isinstance(checkpoint_iteration, torch.Tensor)
        and checkpoint_iteration.numel() == 1
        else -1
    )
    contact_settings = _diagnostic_contact_settings(base_env)
    detailed_contact = _create_detailed_foot_contact_view(base_env)
    filtered_contact_available = str(base_env.device).startswith("cuda")
    constraint_groups = np.asarray([constraint["group"] for constraint in constraints])
    constraint_margins = np.asarray([constraint["margin"] for constraint in constraints], dtype=np.float32)
    constraint_joint_names = (
        np.concatenate([constraint["joint_names"] for constraint in constraints])
        if constraints else np.asarray([])
    )
    return {
        "path": output_path,
        "stride": int(stride),
        "step_dt": float(base_env.step_dt),
        "schema_version": _DIAGNOSTIC_SCHEMA_VERSION,
        "sample_timing": (
            "state/reference/policy/submitted/contact are pre-step; reward/done/executed_action "
            "are the resulting transition"
        ),
        "action_value_semantics": np.asarray(
            [
                "policy_action: raw policy output before playback constraints",
                "submitted_action: action passed to env.step after playback constraints",
                "effective_action: action after action-layer projection, limits, and filtering",
            ]
        ),
        "joint_ids": torch.as_tensor(joint_ids, dtype=torch.long, device=base_env.device),
        "joint_names": np.asarray(_ARM_DIAGNOSTIC_JOINT_NAMES),
        "action_ids": torch.as_tensor(arm_action_ids, dtype=torch.long, device=base_env.device),
        "direct_upper_body_latent": direct_upper_latent,
        "trunk_joint_ids": torch.as_tensor(trunk_joint_ids, dtype=torch.long, device=base_env.device),
        "trunk_joint_names": np.asarray(_TRUNK_DIAGNOSTIC_JOINT_NAMES),
        "trunk_action_ids": torch.as_tensor(trunk_action_ids, dtype=torch.long, device=base_env.device),
        "trunk_full_action_ids": torch.as_tensor(
            trunk_full_action_ids, dtype=torch.long, device=base_env.device
        ),
        "trunk_body_ids": torch.as_tensor(trunk_body_ids, dtype=torch.long, device=base_env.device),
        "trunk_reference_body_ids": torch.as_tensor(
            trunk_reference_body_ids, dtype=torch.long, device=base_env.device
        ),
        "trunk_body_names": np.asarray(_TRUNK_DIAGNOSTIC_BODY_NAMES),
        "reward_term_names": reward_term_names,
        "task_state_names": np.asarray(["idle", "dribble", "stop"]),
        "ball_spawn_source_names": np.asarray(["fallback_pelvis_local_front", "reference_first_contact"]),
        "constraint_group": "none" if not constraints else "+".join(constraint_groups.tolist()),
        "constraint_margin": float(constraint_margins[0]) if len(constraints) == 1 else np.nan,
        "constraint_joint_names": constraint_joint_names,
        "constraint_groups": constraint_groups,
        "constraint_margins": constraint_margins,
        "waist_roll_stiffness_scale": float(waist_roll_stiffness_scale),
        "waist_roll_damping_scale": float(waist_roll_stiffness_scale) ** 0.5,
        "contact_settings": contact_settings,
        "detailed_contact": detailed_contact,
        "physx_contact_sensor_pattern": detailed_contact["sensor_pattern"],
        "physx_contact_filter_patterns": detailed_contact["filter_patterns"],
        "physx_contact_filter_paths": detailed_contact["filter_paths"],
        "physx_contact_body_names": detailed_contact["filter_names"],
        "physx_contact_foot_names": detailed_contact["foot_body_names"],
        "physx_contact_points_per_foot": int(detailed_contact["points_per_foot"]),
        "physx_contact_backend": "early_tensor_contact_view",
        "filtered_contact_available": filtered_contact_available,
        "contact_region_frame": "foot_yaw",
        "locomotion_command_frame": str(
            getattr(command.cfg, "locomotion_command_frame", "world")
        ),
        "contact_body_names": np.asarray(contact_settings["body_names"]),
        "contact_surface": contact_settings["contact_surface"],
        "contact_force_threshold": contact_settings["contact_force_threshold"],
        "contact_cg_gated": contact_settings["cg_gated"],
        "contact_cg_surface_gated": contact_settings["cg_surface_gated"],
        "s2_checkpoint_iteration": checkpoint_iteration,
        "s2_curriculum_level_entry_iteration": level_entry_iteration,
        "step": [],
        "motion_idx": [],
        "ball_spawn_source": [],
        "ball_spawn_reference_contact_frame": [],
        "ball_spawn_reference_local": [],
        "style_phase": [],
        "style_cycle_length": [],
        "style_source_first_frame": [],
        "style_source_second_frame": [],
        "style_seam_blend": [],
        "style_in_seam_bridge": [],
        "segment_idx": [],
        "task_state": [],
        "command_mode": [],
        "command_heading": [],
        "effective_command_speed": [],
        "effective_command_heading": [],
        "active_command_lin_vel_w": [],
        "active_command_ang_vel_w": [],
        "reference_twist_local": [],
        "active_twist_local": [],
        "actual_twist_local": [],
        "twist_local_error": [],
        "twist_blend_alpha": [],
        "command_heading_valid": [],
        "pelvis_yaw": [],
        "reference_joint_pos": [],
        "reference_joint_vel": [],
        "actual_joint_pos": [],
        "actual_joint_vel": [],
        "policy_action": [],
        "submitted_action": [],
        "effective_action": [],
        "applied_action": [],
        "action_snapshot_available": [],
        "post_step_state_valid": [],
        "upper_policy_latent": [],
        "trunk_reference_joint_pos": [],
        "trunk_reference_joint_vel": [],
        "trunk_actual_joint_pos": [],
        "trunk_actual_joint_vel": [],
        "trunk_policy_action": [],
        "trunk_submitted_action": [],
        "trunk_effective_action": [],
        "trunk_applied_action": [],
        "trunk_processed_joint_target": [],
        "trunk_target_minus_reference": [],
        "trunk_post_step_actual_joint_pos": [],
        "trunk_post_step_target_error": [],
        "trunk_soft_joint_pos_limits": [],
        "trunk_actual_limit_margin": [],
        "trunk_target_limit_margin": [],
        "trunk_computed_torque": [],
        "trunk_applied_torque": [],
        "trunk_effort_limit": [],
        "trunk_computed_effort_utilization": [],
        "trunk_effort_utilization": [],
        "trunk_effort_saturated": [],
        "pelvis_rpy": [],
        "torso_rpy": [],
        "torso_minus_pelvis_rpy": [],
        "reference_pelvis_rpy": [],
        "reference_torso_rpy": [],
        "reference_torso_minus_pelvis_rpy": [],
        "torso_minus_pelvis_rpy_error": [],
        "pelvis_ang_vel_w": [],
        "torso_ang_vel_w": [],
        "torso_minus_pelvis_ang_vel_w": [],
        "ball_pelvis_xy_distance": [],
        "motion_frame": [],
        "ball_pos_w": [],
        "ball_lin_vel_w": [],
        "expected_foot_pos_w": [],
        "expected_foot_lin_vel_w": [],
        "ball_minus_expected_foot_lin_vel_w": [],
        "ball_contact": [],
        "contact_net_force_w": [],
        "contact_force_magnitude": [],
        "contact_body_idx": [],
        "contact_foot": [],
        "contact_legal_ankle": [],
        "contact_generic_instep": [],
        "contact_inside_instep": [],
        "contact_outside_instep": [],
        "contact_requested_surface_match": [],
        "contact_gentle": [],
        "contact_legal_touch": [],
        "contact_cg_gate_pass": [],
        "contact_cg_eligible": [],
        "contact_ball_offset_foot_yaw": [],
        "contact_medial_offset": [],
        "physx_foot_contact": [],
        "physx_robot_contact": [],
        "physx_contact_body_idx": [],
        "physx_contact_body_force": [],
        "physx_contact_foot": [],
        "physx_foot_contact_substeps": [],
        "physx_foot_contact_count": [],
        "physx_foot_contact_truncated": [],
        "physx_foot_contact_force": [],
        "physx_foot_contact_point_w": [],
        "physx_foot_contact_normal_w": [],
        "physx_foot_contact_separation": [],
        "physx_foot_contact_point_foot": [],
        "physx_foot_contact_normal_foot": [],
        "physx_foot_relative_point_velocity_w": [],
        "physx_foot_relative_point_velocity_foot": [],
        "physx_foot_normal_force_w": [],
        "physx_foot_friction_force_w": [],
        "physx_foot_normal_impulse_w": [],
        "physx_foot_force_weighted_point_w": [],
        "physx_foot_force_weighted_point_foot": [],
        "physx_ball_center_foot": [],
        "physx_contact_motion_frame": [],
        "physx_contact_event_id": [],
        "physx_contact_event_frame": [],
        "physx_contact_expected_foot": [],
        "physx_contact_expected_surface": [],
        "physx_contact_frame_error": [],
        "cg_label_available": [],
        "cg_reference_contact": [],
        "cg_reference_foot": [],
        "cg_reference_surface": [],
        "cg_contact_foot_match": [],
        "s2_contact_window": [],
        "s2_reward_evaluation_frame": [],
        "s2_curriculum_level": [],
        "s2_episode_curriculum_level": [],
        "s2_episode_curriculum_audit": [],
        "s2_episode_contact_count": [],
        "s2_reset_mode": [],
        "s2_acquisition_history_steps_before_contact": [],
        "s2_premature_contact_count": [],
        "s2_touch_occurrence_rate": [],
        "s2_valid_contact_rate": [],
        "s2_correct_side_rate": [],
        "s2_touch_force_mean": [],
        "s2_touch_force_max": [],
        "s2_contact_event_id": [],
        "s2_contact_event_frame": [],
        "s2_expected_foot": [],
        "s2_expected_side": [],
        "s2_expected_surface": [],
        "s2_reference_foot_pos_w": [],
        "s2_reference_foot_yaw": [],
        "s2_reference_pelvis_pos_w": [],
        "s2_reference_pelvis_yaw": [],
        "s2_actual_pelvis_pos_w": [],
        "s2_actual_pelvis_yaw": [],
        "s2_ball_offset_reference_foot": [],
        "s2_reference_target_region_distance": [],
        "s2_reference_contact_anchor_w": [],
        "s2_contact_anchor_w": [],
        "s2_contact_anchor_distance_xy": [],
        "s2_contact_anchor_quality": [],
        "s2_contact_anchor_offset_actual_foot_yaw": [],
        "s2_contact_anchor_offset_actual_foot_full": [],
        "s2_actual_foot_pos_w": [],
        "s2_actual_foot_yaw": [],
        "s2_ball_offset_actual_foot": [],
        "s2_target_region_distance": [],
        "s2_physical_foot_ball_distance": [],
        "s2_critic_sparse_context": [],
        "s2_sequence_cursor": [],
        "s2_sequence_failed": [],
        "s2_command_locked_next_anchor_w": [],
        "s2_command_locked_next_anchor_valid": [],
        "s2_critic_locked_anchor_world_error": [],
        "s2_approach_progress": [],
        "s2_coarse_contact_approach": [],
        "s2_immediate_contact": [],
        "s2_valid_contact_event": [],
        "s2_ordered_transition_completion": [],
        "s2_contact_has_force": [],
        "s2_contact_previous_has_force": [],
        "s2_contact_new_physical_onset": [],
        "s2_contact_event_claim": [],
        "s2_contact_event_owns_physical": [],
        "s2_contact_closest_body": [],
        "s2_contact_correct_foot": [],
        "s2_contact_side_correct": [],
        "s2_contact_coarse_longitudinal_pass": [],
        "s2_contact_anchor_valid_touch": [],
        "s2_contact_rearmed": [],
        "s2_contact_release_off_count": [],
        "s2_contact_release_confirmed": [],
        "s2_contact_left_claimed_region": [],
        "s2_contact_claimed_region_distance": [],
        "s2_contact_physical_episode": [],
        "s2_contact_claimed_episode": [],
        "s2_contact_claimed_event": [],
        "s2_contact_region_distance": [],
        "s2_contact_region_valid": [],
        "s2_contact_region_nearest_point_id": [],
        "s2_contact_region_nearest_timing": [],
        "s2_ball_in_actual_foot_full": [],
        "s2_surface_style": [],
        "s2_release_pending": [],
        "s2_release_actual_touch_frame": [],
        "s2_release_next_anchor_w": [],
        "s2_release_remaining_time": [],
        "s2_release_target_distance": [],
        "s2_release_target_direction": [],
        "s2_release_nominal_speed": [],
        "s2_release_parallel_speed": [],
        "s2_release_lateral_speed": [],
        "s2_release_speed_ratio": [],
        "s2_release_separation_score": [],
        "s2_release_score": [],
        "s2_release_v2_score": [],
        "s2_release_quality_pass": [],
        "s2_locked_next_region_active": [],
        "s2_locked_next_anchor_distance": [],
        "s2_locked_next_region_arrival_pulse": [],
        "s2_locked_next_region_arrival": [],
        "s2_locked_next_region_min_distance": [],
        "s2_locked_euclidean_positive_progress": [],
        "s2_quality_streak": [],
        "s2_second_touch_conditional_success": [],
        "s2_terminal_transition_valid": [],
        "s2_terminal_transition_success": [],
        "s2_terminal_transition_start_distance": [],
        "s2_terminal_transition_end_distance": [],
        "s2_terminal_transition_net_progress": [],
        "s2_terminal_window_position_outcome": [],
        "s2_window_outcome_active": [],
        "s2_window_outcome_source_event_id": [],
        "s2_window_outcome_target_event_id": [],
        "s2_window_outcome_candidate_frame": [],
        "s2_window_outcome_start_distance": [],
        "s2_window_outcome_score_sum": [],
        "s2_window_outcome_sample_count": [],
        "s2_window_outcome_score_average": [],
        "s2_window_outcome_score_end": [],
        "s2_window_outcome_settlement_quality": [],
        "s2_window_outcome_settlement_pulse": [],
        "s2_window_outcome_pulse": [],
        "s2_window_outcome_emitted": [],
        "s2_window_outcome_aborted_ball_lost": [],
        "s2_window_outcome_aborted_other": [],
        "s2_locked_live_target_w": [],
        "s2_locked_fixed_target_w": [],
        "s2_fixed_target_shift_xy": [],
        "s2_fixed_target_shift_distance": [],
        "s2_fixed_target_inverse_error": [],
        "s2_fixed_window_distance": [],
        "s2_live_window_distance": [],
        "s2_fixed_window_min_distance": [],
        "s2_live_window_min_distance": [],
        "s2_fixed_window_min_frame": [],
        "s2_terminal_release": [],
        "s2_terminal_transition_window": [],
        "s2_terminal_progress_raw_rate": [],
        "s2_terminal_progress_clipped_rate": [],
        "s2_terminal_progress_positive_rate": [],
        "s2_terminal_progress_negative_rate": [],
        "s2_t2_region_progress": [],
        "s2_t2_region_arrival": [],
        "s2_t2_region_distance": [],
        "s2_t2_success": [],
        "s2_t2_evaluated": [],
        "cg_flow_label_available": [],
        "cg_flow_valid": [],
        "cg_flow_anchor_frame": [],
        "cg_flow_direction_local": [],
        "cg_flow_distance": [],
        "cg_flow_duration": [],
        "cg_flow_nominal_speed": [],
        "cg_flow_latched": [],
        "cg_flow_release_active": [],
        "cg_flow_target_direction_world": [],
        "cg_flow_parallel_speed": [],
        "cg_flow_lateral_speed": [],
        "cg_flow_release_reward": [],
        "cg_flow_progress": [],
        "cg_flow_progress_rate": [],
        "cg_flow_lateral_offset": [],
        "cg_flow_progress_reward": [],
        "ball_xy_speed": [],
        "ball_command_forward_speed": [],
        "pelvis_xy_speed": [],
        "command_target_speed": [],
        "command_target_heading": [],
        "foot_reference_position_error": [],
        "heading_error": [],
        "no_contact_count": [],
        "no_contact_task_active": [],
        "no_contact_recovery_active": [],
        "no_contact_proximity_recovery_active": [],
        "no_contact_relative_speed": [],
        "idle_active": [],
        "idle_pelvis_speed": [],
        "idle_pelvis_angular_speed": [],
        "stop_active": [],
        "stop_settled": [],
        "stop_success": [],
        "stop_settle_elapsed_s": [],
        "stop_pelvis_speed": [],
        "stop_pelvis_angular_speed": [],
        "stop_ball_speed": [],
        "stop_forward_offset": [],
        "stop_lateral_offset": [],
        "stop_position_score": [],
        "stop_speed_score": [],
        "manifold_raw_upper_target": [],
        "manifold_reference_upper_target": [],
        "manifold_constrained_upper_target": [],
        "manifold_projected_upper_target": [],
        "manifold_joint_limited_upper_target": [],
        "manifold_executed_upper_target": [],
        "manifold_latent": [],
        "manifold_projection_error": [],
        "manifold_projection_error_after_reference_constraint": [],
        "manifold_nullspace_residual": [],
        "manifold_latent_clip_fraction": [],
        "manifold_reference_overflow": [],
        "manifold_reference_clamp_fraction": [],
        "manifold_joint_limit_clamp_fraction": [],
        "manifold_filter_lag": [],
        "trunk_pitch_raw_target": [],
        "trunk_pitch_reference_target": [],
        "trunk_pitch_soft_target": [],
        "trunk_pitch_filtered_target": [],
        "trunk_pitch_reference_overflow": [],
        "trunk_pitch_turn_relaxation": [],
        "trunk_pitch_active_lower_deviation": [],
        "trunk_pitch_active_upper_deviation": [],
        "trunk_pitch_active_cutoff_frequency_hz": [],
        "step_reward": [],
        "reward_terms": [],
        "done": [],
        "termination_reason": [],
    }


def _append_diagnostic(
    diagnostic: dict,
    env,
    policy_actions: torch.Tensor,
    applied_actions: torch.Tensor,
    step: int,
) -> bool:
    """Record the pre-step reference/state plus policy and applied actions for env 0."""
    if step % diagnostic["stride"] != 0:
        return False

    base_env = _resolve_base_env(env)
    command = base_env.command_manager.get_term("motion")
    robot = base_env.scene[command.cfg.asset_name]
    ids = diagnostic["joint_ids"]
    action_ids = diagnostic["action_ids"]
    trunk_ids = diagnostic["trunk_joint_ids"]
    trunk_action_ids = diagnostic["trunk_action_ids"]
    trunk_body_ids = diagnostic["trunk_body_ids"]
    trunk_reference_body_ids = diagnostic["trunk_reference_body_ids"]

    def _cpu(value: torch.Tensor) -> np.ndarray:
        return value[0].detach().cpu().numpy().copy()

    diagnostic["step"].append(int(step))
    diagnostic["motion_idx"].append(int(command.motion_idx[0].item()))
    if hasattr(command, "ball_spawn_reference_info"):
        spawn_info = command.ball_spawn_reference_info()
        diagnostic["ball_spawn_source"].append(int(spawn_info["source"][0].item()))
        diagnostic["ball_spawn_reference_contact_frame"].append(
            int(spawn_info["reference_contact_frame"][0].item())
        )
        diagnostic["ball_spawn_reference_local"].append(_cpu(spawn_info["reference_local"]))
    else:
        diagnostic["ball_spawn_source"].append(-1)
        diagnostic["ball_spawn_reference_contact_frame"].append(-1)
        diagnostic["ball_spawn_reference_local"].append(np.full(3, np.nan, dtype=np.float32))
    diagnostic["style_phase"].append(int(command.style_phase_steps[0].item()))
    if hasattr(command, "style_phase_reference_info"):
        style_info = command.style_phase_reference_info()
        diagnostic["style_cycle_length"].append(int(style_info["cycle_length"][0].item()))
        diagnostic["style_source_first_frame"].append(int(style_info["source_first_frame"][0].item()))
        diagnostic["style_source_second_frame"].append(int(style_info["source_second_frame"][0].item()))
        diagnostic["style_seam_blend"].append(float(style_info["seam_blend"][0].item()))
        diagnostic["style_in_seam_bridge"].append(bool(style_info["in_seam_bridge"][0].item()))
    else:
        diagnostic["style_cycle_length"].append(int(command.motion_length[0].item()))
        diagnostic["style_source_first_frame"].append(int(command.style_phase_steps[0].item()))
        diagnostic["style_source_second_frame"].append(int(command.style_phase_steps[0].item()))
        diagnostic["style_seam_blend"].append(0.0)
        diagnostic["style_in_seam_bridge"].append(False)
    diagnostic["segment_idx"].append(int(command._locomotion_segment_idx[0].item()))
    task_state = getattr(command, "locomotion_task_state", None)
    diagnostic["task_state"].append(1 if task_state is None else int(task_state[0].item()))
    active_command = _active_locomotion_command(command)
    active_speed = active_command["speed"]
    active_heading = active_command["heading"]
    heading_valid = active_command["heading_valid"]
    diagnostic["command_mode"].append(str(active_command["mode"]))
    diagnostic["command_heading"].append(float(active_heading[0].item()))
    diagnostic["command_target_speed"].append(float(active_command["target_speed"][0].item()))
    diagnostic["command_target_heading"].append(float(active_command["target_heading"][0].item()))
    diagnostic["effective_command_speed"].append(float(active_speed[0].item()))
    diagnostic["effective_command_heading"].append(float(active_heading[0].item()))
    diagnostic["active_command_lin_vel_w"].append(_cpu(active_command["lin_vel_w"]))
    diagnostic["active_command_ang_vel_w"].append(_cpu(active_command["ang_vel_w"]))
    diagnostic["reference_twist_local"].append(_cpu(active_command["reference_twist_b"]))
    diagnostic["active_twist_local"].append(_cpu(active_command["twist_b"]))
    diagnostic["twist_blend_alpha"].append(float(active_command["blend_alpha"]))
    diagnostic["command_heading_valid"].append(bool(heading_valid[0].item()))

    pelvis_id = robot.body_names.index("pelvis")
    pelvis_quat = robot.data.body_quat_w[0, pelvis_id]
    pelvis_yaw_inv = quat_inv(yaw_quat(robot.data.body_quat_w[:, pelvis_id]))
    pelvis_lin_vel_b = quat_apply(pelvis_yaw_inv, robot.data.body_lin_vel_w[:, pelvis_id])
    pelvis_ang_vel_b = quat_apply(pelvis_yaw_inv, robot.data.body_ang_vel_w[:, pelvis_id])
    actual_twist_b = torch.stack(
        (pelvis_lin_vel_b[:, 0], pelvis_lin_vel_b[:, 1], pelvis_ang_vel_b[:, 2]), dim=-1
    )
    diagnostic["actual_twist_local"].append(_cpu(actual_twist_b))
    diagnostic["twist_local_error"].append(_cpu(active_command["twist_b"] - actual_twist_b))
    diagnostic["pelvis_yaw"].append(float(torch.atan2(
        2.0 * (pelvis_quat[0] * pelvis_quat[3] + pelvis_quat[1] * pelvis_quat[2]),
        1.0 - 2.0 * (pelvis_quat[2].square() + pelvis_quat[3].square()),
    ).item()))
    diagnostic["reference_joint_pos"].append(_cpu(command.joint_pos[:, ids]))
    diagnostic["reference_joint_vel"].append(_cpu(command.joint_vel[:, ids]))
    diagnostic["actual_joint_pos"].append(_cpu(robot.data.joint_pos[:, ids]))
    diagnostic["actual_joint_vel"].append(_cpu(robot.data.joint_vel[:, ids]))
    action_term = base_env.action_manager.get_term("joint_pos")
    if diagnostic["direct_upper_body_latent"]:
        # A direct-latent policy has no one-to-one arm action. Keep arm-shaped
        # entries NaN and record the actual latent policy coordinates instead.
        arm_nan = torch.full(
            (policy_actions.shape[0], len(action_ids)), torch.nan,
            dtype=policy_actions.dtype, device=policy_actions.device,
        )
        latent_ids = getattr(action_term, "_latent_policy_action_ids", None)
        latent = (
            policy_actions[:, latent_ids]
            if isinstance(latent_ids, torch.Tensor)
            else torch.full((policy_actions.shape[0], 0), torch.nan, device=policy_actions.device)
        )
        diagnostic["policy_action"].append(_cpu(arm_nan))
        diagnostic["submitted_action"].append(_cpu(arm_nan))
        diagnostic["applied_action"].append(_cpu(arm_nan))
        diagnostic["upper_policy_latent"].append(_cpu(latent))
    else:
        diagnostic["policy_action"].append(_cpu(policy_actions[:, action_ids]))
        diagnostic["submitted_action"].append(_cpu(applied_actions[:, action_ids]))
        # Kept as a compatibility alias; use submitted_action/effective_action
        # in new analysis code.
        diagnostic["applied_action"].append(_cpu(applied_actions[:, action_ids]))
        diagnostic["upper_policy_latent"].append(np.empty(0, dtype=np.float32))
    diagnostic["trunk_reference_joint_pos"].append(_cpu(command.joint_pos[:, trunk_ids]))
    diagnostic["trunk_reference_joint_vel"].append(_cpu(command.joint_vel[:, trunk_ids]))
    diagnostic["trunk_actual_joint_pos"].append(_cpu(robot.data.joint_pos[:, trunk_ids]))
    diagnostic["trunk_actual_joint_vel"].append(_cpu(robot.data.joint_vel[:, trunk_ids]))
    diagnostic["trunk_policy_action"].append(_cpu(policy_actions[:, trunk_action_ids]))
    diagnostic["trunk_submitted_action"].append(_cpu(applied_actions[:, trunk_action_ids]))
    diagnostic["trunk_applied_action"].append(_cpu(applied_actions[:, trunk_action_ids]))

    trunk_quat = robot.data.body_quat_w[0, trunk_body_ids]
    trunk_rpy = _world_quat_to_rpy(trunk_quat)
    torso_minus_pelvis_rpy = torch.atan2(
        torch.sin(trunk_rpy[1] - trunk_rpy[0]),
        torch.cos(trunk_rpy[1] - trunk_rpy[0]),
    )
    reference_trunk_rpy = _world_quat_to_rpy(
        command.body_quat_relative_w[0, trunk_reference_body_ids]
    )
    reference_torso_minus_pelvis_rpy = torch.atan2(
        torch.sin(reference_trunk_rpy[1] - reference_trunk_rpy[0]),
        torch.cos(reference_trunk_rpy[1] - reference_trunk_rpy[0]),
    )
    torso_minus_pelvis_rpy_error = torch.atan2(
        torch.sin(torso_minus_pelvis_rpy - reference_torso_minus_pelvis_rpy),
        torch.cos(torso_minus_pelvis_rpy - reference_torso_minus_pelvis_rpy),
    )
    trunk_ang_vel_w = robot.data.body_ang_vel_w[0, trunk_body_ids]
    diagnostic["pelvis_rpy"].append(_cpu(trunk_rpy[None, 0]))
    diagnostic["torso_rpy"].append(_cpu(trunk_rpy[None, 1]))
    diagnostic["torso_minus_pelvis_rpy"].append(_cpu(torso_minus_pelvis_rpy[None]))
    diagnostic["reference_pelvis_rpy"].append(_cpu(reference_trunk_rpy[None, 0]))
    diagnostic["reference_torso_rpy"].append(_cpu(reference_trunk_rpy[None, 1]))
    diagnostic["reference_torso_minus_pelvis_rpy"].append(
        _cpu(reference_torso_minus_pelvis_rpy[None])
    )
    diagnostic["torso_minus_pelvis_rpy_error"].append(_cpu(torso_minus_pelvis_rpy_error[None]))
    diagnostic["pelvis_ang_vel_w"].append(_cpu(trunk_ang_vel_w[None, 0]))
    diagnostic["torso_ang_vel_w"].append(_cpu(trunk_ang_vel_w[None, 1]))
    diagnostic["torso_minus_pelvis_ang_vel_w"].append(_cpu((trunk_ang_vel_w[1] - trunk_ang_vel_w[0])[None]))
    soccer_ball = base_env.scene["soccer_ball"]
    diagnostic["motion_frame"].append(int(command.time_steps[0].item()))
    diagnostic["ball_pos_w"].append(_cpu(soccer_ball.data.root_pos_w[:, :3]))
    diagnostic["ball_lin_vel_w"].append(_cpu(soccer_ball.data.root_lin_vel_w[:, :3]))
    ball_delta_xy = soccer_ball.data.root_pos_w[0, :2] - robot.data.body_pos_w[0, pelvis_id, :2]
    diagnostic["ball_pelvis_xy_distance"].append(float(torch.norm(ball_delta_xy).item()))
    contact_settings = diagnostic["contact_settings"]
    contact = dribbling_contact_telemetry(
        base_env,
        command_name=contact_settings["command_name"],
        ball_sensor_name=contact_settings["ball_sensor_name"],
        all_body_cfg=contact_settings["all_body_cfg"],
        num_ankle_links=contact_settings["num_ankle_links"],
        contact_surface=contact_settings["contact_surface"],
        medial_y_min=contact_settings["medial_y_min"],
        cg_surface_gated=contact_settings["cg_surface_gated"],
        contact_force_threshold=contact_settings["contact_force_threshold"],
    )
    diagnostic["ball_contact"].append(bool(contact["has_contact"][0].item()))
    diagnostic["contact_net_force_w"].append(
        _cpu(soccer_ball_contact_net_force_w(base_env, contact_settings["ball_sensor_name"]))
    )
    diagnostic["contact_force_magnitude"].append(float(contact["force_magnitude"][0].item()))
    diagnostic["contact_body_idx"].append(int(contact["contact_body_idx"][0].item()))
    diagnostic["contact_foot"].append(int(contact["contact_foot"][0].item()))
    diagnostic["contact_legal_ankle"].append(bool(contact["legal_ankle"][0].item()))
    diagnostic["contact_generic_instep"].append(bool(contact["generic_instep"][0].item()))
    diagnostic["contact_inside_instep"].append(bool(contact["inside_instep"][0].item()))
    diagnostic["contact_outside_instep"].append(bool(contact["outside_instep"][0].item()))
    diagnostic["contact_requested_surface_match"].append(bool(contact["requested_surface_match"][0].item()))
    diagnostic["contact_gentle"].append(bool(contact["gentle"][0].item()))
    diagnostic["contact_legal_touch"].append(bool(contact["legal_touch"][0].item()))
    diagnostic["contact_ball_offset_foot_yaw"].append(_cpu(contact["ball_offset_foot_yaw"]))
    diagnostic["contact_medial_offset"].append(float(contact["medial_offset"][0].item()))
    detailed_contact = _consume_detailed_foot_contact(base_env, diagnostic["detailed_contact"])
    diagnostic["physx_foot_contact"].append(bool(detailed_contact["has_contact"]))
    diagnostic["physx_robot_contact"].append(bool(detailed_contact["any_robot_contact"]))
    diagnostic["physx_contact_body_idx"].append(int(detailed_contact["contact_body_idx"]))
    diagnostic["physx_contact_body_force"].append(float(detailed_contact["contact_body_force"]))
    diagnostic["physx_contact_foot"].append(int(detailed_contact["contact_foot"]))
    for key, sample_key in (
        ("physx_foot_contact_substeps", "contact_substeps"),
        ("physx_foot_contact_count", "count"),
        ("physx_foot_contact_truncated", "truncated"),
        ("physx_foot_contact_force", "force"),
        ("physx_foot_contact_point_w", "point_w"),
        ("physx_foot_contact_normal_w", "normal_w"),
        ("physx_foot_contact_separation", "separation"),
        ("physx_foot_contact_point_foot", "point_foot"),
        ("physx_foot_contact_normal_foot", "normal_foot"),
        ("physx_foot_relative_point_velocity_w", "relative_point_velocity_w"),
        ("physx_foot_relative_point_velocity_foot", "relative_point_velocity_foot"),
        ("physx_foot_normal_force_w", "normal_force_w"),
        ("physx_foot_friction_force_w", "friction_force_w"),
        ("physx_foot_normal_impulse_w", "normal_impulse_w"),
        ("physx_foot_force_weighted_point_w", "force_weighted_point_w"),
        ("physx_foot_force_weighted_point_foot", "force_weighted_point_foot"),
        ("physx_ball_center_foot", "ball_center_foot"),
    ):
        diagnostic[key].append(detailed_contact[sample_key])
    for key, sample_key in (
        ("physx_contact_motion_frame", "motion_frame"),
        ("physx_contact_event_id", "event_id"),
        ("physx_contact_event_frame", "event_frame"),
        ("physx_contact_expected_foot", "expected_foot"),
        ("physx_contact_expected_surface", "expected_surface"),
        ("physx_contact_frame_error", "frame_error"),
    ):
        diagnostic[key].append(detailed_contact[sample_key])
    cg_labeled = getattr(command, "motion_has_dribble_cg_label", None)
    cg_ref_contact = getattr(command, "dribble_cg_contact_ref", None)
    cg_ref_foot = getattr(command, "dribble_cg_foot_ref", None)
    cg_ref_surface = getattr(command, "dribble_cg_surface_ref", None)
    has_cg_label = isinstance(cg_labeled, torch.Tensor) and bool(cg_labeled[0].item())
    ref_contact = bool(cg_ref_contact[0].item()) if isinstance(cg_ref_contact, torch.Tensor) else False
    ref_foot = int(cg_ref_foot[0].item()) if isinstance(cg_ref_foot, torch.Tensor) else -1
    ref_surface = int(cg_ref_surface[0].item()) if isinstance(cg_ref_surface, torch.Tensor) else -1
    actual_foot = int(contact["contact_foot"][0].item())
    diagnostic["cg_label_available"].append(has_cg_label)
    diagnostic["cg_reference_contact"].append(ref_contact)
    diagnostic["cg_reference_foot"].append(ref_foot)
    diagnostic["cg_reference_surface"].append(ref_surface)
    diagnostic["cg_contact_foot_match"].append(
        bool(contact["has_contact"][0].item()) and has_cg_label and ref_contact and actual_foot == ref_foot
    )
    s2_event_id = getattr(command, "s2_contact_event_id_ref", None)
    s2_event_frame = getattr(command, "s2_contact_event_frame_ref", None)
    s2_expected_foot = getattr(command, "s2_contact_event_foot_ref", None)
    s2_expected_side = getattr(command, "s2_contact_event_side_ref", None)
    s2_expected_surface = getattr(command, "s2_contact_event_surface_ref", None)
    s2_window = getattr(command, "s2_contact_window_ref", None)
    expected_foot_value = (
        int(s2_expected_foot[0].item())
        if isinstance(s2_expected_foot, torch.Tensor) else -1
    )
    if expected_foot_value in (0, 1):
        expected_foot_name = (
            "right_ankle_roll_link" if expected_foot_value == 1 else "left_ankle_roll_link"
        )
        expected_foot_body_id = robot.body_names.index(expected_foot_name)
        expected_foot_pos_w = robot.data.body_pos_w[:, expected_foot_body_id]
        expected_foot_lin_vel_w = robot.data.body_lin_vel_w[:, expected_foot_body_id]
        diagnostic["expected_foot_pos_w"].append(_cpu(expected_foot_pos_w))
        diagnostic["expected_foot_lin_vel_w"].append(_cpu(expected_foot_lin_vel_w))
        diagnostic["ball_minus_expected_foot_lin_vel_w"].append(
            _cpu(soccer_ball.data.root_lin_vel_w[:, :3] - expected_foot_lin_vel_w)
        )
    else:
        diagnostic["expected_foot_pos_w"].append(np.full(3, np.nan, dtype=np.float32))
        diagnostic["expected_foot_lin_vel_w"].append(np.full(3, np.nan, dtype=np.float32))
        diagnostic["ball_minus_expected_foot_lin_vel_w"].append(
            np.full(3, np.nan, dtype=np.float32)
        )
    curriculum_level = getattr(command, "s2_curriculum_level", None)
    episode_curriculum_level = getattr(command, "s2_episode_curriculum_level", None)
    episode_curriculum_audit = getattr(command, "s2_episode_curriculum_audit", None)
    episode_contact_count = getattr(command, "s2_episode_contact_count", None)
    episode_reset_mode = getattr(command, "s2_episode_reset_mode", None)
    diagnostic["s2_curriculum_level"].append(
        int(curriculum_level.item()) if isinstance(curriculum_level, torch.Tensor) else -1
    )
    diagnostic["s2_episode_curriculum_level"].append(
        int(episode_curriculum_level[0].item())
        if isinstance(episode_curriculum_level, torch.Tensor) else -1
    )
    diagnostic["s2_episode_curriculum_audit"].append(
        bool(episode_curriculum_audit[0].item())
        if isinstance(episode_curriculum_audit, torch.Tensor) else True
    )
    diagnostic["s2_episode_contact_count"].append(
        int(episode_contact_count[0].item())
        if isinstance(episode_contact_count, torch.Tensor) else 0
    )
    diagnostic["s2_reset_mode"].append(
        int(episode_reset_mode[0].item())
        if isinstance(episode_reset_mode, torch.Tensor) else 0
    )
    sparse_context = s2_sparse_ball_reference_context(base_env, command_name="motion")
    sparse_context_0 = sparse_context[0]
    diagnostic["s2_critic_sparse_context"].append(_cpu(sparse_context))
    diagnostic["s2_sequence_cursor"].append(
        int(command.s2_sequence_cursor[0].item())
        if isinstance(getattr(command, "s2_sequence_cursor", None), torch.Tensor)
        else -1
    )
    diagnostic["s2_sequence_failed"].append(bool(sparse_context_0[-1].item()))
    locked_anchor = getattr(command, "s2_locked_next_anchor_w", None)
    locked_valid = getattr(command, "s2_locked_next_anchor_valid", None)
    locked_valid_0 = (
        isinstance(locked_valid, torch.Tensor) and bool(locked_valid[0].item())
    )
    diagnostic["s2_command_locked_next_anchor_w"].append(
        _cpu(locked_anchor)
        if isinstance(locked_anchor, torch.Tensor) and locked_valid_0
        else np.full(3, np.nan, dtype=np.float32)
    )
    diagnostic["s2_command_locked_next_anchor_valid"].append(locked_valid_0)
    # A completed C2 episode can leave the command-side latch populated while
    # the critic correctly masks its next-anchor fields because no next event
    # exists.  Compare frames only when the critic's next_valid bit is active.
    critic_next_valid_0 = bool(sparse_context_0[13].item())
    if locked_valid_0 and critic_next_valid_0:
        critic_next_anchor_w = command.robot_pelvis_pos_w[0] + quat_apply(
            yaw_quat(command.robot_pelvis_quat_w[0:1]),
            sparse_context_0[3:6].unsqueeze(0),
        )[0]
        diagnostic["s2_critic_locked_anchor_world_error"].append(
            float(torch.linalg.vector_norm(critic_next_anchor_w - locked_anchor[0]).item())
        )
    else:
        diagnostic["s2_critic_locked_anchor_world_error"].append(np.nan)
    for name, default in (
        ("s2_acquisition_history_steps_before_contact", 0.0),
        ("s2_reward_evaluation_frame", -1.0),
        ("s2_approach_progress", 0.0),
        ("s2_coarse_contact_approach", 0.0),
        ("s2_immediate_contact", 0.0),
        ("s2_valid_contact_event", 0.0),
        ("s2_ordered_transition_completion", 0.0),
        ("s2_contact_has_force", False),
        ("s2_contact_previous_has_force", False),
        ("s2_contact_new_physical_onset", False),
        ("s2_contact_event_claim", False),
        ("s2_contact_event_owns_physical", False),
        ("s2_contact_closest_body", -1),
        ("s2_contact_correct_foot", False),
        ("s2_contact_side_correct", False),
        ("s2_contact_coarse_longitudinal_pass", False),
        ("s2_contact_anchor_valid_touch", False),
        ("s2_contact_rearmed", True),
        ("s2_contact_release_off_count", 0.0),
        ("s2_contact_release_confirmed", False),
        ("s2_contact_left_claimed_region", False),
        ("s2_contact_claimed_region_distance", np.nan),
        ("s2_contact_physical_episode", 0.0),
        ("s2_contact_claimed_episode", -1.0),
        ("s2_contact_claimed_event", -1.0),
        ("s2_contact_region_distance", np.nan),
        ("s2_contact_region_valid", False),
        ("s2_contact_region_nearest_point_id", -1.0),
        ("s2_contact_region_nearest_timing", -1.0),
        ("s2_surface_style", 0.0),
        ("s2_release_pending", False),
        ("s2_release_actual_touch_frame", -1),
        ("s2_release_remaining_time", 0.0),
        ("s2_release_target_distance", 0.0),
        ("s2_release_nominal_speed", 0.0),
        ("s2_release_parallel_speed", 0.0),
        ("s2_release_lateral_speed", 0.0),
        ("s2_release_speed_ratio", 0.0),
        ("s2_release_separation_score", 0.0),
        ("s2_release_score", 0.0),
        ("s2_release_v2_score", 0.0),
        ("s2_release_quality_pass", False),
        ("s2_locked_next_region_active", False),
        ("s2_locked_next_anchor_distance", np.nan),
        ("s2_locked_next_region_arrival_pulse", False),
        ("s2_locked_next_region_arrival", False),
        ("s2_locked_next_region_min_distance", np.nan),
        ("s2_locked_euclidean_positive_progress", 0.0),
        ("s2_quality_streak", 0.0),
        ("s2_second_touch_conditional_success", False),
        ("s2_terminal_transition_valid", False),
        ("s2_terminal_transition_success", False),
        ("s2_terminal_transition_start_distance", 0.0),
        ("s2_terminal_transition_end_distance", 0.0),
        ("s2_terminal_transition_net_progress", 0.0),
        ("s2_terminal_window_position_outcome", 0.0),
        ("s2_window_outcome_active", False),
        ("s2_window_outcome_source_event_id", -1.0),
        ("s2_window_outcome_target_event_id", -1.0),
        ("s2_window_outcome_candidate_frame", -1.0),
        ("s2_window_outcome_start_distance", 0.0),
        ("s2_window_outcome_score_sum", 0.0),
        ("s2_window_outcome_sample_count", 0.0),
        ("s2_window_outcome_score_average", 0.0),
        ("s2_window_outcome_score_end", 0.0),
        ("s2_window_outcome_settlement_quality", 0.0),
        ("s2_window_outcome_settlement_pulse", 0.0),
        ("s2_window_outcome_pulse", 0.0),
        ("s2_window_outcome_emitted", False),
        ("s2_window_outcome_aborted_ball_lost", False),
        ("s2_window_outcome_aborted_other", False),
        ("s2_fixed_target_shift_distance", 0.0),
        ("s2_fixed_window_distance", np.nan),
        ("s2_live_window_distance", np.nan),
        ("s2_fixed_window_min_distance", np.nan),
        ("s2_live_window_min_distance", np.nan),
        ("s2_fixed_window_min_frame", -1.0),
        ("s2_terminal_release", False),
        ("s2_terminal_transition_window", False),
        ("s2_terminal_progress_raw_rate", 0.0),
        ("s2_terminal_progress_clipped_rate", 0.0),
        ("s2_terminal_progress_positive_rate", 0.0),
        ("s2_terminal_progress_negative_rate", 0.0),
        ("s2_t2_region_progress", 0.0),
        ("s2_t2_region_arrival", 0.0),
        ("s2_t2_region_distance", np.nan),
        ("s2_t2_success", False),
        ("s2_t2_evaluated", False),
    ):
        diagnostic[name].append(default)
    fixed_inverse_error = getattr(command, "s2_last_fixed_inverse_error", None)
    diagnostic["s2_fixed_target_inverse_error"].append(
        float(fixed_inverse_error[0].item())
        if isinstance(fixed_inverse_error, torch.Tensor) else np.nan
    )
    diagnostic["s2_release_next_anchor_w"].append(
        np.full(3, np.nan, dtype=np.float32)
    )
    diagnostic["s2_locked_live_target_w"].append(np.full(3, np.nan, dtype=np.float32))
    diagnostic["s2_locked_fixed_target_w"].append(np.full(3, np.nan, dtype=np.float32))
    diagnostic["s2_fixed_target_shift_xy"].append(np.full(2, np.nan, dtype=np.float32))
    diagnostic["s2_ball_in_actual_foot_full"].append(
        np.full(3, np.nan, dtype=np.float32)
    )
    diagnostic["s2_release_target_direction"].append(
        np.full(2, np.nan, dtype=np.float32)
    )
    premature_contact_count = command.metrics.get("s2_premature_contact_count")
    diagnostic["s2_premature_contact_count"].append(
        float(premature_contact_count[0].item())
        if isinstance(premature_contact_count, torch.Tensor) else 0.0
    )
    for metric_name in (
        "s2_touch_occurrence_rate",
        "s2_valid_contact_rate",
        "s2_correct_side_rate",
        "s2_touch_force_mean",
        "s2_touch_force_max",
        "s2_contact_anchor_quality",
    ):
        metric_value = command.metrics.get(metric_name)
        diagnostic[metric_name].append(
            float(metric_value[0].item())
            if isinstance(metric_value, torch.Tensor)
            else 0.0
        )

    # Record the reference/actual pelvis transform explicitly.  The S2
    # contact anchors are mapped through the live reference-to-robot
    # alignment, so a contact miss cannot be diagnosed from body-error
    # aggregates alone.
    try:
        pelvis_reference_index = command.cfg.body_names.index("pelvis")
        # Use the live-aligned targets consumed by motion tracking, not the
        # raw NPZ world pose.  Flat reset may rotate the whole robot/reference
        # pair into task-forward.
        reference_pelvis_pos = command.body_pos_relative_w[:, pelvis_reference_index]
        reference_pelvis_quat = command.body_quat_relative_w[:, pelvis_reference_index]
        actual_pelvis_pos = command.robot_body_pos_w[:, pelvis_reference_index]
        actual_pelvis_quat = command.robot_body_quat_w[:, pelvis_reference_index]
        reference_pelvis_yaw = _world_quat_to_rpy(reference_pelvis_quat)[:, 2]
        actual_pelvis_yaw = _world_quat_to_rpy(actual_pelvis_quat)[:, 2]
        diagnostic["s2_reference_pelvis_pos_w"].append(_cpu(reference_pelvis_pos))
        diagnostic["s2_reference_pelvis_yaw"].append(
            float(reference_pelvis_yaw[0].item())
        )
        diagnostic["s2_actual_pelvis_pos_w"].append(_cpu(actual_pelvis_pos))
        diagnostic["s2_actual_pelvis_yaw"].append(float(actual_pelvis_yaw[0].item()))
    except (AttributeError, ValueError):
        diagnostic["s2_reference_pelvis_pos_w"].append(
            np.full(3, np.nan, dtype=np.float32)
        )
        diagnostic["s2_reference_pelvis_yaw"].append(np.nan)
        diagnostic["s2_actual_pelvis_pos_w"].append(
            np.full(3, np.nan, dtype=np.float32)
        )
        diagnostic["s2_actual_pelvis_yaw"].append(np.nan)

    if isinstance(s2_event_id, torch.Tensor) and hasattr(command, "s2_contact_reference_foot_pose_w"):
        reference_foot_pos, reference_foot_yaw = command.s2_contact_reference_foot_pose_w()
        ball_offset_reference_foot = quat_apply_inverse(
            reference_foot_yaw,
            soccer_ball.data.root_pos_w[:, :3] - reference_foot_pos,
        )
        def _s2_proximity_geometry(
            ball_offset: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            physical_distance = torch.linalg.vector_norm(ball_offset, dim=-1)
            reach_error = torch.relu(physical_distance - 0.25)
            signed_lateral = torch.where(
                s2_expected_side == 0, ball_offset[:, 1], -ball_offset[:, 1]
            )
            side_error = torch.relu(0.04 - signed_lateral)
            target_distance = torch.sqrt(reach_error.square() + side_error.square())
            return target_distance, physical_distance

        reference_region_distance, _ = _s2_proximity_geometry(
            ball_offset_reference_foot
        )
        if hasattr(command, "s2_contact_reference_anchor_target_w"):
            reference_contact_anchor_w, reference_contact_anchor_valid = (
                command.s2_contact_reference_anchor_target_w(s2_event_id)
            )
            diagnostic["s2_reference_contact_anchor_w"].append(
                _cpu(reference_contact_anchor_w)
                if bool(reference_contact_anchor_valid[0].item())
                else np.full(3, np.nan, dtype=np.float32)
            )
        else:
            diagnostic["s2_reference_contact_anchor_w"].append(
                np.full(3, np.nan, dtype=np.float32)
            )
        if expected_foot_value in (0, 1):
            foot_name = "right_ankle_roll_link" if expected_foot_value == 1 else "left_ankle_roll_link"
            actual_foot_index = command.cfg.body_names.index(foot_name)
            actual_foot_pos = command.robot_body_pos_w[:, actual_foot_index]
            actual_foot_yaw = yaw_quat(command.robot_body_quat_w[:, actual_foot_index])
            ball_offset_actual_foot = quat_apply_inverse(
                actual_foot_yaw,
                soccer_ball.data.root_pos_w[:, :3] - actual_foot_pos,
            )
            (
                actual_region_distance,
                actual_physical_distance,
            ) = _s2_proximity_geometry(ball_offset_actual_foot)
            actual_yaw = _world_quat_to_rpy(actual_foot_yaw)[:, 2]
            diagnostic["s2_actual_foot_pos_w"].append(_cpu(actual_foot_pos))
            diagnostic["s2_actual_foot_yaw"].append(float(actual_yaw[0].item()))
            diagnostic["s2_ball_offset_actual_foot"].append(_cpu(ball_offset_actual_foot))
            diagnostic["s2_target_region_distance"].append(float(actual_region_distance[0].item()))
            diagnostic["s2_physical_foot_ball_distance"].append(
                float(actual_physical_distance[0].item())
            )
            contact_anchor_w, contact_anchor_valid = command.s2_contact_anchor_target_w(
                s2_event_id
            )
            anchor_distance_xy = torch.linalg.vector_norm(
                soccer_ball.data.root_pos_w[:, :2] - contact_anchor_w[:, :2], dim=-1
            )
            anchor_offset_actual_foot_yaw = quat_apply_inverse(
                actual_foot_yaw, contact_anchor_w - actual_foot_pos
            )
            actual_foot_quat = command.robot_body_quat_w[:, actual_foot_index]
            anchor_offset_actual_foot_full = quat_apply_inverse(
                actual_foot_quat, contact_anchor_w - actual_foot_pos
            )
            anchor_valid = bool(contact_anchor_valid[0].item())
            diagnostic["s2_contact_anchor_w"].append(
                _cpu(contact_anchor_w) if anchor_valid else np.full(3, np.nan, dtype=np.float32)
            )
            diagnostic["s2_contact_anchor_distance_xy"].append(
                float(anchor_distance_xy[0].item()) if anchor_valid else np.nan
            )
            diagnostic["s2_contact_anchor_offset_actual_foot_yaw"].append(
                _cpu(anchor_offset_actual_foot_yaw)
                if anchor_valid else np.full(3, np.nan, dtype=np.float32)
            )
            diagnostic["s2_contact_anchor_offset_actual_foot_full"].append(
                _cpu(anchor_offset_actual_foot_full)
                if anchor_valid else np.full(3, np.nan, dtype=np.float32)
            )
        else:
            diagnostic["s2_actual_foot_pos_w"].append(np.full(3, np.nan, dtype=np.float32))
            diagnostic["s2_actual_foot_yaw"].append(np.nan)
            diagnostic["s2_ball_offset_actual_foot"].append(np.full(3, np.nan, dtype=np.float32))
            diagnostic["s2_target_region_distance"].append(np.nan)
            diagnostic["s2_physical_foot_ball_distance"].append(np.nan)
            diagnostic["s2_contact_anchor_w"].append(np.full(3, np.nan, dtype=np.float32))
            diagnostic["s2_contact_anchor_distance_xy"].append(np.nan)
            diagnostic["s2_contact_anchor_offset_actual_foot_yaw"].append(
                np.full(3, np.nan, dtype=np.float32)
            )
            diagnostic["s2_contact_anchor_offset_actual_foot_full"].append(
                np.full(3, np.nan, dtype=np.float32)
            )
        diagnostic["s2_contact_window"].append(bool(s2_window[0].item()))
        diagnostic["s2_contact_event_id"].append(int(s2_event_id[0].item()))
        diagnostic["s2_contact_event_frame"].append(int(s2_event_frame[0].item()))
        diagnostic["s2_expected_foot"].append(expected_foot_value)
        diagnostic["s2_expected_side"].append(int(s2_expected_side[0].item()))
        diagnostic["s2_expected_surface"].append(
            int(s2_expected_surface[0].item())
            if isinstance(s2_expected_surface, torch.Tensor) else -1
        )
        diagnostic["s2_reference_foot_pos_w"].append(_cpu(reference_foot_pos))
        reference_yaw = _world_quat_to_rpy(reference_foot_yaw)[:, 2]
        diagnostic["s2_reference_foot_yaw"].append(float(reference_yaw[0].item()))
        diagnostic["s2_ball_offset_reference_foot"].append(_cpu(ball_offset_reference_foot))
        diagnostic["s2_reference_target_region_distance"].append(
            float(reference_region_distance[0].item())
        )
    else:
        diagnostic["s2_contact_window"].append(False)
        diagnostic["s2_contact_event_id"].append(-1)
        diagnostic["s2_contact_event_frame"].append(-1)
        diagnostic["s2_expected_foot"].append(-1)
        diagnostic["s2_expected_side"].append(-1)
        diagnostic["s2_expected_surface"].append(-1)
        diagnostic["s2_reference_foot_pos_w"].append(np.full(3, np.nan, dtype=np.float32))
        diagnostic["s2_reference_foot_yaw"].append(np.nan)
        diagnostic["s2_ball_offset_reference_foot"].append(np.full(3, np.nan, dtype=np.float32))
        diagnostic["s2_reference_target_region_distance"].append(np.nan)
        diagnostic["s2_reference_contact_anchor_w"].append(
            np.full(3, np.nan, dtype=np.float32)
        )
        diagnostic["s2_actual_foot_pos_w"].append(np.full(3, np.nan, dtype=np.float32))
        diagnostic["s2_actual_foot_yaw"].append(np.nan)
        diagnostic["s2_ball_offset_actual_foot"].append(np.full(3, np.nan, dtype=np.float32))
        diagnostic["s2_target_region_distance"].append(np.nan)
        diagnostic["s2_physical_foot_ball_distance"].append(np.nan)
        diagnostic["s2_contact_anchor_w"].append(np.full(3, np.nan, dtype=np.float32))
        diagnostic["s2_contact_anchor_distance_xy"].append(np.nan)
        diagnostic["s2_contact_anchor_offset_actual_foot_yaw"].append(
            np.full(3, np.nan, dtype=np.float32)
        )
        diagnostic["s2_contact_anchor_offset_actual_foot_full"].append(
            np.full(3, np.nan, dtype=np.float32)
        )
    flow_available = getattr(command, "motion_has_dribble_cg_flow_label", None)
    flow_valid = getattr(command, "dribble_cg_flow_valid_ref", None)
    flow_anchor = getattr(command, "dribble_cg_flow_anchor_frame_ref", None)
    flow_direction = getattr(command, "dribble_cg_flow_dir_local_ref", None)
    flow_distance = getattr(command, "dribble_cg_flow_distance_ref", None)
    flow_duration = getattr(command, "dribble_cg_flow_duration_ref", None)
    flow_available_0 = isinstance(flow_available, torch.Tensor) and bool(flow_available[0].item())
    flow_valid_0 = isinstance(flow_valid, torch.Tensor) and bool(flow_valid[0].item())
    flow_distance_0 = float(flow_distance[0].item()) if isinstance(flow_distance, torch.Tensor) else -1.0
    flow_duration_0 = float(flow_duration[0].item()) if isinstance(flow_duration, torch.Tensor) else -1.0
    diagnostic["cg_flow_label_available"].append(flow_available_0)
    diagnostic["cg_flow_valid"].append(flow_valid_0)
    diagnostic["cg_flow_anchor_frame"].append(
        int(flow_anchor[0].item()) if isinstance(flow_anchor, torch.Tensor) else -1
    )
    diagnostic["cg_flow_direction_local"].append(
        _cpu(flow_direction) if isinstance(flow_direction, torch.Tensor) else np.full(2, np.nan, dtype=np.float32)
    )
    diagnostic["cg_flow_distance"].append(flow_distance_0)
    diagnostic["cg_flow_duration"].append(flow_duration_0)
    diagnostic["cg_flow_nominal_speed"].append(
        flow_distance_0 / flow_duration_0 if flow_valid_0 and flow_duration_0 > 0.0 else np.nan
    )
    diagnostic["cg_flow_latched"].append(False)
    diagnostic["cg_flow_release_active"].append(False)
    diagnostic["cg_flow_target_direction_world"].append(np.full(2, np.nan, dtype=np.float32))
    diagnostic["cg_flow_parallel_speed"].append(np.nan)
    diagnostic["cg_flow_lateral_speed"].append(np.nan)
    diagnostic["cg_flow_release_reward"].append(np.nan)
    diagnostic["cg_flow_progress"].append(np.nan)
    diagnostic["cg_flow_progress_rate"].append(np.nan)
    diagnostic["cg_flow_lateral_offset"].append(np.nan)
    diagnostic["cg_flow_progress_reward"].append(np.nan)
    cg_gate_pass = not (contact_settings["cg_gated"] and has_cg_label) or ref_contact
    diagnostic["contact_cg_gate_pass"].append(cg_gate_pass)
    diagnostic["contact_cg_eligible"].append(
        bool(contact["legal_touch"][0].item()) and cg_gate_pass
    )
    ball_vel_xy = soccer_ball.data.root_lin_vel_w[0, :2]
    diagnostic["ball_xy_speed"].append(float(torch.norm(ball_vel_xy).item()))
    if bool(heading_valid[0].item()):
        command_dir = torch.stack((torch.cos(active_heading[0]), torch.sin(active_heading[0])))
        ball_forward_speed = float(torch.dot(ball_vel_xy, command_dir).item())
    else:
        ball_forward_speed = np.nan
    diagnostic["ball_command_forward_speed"].append(ball_forward_speed)
    diagnostic["pelvis_xy_speed"].append(float(torch.norm(robot.data.body_lin_vel_w[0, pelvis_id, :2]).item()))
    foot_ref_ids = [command.cfg.body_names.index(name) for name in _FOOT_DIAGNOSTIC_BODY_NAMES]
    foot_error = torch.norm(
        command.robot_body_pos_w[0, foot_ref_ids] - command.body_pos_relative_w[0, foot_ref_ids], dim=-1
    ).mean()
    diagnostic["foot_reference_position_error"].append(float(foot_error.item()))
    is_local_twist = str(getattr(command.cfg, "locomotion_command_frame", "world")) == "pelvis_local"
    if bool(heading_valid[0].item()) and not is_local_twist:
        heading_error = torch.atan2(
            torch.sin(active_heading[0] - diagnostic["pelvis_yaw"][-1]),
            torch.cos(active_heading[0] - diagnostic["pelvis_yaw"][-1]),
        )
        diagnostic["heading_error"].append(float(heading_error.item()))
    else:
        diagnostic["heading_error"].append(np.nan)
    no_contact_count = getattr(base_env, "_dribbling_no_contact_count", None)
    diagnostic["no_contact_count"].append(
        np.nan if no_contact_count is None else float(no_contact_count[0].item())
    )
    no_contact_task_active = getattr(base_env, "_dribbling_no_contact_task_active", None)
    diagnostic["no_contact_task_active"].append(
        False if no_contact_task_active is None else bool(no_contact_task_active[0].item())
    )
    no_contact_recovery = getattr(base_env, "_dribbling_no_contact_recovery_active", None)
    diagnostic["no_contact_recovery_active"].append(
        False if no_contact_recovery is None else bool(no_contact_recovery[0].item())
    )
    proximity_recovery = getattr(base_env, "_dribbling_no_contact_proximity_recovery_active", None)
    diagnostic["no_contact_proximity_recovery_active"].append(
        False if proximity_recovery is None else bool(proximity_recovery[0].item())
    )
    relative_speed = getattr(base_env, "_dribbling_no_contact_relative_speed", None)
    diagnostic["no_contact_relative_speed"].append(
        np.nan if relative_speed is None else float(relative_speed[0].item())
    )
    for key, attr in (
        ("idle_active", "_dribbling_idle_active"),
        ("idle_pelvis_speed", "_dribbling_idle_pelvis_speed"),
        ("idle_pelvis_angular_speed", "_dribbling_idle_pelvis_angular_speed"),
        ("stop_active", "_dribbling_stop_active"),
        ("stop_settled", "_dribbling_stop_settled"),
        ("stop_success", "_dribbling_stop_success"),
        ("stop_settle_elapsed_s", "_dribbling_stop_settle_elapsed_s"),
        ("stop_pelvis_speed", "_dribbling_stop_pelvis_speed"),
        ("stop_pelvis_angular_speed", "_dribbling_stop_pelvis_angular_speed"),
        ("stop_ball_speed", "_dribbling_stop_ball_speed"),
        ("stop_forward_offset", "_dribbling_stop_forward_offset"),
        ("stop_lateral_offset", "_dribbling_stop_lateral_offset"),
        ("stop_position_score", "_dribbling_stop_position_score"),
        ("stop_speed_score", "_dribbling_stop_speed_score"),
    ):
        value = getattr(base_env, attr, None)
        if value is None:
            diagnostic[key].append(False if key.endswith(("active", "settled", "success")) else np.nan)
        elif key.endswith(("active", "settled", "success")):
            diagnostic[key].append(bool(value[0].item()))
        else:
            diagnostic[key].append(float(value[0].item()))
    # Filled with the reward returned by the immediately following env.step().
    diagnostic["step_reward"].append(np.nan)
    return True


def _append_upper_body_manifold_diagnostic(diagnostic: dict, env, dones: torch.Tensor) -> None:
    """Record post-step arm projection and trunk target/actuator telemetry."""
    base_env = _resolve_base_env(env)
    action_term = base_env.action_manager.get_term("joint_pos")
    snapshot = getattr(action_term, "diagnostic_snapshot", {})
    snapshot_available = isinstance(snapshot, dict) and bool(snapshot)
    post_step_state_valid = not bool(dones[0].item())

    def _env0(name: str, fallback_shape: tuple[int, ...]) -> np.ndarray:
        value = snapshot.get(name, getattr(action_term, name, None)) if snapshot_available else getattr(action_term, name, None)
        if not isinstance(value, torch.Tensor):
            return np.full(fallback_shape, np.nan, dtype=np.float32)
        return value[0].detach().cpu().numpy().copy()

    def _tensor(name: str):
        if snapshot_available and isinstance(snapshot.get(name), torch.Tensor):
            return snapshot[name]
        return getattr(action_term, name, None)

    upper_dim = len(_ARM_DIAGNOSTIC_JOINT_NAMES)
    diagnostic["manifold_raw_upper_target"].append(
        _env0("manifold_raw_upper_target", (upper_dim,))
    )
    diagnostic["manifold_reference_upper_target"].append(
        _env0("manifold_reference_upper_target", (upper_dim,))
    )
    diagnostic["manifold_constrained_upper_target"].append(
        _env0("manifold_constrained_upper_target", (upper_dim,))
    )
    diagnostic["manifold_projected_upper_target"].append(
        _env0("manifold_projected_upper_target", (upper_dim,))
    )
    diagnostic["manifold_joint_limited_upper_target"].append(
        _env0("manifold_joint_limited_upper_target", (upper_dim,))
    )
    diagnostic["manifold_executed_upper_target"].append(
        _env0("manifold_executed_upper_target", (upper_dim,))
    )
    diagnostic["manifold_latent"].append(_env0("manifold_latent", (0,)))
    diagnostic["manifold_projection_error"].append(
        float(_env0("manifold_projection_error", ()).item())
    )
    diagnostic["manifold_projection_error_after_reference_constraint"].append(
        float(_env0("manifold_projection_error_after_reference_constraint", ()).item())
    )
    diagnostic["manifold_nullspace_residual"].append(
        float(_env0("manifold_nullspace_residual", ()).item())
    )
    diagnostic["manifold_latent_clip_fraction"].append(
        float(_env0("manifold_latent_clip_fraction", ()).item())
    )
    diagnostic["manifold_reference_overflow"].append(
        _env0("manifold_reference_overflow", (upper_dim,))
    )
    diagnostic["manifold_reference_clamp_fraction"].append(
        float(_env0("manifold_reference_clamp_fraction", ()).item())
    )
    diagnostic["manifold_joint_limit_clamp_fraction"].append(
        float(_env0("manifold_joint_limit_clamp_fraction", ()).item())
    )
    diagnostic["manifold_filter_lag"].append(
        float(_env0("manifold_filter_lag", ()).item())
    )
    diagnostic["trunk_pitch_raw_target"].append(
        float(_env0("trunk_pitch_raw_target", ()).item())
    )
    diagnostic["trunk_pitch_reference_target"].append(
        float(_env0("trunk_pitch_reference_target", ()).item())
    )
    diagnostic["trunk_pitch_soft_target"].append(
        float(_env0("trunk_pitch_soft_target", ()).item())
    )
    diagnostic["trunk_pitch_filtered_target"].append(
        float(_env0("trunk_pitch_filtered_target", ()).item())
    )
    diagnostic["trunk_pitch_reference_overflow"].append(
        float(_env0("trunk_pitch_reference_overflow", ()).item())
    )
    diagnostic["trunk_pitch_turn_relaxation"].append(
        float(_env0("trunk_pitch_turn_relaxation", ()).item())
    )
    diagnostic["trunk_pitch_active_lower_deviation"].append(
        float(_env0("trunk_pitch_active_lower_deviation", ()).item())
    )
    diagnostic["trunk_pitch_active_upper_deviation"].append(
        float(_env0("trunk_pitch_active_upper_deviation", ()).item())
    )
    diagnostic["trunk_pitch_active_cutoff_frequency_hz"].append(
        float(_env0("trunk_pitch_active_cutoff_frequency_hz", ()).item())
    )

    command = base_env.command_manager.get_term("motion")
    robot = base_env.scene[command.cfg.asset_name]
    trunk_ids = diagnostic["trunk_joint_ids"]
    trunk_action_ids = diagnostic["trunk_action_ids"]
    trunk_full_action_ids = diagnostic["trunk_full_action_ids"]
    effective_raw_actions = _tensor("effective_raw_actions")
    if isinstance(effective_raw_actions, torch.Tensor):
        diagnostic["effective_action"].append(
            effective_raw_actions[0, diagnostic["action_ids"]].detach().cpu().numpy().copy()
        )
        diagnostic["trunk_effective_action"].append(
            effective_raw_actions[0, trunk_full_action_ids].detach().cpu().numpy().copy()
        )
    else:
        diagnostic["effective_action"].append(
            np.full((len(diagnostic["action_ids"]),), np.nan, dtype=np.float32)
        )
        diagnostic["trunk_effective_action"].append(
            np.full((len(_TRUNK_DIAGNOSTIC_JOINT_NAMES),), np.nan, dtype=np.float32)
        )
    diagnostic["action_snapshot_available"].append(snapshot_available)
    diagnostic["post_step_state_valid"].append(post_step_state_valid)
    processed_actions = _tensor("executed_joint_targets")
    if processed_actions is None:
        processed_actions = getattr(action_term, "_processed_actions", None)

    if isinstance(processed_actions, torch.Tensor):
        trunk_target = processed_actions[0, trunk_full_action_ids]
    else:
        trunk_target = torch.full(
            (len(_TRUNK_DIAGNOSTIC_JOINT_NAMES),),
            float("nan"),
            dtype=robot.data.joint_pos.dtype,
            device=base_env.device,
    )
    trunk_post_step_pos = robot.data.joint_pos[0, trunk_ids]
    # This row's target was formed from the reference recorded before
    # env.step(). The command style phase may already have advanced now.
    trunk_reference = torch.as_tensor(
        diagnostic["trunk_reference_joint_pos"][-1],
        dtype=trunk_target.dtype,
        device=base_env.device,
    )
    soft_limits = robot.data.soft_joint_pos_limits[0, trunk_ids]
    trunk_actual_limit_margin = torch.minimum(
        trunk_post_step_pos - soft_limits[:, 0], soft_limits[:, 1] - trunk_post_step_pos
    )
    trunk_target_limit_margin = torch.minimum(
        trunk_target - soft_limits[:, 0], soft_limits[:, 1] - trunk_target
    )
    computed_torque = getattr(robot.data, "computed_torque", None)
    if isinstance(computed_torque, torch.Tensor):
        trunk_computed_torque = computed_torque[0, trunk_ids]
    else:
        trunk_computed_torque = torch.full_like(trunk_target, float("nan"))
    applied_torque = getattr(robot.data, "applied_torque", None)
    if isinstance(applied_torque, torch.Tensor):
        trunk_applied_torque = applied_torque[0, trunk_ids]
    else:
        trunk_applied_torque = torch.full_like(trunk_target, float("nan"))
    effort_limit = torch.as_tensor(_TRUNK_EFFORT_LIMITS, dtype=trunk_target.dtype, device=base_env.device)
    computed_effort_utilization = torch.abs(trunk_computed_torque) / effort_limit
    effort_utilization = torch.abs(trunk_applied_torque) / effort_limit
    if not post_step_state_valid:
        # Isaac Lab may already have reset the articulation. The saved action
        # target still belongs to this transition, but live body/joint/torque
        # data now belongs to the next episode and must not be mixed in.
        trunk_post_step_pos = torch.full_like(trunk_post_step_pos, torch.nan)
        trunk_actual_limit_margin = torch.full_like(trunk_actual_limit_margin, torch.nan)
        trunk_computed_torque = torch.full_like(trunk_computed_torque, torch.nan)
        trunk_applied_torque = torch.full_like(trunk_applied_torque, torch.nan)
        computed_effort_utilization = torch.full_like(computed_effort_utilization, torch.nan)
        effort_utilization = torch.full_like(effort_utilization, torch.nan)

    def _trunk_cpu(value: torch.Tensor) -> np.ndarray:
        return value.detach().cpu().numpy().copy()

    diagnostic["trunk_processed_joint_target"].append(_trunk_cpu(trunk_target))
    diagnostic["trunk_target_minus_reference"].append(_trunk_cpu(trunk_target - trunk_reference))
    diagnostic["trunk_post_step_actual_joint_pos"].append(_trunk_cpu(trunk_post_step_pos))
    diagnostic["trunk_post_step_target_error"].append(_trunk_cpu(trunk_post_step_pos - trunk_target))
    diagnostic["trunk_soft_joint_pos_limits"].append(_trunk_cpu(soft_limits))
    diagnostic["trunk_actual_limit_margin"].append(_trunk_cpu(trunk_actual_limit_margin))
    diagnostic["trunk_target_limit_margin"].append(_trunk_cpu(trunk_target_limit_margin))
    diagnostic["trunk_computed_torque"].append(_trunk_cpu(trunk_computed_torque))
    diagnostic["trunk_applied_torque"].append(_trunk_cpu(trunk_applied_torque))
    diagnostic["trunk_effort_limit"].append(_trunk_cpu(effort_limit))
    diagnostic["trunk_computed_effort_utilization"].append(_trunk_cpu(computed_effort_utilization))
    diagnostic["trunk_effort_utilization"].append(_trunk_cpu(effort_utilization))
    effort_saturated = (
        effort_utilization >= 0.98
        if post_step_state_valid
        else torch.full_like(effort_utilization, torch.nan)
    )
    diagnostic["trunk_effort_saturated"].append(_trunk_cpu(effort_saturated))


def _complete_cg_flow_diagnostic(diagnostic: dict, env) -> None:
    """Attach post-transition S2 release and legacy flow telemetry."""
    base_env = _resolve_base_env(env)
    s2_telemetry = getattr(base_env, "_dribbling_s2_release_telemetry", None)
    if isinstance(s2_telemetry, dict):
        def _s2_scalar(name: str) -> float:
            value = s2_telemetry.get(name)
            if not isinstance(value, torch.Tensor) or value.numel() == 0:
                return np.nan
            return float(value[0].item())

        def _s2_bool(name: str) -> bool:
            value = s2_telemetry.get(name)
            return (
                isinstance(value, torch.Tensor)
                and value.numel() > 0
                and bool(value[0].item())
            )

        def _s2_int(name: str, default: int = -1) -> int:
            value = s2_telemetry.get(name)
            if not isinstance(value, torch.Tensor) or value.numel() == 0:
                return int(default)
            return int(value[0].item())

        for diagnostic_name, telemetry_name in (
            ("s2_reward_evaluation_frame", "evaluation_frame"),
            ("s2_approach_progress", "approach_progress"),
            ("s2_coarse_contact_approach", "coarse_contact_approach"),
            ("s2_immediate_contact", "immediate_contact"),
            ("s2_valid_contact_event", "valid_contact_event"),
            (
                "s2_ordered_transition_completion",
                "valid_contact_event",
            ),
            ("s2_contact_release_off_count", "contact_release_off_count"),
            ("s2_contact_claimed_region_distance", "contact_claimed_region_distance"),
            ("s2_contact_physical_episode", "contact_physical_episode"),
            ("s2_contact_claimed_episode", "contact_claimed_episode"),
            ("s2_contact_claimed_event", "contact_claimed_event"),
            ("s2_contact_region_distance", "contact_region_distance"),
            ("s2_contact_region_nearest_point_id", "nearest_region_point_id"),
            ("s2_contact_region_nearest_timing", "nearest_region_point_timing"),
            ("s2_surface_style", "surface_style"),
            ("s2_release_actual_touch_frame", "release_actual_touch_frame"),
            ("s2_release_remaining_time", "release_remaining_time"),
            ("s2_release_target_distance", "release_target_distance"),
            ("s2_release_nominal_speed", "release_nominal_speed"),
            ("s2_release_parallel_speed", "release_parallel_speed"),
            ("s2_release_lateral_speed", "release_lateral_speed"),
            ("s2_release_speed_ratio", "release_speed_ratio"),
            ("s2_release_separation_score", "release_separation_score"),
            ("s2_release_score", "release_score"),
            ("s2_locked_next_anchor_distance", "locked_next_anchor_distance"),
            (
                "s2_locked_next_region_min_distance",
                "locked_next_region_min_distance",
            ),
            (
                "s2_locked_euclidean_positive_progress",
                "locked_euclidean_positive_progress",
            ),
            ("s2_quality_streak", "quality_streak"),
            (
                "s2_terminal_transition_start_distance",
                "terminal_transition_start_distance",
            ),
            (
                "s2_terminal_transition_end_distance",
                "terminal_transition_end_distance",
            ),
            (
                "s2_terminal_transition_net_progress",
                "terminal_transition_net_progress",
            ),
            (
                "s2_terminal_window_position_outcome",
                "terminal_window_position_outcome",
            ),
            ("s2_window_outcome_source_event_id", "window_outcome_source_event_id"),
            ("s2_window_outcome_target_event_id", "window_outcome_target_event_id"),
            ("s2_window_outcome_candidate_frame", "window_outcome_candidate_frame"),
            ("s2_window_outcome_start_distance", "window_outcome_start_distance"),
            ("s2_window_outcome_score_sum", "window_outcome_score_sum"),
            ("s2_window_outcome_sample_count", "window_outcome_sample_count"),
            ("s2_window_outcome_score_average", "window_outcome_score_average"),
            ("s2_window_outcome_score_end", "window_outcome_score_end"),
            ("s2_window_outcome_settlement_quality", "window_outcome_settlement_quality"),
            ("s2_window_outcome_settlement_pulse", "window_outcome_settlement_pulse"),
            ("s2_window_outcome_pulse", "window_outcome_pulse"),
            ("s2_fixed_target_shift_distance", "fixed_target_shift_distance"),
            ("s2_fixed_window_distance", "fixed_window_distance"),
            ("s2_live_window_distance", "live_window_distance"),
            ("s2_fixed_window_min_distance", "fixed_window_min_distance"),
            ("s2_live_window_min_distance", "live_window_min_distance"),
            ("s2_fixed_window_min_frame", "fixed_window_min_frame"),
            ("s2_terminal_progress_raw_rate", "terminal_progress_raw_rate"),
            (
                "s2_terminal_progress_clipped_rate",
                "terminal_progress_clipped_rate",
            ),
            (
                "s2_terminal_progress_positive_rate",
                "terminal_progress_positive_rate",
            ),
            (
                "s2_terminal_progress_negative_rate",
                "terminal_progress_negative_rate",
            ),
            (
                "s2_acquisition_history_steps_before_contact",
                "acquisition_history_steps_before_contact",
            ),
        ):
            diagnostic[diagnostic_name][-1] = _s2_scalar(telemetry_name)
        for diagnostic_name, telemetry_name in (
            ("s2_contact_has_force", "contact_has_force"),
            ("s2_contact_previous_has_force", "contact_previous_has_force"),
            ("s2_contact_new_physical_onset", "contact_new_physical_onset"),
            ("s2_contact_event_claim", "contact_event_claim"),
            (
                "s2_contact_event_owns_physical",
                "contact_event_owns_physical",
            ),
            ("s2_contact_correct_foot", "contact_correct_foot"),
            ("s2_contact_side_correct", "contact_side_correct"),
            (
                "s2_contact_coarse_longitudinal_pass",
                "contact_coarse_longitudinal_pass",
            ),
            (
                "s2_contact_anchor_valid_touch",
                "contact_anchor_valid_touch",
            ),
            ("s2_contact_rearmed", "contact_rearmed"),
            ("s2_contact_release_confirmed", "contact_release_confirmed"),
            ("s2_contact_left_claimed_region", "contact_left_claimed_region"),
            ("s2_contact_region_valid", "contact_region_valid"),
            ("s2_release_pending", "release_pending"),
            ("s2_release_quality_pass", "release_quality_pass"),
            ("s2_locked_next_region_active", "locked_next_region_active"),
            (
                "s2_locked_next_region_arrival_pulse",
                "locked_next_region_arrival_pulse",
            ),
            ("s2_locked_next_region_arrival", "locked_next_region_arrival"),
            (
                "s2_second_touch_conditional_success",
                "second_touch_conditional_success",
            ),
            ("s2_terminal_transition_valid", "terminal_transition_valid"),
            ("s2_terminal_transition_success", "terminal_transition_success"),
            ("s2_terminal_release", "terminal_release"),
            ("s2_window_outcome_active", "window_outcome_active"),
            ("s2_window_outcome_emitted", "window_outcome_emitted"),
            ("s2_window_outcome_aborted_ball_lost", "window_outcome_aborted_ball_lost"),
            ("s2_window_outcome_aborted_other", "window_outcome_aborted_other"),
            (
                "s2_terminal_transition_window",
                "terminal_transition_window",
            ),
        ):
            diagnostic[diagnostic_name][-1] = _s2_bool(telemetry_name)
        diagnostic["s2_contact_closest_body"][-1] = _s2_int(
            "contact_closest_body"
        )
        for diagnostic_name, telemetry_name in (
            ("s2_release_next_anchor_w", "release_next_anchor_w"),
            ("s2_release_target_direction", "release_target_direction"),
            ("s2_ball_in_actual_foot_full", "ball_in_actual_foot_full"),
            ("s2_locked_live_target_w", "locked_live_target_w"),
            ("s2_locked_fixed_target_w", "locked_fixed_target_w"),
            ("s2_fixed_target_shift_xy", "fixed_target_shift_xy"),
        ):
            value = s2_telemetry.get(telemetry_name)
            if isinstance(value, torch.Tensor) and value.numel() > 0:
                diagnostic[diagnostic_name][-1] = (
                    value[0].detach().cpu().numpy().copy()
                )

    command = base_env.command_manager.get_term("motion")
    for diagnostic_name, metric_name, as_bool in (
        ("s2_t2_region_progress", "s2_t2_region_progress", False),
        ("s2_t2_region_arrival", "s2_t2_region_arrival", False),
        ("s2_t2_region_distance", "s2_t2_region_distance", False),
        ("s2_t2_success", "s2_t2_success", True),
        ("s2_t2_evaluated", "s2_terminal_transition_valid", True),
    ):
        value = command.metrics.get(metric_name)
        if isinstance(value, torch.Tensor) and value.numel() > 0:
            diagnostic[diagnostic_name][-1] = (
                bool(value[0].item()) if as_bool else float(value[0].item())
            )

    telemetry = getattr(base_env, "_dribbling_cg_flow_telemetry", None)
    if not isinstance(telemetry, dict):
        return

    def _scalar(name: str) -> float:
        value = telemetry.get(name)
        if not isinstance(value, torch.Tensor) or value.numel() == 0:
            return np.nan
        return float(value[0].item())

    def _boolean(name: str) -> bool:
        value = telemetry.get(name)
        return isinstance(value, torch.Tensor) and value.numel() > 0 and bool(value[0].item())

    direction = telemetry.get("target_direction_world")
    diagnostic["cg_flow_latched"][-1] = _boolean("latched")
    diagnostic["cg_flow_release_active"][-1] = _boolean("release_active")
    if isinstance(direction, torch.Tensor) and direction.numel() >= 2:
        diagnostic["cg_flow_target_direction_world"][-1] = (
            direction[0].detach().cpu().numpy().copy()
        )
    diagnostic["cg_flow_parallel_speed"][-1] = _scalar("parallel_speed")
    diagnostic["cg_flow_lateral_speed"][-1] = _scalar("lateral_speed")
    diagnostic["cg_flow_release_reward"][-1] = _scalar("release_reward")
    diagnostic["cg_flow_progress"][-1] = _scalar("progress")
    diagnostic["cg_flow_progress_rate"][-1] = _scalar("progress_rate")
    diagnostic["cg_flow_lateral_offset"][-1] = _scalar("lateral_offset")
    diagnostic["cg_flow_progress_reward"][-1] = _scalar("progress_reward")


def _summarize_first_two_s2_touches(arrays: dict[str, np.ndarray]) -> None:
    """Print outcome-based O1/O2 contact quality without changing the task.

    A valid-touch pulse identifies the accepted collision.  For each pulse we
    integrate the measured ball force over the contiguous collision, inspect
    pre-impact foot/ball relative velocity, and project the following 0.16 and
    0.20 seconds of ball displacement onto the latched next-anchor direction.
    """
    if not bool(
        np.asarray(arrays.get("filtered_contact_available", False)).reshape(-1)[0]
    ):
        print("[INFO] O1/O2 filtered-contact quality: UNAVAILABLE on CPU backend.")
        return
    required = (
        "s2_immediate_contact",
        "s2_contact_event_id",
        "ball_contact",
        "contact_net_force_w",
        "ball_pos_w",
        "ball_minus_expected_foot_lin_vel_w",
        "s2_ball_offset_actual_foot",
        "s2_release_target_direction",
        "s2_physical_foot_ball_distance",
        "done",
    )
    if any(name not in arrays for name in required):
        return
    dt = float(np.asarray(arrays.get("step_dt", 0.02)).reshape(-1)[0])
    recorded_steps = np.asarray(arrays.get("step", []), dtype=np.int64)
    stride = (
        max(1, int(round(float(np.median(np.diff(recorded_steps))))))
        if recorded_steps.size > 1 else 1
    )
    sample_dt = dt * stride
    episode_id = np.cumsum(np.r_[0, arrays["done"][:-1].astype(np.int64)])
    pulses = np.flatnonzero(arrays["s2_immediate_contact"] > 0.5)
    eligible_episodes: list[int] = []
    o1_episode: dict[int, bool] = {}
    o2_episode: dict[int, bool] = {}
    fall_episode: dict[int, bool] = {}
    term_reasons = np.asarray(
        arrays.get("termination_reason", np.full(episode_id.size, ""))
    ).astype(str)
    for ep in np.unique(episode_id):
        indices = np.flatnonzero(episode_id == ep)
        # Count only episodes for which the O1 labelled window was observed;
        # this excludes a truncated tail at the end of fixed-length playback.
        saw_o1 = bool(
            np.any(
                arrays["s2_contact_window"][indices].astype(bool)
                & (arrays["s2_contact_event_id"][indices] == 0)
            )
        )
        if not saw_o1:
            continue
        ep_pulses = pulses[episode_id[pulses] == ep]
        o1 = bool(np.any(arrays["s2_contact_event_id"][ep_pulses] == 0))
        o2 = bool(np.any(arrays["s2_contact_event_id"][ep_pulses] == 1))
        eligible_episodes.append(int(ep))
        o1_episode[int(ep)] = o1
        o2_episode[int(ep)] = o2
        fall_episode[int(ep)] = bool(
            np.any(
                np.char.find(term_reasons[indices], "anchor_pos_z") >= 0
            )
            or np.any(np.char.find(term_reasons[indices], "anchor_ori") >= 0)
        )
    n_episodes = len(eligible_episodes)
    n_o1 = sum(o1_episode.values())
    n_o1_o2 = sum(
        o1_episode[ep] and o2_episode[ep] for ep in eligible_episodes
    )
    p_o1 = n_o1 / n_episodes if n_episodes else np.nan
    p_o2_given_o1 = n_o1_o2 / n_o1 if n_o1 else np.nan
    p_c2 = n_o1_o2 / n_episodes if n_episodes else np.nan
    fall_rate = (
        sum(fall_episode.values()) / n_episodes if n_episodes else np.nan
    )
    print(
        "[INFO] C2 deterministic episode audit: "
        f"episodes={n_episodes}  P(O1)={p_o1:.3f}  "
        f"P(O2|O1)={p_o2_given_o1:.3f}  P(C2)={p_c2:.3f}  "
        f"fall_rate={fall_rate:.3f}"
    )
    rows: list[dict[str, float]] = []
    for pulse in pulses:
        event_id = int(arrays["s2_contact_event_id"][pulse])
        if event_id not in (0, 1):
            continue
        ep = episode_id[pulse]
        ep_indices = np.flatnonzero(episode_id == ep)
        ep_start, ep_end = int(ep_indices[0]), int(ep_indices[-1])

        # The archive is pre-step while immediate_contact is filled post-step;
        # include one neighboring sample when locating the collision run.
        contact_candidates = np.flatnonzero(
            arrays["ball_contact"][ep_start : ep_end + 1]
        ) + ep_start
        if contact_candidates.size:
            seed = int(contact_candidates[np.argmin(np.abs(contact_candidates - pulse))])
            left = seed
            right = seed
            while left > ep_start and bool(arrays["ball_contact"][left - 1]):
                left -= 1
            while right < ep_end and bool(arrays["ball_contact"][right + 1]):
                right += 1
            impulse_w = np.sum(arrays["contact_net_force_w"][left : right + 1], axis=0) * sample_dt
            pre_index = max(ep_start, left - 1)
        else:
            impulse_w = np.full(3, np.nan, dtype=np.float32)
            pre_index = max(ep_start, pulse - 1)

        direction = arrays["s2_release_target_direction"][pulse]
        if not np.all(np.isfinite(direction)) or np.linalg.norm(direction) < 1.0e-6:
            future = np.arange(pulse, min(ep_end + 1, pulse + 3))
            valid = future[np.all(np.isfinite(arrays["s2_release_target_direction"][future]), axis=1)]
            direction = (
                arrays["s2_release_target_direction"][valid[0]]
                if valid.size else np.full(2, np.nan, dtype=np.float32)
            )
        if np.all(np.isfinite(direction)):
            direction = direction / max(float(np.linalg.norm(direction)), 1.0e-6)
            lateral = np.asarray([-direction[1], direction[0]])
        else:
            lateral = np.full(2, np.nan, dtype=np.float32)

        row: dict[str, float] = {
            "event": float(event_id),
            "onset_frame": float(arrays["motion_frame"][pulse]),
            "evaluation_frame": float(
                np.asarray(arrays.get("s2_reward_evaluation_frame", arrays["motion_frame"]))[pulse]
            ),
            "contact_x": float(arrays["s2_ball_offset_actual_foot"][pulse, 0]),
            "contact_y": float(arrays["s2_ball_offset_actual_foot"][pulse, 1]),
            "contact_z": float(arrays["s2_ball_offset_actual_foot"][pulse, 2]),
            "side_margin": float(abs(arrays["s2_ball_offset_actual_foot"][pulse, 1])),
            "anchor_quality": float(
                np.asarray(arrays.get("s2_contact_anchor_quality", np.full(episode_id.size, np.nan)))[pulse]
            ),
            "v3_region_distance": float(
                np.asarray(arrays.get("s2_contact_region_distance", np.full(episode_id.size, np.nan)))[pulse]
            ),
            "v3_nearest_timing": float(
                np.asarray(arrays.get("s2_contact_region_nearest_timing", np.full(episode_id.size, np.nan)))[pulse]
            ),
            "foot_pos_error": float(
                np.asarray(arrays.get("foot_reference_position_error", np.full(episode_id.size, np.nan)))[pulse]
            ),
            "pre_rel_vx": float(arrays["ball_minus_expected_foot_lin_vel_w"][pre_index, 0]),
            "pre_rel_vy": float(arrays["ball_minus_expected_foot_lin_vel_w"][pre_index, 1]),
            "impulse": float(np.linalg.norm(impulse_w[:2])),
            "pre_rel_forward": float(
                np.dot(arrays["ball_minus_expected_foot_lin_vel_w"][pre_index, :2], direction)
            ),
            "pre_rel_lateral": float(
                np.dot(arrays["ball_minus_expected_foot_lin_vel_w"][pre_index, :2], lateral)
            ),
            "impulse_forward": float(np.dot(impulse_w[:2], direction)),
            "impulse_lateral": float(np.dot(impulse_w[:2], lateral)),
        }
        reference_quat = np.asarray(
            arrays.get("s2_reference_foot_yaw", np.full((episode_id.size, 4), np.nan))
        )[pulse]
        actual_quat = np.asarray(
            arrays.get("s2_actual_foot_yaw", np.full((episode_id.size, 4), np.nan))
        )[pulse]
        if reference_quat.shape == (4,) and actual_quat.shape == (4,) and np.all(
            np.isfinite(np.r_[reference_quat, actual_quat])
        ):
            quat_dot = float(np.clip(abs(np.dot(reference_quat, actual_quat)), 0.0, 1.0))
            row["foot_ori_error"] = float(2.0 * np.arccos(quat_dot))
        else:
            row["foot_ori_error"] = np.nan
        for horizon_s in (0.16, 0.20):
            horizon_samples = max(1, int(round(horizon_s / sample_dt)))
            end = min(ep_end, pulse + horizon_samples)
            displacement = arrays["ball_pos_w"][end, :2] - arrays["ball_pos_w"][pulse, :2]
            suffix = str(int(round(horizon_s * 1000.0)))
            row[f"forward_{suffix}"] = float(np.dot(displacement, direction))
            row[f"lateral_{suffix}"] = float(np.dot(displacement, lateral))
            row[f"separation_{suffix}"] = float(
                arrays["s2_physical_foot_ball_distance"][end]
            )
        rows.append(row)

    if not rows:
        print("[INFO] O1/O2 quality: no accepted first-two-touch pulses recorded.")
        return
    print("[INFO] O1/O2 ordered physical-contact diagnostics (mean over episodes):")
    for event_id, label in ((0, "O1"), (1, "O2")):
        selected = [row for row in rows if int(row["event"]) == event_id]
        if not selected:
            print(f"[INFO]   {label}: n=0")
            continue
        mean = lambda key: float(np.nanmean([row[key] for row in selected]))
        print(
            f"[INFO]   {label}: n={len(selected)}  "
            f"onset_frame={mean('onset_frame'):.2f}  "
            f"reward_frame={mean('evaluation_frame'):.2f}  "
            f"contact_foot_xyz=({mean('contact_x'):+.3f},{mean('contact_y'):+.3f},"
            f"{mean('contact_z'):+.3f}) m  side_margin={mean('side_margin'):.3f} m  "
            f"anchor_quality={mean('anchor_quality'):.3f}  "
            f"v3_distance={mean('v3_region_distance'):.3f} m  "
            f"v3_timing={mean('v3_nearest_timing'):+.2f} fr  "
            f"foot_ref_err=(pos={mean('foot_pos_error'):.3f} m, "
            f"ori={mean('foot_ori_error'):.3f} rad)  "
            f"pre_rel_v=(forward={mean('pre_rel_forward'):+.3f}, lateral={mean('pre_rel_lateral'):+.3f}) m/s  "
            f"impulse=(norm={mean('impulse'):.3f}, forward={mean('impulse_forward'):+.3f}, "
            f"lateral={mean('impulse_lateral'):+.3f}) N*s  "
            f"disp@160ms=(forward={mean('forward_160'):+.3f}, lateral={mean('lateral_160'):+.3f}) m  "
            f"separation@160ms={mean('separation_160'):.3f} m  "
            f"disp@200ms=(forward={mean('forward_200'):+.3f}, lateral={mean('lateral_200'):+.3f}) m  "
            f"separation@200ms={mean('separation_200'):.3f} m"
        )

    # Keep the two meanings of "anchor error" explicit.  Contact geometry is
    # reconstructed from the current actual foot; locked-target error is
    # measured against the fixed target created at O1.
    if "s2_locked_next_anchor_distance" in arrays:
        locked_min: list[float] = []
        contact_geometry_min: list[float] = []
        arrived: list[float] = []
        o2_valid: list[float] = []
        for ep in np.unique(episode_id):
            window = np.flatnonzero(
                (episode_id == ep)
                & arrays["s2_contact_window"].astype(bool)
                & (arrays["s2_contact_event_id"] == 1)
            )
            if not window.size:
                continue
            locked_values = arrays["s2_locked_next_anchor_distance"][window]
            locked_values = locked_values[np.isfinite(locked_values)]
            geometry_values = arrays["s2_contact_anchor_distance_xy"][window]
            geometry_values = geometry_values[np.isfinite(geometry_values)]
            locked_min.append(
                float(np.min(locked_values)) if locked_values.size else np.nan
            )
            contact_geometry_min.append(
                float(np.min(geometry_values)) if geometry_values.size else np.nan
            )
            arrived.append(float(np.any(arrays["s2_locked_next_region_arrival"][window])))
            o2_valid.append(float(np.any(arrays["s2_immediate_contact"][window] > 0.5)))
        if locked_min:
            print(
                "[INFO] O1→O2 target audit: "
                f"episodes={len(locked_min)}  "
                f"locked_O2_min={float(np.nanmean(locked_min)):.3f} m  "
                f"current-foot_contact_geometry_min={float(np.nanmean(contact_geometry_min)):.3f} m  "
                f"locked_region_15cm_rate={float(np.mean(arrived)):.3f}  "
                f"O2_valid_contact_rate={float(np.mean(o2_valid)):.3f}"
            )


def _summarize_s2_terminal_reward_audit(arrays: dict[str, np.ndarray]) -> None:
    """Report the actual signed O2->I3 progress lifecycle without changing reward."""
    required = (
        "done",
        "s2_terminal_release",
        "s2_terminal_progress_raw_rate",
        "s2_terminal_progress_clipped_rate",
        "s2_terminal_progress_positive_rate",
        "s2_terminal_progress_negative_rate",
        "s2_terminal_transition_window",
        "s2_terminal_window_position_outcome",
        "s2_immediate_contact",
        "s2_valid_contact_event",
    )
    if any(name not in arrays for name in required):
        return

    done = np.asarray(arrays["done"], dtype=bool)
    episode_id = np.cumsum(np.r_[0, done[:-1].astype(np.int64)])
    dt = float(np.asarray(arrays.get("step_dt", 0.02)).reshape(-1)[0])
    terminal = np.asarray(arrays["s2_terminal_release"], dtype=bool)
    window = np.asarray(arrays["s2_terminal_transition_window"], dtype=bool)
    rows: list[tuple[float, ...]] = []
    for episode in np.unique(episode_id[terminal]):
        selected = (episode_id == episode) & terminal
        if not np.any(selected):
            continue
        positive = float(
            np.nansum(arrays["s2_terminal_progress_positive_rate"][selected]) * dt
        )
        negative = float(
            np.nansum(arrays["s2_terminal_progress_negative_rate"][selected]) * dt
        )
        raw = float(
            np.nansum(arrays["s2_terminal_progress_raw_rate"][selected]) * dt
        )
        clipped = float(
            np.nansum(arrays["s2_terminal_progress_clipped_rate"][selected]) * dt
        )
        selected_window = (episode_id == episode) & window
        window_outcome = (
            float(np.nanmean(arrays["s2_terminal_window_position_outcome"][selected_window]))
            if np.any(selected_window)
            else np.nan
        )
        episode_mask = episode_id == episode
        candidate_pulses = float(
            np.count_nonzero(arrays["s2_immediate_contact"][episode_mask] > 0.5)
        )
        closure_pulses = float(
            np.count_nonzero(arrays["s2_valid_contact_event"][episode_mask] > 0.5)
        )
        rows.append(
            (
                positive,
                negative,
                raw,
                clipped,
                window_outcome,
                candidate_pulses,
                closure_pulses,
            )
        )
    if not rows:
        print("[INFO] O2→I3 reward audit: no terminal-release episode recorded.")
        return
    values = np.asarray(rows, dtype=np.float64)
    print(
        "[INFO] O2→I3 reward audit (unweighted distance-equivalent return): "
        f"episodes={values.shape[0]}  "
        f"positive={np.nanmean(values[:, 0]):+.4f} m  "
        f"negative={np.nanmean(values[:, 1]):+.4f} m  "
        f"raw_net={np.nanmean(values[:, 2]):+.4f} m  "
        f"clipped_net={np.nanmean(values[:, 3]):+.4f} m  "
        f"clip_bias={np.nanmean(values[:, 3] - values[:, 2]):+.4f} m  "
        f"I3_window_outcome={np.nanmean(values[:, 4]):+.3f}  "
        f"candidate_pulses={np.nanmean(values[:, 5]):.2f}  "
        f"closure_pulses={np.nanmean(values[:, 6]):.2f}  "
        f"legacy_contact_return@20/15="
        f"{np.nanmean(values[:, 5] * 20.0 * dt + values[:, 6] * 15.0 * dt):+.3f}"
    )

    online_required = (
        "s2_window_outcome_sample_count",
        "s2_window_outcome_score_sum",
        "s2_window_outcome_score_average",
        "s2_window_outcome_score_end",
        "s2_window_outcome_settlement_quality",
        "s2_window_outcome_settlement_pulse",
        "s2_window_outcome_pulse",
        "s2_window_outcome_emitted",
        "s2_window_outcome_aborted_ball_lost",
        "s2_window_outcome_aborted_other",
    )
    if any(name not in arrays for name in online_required):
        return
    checked = 0
    max_count_error = max_sum_error = max_avg_error = max_pulse_error = 0.0
    max_end_error = max_quality_error = max_settlement_pulse_error = 0.0
    zero_count = duplicate_pulse = aborted_ball_lost = aborted_other = 0
    emitted = np.asarray(arrays["s2_window_outcome_emitted"], dtype=bool)
    aborted_ball_lost_values = np.asarray(
        arrays["s2_window_outcome_aborted_ball_lost"], dtype=bool
    )
    aborted_other_values = np.asarray(
        arrays["s2_window_outcome_aborted_other"], dtype=bool
    )
    offline_values = np.asarray(
        arrays["s2_terminal_window_position_outcome"], dtype=np.float64
    )
    for episode in np.unique(episode_id[terminal]):
        episode_mask = episode_id == episode
        pulse_idx = np.flatnonzero(episode_mask & emitted)
        duplicate_pulse += max(0, int(pulse_idx.size) - 1)
        aborted_ball_lost += int(np.any(episode_mask & aborted_ball_lost_values))
        aborted_other += int(np.any(episode_mask & aborted_other_values))
        if pulse_idx.size == 0:
            zero_count += int(not np.any(episode_mask & window))
            continue
        idx = int(pulse_idx[0])
        offline_scores = offline_values[episode_mask & window]
        offline_count = float(offline_scores.size)
        offline_sum = float(np.nansum(offline_scores))
        offline_avg = offline_sum / offline_count if offline_count > 0 else 0.0
        online_count = float(arrays["s2_window_outcome_sample_count"][idx])
        online_sum = float(arrays["s2_window_outcome_score_sum"][idx])
        online_avg = float(arrays["s2_window_outcome_score_average"][idx])
        online_pulse = float(arrays["s2_window_outcome_pulse"][idx])
        online_end = float(arrays["s2_window_outcome_score_end"][idx])
        online_quality = float(arrays["s2_window_outcome_settlement_quality"][idx])
        online_settlement_pulse = float(
            arrays["s2_window_outcome_settlement_pulse"][idx]
        )
        offline_end = float(offline_scores[-1]) if offline_count > 0 else 0.0
        offline_quality = float(np.clip(min(offline_avg, offline_end), 0.0, 1.0))
        checked += 1
        max_count_error = max(max_count_error, abs(online_count - offline_count))
        max_sum_error = max(max_sum_error, abs(online_sum - offline_sum))
        max_avg_error = max(max_avg_error, abs(online_avg - offline_avg))
        max_pulse_error = max(max_pulse_error, abs(online_pulse - offline_avg / dt))
        max_end_error = max(max_end_error, abs(online_end - offline_end))
        max_quality_error = max(max_quality_error, abs(online_quality - offline_quality))
        max_settlement_pulse_error = max(
            max_settlement_pulse_error,
            abs(online_settlement_pulse - offline_quality),
        )
    print(
        "[INFO] Window-outcome online/offline audit: "
        f"checked={checked}  max_count_error={max_count_error:.3e}  "
        f"max_sum_error={max_sum_error:.3e}  max_avg_error={max_avg_error:.3e}  "
        f"max_pulse_error={max_pulse_error:.3e}  zero_count={zero_count}  "
        f"max_end_error={max_end_error:.3e}  max_quality_error={max_quality_error:.3e}  "
        f"max_settlement_pulse_error={max_settlement_pulse_error:.3e}  "
        f"duplicate_pulses={duplicate_pulse}  aborted_ball_lost={aborted_ball_lost}  "
        f"aborted_other={aborted_other}"
    )
    settlement_return = float(
        np.nansum(arrays["s2_window_outcome_settlement_pulse"]) * dt * 15.0
        / max(values.shape[0], 1)
    )
    delayed_contact_return = float(
        np.nanmean(values[:, 5] * 5.0 * dt + values[:, 6] * 15.0 * dt)
    )
    print(
        "[INFO] Delayed-settlement hypothetical/active budget: "
        f"discovery+closure@5/15={delayed_contact_return:+.3f}  "
        f"settlement@15={settlement_return:+.3f}  "
        f"total={delayed_contact_return + settlement_return:+.3f}"
    )

    fixed_required = (
        "s2_locked_live_target_w",
        "s2_locked_fixed_target_w",
        "s2_fixed_target_shift_distance",
        "s2_fixed_target_inverse_error",
        "s2_fixed_window_min_distance",
        "s2_live_window_min_distance",
        "s2_fixed_window_min_frame",
    )
    if any(name not in arrays for name in fixed_required):
        return
    fixed_rows = []
    fixed_targets = np.asarray(arrays["s2_locked_fixed_target_w"], dtype=np.float64)
    live_targets = np.asarray(arrays["s2_locked_live_target_w"], dtype=np.float64)
    for episode in np.unique(episode_id[terminal]):
        selected = (episode_id == episode) & window
        if not np.any(selected):
            continue
        idx = np.flatnonzero(selected)[-1]
        fixed_min = float(arrays["s2_fixed_window_min_distance"][idx])
        live_min = float(arrays["s2_live_window_min_distance"][idx])
        shift = float(arrays["s2_fixed_target_shift_distance"][idx])
        inverse_error = float(arrays["s2_fixed_target_inverse_error"][idx])
        min_frame = float(arrays["s2_fixed_window_min_frame"][idx])
        target_variation = float(
            np.nanmax(
                np.linalg.norm(
                    fixed_targets[selected] - fixed_targets[selected][0], axis=1
                )
            )
        )
        fixed_rows.append((fixed_min, live_min, shift, fixed_min - live_min, min_frame, target_variation, inverse_error))
    if fixed_rows:
        fixed_values = np.asarray(fixed_rows, dtype=np.float64)
        print(
            "[INFO] Fixed/live I3 target audit (conditioned on O2 terminal release): "
            f"episodes={len(fixed_rows)}  fixed_Dmin={np.nanmean(fixed_values[:, 0]):.3f} m  "
            f"live_Dmin={np.nanmean(fixed_values[:, 1]):.3f} m  "
            f"target_shift={np.nanmean(fixed_values[:, 2]):.3f} m  "
            f"fixed_minus_live_Dmin={np.nanmean(fixed_values[:, 3]):+.3f} m  "
            f"fixed_min_frame={np.nanmean(fixed_values[:, 4]):.2f}  "
            f"fixed_target_variation_max={np.nanmax(fixed_values[:, 5]):.3e} m  "
            f"inverse_closure_error_max={np.nanmax(fixed_values[:, 6]):.3e} m"
        )


def _summarize_o1_to_o2_acquisition(arrays: dict[str, np.ndarray]) -> None:
    """Separate poor O1 delivery from failure of the O2 foot to acquire the ball."""
    required = (
        "done",
        "motion_frame",
        "s2_contact_event_id",
        "s2_contact_window",
        "s2_sequence_cursor",
        "s2_physical_foot_ball_distance",
        "s2_ball_offset_actual_foot",
        "s2_coarse_contact_approach",
        "s2_immediate_contact",
    )
    if any(name not in arrays for name in required):
        return

    done = np.asarray(arrays["done"], dtype=bool)
    episode_id = np.cumsum(np.r_[0, done[:-1].astype(np.int64)])
    dt = float(np.asarray(arrays.get("step_dt", 0.02)).reshape(-1)[0])
    recorded_steps = np.asarray(arrays.get("step", []), dtype=np.int64)
    stride = (
        max(1, int(round(float(np.median(np.diff(recorded_steps))))))
        if recorded_steps.size > 1 else 1
    )
    sample_dt = dt * stride
    event_id = np.asarray(arrays["s2_contact_event_id"], dtype=np.int64)
    contact_window = np.asarray(arrays["s2_contact_window"], dtype=bool)
    cursor = np.asarray(arrays["s2_sequence_cursor"], dtype=np.float32)
    distance = np.asarray(arrays["s2_physical_foot_ball_distance"], dtype=np.float32)
    local_ball = np.asarray(arrays["s2_ball_offset_actual_foot"], dtype=np.float32)
    coarse = np.asarray(arrays["s2_coarse_contact_approach"], dtype=np.float32)
    physical_candidate = np.asarray(arrays["s2_immediate_contact"], dtype=np.float32) > 0.5

    rows: list[dict[str, float]] = []
    for ep in np.unique(episode_id):
        ep_indices = np.flatnonzero(episode_id == ep)
        o2_window = ep_indices[contact_window[ep_indices] & (event_id[ep_indices] == 1)]
        if not o2_window.size:
            continue
        o1_qualified = bool(np.any(cursor[ep_indices] >= 1.0))
        if not o1_qualified:
            continue
        finite = o2_window[np.isfinite(distance[o2_window])]
        if finite.size:
            min_index = int(finite[np.argmin(distance[finite])])
            min_distance = float(distance[min_index])
            min_frame = float(arrays["motion_frame"][min_index])
            min_local = local_ball[min_index]
        else:
            min_distance = np.nan
            min_frame = np.nan
            min_local = np.full(3, np.nan, dtype=np.float32)
        o2_phase = ep_indices[(event_id[ep_indices] == 1) & (cursor[ep_indices] >= 1.0)]
        rows.append(
            {
                "min_distance": min_distance,
                "min_frame": min_frame,
                "local_x": float(min_local[0]),
                "local_y": float(min_local[1]),
                "local_z": float(min_local[2]),
                "coarse_integral": float(np.nansum(coarse[o2_phase]) * sample_dt),
                "coarse_positive_integral": float(
                    np.nansum(np.maximum(coarse[o2_phase], 0.0)) * sample_dt
                ),
                "o2_physical": float(np.any(physical_candidate[o2_window])),
            }
        )

    if not rows:
        print("[INFO] O1→O2 acquisition audit: no O1-qualified episode reached the O2 window.")
        return

    def mean(name: str) -> float:
        values = np.asarray([row[name] for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        return float(np.mean(finite)) if finite.size else np.nan

    print(
        "[INFO] O1→O2 acquisition audit (conditioned on ordered physical O1): "
        f"episodes={len(rows)}  "
        f"O2_physical_rate={mean('o2_physical'):.3f}  "
        f"O2_window_min_foot_ball={mean('min_distance'):.3f} m  "
        f"min_frame={mean('min_frame'):.2f}  "
        f"ball_in_O2_foot_at_min=({mean('local_x'):+.3f},{mean('local_y'):+.3f},"
        f"{mean('local_z'):+.3f}) m  "
        f"O2_coarse_integral={mean('coarse_integral'):+.4f}  "
        f"O2_coarse_positive={mean('coarse_positive_integral'):+.4f}"
    )


def _summarize_s2_release_gate(arrays: dict[str, np.ndarray]) -> None:
    """Print the O1/O2 release decision made by the training state machine."""
    required = (
        "motion_frame",
        "done",
        "s2_contact_event_id",
        "s2_contact_window",
        "s2_immediate_contact",
        "s2_contact_new_physical_onset",
        "s2_contact_release_off_count",
        "s2_contact_release_confirmed",
        "s2_contact_left_claimed_region",
        "s2_contact_claimed_region_distance",
    )
    if any(name not in arrays for name in required):
        return
    episode_id = np.cumsum(np.r_[0, arrays["done"][:-1].astype(np.int64)])
    rows = []
    for ep in np.unique(episode_id):
        idx = np.flatnonzero(episode_id == ep)
        if not np.any(
            arrays["s2_contact_window"][idx].astype(bool)
            & (arrays["s2_contact_event_id"][idx] == 0)
        ):
            continue
        accepted = idx[arrays["s2_immediate_contact"][idx] > 0.5]
        o1 = accepted[arrays["s2_contact_event_id"][accepted] == 0]
        o2 = accepted[arrays["s2_contact_event_id"][accepted] == 1]
        onset = idx[arrays["s2_contact_new_physical_onset"][idx].astype(bool)]
        release = idx[arrays["s2_contact_release_confirmed"][idx].astype(bool)]
        after_o1 = idx[idx > o1[0]] if o1.size else idx
        before_o2_end = after_o1[arrays["motion_frame"][after_o1] <= 53]
        region_distance = arrays["s2_contact_claimed_region_distance"][before_o2_end]
        finite_region = region_distance[np.isfinite(region_distance)]
        rows.append(
            (
                int(ep),
                int(arrays["motion_frame"][o1[0]]) if o1.size else -1,
                int(arrays["motion_frame"][o2[0]]) if o2.size else -1,
                ",".join(str(int(v)) for v in arrays["motion_frame"][onset]) or "-",
                ",".join(str(int(v)) for v in arrays["motion_frame"][release]) or "-",
                int(np.max(arrays["s2_contact_release_off_count"][before_o2_end]))
                if before_o2_end.size else 0,
                float(np.min(finite_region)) if finite_region.size else np.nan,
                bool(np.any(arrays["s2_contact_left_claimed_region"][before_o2_end]))
                if before_o2_end.size else False,
            )
        )
    if not rows:
        return
    print("[INFO] O1/O2 release-gate audit (training-state timestamps):")
    print("[INFO]   ep  O1  O2  physical_onsets  releases  max_off  min_prev_region  left_region")
    for ep, o1, o2, onset, release, max_off, min_region, left_region in rows[:32]:
        print(
            f"[INFO]   {ep:2d}  {o1:2d}  {o2:2d}  {onset:15s}  {release:8s}  "
            f"{max_off:7d}  {min_region:15.4f}  {int(left_region)}"
        )


def _summarize_s2_candidate_gate(arrays: dict[str, np.ndarray]) -> None:
    """Show exactly which condition rejected each claimed physical contact."""
    required = (
        "motion_frame",
        "done",
        "s2_contact_event_id",
        "s2_contact_event_claim",
        "s2_contact_event_owns_physical",
        "s2_contact_closest_body",
        "s2_contact_correct_foot",
        "s2_contact_side_correct",
        "s2_contact_coarse_longitudinal_pass",
        "s2_contact_anchor_valid_touch",
        "s2_sequence_cursor",
        "s2_immediate_contact",
        "s2_ball_in_actual_foot_full",
    )
    if any(name not in arrays for name in required):
        return
    claims = np.flatnonzero(arrays["s2_contact_event_claim"].astype(bool))
    if not claims.size:
        return
    episode_id = np.cumsum(np.r_[0, arrays["done"][:-1].astype(np.int64)])
    print("[INFO] S2 physical-candidate gate audit:")
    print(
        "[INFO]   ep frame evt cur body own foot side xpos accepted "
        "ball_in_expected_foot(x,y,z)"
    )
    for i in claims[:64]:
        local = np.asarray(arrays["s2_ball_in_actual_foot_full"][i]).reshape(-1)
        if local.size >= 3:
            local_text = f"({local[0]:+.3f},{local[1]:+.3f},{local[2]:+.3f})"
        else:
            local_text = "(nan,nan,nan)"
        print(
            f"[INFO]   {int(episode_id[i]):2d} {int(arrays['motion_frame'][i]):5d} "
            f"{int(arrays['s2_contact_event_id'][i]):3d} "
            f"{int(arrays['s2_sequence_cursor'][i]):3d} "
            f"{int(arrays['s2_contact_closest_body'][i]):4d} "
            f"{int(bool(arrays['s2_contact_event_owns_physical'][i])):3d} "
            f"{int(bool(arrays['s2_contact_correct_foot'][i])):4d} "
            f"{int(bool(arrays['s2_contact_side_correct'][i])):4d} "
            f"{int(bool(arrays['s2_contact_coarse_longitudinal_pass'][i])):4d} "
            f"{int(bool(arrays['s2_immediate_contact'][i] > 0.5)):8d} "
            f"{local_text}"
        )


def _summarize_physx_foot_contacts(arrays: dict[str, np.ndarray]) -> None:
    """Summarize exact PhysX ball-foot manifolds without defining a task gate."""
    required = (
        "physx_foot_contact",
        "physx_robot_contact",
        "physx_contact_body_idx",
        "physx_contact_foot",
        "physx_foot_contact_force",
        "physx_foot_force_weighted_point_foot",
        "physx_ball_center_foot",
        "physx_foot_relative_point_velocity_foot",
        "physx_contact_frame_error",
        "physx_contact_expected_foot",
        "physx_contact_expected_surface",
        "ball_contact",
        "done",
    )
    if any(name not in arrays for name in required):
        return

    available = bool(
        np.asarray(arrays.get("filtered_contact_available", False)).reshape(-1)[0]
    )
    if not available:
        net_force_events = int(
            np.count_nonzero(np.asarray(arrays["ball_contact"], dtype=bool))
        )
        print(
            "[INFO] Filtered/detailed PhysX foot-contact audit: UNAVAILABLE "
            f"on CPU backend; ball net-force event samples={net_force_events}. "
            "These samples are not assigned to a link or foot."
        )
        return

    exact = arrays["physx_foot_contact"].astype(bool)
    robot_contact = arrays["physx_robot_contact"].astype(bool)
    legacy = arrays["ball_contact"].astype(bool)
    if not exact.size:
        return
    print(
        "[INFO] Exact PhysX foot-contact audit: "
        f"foot_samples={int(np.count_nonzero(exact))}/{exact.size}  "
        f"tracked_body_subset_samples={int(np.count_nonzero(robot_contact))}  "
        f"legacy_and_exact={int(np.count_nonzero(legacy & exact))}  "
        f"legacy_only={int(np.count_nonzero(legacy & ~exact))}  "
        f"exact_only={int(np.count_nonzero(~legacy & exact))}"
    )
    if np.any(robot_contact):
        body_indices = arrays["physx_contact_body_idx"][robot_contact].astype(np.int64)
        body_names = arrays.get("physx_contact_body_names", np.asarray([]))
        unique_indices, body_counts = np.unique(body_indices, return_counts=True)
        body_summary = ", ".join(
            f"{body_names[index] if 0 <= index < len(body_names) else index}={count}"
            for index, count in zip(unique_indices, body_counts)
        )
        print(f"[INFO] Exact PhysX contacting bodies: {body_summary}")
    if not np.any(exact):
        print("[INFO] Exact PhysX foot-contact audit: no ball-foot contact manifold was sampled.")
        return

    # Treat the first sampled step of each contiguous manifold as one impact.
    previous_exact = np.r_[False, exact[:-1]]
    previous_done = np.r_[True, arrays["done"][:-1].astype(bool)]
    impact_indices = np.flatnonzero(exact & (~previous_exact | previous_done))
    foot = arrays["physx_contact_foot"].astype(np.int64)
    expected_foot = arrays["physx_contact_expected_foot"].astype(np.int64)
    expected_surface = arrays["physx_contact_expected_surface"].astype(np.int64)
    correct_foot = foot[impact_indices] == expected_foot[impact_indices]
    labelled = np.isin(expected_foot[impact_indices], (0, 1))
    foot_match = float(np.mean(correct_foot[labelled])) if np.any(labelled) else np.nan
    timing = arrays["physx_contact_frame_error"][impact_indices]
    timing = timing[np.isfinite(timing)]
    timing_mean = float(np.mean(timing)) if timing.size else np.nan
    timing_abs_p90 = float(np.percentile(np.abs(timing), 90)) if timing.size else np.nan
    print(
        "[INFO] Exact PhysX impacts: "
        f"n={impact_indices.size}  expected_foot_match={foot_match:.3f}  "
        f"frame_error_mean={timing_mean:+.2f}  abs_frame_error_p90={timing_abs_p90:.2f}"
    )

    event_at_impact = arrays["physx_contact_event_id"][impact_indices].astype(np.int64)
    for event_id in sorted(np.unique(event_at_impact[event_at_impact >= 0]).tolist()):
        selected = impact_indices[event_at_impact == event_id]
        selected_foot = foot[selected]
        resolved = np.isin(selected_foot, (0, 1))
        selected = selected[resolved]
        selected_foot = selected_foot[resolved]
        if not selected.size:
            continue
        point_foot = arrays["physx_foot_force_weighted_point_foot"][
            selected, selected_foot
        ]
        ball_center_foot = arrays["physx_ball_center_foot"][selected, selected_foot]
        impulse_w = arrays["physx_foot_normal_impulse_w"][selected, selected_foot]
        point_force = arrays["physx_foot_contact_force"][selected, selected_foot]
        point_relative_velocity = arrays["physx_foot_relative_point_velocity_foot"][
            selected, selected_foot
        ]
        weights = np.maximum(point_force, 0.0)
        weight_sum = np.sum(weights, axis=1, keepdims=True)
        relative_velocity_foot = np.nansum(
            point_relative_velocity * weights[..., None], axis=1
        ) / np.maximum(weight_sum, 1.0e-8)
        frame_error = arrays["physx_contact_frame_error"][selected]

        def _finite_vector_mean(values: np.ndarray) -> np.ndarray:
            finite = values[np.all(np.isfinite(values), axis=1)]
            return np.mean(finite, axis=0) if finite.size else np.full(3, np.nan)

        point_mean = _finite_vector_mean(point_foot)
        center_mean = _finite_vector_mean(ball_center_foot)
        relative_velocity_mean = _finite_vector_mean(relative_velocity_foot)
        impulse_norm = np.linalg.norm(impulse_w, axis=1)
        impulse_norm = impulse_norm[np.isfinite(impulse_norm)]
        frame_error = frame_error[np.isfinite(frame_error)]
        label = ("O1", "O2", "I3", "I4", "O5", "O6", "I7", "I8")
        event_label = label[event_id] if event_id < len(label) else f"event_{event_id}"
        print(
            f"[INFO]   Exact {event_label}: n={selected.size}  "
            f"point_foot=({point_mean[0]:+.3f},{point_mean[1]:+.3f},{point_mean[2]:+.3f}) m  "
            f"ball_center_foot=({center_mean[0]:+.3f},{center_mean[1]:+.3f},{center_mean[2]:+.3f}) m  "
            f"pre_rel_v_foot=({relative_velocity_mean[0]:+.3f},"
            f"{relative_velocity_mean[1]:+.3f},{relative_velocity_mean[2]:+.3f}) m/s  "
            f"impulse={float(np.mean(impulse_norm)) if impulse_norm.size else np.nan:.3f} N*s  "
            f"frame_error={float(np.mean(frame_error)) if frame_error.size else np.nan:+.2f}"
        )

    for surface_id, surface_name in ((1, "outside"), (0, "inside")):
        selected = impact_indices[
            expected_surface[impact_indices] == surface_id
        ]
        if not selected.size:
            print(f"[INFO]   {surface_name}: n=0")
            continue
        selected_foot = foot[selected]
        valid_foot = np.isin(selected_foot, (0, 1))
        selected = selected[valid_foot]
        selected_foot = selected_foot[valid_foot]
        if not selected.size:
            print(f"[INFO]   {surface_name}: n=0 with a resolved contacting foot")
            continue

        point_foot = arrays["physx_foot_force_weighted_point_foot"][selected, selected_foot]
        ball_center_foot = arrays["physx_ball_center_foot"][selected, selected_foot]
        force_values = arrays["physx_foot_contact_force"][selected, selected_foot]
        relative_velocity = arrays["physx_foot_relative_point_velocity_foot"][selected, selected_foot]
        force_sum = np.sum(force_values, axis=1)
        weights = np.maximum(force_values, 0.0)
        weight_sum = np.sum(weights, axis=1, keepdims=True)
        weighted_rel_v = np.nansum(relative_velocity * weights[..., None], axis=1) / np.maximum(
            weight_sum, 1.0e-8
        )

        def _axis_summary(values: np.ndarray) -> str:
            finite_rows = values[np.all(np.isfinite(values), axis=1)]
            if not finite_rows.size:
                return "unavailable"
            mean = np.mean(finite_rows, axis=0)
            p10 = np.percentile(finite_rows, 10, axis=0)
            p90 = np.percentile(finite_rows, 90, axis=0)
            return (
                f"mean=({mean[0]:+.3f},{mean[1]:+.3f},{mean[2]:+.3f}) "
                f"p10=({p10[0]:+.3f},{p10[1]:+.3f},{p10[2]:+.3f}) "
                f"p90=({p90[0]:+.3f},{p90[1]:+.3f},{p90[2]:+.3f})"
            )

        print(
            f"[INFO]   {surface_name}: n={selected.size}  "
            f"normal_force_mean={float(np.mean(force_sum)):.1f} N  "
            f"contact_point_foot[{_axis_summary(point_foot)}]"
        )
        print(
            f"[INFO]     ball_center_foot[{_axis_summary(ball_center_foot)}]  "
            f"relative_point_velocity_foot[{_axis_summary(weighted_rel_v)}]"
        )


def _append_physical_contact_events(arrays: dict[str, np.ndarray]) -> None:
    """Aggregate contiguous filtered foot-ball contact into event-level rows.

    This is calibration telemetry only.  It deliberately starts from the raw
    filtered PhysX fact rather than the current S2 valid/qualified definition,
    and never changes reward, cursor, termination, or curriculum state.
    """
    required = (
        "physx_foot_contact",
        "physx_contact_foot",
        "physx_contact_body_force",
        "physx_foot_normal_impulse_w",
        "physx_ball_center_foot",
        "motion_frame",
        "done",
    )
    if any(name not in arrays for name in required):
        return
    available = bool(
        np.asarray(arrays.get("filtered_contact_available", False)).reshape(-1)[0]
    )
    if not available:
        print(
            "[INFO] Physical contact calibration: UNAVAILABLE on CPU backend; "
            "no filtered event rows were generated."
        )
        return
    active = np.asarray(arrays["physx_foot_contact"], dtype=bool)
    foot = np.asarray(arrays["physx_contact_foot"], dtype=np.int64)
    done = np.asarray(arrays["done"], dtype=bool)
    sample_dt = float(np.asarray(arrays.get("step_dt", 0.02)).reshape(-1)[0])
    steps = np.asarray(arrays.get("step", np.arange(active.size)), dtype=np.int64)
    if steps.size > 1:
        sample_dt *= max(1, int(round(float(np.median(np.diff(steps))))))
    episode = np.cumsum(np.r_[0, done[:-1].astype(np.int64)])

    rows: list[dict[str, object]] = []
    index = 0
    while index < active.size:
        if not active[index] or foot[index] not in (0, 1):
            index += 1
            continue
        start = index
        event_foot = int(foot[index])
        while (
            index + 1 < active.size
            and active[index + 1]
            and int(foot[index + 1]) == event_foot
            and int(episode[index + 1]) == int(episode[start])
        ):
            index += 1
        end = index
        segment = slice(start, end + 1)
        force = np.asarray(arrays["physx_contact_body_force"])[segment]
        finite_force = np.where(np.isfinite(force), force, -np.inf)
        peak_local = int(np.argmax(finite_force)) if finite_force.size else 0
        peak = start + peak_local
        impulses = np.asarray(arrays["physx_foot_normal_impulse_w"])[segment, event_foot]
        impulse_w = np.nansum(impulses, axis=0)
        ball_center_foot = np.asarray(arrays["physx_ball_center_foot"])[peak, event_foot]

        default_int = np.full(active.size, -1, dtype=np.int64)
        expected_foot = int(np.asarray(arrays.get("s2_expected_foot", default_int))[peak])
        expected_side = int(np.asarray(arrays.get("s2_expected_side", default_int))[peak])
        expected_surface = int(np.asarray(arrays.get("s2_expected_surface", default_int))[peak])
        in_window = bool(
            np.asarray(arrays.get("s2_contact_window", np.zeros(active.size, dtype=bool)))[peak]
        )
        event_id = int(np.asarray(arrays.get("s2_contact_event_id", default_int))[peak])
        ball_offset = np.asarray(
            arrays.get("s2_ball_offset_actual_foot", np.full((active.size, 3), np.nan))
        )[peak]
        side_correct = bool(
            np.isfinite(ball_offset[1])
            and expected_side in (0, 1)
            and ((expected_side == 0 and ball_offset[1] >= 0.0)
                 or (expected_side == 1 and ball_offset[1] <= 0.0))
        )
        rows.append(
            {
                "episode": int(episode[start]),
                "start_sample": start,
                "end_sample": end,
                "start_frame": int(arrays["motion_frame"][start]),
                "end_frame": int(arrays["motion_frame"][end]),
                "duration": (end - start + 1) * sample_dt,
                "foot": event_foot,
                "event_id": event_id,
                "expected_foot": expected_foot,
                "expected_side": expected_side,
                "expected_surface": expected_surface,
                "in_window": in_window,
                "correct_foot": event_foot == expected_foot,
                "correct_side": side_correct,
                "peak_force": float(force[peak_local]) if force.size else np.nan,
                "impulse_w": impulse_w,
                "ball_center_foot": ball_center_foot,
                "ball_offset_actual_foot": ball_offset,
                "anchor_error_xy": float(
                    np.asarray(arrays.get("s2_contact_anchor_distance_xy", np.full(active.size, np.nan)))[peak]
                ),
                "foot_reference_error": float(
                    np.asarray(arrays.get("foot_reference_position_error", np.full(active.size, np.nan)))[peak]
                ),
                "ball_velocity_w": np.asarray(arrays.get("ball_lin_vel_w"))[peak],
            }
        )
        index += 1

    scalar_fields = (
        "episode", "start_sample", "end_sample", "start_frame", "end_frame",
        "duration", "foot", "event_id", "expected_foot", "expected_side",
        "expected_surface", "in_window", "correct_foot", "correct_side",
        "peak_force", "anchor_error_xy", "foot_reference_error",
    )
    vector_fields = ("impulse_w", "ball_center_foot", "ball_offset_actual_foot", "ball_velocity_w")
    for field in scalar_fields:
        arrays[f"physical_event_{field}"] = np.asarray([row[field] for row in rows])
    for field in vector_fields:
        arrays[f"physical_event_{field}"] = (
            np.asarray([row[field] for row in rows], dtype=np.float32)
            if rows else np.empty((0, 3), dtype=np.float32)
        )
    print(
        f"[INFO] Physical contact calibration: {len(rows)} filtered foot-ball events "
        f"from {int(np.sum(active))} contact samples."
    )


def _summarize_s2_calibration_smoke(arrays: dict[str, np.ndarray]) -> None:
    context = np.asarray(arrays.get("s2_critic_sparse_context", []), dtype=np.float32)
    finite = bool(context.size and np.all(np.isfinite(context)))
    max_abs = float(np.max(np.abs(context))) if context.size else np.nan
    locked_error = np.asarray(
        arrays.get("s2_critic_locked_anchor_world_error", []), dtype=np.float32
    )
    locked_error = locked_error[np.isfinite(locked_error)]
    max_locked_error = float(np.max(locked_error)) if locked_error.size else np.nan
    failed = np.asarray(arrays.get("s2_sequence_failed", []), dtype=bool)
    done = np.asarray(arrays.get("done", np.zeros(failed.size, dtype=bool)), dtype=bool)
    failed_unlatched = 0
    for index in np.flatnonzero(failed):
        if index + 1 < failed.size and not done[index] and not failed[index + 1]:
            failed_unlatched += 1
    print(
        "[INFO] S2 calibration smoke: "
        f"critic19_finite={finite}  critic19_max_abs={max_abs:.4f}  "
        f"locked_world_error_max={max_locked_error:.3e} m  "
        f"sequence_failed_unlatched={failed_unlatched}"
    )


def _summarize_s2_o1_geometry(arrays: dict[str, np.ndarray]) -> None:
    """Audit raw O1 geometry even when no physical contact occurs."""
    required = (
        "motion_frame",
        "done",
        "ball_pos_w",
        "s2_contact_window",
        "s2_contact_event_id",
        "s2_reference_foot_pos_w",
        "s2_actual_foot_pos_w",
        "s2_reference_contact_anchor_w",
        "s2_contact_anchor_w",
        "s2_reference_pelvis_pos_w",
        "s2_actual_pelvis_pos_w",
        "s2_reference_pelvis_yaw",
        "s2_actual_pelvis_yaw",
        "s2_physical_foot_ball_distance",
    )
    if any(name not in arrays for name in required):
        return

    frames = np.asarray(arrays["motion_frame"], dtype=np.int64)
    done = np.asarray(arrays["done"], dtype=bool)
    event_id = np.asarray(arrays["s2_contact_event_id"], dtype=np.int64)
    in_window = np.asarray(arrays["s2_contact_window"], dtype=bool)
    o1 = in_window & (event_id == 0)
    if not np.any(o1):
        labelled = np.unique(event_id[event_id >= 0]).tolist()
        print(
            "[INFO] O1 geometry audit: no event_index=0 contact-window samples; "
            f"observed_event_ids={labelled}. Check phase/contact-label alignment."
        )
        return

    episode = np.cumsum(np.r_[0, done[:-1].astype(np.int64)])
    ball = np.asarray(arrays["ball_pos_w"], dtype=np.float64)
    reference_foot = np.asarray(arrays["s2_reference_foot_pos_w"], dtype=np.float64)
    actual_foot = np.asarray(arrays["s2_actual_foot_pos_w"], dtype=np.float64)
    reference_anchor = np.asarray(
        arrays["s2_reference_contact_anchor_w"], dtype=np.float64
    )
    actual_foot_target = np.asarray(arrays["s2_contact_anchor_w"], dtype=np.float64)
    reference_pelvis = np.asarray(
        arrays["s2_reference_pelvis_pos_w"], dtype=np.float64
    )
    actual_pelvis = np.asarray(arrays["s2_actual_pelvis_pos_w"], dtype=np.float64)
    reference_pelvis_yaw = np.asarray(
        arrays["s2_reference_pelvis_yaw"], dtype=np.float64
    )
    actual_pelvis_yaw = np.asarray(arrays["s2_actual_pelvis_yaw"], dtype=np.float64)
    direct_ankle_ball = np.asarray(
        arrays["s2_physical_foot_ball_distance"], dtype=np.float64
    )
    v3_distance = np.asarray(
        arrays.get("s2_contact_region_distance", np.full(frames.size, np.nan)),
        dtype=np.float64,
    )
    v3_valid = np.asarray(
        arrays.get("s2_contact_region_valid", np.zeros(frames.size, dtype=bool)),
        dtype=bool,
    )
    v3_distance = np.where(v3_valid, v3_distance, np.nan)
    ball_local_full = np.asarray(
        arrays.get("s2_ball_in_actual_foot_full", np.full((frames.size, 3), np.nan)),
        dtype=np.float64,
    )
    anchor_local_full = np.asarray(
        arrays.get(
            "s2_contact_anchor_offset_actual_foot_full",
            np.full((frames.size, 3), np.nan),
        ),
        dtype=np.float64,
    )

    def _norm(delta: np.ndarray) -> float:
        return float(np.linalg.norm(delta)) if np.all(np.isfinite(delta)) else np.nan

    def _yaw_error(reference: float, actual: float) -> float:
        if not np.isfinite(reference) or not np.isfinite(actual):
            return np.nan
        return float(np.arctan2(np.sin(actual - reference), np.cos(actual - reference)))

    rows: list[dict[str, object]] = []
    for episode_id in np.unique(episode[o1]):
        selected = np.flatnonzero(o1 & (episode == episode_id))
        finite = selected[np.isfinite(direct_ankle_ball[selected])]
        if not finite.size:
            continue
        closest = int(finite[np.argmin(direct_ankle_ball[finite])])
        episode_samples = np.flatnonzero(episode == episode_id)
        spawn = ball[int(episode_samples[0])] if episode_samples.size else np.full(3, np.nan)
        rows.append(
            {
                "episode": int(episode_id),
                "index": closest,
                "frame": int(frames[closest]),
                "spawn": spawn,
                "ankle_ball": float(direct_ankle_ball[closest]),
                "reference_foot_ball": _norm(ball[closest] - reference_foot[closest]),
                "foot_tracking": _norm(actual_foot[closest] - reference_foot[closest]),
                "reference_anchor_error": _norm(ball[closest] - reference_anchor[closest]),
                "actual_target_error": _norm(ball[closest] - actual_foot_target[closest]),
                "local_anchor_error": _norm(
                    ball_local_full[closest] - anchor_local_full[closest]
                ),
                "pelvis_tracking": _norm(actual_pelvis[closest] - reference_pelvis[closest]),
                "pelvis_yaw_error": _yaw_error(
                    reference_pelvis_yaw[closest], actual_pelvis_yaw[closest]
                ),
                "v3_distance": float(v3_distance[closest]),
            }
        )

    if not rows:
        print("[INFO] O1 geometry audit: O1 samples exist but direct ankle-ball distance is unavailable.")
        return

    def _mean(name: str) -> float:
        values = np.asarray([row[name] for row in rows], dtype=np.float64)
        values = values[np.isfinite(values)]
        return float(np.mean(values)) if values.size else np.nan

    print(
        "[INFO] O1 raw geometry audit (closest direct ankle-ball sample per episode): "
        f"episodes={len(rows)}  ankle_ball={_mean('ankle_ball'):.3f} m  "
        f"reference_foot_ball={_mean('reference_foot_ball'):.3f} m  "
        f"foot_tracking={_mean('foot_tracking'):.3f} m  "
        f"reference_anchor_error={_mean('reference_anchor_error'):.3f} m  "
        f"actual_target_error={_mean('actual_target_error'):.3f} m  "
        f"local_anchor_error={_mean('local_anchor_error'):.3f} m  "
        f"pelvis_tracking={_mean('pelvis_tracking'):.3f} m  "
        f"pelvis_yaw_error={_mean('pelvis_yaw_error'):+.3f} rad  "
        f"legacy_v3_distance={_mean('v3_distance'):.3f} m"
    )
    print(
        "[INFO]   ep frame spawn_w(x,y,z)       ankle-ball ref-foot-ball "
        "foot-track ref-anchor current-target local-anchor pelvis-xyz yaw-error legacy-v3"
    )
    for row in rows[:16]:
        spawn = np.asarray(row["spawn"])
        print(
            f"[INFO]   {row['episode']:2d} {row['frame']:5d} "
            f"({spawn[0]:+.3f},{spawn[1]:+.3f},{spawn[2]:+.3f})  "
            f"{row['ankle_ball']:.3f}      {row['reference_foot_ball']:.3f}        "
            f"{row['foot_tracking']:.3f}      {row['reference_anchor_error']:.3f}      "
            f"{row['actual_target_error']:.3f}         {row['local_anchor_error']:.3f}       "
            f"{row['pelvis_tracking']:.3f}      {row['pelvis_yaw_error']:+.3f}     "
            f"{row['v3_distance']:.3f}"
        )

    closest_indices = np.asarray([row["index"] for row in rows], dtype=np.int64)
    for label, values in (
        ("ball_w", ball),
        ("reference_foot_w", reference_foot),
        ("actual_foot_w", actual_foot),
        ("reference_anchor_w", reference_anchor),
        ("current_actual-foot_target_w", actual_foot_target),
    ):
        selected = values[closest_indices]
        selected = selected[np.all(np.isfinite(selected), axis=1)]
        mean = np.mean(selected, axis=0) if selected.size else np.full(3, np.nan)
        print(f"[INFO]   mean {label}=({mean[0]:+.3f},{mean[1]:+.3f},{mean[2]:+.3f}) m")


def _save_diagnostic(diagnostic: dict) -> None:
    """Persist the trace in a self-describing NumPy archive."""
    if not diagnostic["step"]:
        print("[WARN] Diagnostic requested but no samples were recorded.")
        return
    metadata_keys = {
        "path", "stride", "step_dt", "joint_ids", "joint_names", "action_ids", "reward_term_names", "task_state_names",
        "ball_spawn_source_names",
        "trunk_joint_ids", "trunk_joint_names", "trunk_action_ids", "trunk_full_action_ids", "trunk_body_ids",
        "trunk_reference_body_ids", "trunk_body_names",
        "constraint_group", "constraint_margin", "constraint_joint_names", "constraint_groups",
        "constraint_margins", "waist_roll_stiffness_scale", "waist_roll_damping_scale",
        "direct_upper_body_latent", "schema_version", "sample_timing", "action_value_semantics",
        "contact_settings", "contact_region_frame", "contact_body_names", "contact_surface",
        "contact_force_threshold", "contact_cg_gated", "contact_cg_surface_gated", "locomotion_command_frame",
        "detailed_contact", "physx_contact_sensor_pattern", "physx_contact_filter_patterns",
        "physx_contact_filter_paths", "physx_contact_body_names", "physx_contact_foot_names",
        "physx_contact_points_per_foot", "physx_contact_backend",
        "filtered_contact_available",
    }
    arrays = {}
    for key, value in diagnostic.items():
        if key in metadata_keys:
            continue
        try:
            arrays[key] = np.asarray(value)
        except ValueError as exc:
            shapes = sorted({np.asarray(item).shape for item in value})
            raise ValueError(
                f"Diagnostic field {key!r} has inconsistent per-sample shapes: {shapes}"
            ) from exc
    arrays["joint_names"] = diagnostic["joint_names"]
    arrays["trunk_joint_names"] = diagnostic["trunk_joint_names"]
    arrays["trunk_body_names"] = diagnostic["trunk_body_names"]
    arrays["reward_term_names"] = diagnostic["reward_term_names"]
    arrays["task_state_names"] = diagnostic["task_state_names"]
    arrays["ball_spawn_source_names"] = diagnostic["ball_spawn_source_names"]
    arrays["constraint_group"] = np.asarray(diagnostic["constraint_group"])
    arrays["constraint_margin"] = np.asarray(diagnostic["constraint_margin"])
    arrays["constraint_joint_names"] = diagnostic["constraint_joint_names"]
    arrays["constraint_groups"] = diagnostic["constraint_groups"]
    arrays["constraint_margins"] = diagnostic["constraint_margins"]
    arrays["waist_roll_stiffness_scale"] = np.asarray(diagnostic["waist_roll_stiffness_scale"])
    arrays["waist_roll_damping_scale"] = np.asarray(diagnostic["waist_roll_damping_scale"])
    arrays["direct_upper_body_latent"] = np.asarray(diagnostic["direct_upper_body_latent"])
    arrays["schema_version"] = np.asarray(diagnostic["schema_version"])
    arrays["step_dt"] = np.asarray(diagnostic["step_dt"], dtype=np.float32)
    arrays["sample_timing"] = np.asarray(diagnostic["sample_timing"])
    arrays["action_value_semantics"] = diagnostic["action_value_semantics"]
    arrays["contact_region_frame"] = np.asarray(diagnostic["contact_region_frame"])
    arrays["contact_body_names"] = diagnostic["contact_body_names"]
    arrays["contact_surface"] = np.asarray(diagnostic["contact_surface"])
    arrays["contact_force_threshold"] = np.asarray(diagnostic["contact_force_threshold"])
    arrays["contact_cg_gated"] = np.asarray(diagnostic["contact_cg_gated"])
    arrays["contact_cg_surface_gated"] = np.asarray(diagnostic["contact_cg_surface_gated"])
    arrays["locomotion_command_frame"] = np.asarray(diagnostic["locomotion_command_frame"])
    arrays["physx_contact_sensor_pattern"] = np.asarray(
        diagnostic["physx_contact_sensor_pattern"]
    )
    arrays["physx_contact_filter_patterns"] = diagnostic["physx_contact_filter_patterns"]
    arrays["physx_contact_filter_paths"] = diagnostic["physx_contact_filter_paths"]
    arrays["physx_contact_body_names"] = diagnostic["physx_contact_body_names"]
    arrays["physx_tracked_body_subset_names"] = diagnostic["physx_contact_body_names"]
    arrays["physx_contact_foot_names"] = diagnostic["physx_contact_foot_names"]
    arrays["physx_contact_points_per_foot"] = np.asarray(
        diagnostic["physx_contact_points_per_foot"], dtype=np.int32
    )
    arrays["physx_contact_backend"] = np.asarray(diagnostic["physx_contact_backend"])
    arrays["filtered_contact_available"] = np.asarray(
        diagnostic["filtered_contact_available"], dtype=bool
    )
    # Explicit names prevent an unfiltered ball net-force pulse or an ankle
    # origin distance from being interpreted as a verified foot contact or a
    # collision-surface clearance.
    arrays["ball_net_force_event"] = np.asarray(arrays.get("ball_contact", []), dtype=bool)
    arrays["ankle_origin_to_ball_center_distance"] = np.asarray(
        arrays.get("s2_physical_foot_ball_distance", []), dtype=np.float32
    )
    _append_physical_contact_events(arrays)
    np.savez_compressed(diagnostic["path"], **arrays)
    _summarize_s2_calibration_smoke(arrays)
    _summarize_s2_o1_geometry(arrays)
    _summarize_physx_foot_contacts(arrays)
    _summarize_s2_release_gate(arrays)
    _summarize_s2_candidate_gate(arrays)
    _summarize_s2_terminal_reward_audit(arrays)
    _summarize_o1_to_o2_acquisition(arrays)
    _summarize_first_two_s2_touches(arrays)

    def _finite_mean(values: np.ndarray) -> float:
        finite = values[np.isfinite(values)]
        return float(np.mean(finite)) if finite.size else np.nan

    contact_rate = float(np.mean(arrays["ball_contact"]))
    ball_distance = float(np.mean(arrays["ball_pelvis_xy_distance"]))
    ball_speed = float(np.mean(arrays["ball_xy_speed"]))
    ball_forward_speed = _finite_mean(arrays["ball_command_forward_speed"])
    pelvis_speed = float(np.mean(arrays["pelvis_xy_speed"]))
    foot_error = float(np.mean(arrays["foot_reference_position_error"]))
    heading_error = _finite_mean(np.abs(arrays["heading_error"]))
    local_twist_abs_error = np.abs(arrays["twist_local_error"])
    local_vx_error = _finite_mean(local_twist_abs_error[:, 0])
    local_vy_error = _finite_mean(local_twist_abs_error[:, 1])
    local_wz_error = _finite_mean(local_twist_abs_error[:, 2])
    final_twist_blend_alpha = float(arrays["twist_blend_alpha"][-1])
    task_state_names = np.asarray(["idle", "dribble", "stop"])
    task_state_counts = np.bincount(arrays["task_state"].astype(np.int64), minlength=3)[:3]
    task_state_summary = ", ".join(
        f"{name}={count}" for name, count in zip(task_state_names, task_state_counts)
    )
    stop_mask = arrays["stop_active"].astype(bool)
    stop_settle_rate = (
        float(np.mean(arrays["stop_settled"][stop_mask])) if np.any(stop_mask) else np.nan
    )
    stop_successes = int(np.count_nonzero(arrays["stop_success"].astype(bool)))
    stop_ball_speed = (
        float(np.mean(arrays["stop_ball_speed"][stop_mask])) if np.any(stop_mask) else np.nan
    )
    stop_pelvis_speed = (
        float(np.mean(arrays["stop_pelvis_speed"][stop_mask])) if np.any(stop_mask) else np.nan
    )
    stop_pelvis_angular_speed = (
        float(np.mean(arrays["stop_pelvis_angular_speed"][stop_mask])) if np.any(stop_mask) else np.nan
    )
    arm_error = float(np.mean(np.abs(arrays["actual_joint_pos"] - arrays["reference_joint_pos"])))
    trunk_error = float(
        np.mean(np.abs(arrays["trunk_actual_joint_pos"] - arrays["trunk_reference_joint_pos"]))
    )
    trunk_action_delta = np.diff(arrays["trunk_effective_action"], axis=0)
    finite_trunk_action_delta = np.linalg.norm(trunk_action_delta, axis=1)
    finite_trunk_action_delta = finite_trunk_action_delta[np.isfinite(finite_trunk_action_delta)]
    trunk_action_step = (
        float(np.mean(finite_trunk_action_delta)) if finite_trunk_action_delta.size else np.nan
    )
    action_snapshot_coverage = float(np.mean(arrays["action_snapshot_available"].astype(np.float32)))
    post_step_state_coverage = float(np.mean(arrays["post_step_state_valid"].astype(np.float32)))
    contact_mask = arrays["ball_contact"].astype(bool)
    legal_touch_rate = (
        float(np.mean(arrays["contact_legal_touch"][contact_mask])) if np.any(contact_mask) else np.nan
    )
    cg_eligible_touch_rate = (
        float(np.mean(arrays["contact_cg_eligible"][contact_mask])) if np.any(contact_mask) else np.nan
    )
    generic_instep_rate = (
        float(np.mean(arrays["contact_generic_instep"][contact_mask])) if np.any(contact_mask) else np.nan
    )
    cg_expected_touch_mask = arrays["cg_label_available"].astype(bool) & arrays["cg_reference_contact"].astype(bool)
    cg_foot_match_rate = (
        float(np.mean(arrays["cg_contact_foot_match"][cg_expected_touch_mask]))
        if np.any(cg_expected_touch_mask)
        else np.nan
    )
    s2_window_mask = arrays["s2_contact_window"].astype(bool)
    s2_region_distance = (
        _finite_mean(arrays["s2_target_region_distance"][s2_window_mask])
        if np.any(s2_window_mask)
        else np.nan
    )
    s2_reference_region_distance = (
        _finite_mean(arrays["s2_reference_target_region_distance"][s2_window_mask])
        if np.any(s2_window_mask)
        else np.nan
    )
    s2_physical_distance = (
        _finite_mean(arrays["s2_physical_foot_ball_distance"][s2_window_mask])
        if np.any(s2_window_mask)
        else np.nan
    )
    s2_release_mask = arrays["s2_release_pending"].astype(bool)
    s2_release_score = (
        _finite_mean(arrays["s2_release_score"][s2_release_mask])
        if np.any(s2_release_mask)
        else np.nan
    )
    s2_release_speed_ratio = (
        _finite_mean(arrays["s2_release_speed_ratio"][s2_release_mask])
        if np.any(s2_release_mask)
        else np.nan
    )
    s2_release_quality_pulses = int(
        np.count_nonzero(arrays["s2_release_quality_pass"])
    )
    torso_rel_tilt = float(
        np.mean(np.linalg.norm(arrays["torso_minus_pelvis_rpy"][:, :2], axis=1))
    )
    torso_rel_tilt_error = float(
        np.mean(np.linalg.norm(arrays["torso_minus_pelvis_rpy_error"][:, :2], axis=1))
    )
    torso_rel_ang_vel = float(
        np.mean(np.linalg.norm(arrays["torso_minus_pelvis_ang_vel_w"], axis=1))
    )
    trunk_target_error = float(np.nanmean(np.abs(arrays["trunk_post_step_target_error"])))
    trunk_target_reference_offset = float(np.nanmean(np.abs(arrays["trunk_target_minus_reference"])))
    trunk_pitch_filter_values = np.abs(
        arrays["trunk_pitch_filtered_target"] - arrays["trunk_pitch_raw_target"]
    )
    finite_trunk_pitch_filter_values = trunk_pitch_filter_values[
        np.isfinite(trunk_pitch_filter_values)
    ]
    trunk_pitch_filter_delta = (
        float(np.mean(finite_trunk_pitch_filter_values))
        if finite_trunk_pitch_filter_values.size
        else np.nan
    )
    finite_trunk_pitch_overflow = arrays["trunk_pitch_reference_overflow"][
        np.isfinite(arrays["trunk_pitch_reference_overflow"])
    ]
    trunk_pitch_overflow = (
        float(np.mean(finite_trunk_pitch_overflow)) if finite_trunk_pitch_overflow.size else np.nan
    )
    trunk_effort_utilization = arrays["trunk_effort_utilization"]
    finite_effort_utilization = trunk_effort_utilization[np.isfinite(trunk_effort_utilization)]
    effort_utilization_p95 = (
        float(np.percentile(finite_effort_utilization, 95)) if finite_effort_utilization.size else np.nan
    )
    computed_effort_utilization = arrays["trunk_computed_effort_utilization"]
    finite_computed_effort_utilization = computed_effort_utilization[
        np.isfinite(computed_effort_utilization)
    ]
    computed_effort_utilization_p95 = (
        float(np.percentile(finite_computed_effort_utilization, 95))
        if finite_computed_effort_utilization.size else np.nan
    )
    effort_saturation_fraction = float(np.nanmean(arrays["trunk_effort_saturated"]))
    trunk_limit_margin = arrays["trunk_actual_limit_margin"]
    finite_limit_margin = trunk_limit_margin[np.isfinite(trunk_limit_margin)]
    limit_margin_p05 = (
        float(np.percentile(finite_limit_margin, 5)) if finite_limit_margin.size else np.nan
    )
    terminations = int(np.sum(arrays["done"]))
    finite_projection = arrays["manifold_projection_error"][
        np.isfinite(arrays["manifold_projection_error"])
    ]
    finite_nullspace = arrays["manifold_nullspace_residual"][
        np.isfinite(arrays["manifold_nullspace_residual"])
    ]
    finite_clip = arrays["manifold_latent_clip_fraction"][
        np.isfinite(arrays["manifold_latent_clip_fraction"])
    ]
    projection_error = float(np.mean(finite_projection)) if finite_projection.size else np.nan
    nullspace_residual = float(np.mean(finite_nullspace)) if finite_nullspace.size else np.nan
    latent_clip = float(np.mean(finite_clip)) if finite_clip.size else np.nan
    term_reasons = arrays["termination_reason"][arrays["termination_reason"] != ""]
    unique_reasons, reason_counts = np.unique(term_reasons, return_counts=True)
    reason_summary = ", ".join(f"{name}={count}" for name, count in zip(unique_reasons, reason_counts)) or "none"
    print(f"[INFO] Diagnostic ({len(diagnostic['step'])} samples) → {diagnostic['path']}")
    print(
        "[INFO] Counterfactual metrics: "
        f"contact_rate={contact_rate:.3f}  ball_pelvis_xy={ball_distance:.3f} m  "
        f"legal_touch={legal_touch_rate:.3f}  cg_eligible={cg_eligible_touch_rate:.3f}  "
        f"generic_instep={generic_instep_rate:.3f}  "
        f"cg_foot_match={cg_foot_match_rate:.3f}  "
        f"ball_xy_speed={ball_speed:.3f} m/s  ball_cmd_speed={ball_forward_speed:.3f} m/s  "
        f"pelvis_xy_speed={pelvis_speed:.3f} m/s  "
        f"local_twist_abs_err=(vx={local_vx_error:.3f}, vy={local_vy_error:.3f}, wz={local_wz_error:.3f})  "
        f"twist_blend_alpha={final_twist_blend_alpha:.3f}  "
        f"task_states=({task_state_summary})  stop_settled={stop_settle_rate:.3f}  "
        f"stop_successes={stop_successes}  "
        f"stop_ball_speed={stop_ball_speed:.3f} m/s  stop_pelvis_speed={stop_pelvis_speed:.3f} m/s  "
        f"stop_pelvis_w={stop_pelvis_angular_speed:.3f} rad/s  "
        f"foot_ref_err={foot_error:.3f} m  mean_abs_heading_err={heading_error:.3f} rad  "
        f"arm_joint_err={arm_error:.3f} rad  waist_joint_err={trunk_error:.3f} rad  "
        f"waist_effective_action_step={trunk_action_step:.3f}  action_snapshot={action_snapshot_coverage:.3f}  "
        f"post_state_valid={post_step_state_coverage:.3f}  "
        f"torso_rel_tilt={torso_rel_tilt:.3f} rad  "
        f"torso_rel_tilt_err={torso_rel_tilt_error:.3f} rad  "
        f"torso_rel_ang_vel={torso_rel_ang_vel:.3f} rad/s  manifold_projection={projection_error:.3f} rad  "
        f"manifold_nullspace={nullspace_residual:.3f} rad  "
        f"waist_target_err={trunk_target_error:.3f} rad  "
        f"waist_target_ref_offset={trunk_target_reference_offset:.3f} rad  "
        f"waist_pitch_filter_delta={trunk_pitch_filter_delta:.3f} rad  "
        f"waist_pitch_overflow={trunk_pitch_overflow:.3f} rad  "
        f"waist_computed_effort_util_p95={computed_effort_utilization_p95:.3f}  "
        f"waist_effort_util_p95={effort_utilization_p95:.3f}  "
        f"waist_effort_sat={effort_saturation_fraction:.3f}  "
        f"waist_limit_margin_p05={limit_margin_p05:.3f} rad  "
        f"latent_clip={latent_clip:.3f}  terminations={terminations} ({reason_summary})"
    )
    if np.any(s2_window_mask):
        level_history = arrays["s2_curriculum_level_entry_iteration"].astype(
            np.int64, copy=False
        )
        history_summary = ", ".join(
            f"L{level}={int(iteration)}"
            for level, iteration in enumerate(level_history.tolist())
        )
        print(
            "[INFO] S2 contact labels: "
            f"level={int(arrays['s2_curriculum_level'][-1])}  "
            f"checkpoint_iteration={int(arrays['s2_checkpoint_iteration'])}  "
            f"episode_contacts={int(arrays['s2_episode_contact_count'][-1])}  "
            f"window_samples={int(np.count_nonzero(s2_window_mask))}  "
            f"actual_target_distance={s2_region_distance:.3f} m  "
            f"ankle_origin_to_ball_center_distance={s2_physical_distance:.3f} m  "
            f"release_score={s2_release_score:.3f}  "
            f"release_speed_ratio={s2_release_speed_ratio:.3f}  "
            f"release_quality_pulses={s2_release_quality_pulses}  "
            f"reference_target_distance={s2_reference_region_distance:.3f} m  "
            "side_ids=(left=0, right=1, dead/unknown=-1)"
        )
        print(f"[INFO] S2 level entry iterations: {history_summary or 'unavailable'}")


def _load_local_twist_command_plan(plan_path: str) -> tuple[list[tuple[float, float, float, float]], bool, bool]:
    """Read and validate a reproducible pelvis-local Cartesian command plan."""
    path = pathlib.Path(plan_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Locomotion command plan not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in locomotion command plan {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Locomotion command plan must be a JSON object.")
    frame = str(payload.get("frame", "pelvis_local")).strip().lower()
    if frame != "pelvis_local":
        raise ValueError(
            f"Locomotion command plan frame must be 'pelvis_local', got {payload.get('frame')!r}."
        )
    raw_segments = payload.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise ValueError("Locomotion command plan needs a non-empty 'segments' array.")
    segments: list[tuple[float, float, float, float]] = []
    required = ("vx", "vy", "wz", "duration_s")
    for index, raw_segment in enumerate(raw_segments):
        if not isinstance(raw_segment, dict):
            raise ValueError(f"Plan segment {index} must be an object.")
        missing = [field for field in required if field not in raw_segment]
        if missing:
            raise ValueError(f"Plan segment {index} is missing {missing}.")
        values = tuple(float(raw_segment[field]) for field in required)
        if not all(np.isfinite(value) for value in values):
            raise ValueError(f"Plan segment {index} contains a non-finite value.")
        if values[3] <= 0.0:
            raise ValueError(f"Plan segment {index} duration_s must be positive.")
        segments.append(values)
    return segments, bool(payload.get("loop", True)), bool(payload.get("reset_on_end", False))


def _apply_play_locomotion_command(env, args_cli) -> bool:
    """Apply a JSON local-twist plan, polar segments, or one direct twist."""
    plan_set = args_cli.locomotion_cmd_plan is not None
    polar_set = (
        args_cli.locomotion_cmd_speed is not None
        or args_cli.locomotion_cmd_heading is not None
        or args_cli.locomotion_cmd_duration is not None
        or args_cli.locomotion_task_state is not None
    )
    cartesian_fields = (
        args_cli.locomotion_cmd_vx,
        args_cli.locomotion_cmd_vy,
        args_cli.locomotion_cmd_vz,
        args_cli.locomotion_cmd_wx,
        args_cli.locomotion_cmd_wy,
        args_cli.locomotion_cmd_wz,
    )
    if not plan_set and not polar_set and all(v is None for v in cartesian_fields):
        return False

    base_env = _resolve_base_env(env)
    cmd = base_env.command_manager.get_term("motion")
    if not hasattr(cmd, "set_locomotion_manual_command"):
        print("[WARN] Task motion command has no manual locomotion API; --locomotion_cmd_* ignored.")
        return False

    if plan_set:
        if polar_set or any(value is not None for value in cartesian_fields):
            raise ValueError(
                "--locomotion_cmd_plan cannot be combined with any other --locomotion_cmd_* value."
            )
        if str(getattr(cmd.cfg, "locomotion_command_frame", "world")) != "pelvis_local":
            raise ValueError("--locomotion_cmd_plan requires a pelvis-local locomotion task.")
        if not hasattr(cmd, "set_locomotion_local_twist_sequence"):
            raise RuntimeError("Task motion command does not support local-twist sequences.")
        segments, loop, reset_on_end = _load_local_twist_command_plan(args_cli.locomotion_cmd_plan)
        cmd.set_locomotion_local_twist_sequence(
            segments,
            hold_last=not loop,
            reset_on_end=reset_on_end,
        )
        print(f"[INFO] Local-twist command plan: {args_cli.locomotion_cmd_plan}")
        for index, (vx, vy, wz, duration_s) in enumerate(segments):
            print(
                f"  [{index + 1}] vx={vx:.3f} m/s  vy={vy:.3f} m/s  "
                f"wz={wz:.3f} rad/s  duration={duration_s:.2f} s"
            )
        print("  final segment -> environment reset -> segment 1" if reset_on_end else (
            "  looping: final segment -> first segment" if loop else "  holding: final segment remains active"
        ))
        return True

    if polar_set and hasattr(cmd, "set_locomotion_polar_command"):
        speeds = args_cli.locomotion_cmd_speed
        headings = args_cli.locomotion_cmd_heading
        durations = args_cli.locomotion_cmd_duration
        task_states = args_cli.locomotion_task_state

        if speeds is None:
            if task_states is not None:
                if len(task_states) != 1:
                    raise ValueError("State-only polar control accepts exactly one --locomotion_task_state.")
                speeds = [0.0 if task_states[0] in {"idle", "stop"} else float(
                    torch.norm(cmd.locomotion_manual_lin_vel[0, :2]).item()
                )]
            else:
                speeds = [float(torch.norm(cmd.locomotion_manual_lin_vel[0, :2]).item())]
        if headings is None:
            headings = [
                float(
                    torch.atan2(cmd.locomotion_manual_lin_vel[0, 1], cmd.locomotion_manual_lin_vel[0, 0]).item()
                )
            ]
        if durations is None:
            durations = [2.0]

        n = len(speeds)
        if len(headings) != n or len(durations) != n:
            raise ValueError(
                f"--locomotion_cmd_speed ({n}), --locomotion_cmd_heading ({len(headings)}), "
                f"and --locomotion_cmd_duration ({len(durations)}) must have the same length."
            )
        if task_states is not None and len(task_states) != n:
            raise ValueError(
                f"--locomotion_task_state ({len(task_states)}) must match --locomotion_cmd_speed ({n})."
            )
        is_local_twist = str(getattr(cmd.cfg, "locomotion_command_frame", "world")) == "pelvis_local"
        if is_local_twist:
            if task_states is not None:
                print("[WARN] local-twist tasks ignore --locomotion_task_state and keep DRIBBLE compatibility state.")
            task_states = ["dribble"] * n
        elif task_states is None:
            task_states = ["dribble" if float(speed) > 0.05 else "stop" for speed in speeds]

        wz_raw = args_cli.locomotion_cmd_wz
        if wz_raw is None or len(wz_raw) == 0:
            wz_list = [0.0] * n
        elif len(wz_raw) == 1:
            wz_list = [float(wz_raw[0])] * n
        elif len(wz_raw) == n:
            wz_list = [float(w) for w in wz_raw]
        else:
            raise ValueError(
                f"--locomotion_cmd_wz must be omitted, length 1 (broadcast), or match speed count ({n})."
            )

        if n > 1 and hasattr(cmd, "set_locomotion_polar_sequence"):
            segments = [(speeds[i], headings[i], durations[i], wz_list[i], task_states[i]) for i in range(n)]
            cmd.set_locomotion_polar_sequence(
                segments,
                hold_last=not args_cli.locomotion_cmd_loop,
                reset_on_end=args_cli.locomotion_cmd_reset_on_end,
            )
            print(f"[INFO] Locomotion sequence ({n} segments):")
            for i, (sp, hd, dur, wz, state) in enumerate(segments):
                print(
                    f"  [{i + 1}] state={state.upper():7s}  speed={sp:.3f} m/s  "
                    f"heading={hd:.3f} rad  duration={dur:.2f} s  wz={wz:.3f}"
                )
            if args_cli.locomotion_cmd_reset_on_end:
                print("  final segment -> environment reset -> segment 1")
            elif args_cli.locomotion_cmd_loop:
                print("  looping: final segment -> first segment")
            else:
                print("  holding: final segment remains active")
            return True

        cmd.set_locomotion_command_mode("manual")
        cmd.set_locomotion_polar_command(
            speed=speeds[0],
            heading=headings[0],
            duration_s=durations[0],
            wz=wz_list[0],
            task_state=task_states[0],
        )
        print(
            f"[INFO] Locomotion polar cmd: state={task_states[0].upper()}  speed={speeds[0]:.3f} m/s  "
            f"heading={headings[0]:.3f} rad  "
            f"duration={durations[0]:.2f} s  wz={wz_list[0]:.3f} rad/s"
        )
        return True

    cmd.set_locomotion_command_mode("manual")

    cur_lin = cmd.locomotion_manual_lin_vel[0].tolist()
    cur_ang = cmd.locomotion_manual_ang_vel[0].tolist()
    lin = [
        cur_lin[0] if args_cli.locomotion_cmd_vx is None else args_cli.locomotion_cmd_vx,
        cur_lin[1] if args_cli.locomotion_cmd_vy is None else args_cli.locomotion_cmd_vy,
        cur_lin[2] if args_cli.locomotion_cmd_vz is None else args_cli.locomotion_cmd_vz,
    ]
    wz_values = args_cli.locomotion_cmd_wz
    if wz_values is None or len(wz_values) == 0:
        wz = cur_ang[2]
    elif len(wz_values) == 1:
        wz = float(wz_values[0])
    else:
        raise ValueError("Cartesian manual control accepts at most one --locomotion_cmd_wz value.")
    ang = [
        cur_ang[0] if args_cli.locomotion_cmd_wx is None else args_cli.locomotion_cmd_wx,
        cur_ang[1] if args_cli.locomotion_cmd_wy is None else args_cli.locomotion_cmd_wy,
        wz,
    ]
    cartesian_task_state = None
    if args_cli.locomotion_task_state is not None:
        if len(args_cli.locomotion_task_state) != 1:
            raise ValueError("Cartesian manual control accepts exactly one --locomotion_task_state.")
        cartesian_task_state = args_cli.locomotion_task_state[0]
    is_local_twist = str(getattr(cmd.cfg, "locomotion_command_frame", "world")) == "pelvis_local"
    if is_local_twist and hasattr(cmd, "set_locomotion_local_twist_command"):
        if cartesian_task_state is not None:
            print("[WARN] local-twist tasks keep the compatibility state at DRIBBLE; --locomotion_task_state is ignored.")
        cmd.set_locomotion_local_twist_command(vx=lin[0], vy=lin[1], wz=ang[2])
        frame = "current pelvis local frame"
    else:
        cmd.set_locomotion_manual_command(lin_vel=lin, ang_vel=ang, task_state=cartesian_task_state)
        frame = "task +X/+Y/+Z frame"
    print(
        f"[INFO] Locomotion manual cmd: lin_vel={lin} m/s  ang_vel={ang} rad/s  ({frame})"
    )
    return True


def _get_play_overlay(env) -> str:
    """HUD for dual-view video: speeds, distances, contact, and CG labels."""
    lines: list[str] = []
    try:
        base_env = _resolve_base_env(env)
        cmd = base_env.command_manager.get_term("motion")
        i = 0

        if hasattr(cmd, "motion_idx") and hasattr(cmd, "motion") and hasattr(cmd.motion, "motion_name"):
            mi = int(cmd.motion_idx[i].item())
            names = cmd.motion.motion_name
            n = len(names)
            ref = names[mi] if 0 <= mi < n else f"motion_{mi}"
            lines.append(f"Ref: {ref}  ({mi + 1}/{n})")

        t = int(cmd.time_steps[i].item())
        motion_len = int(cmd.motion_length[i].item())
        if hasattr(cmd, "style_phase_reference_info"):
            style_info = cmd.style_phase_reference_info()
            cycle_len = int(style_info["cycle_length"][i].item())
            source_first = int(style_info["source_first_frame"][i].item())
            source_second = int(style_info["source_second_frame"][i].item())
            seam_blend = float(style_info["seam_blend"][i].item())
            if bool(style_info["in_seam_bridge"][i].item()):
                lines.append(
                    f"Style seam: {source_first}->{source_second}  blend={seam_blend:.2f}"
                )
            else:
                lines.append(f"Style frame: {source_first}  phase {t}/{max(cycle_len - 1, 0)}")
        else:
            lines.append(f"Motion frame: {t}/{max(motion_len - 1, 0)}")
        if hasattr(cmd, "style_phase_wrap_count"):
            wraps = int(cmd.style_phase_wrap_count[i].item())
            lines.append(f"Style wraps: {wraps}")

        pelvis_vel = cmd.robot_anchor_lin_vel_w[i].detach().cpu().numpy()
        pelvis_pos = cmd.robot_pelvis_pos_w[i].detach().cpu().numpy()
        pelvis_sp_xy = float(np.linalg.norm(pelvis_vel[:2]))

        soccer_ball = base_env.scene["soccer_ball"]
        ball_vel = soccer_ball.data.root_lin_vel_w[i].detach().cpu().numpy()
        ball_pos = soccer_ball.data.root_pos_w[i].detach().cpu().numpy()
        ball_sp_xy = float(np.linalg.norm(ball_vel[:2]))

        pelvis_ball_xy = float(np.linalg.norm(ball_pos[:2] - pelvis_pos[:2]))
        pelvis_ball_3d = float(np.linalg.norm(ball_pos - pelvis_pos))

        lines.append(
            f"Pelvis v_xy: {pelvis_sp_xy:.2f} m/s  |  Ball v_xy: {ball_sp_xy:.2f} m/s"
        )

        if hasattr(cmd, "locomotion_lin_vel_command_w"):
            mode = getattr(cmd, "locomotion_command_mode", "?")
            lcmd = cmd.locomotion_lin_vel_command_w()[i].detach().cpu().numpy()
            task_state = getattr(cmd, "locomotion_task_state", None)
            task_state_names = ("IDLE", "DRIBBLE", "STOP")
            task_state_name = "DRIBBLE"
            if isinstance(task_state, torch.Tensor):
                task_state_idx = int(task_state[i].item())
                if 0 <= task_state_idx < len(task_state_names):
                    task_state_name = task_state_names[task_state_idx]

            # heading / arrow helpers — ASCII-only for safe video font rendering
            _ARROWS = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
            def _arrow(rad):
                idx = int(((-float(rad)) / (2 * np.pi) * 8 + 0.5) % 8)
                return _ARROWS[idx]
            def _deg(rad):
                return float(rad) * 180.0 / np.pi

            # For reference (follow) mode locomotion_cmd_speed/heading are not updated;
            # derive them from the velocity command vector instead.
            if mode == "reference":
                cmd_spd = float(np.linalg.norm(lcmd[:2]))
                cmd_hdg = float(np.arctan2(lcmd[1], lcmd[0]))
            else:
                cmd_spd = cmd.locomotion_cmd_speed[i].item()
                cmd_hdg = cmd.locomotion_cmd_heading[i].item()

            # tracking quality: cosine similarity of xy velocity vectors
            cmd_xy = lcmd[:2]
            rob_xy = pelvis_vel[:2]
            cmd_norm = float(np.linalg.norm(cmd_xy))
            rob_norm = float(np.linalg.norm(rob_xy))
            if cmd_norm > 0.05 and rob_norm > 0.05:
                cos_sim = float(np.dot(cmd_xy, rob_xy) / (cmd_norm * rob_norm))
                track_str = f"  track={max(0.0, cos_sim) * 100:3.0f}%"
            else:
                track_str = "  track= n/a"

            # segment / hold info
            seg_note = ""
            if hasattr(cmd, "_locomotion_segment_plans") and cmd._locomotion_segment_plans[i]:
                seg_idx = int(cmd._locomotion_segment_idx[i].item()) + 1
                seg_note = f" seg {seg_idx}/{len(cmd._locomotion_segment_plans[i])}"
            hold_str = ""
            if mode != "reference" and hasattr(cmd, "_locomotion_cmd_hold_steps_remaining"):
                hold_str = f"  hold={int(cmd._locomotion_cmd_hold_steps_remaining[i].item()):4d}"

            # fixed-width columns so digits don't shift layout
            # col widths: spd 4, deg 4, dir 2, track 7, hold 9
            lines.append(
                f"Loco cmd ({mode}{seg_note}):"
                f"  {cmd_spd:4.2f} m/s"
                f"  {_deg(cmd_hdg):+4.0f}deg {_arrow(cmd_hdg):<2s}"
                f"{track_str}"
                f"{hold_str}"
            )
            lines.append(f"Task state         : {task_state_name}")

            # actual robot velocity vs command
            actual_hdg = float(np.arctan2(pelvis_vel[1], pelvis_vel[0]))
            hdg_err_deg = float((_deg(actual_hdg) - _deg(cmd_hdg) + 180) % 360 - 180)
            vx_err = float(pelvis_vel[0] - lcmd[0])
            vy_err = float(pelvis_vel[1] - lcmd[1])

            lines.append(
                f"Robot actual        :"
                f"  {pelvis_sp_xy:4.2f} m/s"
                f"  {_deg(actual_hdg):+4.0f}deg {_arrow(actual_hdg):<2s}"
                f"  hdg_err={hdg_err_deg:+4.0f}deg"
                f"  vx_err={vx_err:+5.2f}"
                f"  vy_err={vy_err:+5.2f}"
            )

        lines.append(
            f"Pelvis-Ball: {pelvis_ball_xy:.2f} m (xy)  |  {pelvis_ball_3d:.2f} m (3D)"
        )

        force_xy = float(
            soccer_ball_contact_force_magnitude(base_env, _BALL_SENSOR_NAME)[i].item()
        )
        sim_touch = force_xy > _CONTACT_FORCE_THRESHOLD

        ankle_parts = [f"contact={'YES' if sim_touch else 'NO'}"]
        if hasattr(cmd, "motion_has_dribble_cg_label") and bool(cmd.motion_has_dribble_cg_label[i].item()):
            ref_contact = bool(cmd.dribble_cg_contact_ref[i].item())
            match = "match" if ref_contact == sim_touch else "MISMATCH"
            ankle_parts.append(f"demo={'YES' if ref_contact else 'NO'}")
            ankle_parts.append(match)
        lines.append(f"Ankle-Ball: {'  |  '.join(ankle_parts)}")

        if hasattr(base_env, "_dribbling_no_contact_count"):
            no_contact_count = float(base_env._dribbling_no_contact_count[i].item())
            command_recovery = bool(base_env._dribbling_no_contact_recovery_active[i].item())
            proximity_recovery_tensor = getattr(
                base_env, "_dribbling_no_contact_proximity_recovery_active", None
            )
            proximity_recovery = (
                False if proximity_recovery_tensor is None else bool(proximity_recovery_tensor[i].item())
            )
            closing = float(base_env._dribbling_no_contact_closing_speed[i].item())
            relative_speed_tensor = getattr(base_env, "_dribbling_no_contact_relative_speed", None)
            relative_speed = 0.0 if relative_speed_tensor is None else float(relative_speed_tensor[i].item())
            recovery = "CMD" if command_recovery else "PROX" if proximity_recovery else "NO"
            lines.append(
                f"No-contact: count={no_contact_count:5.1f}"
                f"  recovery={recovery}"
                f"  closing={closing:+.2f} m/s  relative={relative_speed:.2f} m/s"
            )

        _update_last_termination_reason(base_env, env_idx=i)
        lines.append(f"Last term: {_LAST_TERM_REASON}")

    except Exception as e:
        lines.append(f"HUD error: {e}")

    return "\n".join(lines)

@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    if args_cli.seed is not None:
        env_cfg.seed = int(args_cli.seed)
    if args_cli.disable_obs_noise:
        policy_obs_cfg = getattr(getattr(env_cfg, "observations", None), "policy", None)
        if policy_obs_cfg is None or not hasattr(policy_obs_cfg, "enable_corruption"):
            raise ValueError(
                "--disable_obs_noise was requested, but the task has no policy "
                "observation corruption switch."
            )
        policy_obs_cfg.enable_corruption = False
        print("[INFO] Playback policy-observation corruption disabled.")
    if args_cli.exact_reset_state:
        motion_cfg = getattr(getattr(env_cfg, "commands", None), "motion", None)
        if motion_cfg is None:
            raise ValueError("--exact_reset_state requires a motion command.")
        zero_pose_range = {
            key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        motion_cfg.pose_range = dict(zero_pose_range)
        motion_cfg.velocity_range = dict(zero_pose_range)
        motion_cfg.joint_position_range = (0.0, 0.0)
        print(
            "[INFO] Playback exact reset state enabled: "
            "pose/velocity/joint perturbations disabled."
        )
    if hasattr(env_cfg.commands, "motion"):
        motion_cfg = env_cfg.commands.motion
        replay_ball_overrides = []
        if args_cli.replay_ball_spawn_mode is not None:
            setattr(motion_cfg, "dribble_cg_ball_spawn_mode", args_cli.replay_ball_spawn_mode)
            setattr(motion_cfg, "soccer_ball_spawn_mode", args_cli.replay_ball_spawn_mode)
            replay_ball_overrides.append(f"spawn_mode={args_cli.replay_ball_spawn_mode}")
        if args_cli.replay_ball_fallback_mode is not None:
            setattr(motion_cfg, "dribble_cg_fallback_ball_mode", args_cli.replay_ball_fallback_mode)
            replay_ball_overrides.append(f"fallback={args_cli.replay_ball_fallback_mode}")
        if args_cli.replay_ball_front_distance is not None:
            setattr(motion_cfg, "dribble_cg_front_ball_distance", float(args_cli.replay_ball_front_distance))
            setattr(motion_cfg, "soccer_ball_start_ahead_distance", float(args_cli.replay_ball_front_distance))
            replay_ball_overrides.append(f"front_distance={args_cli.replay_ball_front_distance:.3f}m")
        if args_cli.replay_ball_observation_mode is not None:
            setattr(
                motion_cfg,
                "dribble_cg_replay_ball_observation_mode",
                args_cli.replay_ball_observation_mode,
            )
            replay_ball_overrides.append(f"obs_mode={args_cli.replay_ball_observation_mode}")
        if args_cli.replay_ball_observation_distance is not None:
            setattr(
                motion_cfg,
                "dribble_cg_replay_ball_observation_distance",
                float(args_cli.replay_ball_observation_distance),
            )
            replay_ball_overrides.append(f"obs_distance={args_cli.replay_ball_observation_distance:.3f}m")
        if replay_ball_overrides:
            print("[INFO] Replay-only ball overrides: " + ", ".join(replay_ball_overrides))
    if args_cli.show_s2_contact_regions and hasattr(env_cfg.commands, "motion"):
        env_cfg.commands.motion.debug_vis = True
        setattr(env_cfg.commands.motion, "dribble_cg_s2_debug_regions_only", True)

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

    total_play_steps = _setup_play_episode_limit(
        env_cfg,
        motion_files,
        keep_failure_terms=not args_cli.disable_training_terminations,
    )
    term_note = ", terminations off" if args_cli.disable_training_terminations else ""
    if len(motion_files) > 1:
        print(
            f"[INFO] Sequential playback: {len(motion_files)} references, "
            f"{total_play_steps} steps total (episode cap {env_cfg.episode_length_s:.1f}s{term_note})"
        )
    elif motion_files:
        print(
            f"[INFO] Playback episode cap {env_cfg.episode_length_s:.1f}s "
            f"({total_play_steps} frames{term_note})"
        )

    if args_cli.record_all_motions and len(motion_files) <= 1:
        raise ValueError("--record_all_motions requires --motion_path with multiple .npz files.")

    play_steps = args_cli.video_length
    if play_steps is None:
        if len(motion_files) > 1 or args_cli.video or args_cli.dual_view:
            play_steps = total_play_steps
        else:
            play_steps = None
    elif args_cli.record_all_motions and len(motion_files) > 1:
        print(f"[INFO] --record_all_motions: using explicit --video_length={play_steps}")

    waist_roll_stiffness_scale = _apply_play_waist_roll_stiffness_scale(
        env_cfg, args_cli.waist_roll_stiffness_scale
    )

    if args_cli.diagnostic:
        # A detailed rigid-contact view must be created while sensors initialize,
        # before the first physics step.  Creating it later yields a valid-looking
        # GPU view whose per-point buffers remain empty in Isaac Lab 2.1.
        diagnostic_contact_cfg = env_cfg.scene.soccer_ball_contact
        diagnostic_contact_cfg.class_type = _DiagnosticDetailedContactSensor
        configured_robot_bodies = list(
            getattr(env_cfg.commands.motion, "body_names", ())
        )
        event_filter_foot_names = [
            "right_ankle_roll_link",
            "left_ankle_roll_link",
        ]
        diagnostic_body_names = [
            *event_filter_foot_names,
            *(
                body_name
                for body_name in configured_robot_bodies
                if body_name not in event_filter_foot_names
            ),
        ]
        if len(diagnostic_body_names) <= len(_FOOT_DIAGNOSTIC_BODY_NAMES):
            raise ValueError(
                "Diagnostic all-link contact audit could not resolve the configured robot bodies."
            )
        # The all-link inventory is an independent detailed PhysX view.  Do
        # not replace ``diagnostic_contact_cfg.filter_prim_paths_expr`` here:
        # that high-level matrix is consumed by the S2 contact state machine
        # and must remain the task's right-foot/left-foot pair.
        _DiagnosticDetailedContactSensor.diagnostic_filter_prim_paths_expr = tuple(
            f"/World/envs/env_.*/Robot/{body_name}"
            for body_name in diagnostic_body_names
        )

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    log_dir = os.path.dirname(resume_path)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    _apply_play_locomotion_command(env, args_cli)

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    load_checkpoint_with_obs_expand(ppo_runner, resume_path)
    if args_cli.s2_curriculum_level is not None:
        resolved_level = set_s2_curriculum_level(
            ppo_runner, args_cli.s2_curriculum_level, reset_env=True
        )
        print(f"[INFO] S2 curriculum: playback override set level {resolved_level}.")

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    if args_cli.stochastic_policy:
        if getattr(ppo_runner, "empirical_normalization", False):
            policy = lambda x: ppo_runner.alg.policy.act(ppo_runner.obs_normalizer(x))
        else:
            policy = ppo_runner.alg.policy.act
        print(
            "[INFO] Playback stochastic policy enabled: sampling checkpoint action distribution "
            "(diagnostic only)."
        )

    reference_constraints: list[dict] = []
    if args_cli.upper_body_reference_margin is not None:
        upper_body_constraint = _create_upper_body_reference_constraint(
            env, args_cli.upper_body_reference_margin, args_cli.upper_body_constraint_group
        )
        reference_constraints.append(upper_body_constraint)
        print(
            "[INFO] Play-only upper-body reference constraint enabled: "
            f"group={upper_body_constraint['group']}  "
            f"q_target ∈ q_ref ± {upper_body_constraint['margin']:.3f} rad"
        )

    if args_cli.waist_reference_margin is not None:
        waist_constraint = _create_waist_reference_constraint(env, args_cli.waist_reference_margin)
        reference_constraints.append(waist_constraint)
        print(
            "[INFO] Play-only waist reference constraint enabled: "
            f"q_target within q_ref +/- {waist_constraint['margin']:.3f} rad"
        )

    diagnostic = None
    if args_cli.diagnostic:
        diagnostic = _create_diagnostic(
            env,
            log_dir,
            args_cli.diagnostic_stride,
            reference_constraints,
            waist_roll_stiffness_scale,
        )
        print("[INFO] Diagnostic scope: arms + waist + pelvis/torso motion.")
        print(f"[INFO] Diagnostic enabled (stride={diagnostic['stride']}) → {diagnostic['path']}")

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
    
    # --- Video recorder with HUD overlay (--video and/or --dual_view) ---
    video_recorder = None
    if args_cli.video or args_cli.dual_view:
        from dual_view_recorder import DualViewRecorder, resolve_camera_offsets

        tag = "dual" if args_cli.dual_view else "play"
        video_dir = os.path.join(
            log_dir, "videos", f"{tag}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        front_offset, back_offset = resolve_camera_offsets(layout=args_cli.cam_layout)
        video_recorder = DualViewRecorder(
            env=env.unwrapped if hasattr(env, "unwrapped") else env,
            output_dir=video_dir,
            resolution=(960, 540),
            front_offset=front_offset,
            back_offset=back_offset,
            lookat_offset=0.5,
            fps=30,
            path_tracing=args_cli.path_tracing,
            dual=args_cli.dual_view,
        )
        video_recorder.setup()
        for _ in range(5):
            env.unwrapped.sim.render()
        mode = "dual-view" if args_cli.dual_view else "single-view"
        steps_label = play_steps if play_steps is not None else "unlimited"
        print(f"[INFO] {mode} recording ({steps_label} steps) → {video_dir}")
        print("[INFO] HUD: Ref name, speeds, distances, ankle-ball contact, last term.")
    elif play_steps is not None:
        print(f"[INFO] Running {play_steps} steps (sequential references)")

    # reset environment
    global _LAST_TERM_REASON
    _LAST_TERM_REASON = "-"
    obs, _ = env.get_observations()
    base_env = _resolve_base_env(env)
    counterfactual_ball_state = None
    if args_cli.counterfactual_ball_event is not None:
        counterfactual_ball_state = _create_counterfactual_ball_event_state(
            env,
            args_cli.counterfactual_ball_event,
            args_cli.counterfactual_ball_lead_frames,
        )
        print(
            "[INFO] Frozen-policy counterfactual ball event enabled: "
            f"event={args_cli.counterfactual_ball_event}  "
            f"lead_frames={args_cli.counterfactual_ball_lead_frames}."
        )
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            if _apply_counterfactual_ball_event(env, counterfactual_ball_state):
                # Refresh observations after the simulator-side ball write so
                # a deployable ball observation, when present, is not stale.
                obs, _ = env.get_observations()
            policy_actions = policy(obs)
            actions = policy_actions
            for reference_constraint in reference_constraints:
                actions = _constrain_reference_actions(env, actions, reference_constraint)
            recorded_diagnostic_sample = False
            if diagnostic is not None:
                recorded_diagnostic_sample = _append_diagnostic(
                    diagnostic, env, policy_actions, actions, timestep
                )
            # env stepping
            obs, reward, dones, _ = env.step(actions)
            # ``get_inference_policy`` returns a callable (and wraps it in a
            # normalizer lambda), so it does not expose the recurrent reset
            # method.  Explicitly clear actor/critic memory for environments
            # that reset; otherwise an RNN checkpoint carries the previous
            # episode's hidden state into frame 0 and repeated playback after
            # one failure becomes invalid.
            if bool(torch.any(dones)) and hasattr(ppo_runner.alg.policy, "reset"):
                ppo_runner.alg.policy.reset(dones)
            if counterfactual_ball_state is not None:
                counterfactual_ball_state["injected"][dones.to(torch.bool)] = False
            if recorded_diagnostic_sample:
                _append_upper_body_manifold_diagnostic(diagnostic, env, dones)
                _complete_cg_flow_diagnostic(diagnostic, env)
                diagnostic["step_reward"][-1] = float(reward[0].item())
                diagnostic["done"].append(bool(dones[0].item()))
                diagnostic["reward_terms"].append(_reward_term_values(base_env))
                diagnostic["termination_reason"].append(
                    _active_failure_termination_reason(base_env) if bool(dones[0].item()) else ""
                )

        if video_recorder is not None:
            overlay = _get_play_overlay(env)
            video_recorder.capture(overlay_text=overlay)

        timestep += 1
        if play_steps is not None and timestep >= play_steps:
            break

    if video_recorder is not None:
        video_recorder.save()

    if diagnostic is not None:
        _save_diagnostic(diagnostic)

    if counterfactual_ball_state is not None:
        print(
            "[INFO] Counterfactual ball injections completed: "
            f"{counterfactual_ball_state['count']}"
        )

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
