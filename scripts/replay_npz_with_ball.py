"""Replay a motion ``.npz`` with the G1 robot and visualize the soccer ball trajectory.

Loads ``ball_pos_w`` from the motion file (e.g. after ``synthesize_dribble_ball_traj.py``)
and drives a kinematic soccer ball mesh each frame. Optionally draws a rolling trail of
marker spheres and prints CG contact / foot–ball distance in the terminal.

``--motion_path`` may be a single ``.npz`` or a directory (all ``*.npz`` inside, sorted).

.. code-block:: bash

    # Single clip
    python scripts/replay_npz_with_ball.py \\
        --motion_path motions/dribble-distance/FAST-seg1_unitree_g1.npz

    # Whole folder (plays each clip in order; loops folder by default)
    python scripts/replay_npz_with_ball.py \\
        --motion_path motions/dribble-distance --device cpu

    # Headless MP4 recording (SSH-friendly) with HUD overlay
    python scripts/replay_npz_with_ball.py \\
        --motion_path motions/dribble-distance \\
        --headless --dual_view --no_loop --no_playlist_loop

    # No GPU machine — always use CPU to avoid cudaErrorNoDevice
    python scripts/replay_npz_with_ball.py \\
        --motion_path motions/dribble-distance --device cpu --real_time
"""

from __future__ import annotations

import argparse
import collections
import datetime
import glob
import os
import sys
import time

import numpy as np
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay motion .npz with ball trajectory visualization.")
parser.add_argument(
    "--motion_path",
    type=str,
    required=True,
    help="Path to a motion .npz or a directory of .npz files.",
)
parser.add_argument(
    "--trail_length",
    type=int,
    default=80,
    help="Number of past ball positions drawn as trail markers (0 to disable).",
)
parser.add_argument(
    "--trail_stride",
    type=int,
    default=2,
    help="Subsample trail markers every N frames (reduces clutter).",
)
parser.add_argument(
    "--real_time",
    action="store_true",
    help="Sleep so playback matches motion fps (default: run as fast as sim can render).",
)
parser.add_argument(
    "--loop",
    action="store_true",
    default=True,
    help="Loop the current clip when finished (default: True).",
)
parser.add_argument(
    "--no_loop",
    action="store_false",
    dest="loop",
    help="Stop after one play-through of the current clip.",
)
parser.add_argument(
    "--playlist_loop",
    action="store_true",
    default=True,
    help="When motion_path is a directory, loop the full playlist (default: True).",
)
parser.add_argument(
    "--no_playlist_loop",
    action="store_false",
    dest="playlist_loop",
    help="Stop after playing every file in the directory once.",
)
parser.add_argument(
    "--gap_seconds",
    type=float,
    default=0.8,
    help="Pause between clips when playing a directory (seconds).",
)
parser.add_argument(
    "--video",
    action="store_true",
    default=False,
    help="Record offscreen video (single camera). Implies --headless and enables cameras.",
)
parser.add_argument(
    "--dual_view",
    action="store_true",
    default=False,
    help="Record split-screen front+back video with HUD overlay.",
)
parser.add_argument(
    "--cam_layout",
    type=str,
    default="task_front",
    choices=["diagonal", "task_front", "task_front_side"],
    help="Camera preset. task_front: on +X side, looks ~ -X (robot frontal); diagonal: legacy oblique.",
)
parser.add_argument(
    "--video_output_dir",
    type=str,
    default=None,
    help="Directory for MP4 output (default: <motion_dir>/videos/replay_<timestamp>/).",
)
parser.add_argument(
    "--path_tracing",
    action="store_true",
    default=False,
    help="Use path-tracing renderer for higher quality (slower).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video or args_cli.dual_view:
    args_cli.enable_cameras = True
    args_cli.headless = True
    args_cli.loop = False
    args_cli.playlist_loop = False


def _resolve_sim_device(requested: str | None) -> str:
    """Pick a valid Isaac Sim device; fall back to CPU if CUDA is unavailable."""
    if requested is None or str(requested).strip() == "":
        requested = "cuda:0"
    dev = str(requested).strip().lower()
    if dev == "cuda":
        dev = "cuda:0"
    wants_cuda = dev.startswith("cuda")
    if wants_cuda and not torch.cuda.is_available():
        print(
            "[WARN] No CUDA-capable GPU detected (cudaErrorNoDevice). "
            "Using --device cpu. Set explicitly: --device cpu"
        )
        return "cpu"
    return requested if not wants_cuda or torch.cuda.is_available() else "cpu"


# Ensure AppLauncher / SimulationContext never default to missing CUDA.
args_cli.device = _resolve_sim_device(getattr(args_cli, "device", None))

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from soccer.robots.g1 import G1_CYLINDER_CFG
from soccer.tasks.tracking.config.g1.soccer_flat_env_cfg import SOCCER_ASSET_PATH, SOCCER_BALL_RADIUS
from soccer.tasks.tracking.mdp.commands import MotionLoader

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "rsl_rl"))
from dual_view_recorder import DualViewRecorder, resolve_camera_offsets  # noqa: E402


class _ReplayRecorderEnv:
    """Thin adapter so DualViewRecorder can follow the replay scene robot."""

    def __init__(self, sim: SimulationContext, scene: InteractiveScene):
        self.sim = sim
        self.scene = scene


def collect_motion_files(motion_path: str) -> list[str]:
    """Return sorted list of .npz paths (single file or directory)."""
    if os.path.isfile(motion_path):
        return [motion_path]
    if os.path.isdir(motion_path):
        files = sorted(glob.glob(os.path.join(motion_path, "*.npz")))
        if not files:
            raise ValueError(f"No .npz files found in directory: {motion_path}")
        print(f"[playlist] {len(files)} motion file(s) in {motion_path}")
        for f in files:
            print(f"  - {os.path.basename(f)}")
        return files
    raise ValueError(f"Invalid motion_path: {motion_path} (must be a file or directory)")


class SoccerMotionReplayData:
    """Motion loader extended with optional ball / CG arrays from ``.npz``."""

    def __init__(self, motion_file: str, device: str):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        self.motion_file = motion_file
        self.motion_name = os.path.basename(motion_file)
        data = np.load(motion_file)

        self.fps = float(np.asarray(data["fps"]).reshape(-1)[0]) if "fps" in data.files else 50.0
        self.time_step_total = int(data["joint_pos"].shape[0])

        self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
        self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self._body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)

        self.has_ball = "ball_pos_w" in data.files
        if self.has_ball:
            ball = np.asarray(data["ball_pos_w"], dtype=np.float32).reshape(-1, 3)[: self.time_step_total]
            self.ball_pos_w = torch.tensor(ball, dtype=torch.float32, device=device)
            if self.time_step_total > 1:
                vel = np.zeros_like(ball)
                vel[1:] = (ball[1:] - ball[:-1]) * self.fps
                vel[0] = vel[1]
                self.ball_lin_vel_w = torch.tensor(vel, dtype=torch.float32, device=device)
            else:
                self.ball_lin_vel_w = torch.zeros(1, 3, device=device, dtype=torch.float32)
        else:
            self.ball_pos_w = None
            self.ball_lin_vel_w = None

        self.cg_contact = None
        if "dribble_cg_contact" in data.files:
            cc = np.asarray(data["dribble_cg_contact"], dtype=np.int8).reshape(-1)[: self.time_step_total]
            self.cg_contact = torch.tensor(cc, device=device, dtype=torch.bool)

        self.contact_anchor = None
        if "dribble_contact_anchor" in data.files:
            ca = np.asarray(data["dribble_contact_anchor"], dtype=np.int8).reshape(-1)[: self.time_step_total]
            self.contact_anchor = torch.tensor(ca, device=device, dtype=torch.bool)

        self.cg_foot_ball_dist = None
        if "dribble_cg_foot_ball_dist" in data.files:
            cd = np.asarray(data["dribble_cg_foot_ball_dist"], dtype=np.float32).reshape(-1)[: self.time_step_total]
            self.cg_foot_ball_dist = torch.tensor(cd, device=device, dtype=torch.float32)

        self.cg_foot = None
        if "dribble_cg_foot" in data.files:
            cf = np.asarray(data["dribble_cg_foot"], dtype=np.int8).reshape(-1)[: self.time_step_total]
            self.cg_foot = torch.tensor(cf, device=device, dtype=torch.int8)

        self.cg_surface = None
        if "dribble_cg_surface" in data.files:
            cs = np.asarray(data["dribble_cg_surface"], dtype=np.int8).reshape(-1)[: self.time_step_total]
            self.cg_surface = torch.tensor(cs, device=device, dtype=torch.int8)

        self._body_index = torch.tensor([0], dtype=torch.long, device=device)


@configclass
class ReplayWithBallSceneCfg(InteractiveSceneCfg):
    """Robot + soccer ball + optional debug markers."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    robot: ArticulationCfg = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    soccer_ball: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SoccerBall",
        spawn=sim_utils.UsdFileCfg(
            usd_path=SOCCER_ASSET_PATH,
            activate_contact_sensors=False,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, SOCCER_BALL_RADIUS)),
    )


def _make_trail_markers() -> VisualizationMarkers:
    cfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/BallTrail",
        markers={
            "trail": sim_utils.SphereCfg(
                radius=SOCCER_BALL_RADIUS * 0.55,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.85, 0.1)),
            ),
        },
    )
    return VisualizationMarkers(cfg)


def _make_contact_markers() -> VisualizationMarkers:
    cfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/BallContact",
        markers={
            "contact": sim_utils.SphereCfg(
                # A small red anchor makes the manually labelled contact
                # location visible after the ball itself is hidden.
                radius=SOCCER_BALL_RADIUS * 0.28,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.08, 0.08)),
            ),
        },
    )
    return VisualizationMarkers(cfg)


def _update_trail(
    trail: collections.deque[np.ndarray],
    trail_markers: VisualizationMarkers | None,
    ball_pos: np.ndarray,
    trail_length: int,
    trail_stride: int,
    device: str,
) -> None:
    if trail_markers is None or trail_length <= 0:
        return
    trail.append(ball_pos.copy())
    while len(trail) > trail_length:
        trail.popleft()
    positions = np.stack(list(trail)[:: max(1, trail_stride)], axis=0)
    if positions.shape[0] == 0:
        return
    orientations = np.zeros((positions.shape[0], 4), dtype=np.float32)
    orientations[:, 0] = 1.0
    trail_markers.visualize(
        translations=torch.tensor(positions, dtype=torch.float32, device=device),
        orientations=torch.tensor(orientations, dtype=torch.float32, device=device),
    )


def _load_motion(motion_path: str, device: str) -> tuple[SoccerMotionReplayData, MotionLoader]:
    motion = SoccerMotionReplayData(motion_path, device=device)
    motion_legacy = MotionLoader(motion_path, torch.tensor([0], device=device), device)
    if not motion.has_ball:
        print(
            f"[WARN] {motion.motion_name}: no ball_pos_w — robot only. "
            "Run: python scripts/dribble/synthesize_dribble_ball_traj.py --motion_path <dir>"
        )
    return motion, motion_legacy


def _default_video_output_dir(motion_path: str) -> str:
    base = motion_path if os.path.isdir(motion_path) else os.path.dirname(os.path.abspath(motion_path))
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(base, "videos", f"replay_{ts}")


def _build_video_overlay(
    file_idx: int,
    motion_files: list[str],
    motion: SoccerMotionReplayData,
    frame_idx: int,
) -> str:
    lines = [
        f"Ref: {motion.motion_name}  ({file_idx + 1}/{len(motion_files)})",
        f"Frame: {frame_idx}/{max(motion.time_step_total - 1, 0)}  "
        f"t={frame_idx / motion.fps:.2f}s  fps={motion.fps:.0f}",
    ]
    if motion.has_ball:
        bp = motion.ball_pos_w[frame_idx].detach().cpu().numpy()
        spd = float(torch.norm(motion.ball_lin_vel_w[frame_idx]).item())
        lines.append(
            f"Ball: ({bp[0]:+.2f}, {bp[1]:+.2f}, {bp[2]:.2f}) m  |v|={spd:.2f} m/s"
        )
    if motion.cg_contact is not None:
        contact = int(motion.cg_contact[frame_idx].item())
        lines.append(f"CG contact: {contact}")
    if motion.contact_anchor is not None:
        anchor = int(motion.contact_anchor[frame_idx].item())
        lines.append(f"Contact anchor: {anchor}")
    if motion.cg_foot is not None:
        foot_id = int(motion.cg_foot[frame_idx].item())
        foot_lbl = {0: "L", 1: "R"}.get(foot_id, "-")
        lines.append(f"CG foot: {foot_lbl}")
    if motion.cg_surface is not None:
        surface_id = int(motion.cg_surface[frame_idx].item())
        surface_lbl = {0: "inside", 1: "outside"}.get(surface_id, "-")
        lines.append(f"CG surface: {surface_lbl}")
    if motion.cg_foot_ball_dist is not None:
        d = float(motion.cg_foot_ball_dist[frame_idx].item())
        lines.append(f"Foot-Ball dist: {d:.3f} m" if d >= 0.0 else "Foot-Ball dist: -")
    return "\n".join(lines)


def _make_video_recorder(sim: SimulationContext, scene: InteractiveScene, motion: SoccerMotionReplayData):
    output_dir = args_cli.video_output_dir or _default_video_output_dir(args_cli.motion_path)
    front_offset, back_offset = resolve_camera_offsets(layout=args_cli.cam_layout)
    recorder = DualViewRecorder(
        env=_ReplayRecorderEnv(sim, scene),
        output_dir=output_dir,
        resolution=(960, 540),
        front_offset=front_offset,
        back_offset=back_offset,
        lookat_offset=0.5,
        fps=max(1, int(round(motion.fps))),
        path_tracing=args_cli.path_tracing,
        dual=args_cli.dual_view,
    )
    recorder.setup()
    for _ in range(5):
        sim.render()
    mode = "dual-view" if args_cli.dual_view else "single-view"
    print(f"[video] {mode} recording → {output_dir}")
    return recorder, output_dir


def run_simulator(sim: SimulationContext, scene: InteractiveScene, motion_files: list[str]) -> None:
    robot: Articulation = scene["robot"]
    soccer_ball: RigidObject = scene["soccer_ball"]
    sim_dt = sim.get_physics_dt()
    multi_file = len(motion_files) > 1

    file_idx = 0
    motion, motion_legacy = _load_motion(motion_files[file_idx], sim.device)
    frame_dt = 1.0 / motion.fps if motion.fps > 0 else sim_dt

    time_steps = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)
    trail_markers = _make_trail_markers() if args_cli.trail_length > 0 else None
    contact_markers = _make_contact_markers()
    trail_deque: collections.deque[np.ndarray] = collections.deque()
    record_video = args_cli.video or args_cli.dual_view
    video_recorder = None
    video_output_dir = ""
    if record_video:
        video_recorder, video_output_dir = _make_video_recorder(sim, scene, motion)

    def _save_clip_video() -> None:
        nonlocal video_recorder
        if video_recorder is None or video_recorder.num_frames == 0:
            return
        stem = os.path.splitext(motion.motion_name)[0]
        video_recorder.save(filename=f"{stem}.mp4")

    def _switch_clip(next_idx: int) -> None:
        nonlocal file_idx, motion, motion_legacy, frame_dt, time_steps, trail_deque, video_recorder
        _save_clip_video()
        file_idx = next_idx
        print(f"\n[playlist] ({file_idx + 1}/{len(motion_files)}) {motion_files[file_idx]}")
        motion, motion_legacy = _load_motion(motion_files[file_idx], sim.device)
        frame_dt = 1.0 / motion.fps if motion.fps > 0 else sim_dt
        time_steps[:] = 0
        trail_deque.clear()
        if video_recorder is not None:
            video_recorder._fps = max(1, int(round(motion.fps)))

    print(f"[sim] device={sim.device}")
    print(f"[playlist] clip 1/{len(motion_files)}: {motion_files[0]}")
    if record_video:
        print("[video] loop/playlist_loop disabled for recording")

    while simulation_app.is_running():
        frame_start = time.perf_counter()
        t = int(time_steps[0].item())

        if t >= motion.time_step_total:
            if not multi_file:
                if args_cli.loop:
                    time_steps[:] = 0
                    trail_deque.clear()
                else:
                    print("\n[done] finished clip.")
                    _save_clip_video()
                    break
            else:
                next_idx = file_idx + 1
                if next_idx >= len(motion_files):
                    if args_cli.playlist_loop:
                        next_idx = 0
                    else:
                        print("\n[playlist] finished all clips.")
                        _save_clip_video()
                        break
                if args_cli.gap_seconds > 0:
                    time.sleep(args_cli.gap_seconds)
                _switch_clip(next_idx)
            continue

        time_steps += 1
        ts = torch.tensor([t], dtype=torch.long, device=sim.device)

        root_states = robot.data.default_root_state.clone()
        origin = scene.env_origins
        root_states[:, :3] = motion_legacy.body_pos_w[ts] + origin
        root_states[:, 3:7] = motion_legacy.body_quat_w[ts]
        root_states[:, 7:10] = motion_legacy.body_lin_vel_w[ts]
        root_states[:, 10:] = motion_legacy.body_ang_vel_w[ts]
        robot.write_root_state_to_sim(root_states)
        robot.write_joint_state_to_sim(motion_legacy.joint_pos[ts], motion_legacy.joint_vel[ts])

        if motion.has_ball:
            ball_pos_w = motion.ball_pos_w[t] + origin[0]
            ball_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=sim.device)
            ball_lin_vel = motion.ball_lin_vel_w[t].unsqueeze(0)
            ball_ang_vel = torch.zeros(1, 3, device=sim.device)
            is_contact = motion.cg_contact is not None and bool(motion.cg_contact[t].item())
            # Keep the reference ball visible.  The contact marker below adds
            # a small red point at its labelled position during contact.
            ball_state = torch.cat([ball_pos_w.unsqueeze(0), ball_quat, ball_lin_vel, ball_ang_vel], dim=-1)
            soccer_ball.write_root_state_to_sim(ball_state)

            ball_np = ball_pos_w.detach().cpu().numpy()
            _update_trail(
                trail_deque,
                trail_markers,
                ball_np,
                args_cli.trail_length,
                args_cli.trail_stride,
                sim.device,
            )

            if motion.cg_contact is not None:
                if is_contact:
                    # The ball mesh occludes a marker placed at its centre.
                    # Raise the dot just above the ball so the labelled touch
                    # remains visible from either recording camera.
                    visible_contact_pos = ball_pos_w.unsqueeze(0).clone()
                    visible_contact_pos[:, 2] += SOCCER_BALL_RADIUS + 0.015
                    contact_markers.visualize(
                        translations=visible_contact_pos,
                        orientations=ball_quat,
                    )
                else:
                    contact_markers.visualize(
                        translations=torch.tensor([[0.0, 0.0, -10.0]], device=sim.device),
                        orientations=ball_quat,
                    )

        scene.write_data_to_sim()
        sim.render()
        scene.update(sim_dt)

        pos_lookat = root_states[0, :3].detach().cpu().numpy()
        if not record_video:
            sim.set_camera_view(pos_lookat + np.array([2.2, 2.2, 0.55]), pos_lookat)

        if video_recorder is not None:
            overlay = _build_video_overlay(file_idx, motion_files, motion, t)
            video_recorder.capture(overlay_text=overlay)

        hud = f"\r[{file_idx + 1}/{len(motion_files)}] {motion.motion_name}  "
        hud += f"Frame {t:4d}/{motion.time_step_total - 1}  t={t / motion.fps:6.2f}s"
        if motion.has_ball:
            bp = motion.ball_pos_w[t].detach().cpu().numpy()
            spd = float(torch.norm(motion.ball_lin_vel_w[t]).item())
            hud += f"  ball=({bp[0]:+.2f},{bp[1]:+.2f},{bp[2]:.2f})  |v|={spd:.2f}"
        if motion.cg_contact is not None:
            hud += f"  CG={int(motion.cg_contact[t].item())}"
        if motion.contact_anchor is not None:
            hud += f"  anchor={int(motion.contact_anchor[t].item())}"
        if motion.cg_surface is not None:
            surface_id = int(motion.cg_surface[t].item())
            surface_lbl = {0: "inside", 1: "outside"}.get(surface_id, "-")
            hud += f"  surface={surface_lbl}"
        if motion.cg_foot_ball_dist is not None:
            d = float(motion.cg_foot_ball_dist[t].item())
            if d >= 0.0:
                hud += f"  d_foot_ball={d:.3f}m"
        print(hud, end="", flush=True)

        if args_cli.real_time:
            elapsed = time.perf_counter() - frame_start
            if elapsed < frame_dt:
                time.sleep(frame_dt - elapsed)


def main() -> None:
    motion_files = collect_motion_files(args_cli.motion_path)
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayWithBallSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    run_simulator(sim, scene, motion_files)


if __name__ == "__main__":
    main()
    simulation_app.close()
