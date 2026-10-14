from __future__ import annotations

import math
import numpy as np
import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

from .kick_detection import KickContactTracker
from .task_frame import align_body_quat_yaw_to_task_forward, mimic_anchor_yaw_delta_quat, spawn_ball_ahead_env_local

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Task-level modes are intentionally independent from the velocity command.
# Both IDLE and STOP request zero velocity, but the policy must distinguish
# waiting for a start signal from settling the ball after a completed dribble.
TASK_STATE_IDLE = 0
TASK_STATE_DRIBBLE = 1
TASK_STATE_STOP = 2
TASK_STATE_NAMES = ("idle", "dribble", "stop")
_TASK_STATE_NAME_TO_ID = {name: index for index, name in enumerate(TASK_STATE_NAMES)}


def normalize_locomotion_task_state(value: str | int) -> int:
    """Convert a public task-state name/id to its canonical integer id."""
    if isinstance(value, str):
        normalized = value.lower().strip()
        if normalized not in _TASK_STATE_NAME_TO_ID:
            raise ValueError(
                f"Unsupported locomotion task state {value!r}; expected one of {TASK_STATE_NAMES}."
            )
        return _TASK_STATE_NAME_TO_ID[normalized]
    state = int(value)
    if state < TASK_STATE_IDLE or state > TASK_STATE_STOP:
        raise ValueError(
            f"Unsupported locomotion task state id {state}; expected 0=idle, 1=dribble, or 2=stop."
        )
    return state


def locomotion_task_state_mask(command, active_task_states: Sequence[int] | None = None) -> torch.Tensor:
    """Return a boolean mask for task-state-gated objectives.

    Legacy commands that do not expose a task state are treated as DRIBBLE so
    existing task rewards preserve their original behavior.
    """
    state = getattr(command, "locomotion_task_state", None)
    if not isinstance(state, torch.Tensor):
        state = torch.full(
            (command.num_envs,), TASK_STATE_DRIBBLE, dtype=torch.long, device=command.device
        )
    if active_task_states is None:
        return torch.ones_like(state, dtype=torch.bool)
    allowed = torch.as_tensor(active_task_states, dtype=state.dtype, device=state.device)
    if allowed.numel() == 0:
        return torch.zeros_like(state, dtype=torch.bool)
    return (state.unsqueeze(-1) == allowed.view(1, -1)).any(dim=-1)


class MultiMotionLoader:
    def __init__(self, motion_files: list[str], body_indexes: Sequence[int], device: str = "cpu"):
        assert len(motion_files) > 0, "motion_files must not be empty"
        self.num_files = len(motion_files)
        self._body_indexes = body_indexes
        self.device = device

        # Temporarily store data from each file.
        self.motion_name = []
        self.motion_lengths = []

        joint_pos_list = []
        joint_vel_list = []
        body_pos_w_list = []
        body_quat_w_list = []
        body_lin_vel_w_list = []
        body_ang_vel_w_list = []
        kick_leg_labels = []
        kick_frame_list = []
        kick_end_frame_list = []

        ball_pos_w_list: list[torch.Tensor] = []
        dribble_cg_contact_list: list[torch.Tensor] = []
        dribble_cg_foot_list: list[torch.Tensor] = []
        dribble_cg_surface_list: list[torch.Tensor] = []
        dribble_cg_foot_ball_dist_list: list[torch.Tensor] = []
        dribble_cg_dist_foot_list: list[torch.Tensor] = []
        dribble_cg_flow_dir_local_list: list[torch.Tensor] = []
        dribble_cg_flow_distance_list: list[torch.Tensor] = []
        dribble_cg_flow_duration_list: list[torch.Tensor] = []
        dribble_cg_flow_anchor_frame_list: list[torch.Tensor] = []
        dribble_cg_flow_valid_list: list[torch.Tensor] = []
        motion_has_ball_demo_list: list[bool] = []

        self.fps_list = []

        max_T = 0  # Track maximum frame count.

        for motion_file in motion_files:
            assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
            data = np.load(motion_file)

            self.fps_list.append(data["fps"])
            self.motion_name.append(motion_file.split("/")[-1].split(".")[0])  # Store filename without suffix.
            self.motion_lengths.append(data["joint_pos"].shape[0])

            jp = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
            jv = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
            bp = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
            bq = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
            blv = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
            bav = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)

            joint_pos_list.append(jp)
            joint_vel_list.append(jv)
            body_pos_w_list.append(bp)
            body_quat_w_list.append(bq)
            body_lin_vel_w_list.append(blv)
            body_ang_vel_w_list.append(bav)

            label_value: str | None = None
            if "kick_leg" in data.files:
                raw_label = data["kick_leg"]
                try:
                    label_str = str(raw_label.item()).strip().lower()
                except Exception:
                    label_str = str(raw_label).strip().lower()
                if label_str in {"left", "right"}:
                    label_value = label_str
            kick_leg_labels.append(label_value)

            # Read kick_frame metadata (0-indexed frame where kick contact begins).
            kf_value: int = -1  # -1 means "not annotated" → no gating
            if "kick_frame" in data.files:
                raw_kf = data["kick_frame"]
                try:
                    kf_value = int(np.asarray(raw_kf).flat[0])
                except Exception:
                    kf_value = -1
            kick_frame_list.append(kf_value)

            # Read kick_end_frame metadata (0-indexed frame where kick contact ends).
            kef_value: int = -1
            if "kick_end_frame" in data.files:
                raw_kef = data["kick_end_frame"]
                try:
                    kef_value = int(np.asarray(raw_kef).flat[0])
                except Exception:
                    kef_value = -1
            kick_end_frame_list.append(kef_value)

            T = int(jp.shape[0])

            if "ball_pos_w" in data.files:
                ba = np.asarray(data["ball_pos_w"], dtype=np.float32)
                if ba.shape[0] != T:
                    raise ValueError(
                        f"{motion_file}: ball_pos_w length {ba.shape[0]} != joint_pos length {T}"
                    )
                ball_pos_w_list.append(torch.tensor(ba, dtype=torch.float32, device=device))
                motion_has_ball_demo_list.append(True)
            else:
                ball_pos_w_list.append(torch.zeros((T, 3), dtype=torch.float32, device=device))
                motion_has_ball_demo_list.append(False)

            cg_contact = torch.zeros(T, dtype=torch.int8, device=device)
            cg_foot = torch.full((T,), -1, dtype=torch.int8, device=device)
            # Contact surface id: -1 none/unknown, 0 inside instep,
            # 1 outside instep.  Keep it independent from foot side: the
            # same local region names work for either foot.
            cg_surface = torch.full((T,), -1, dtype=torch.int8, device=device)
            if "dribble_cg_contact" in data.files:
                cc = np.asarray(data["dribble_cg_contact"]).reshape(-1).astype(np.int8)[:T]
                cg_contact[: cc.shape[0]] = torch.as_tensor(cc, device=device, dtype=torch.int8)
            elif kf_value >= 0 and kef_value >= kf_value:
                cg_contact[kf_value : kef_value + 1] = 1
                if label_value == "left":
                    cg_foot[kf_value : kef_value + 1] = 0
                elif label_value == "right":
                    cg_foot[kf_value : kef_value + 1] = 1

            if "dribble_cg_foot" in data.files:
                cf = np.asarray(data["dribble_cg_foot"]).reshape(-1).astype(np.int8)[:T]
                cg_foot[: cf.shape[0]] = torch.as_tensor(cf, device=device, dtype=torch.int8)

            if "dribble_cg_surface" in data.files:
                cs = np.asarray(data["dribble_cg_surface"]).reshape(-1).astype(np.int8)[:T]
                cg_surface[: cs.shape[0]] = torch.as_tensor(cs, device=device, dtype=torch.int8)

            cg_foot_ball_dist = torch.full((T,), -1.0, dtype=torch.float32, device=device)
            if "dribble_cg_foot_ball_dist" in data.files:
                cd = np.asarray(data["dribble_cg_foot_ball_dist"], dtype=np.float32).reshape(-1)[:T]
                cg_foot_ball_dist[: cd.shape[0]] = torch.as_tensor(cd, device=device, dtype=torch.float32)

            cg_dist_foot = torch.full((T,), -1, dtype=torch.int8, device=device)
            if "dribble_cg_dist_foot" in data.files:
                df = np.asarray(data["dribble_cg_dist_foot"], dtype=np.int8).reshape(-1)[:T]
                cg_dist_foot[: df.shape[0]] = torch.as_tensor(df, device=device, dtype=torch.int8)

            flow_keys = {
                "dribble_cg_flow_dir_local",
                "dribble_cg_flow_distance",
                "dribble_cg_flow_duration",
                "dribble_cg_flow_anchor_frame",
                "dribble_cg_flow_valid",
            }
            present_flow_keys = flow_keys.intersection(data.files)
            if present_flow_keys and present_flow_keys != flow_keys:
                missing = sorted(flow_keys - present_flow_keys)
                raise ValueError(f"{motion_file}: incomplete dribble CG flow labels; missing {missing}")

            cg_flow_dir_local = torch.zeros((T, 2), dtype=torch.float32, device=device)
            cg_flow_distance = torch.full((T,), -1.0, dtype=torch.float32, device=device)
            cg_flow_duration = torch.full((T,), -1.0, dtype=torch.float32, device=device)
            cg_flow_anchor_frame = torch.full((T,), -1, dtype=torch.long, device=device)
            cg_flow_valid = torch.zeros(T, dtype=torch.int8, device=device)
            if present_flow_keys:
                flow_dir = np.asarray(data["dribble_cg_flow_dir_local"], dtype=np.float32)
                flow_distance = np.asarray(data["dribble_cg_flow_distance"], dtype=np.float32).reshape(-1)
                flow_duration = np.asarray(data["dribble_cg_flow_duration"], dtype=np.float32).reshape(-1)
                flow_anchor = np.asarray(data["dribble_cg_flow_anchor_frame"], dtype=np.int64).reshape(-1)
                flow_valid = np.asarray(data["dribble_cg_flow_valid"], dtype=np.int8).reshape(-1)
                if flow_dir.shape != (T, 2):
                    raise ValueError(
                        f"{motion_file}: dribble_cg_flow_dir_local shape {flow_dir.shape} != {(T, 2)}"
                    )
                for key, value in (
                    ("dribble_cg_flow_distance", flow_distance),
                    ("dribble_cg_flow_duration", flow_duration),
                    ("dribble_cg_flow_anchor_frame", flow_anchor),
                    ("dribble_cg_flow_valid", flow_valid),
                ):
                    if value.shape[0] != T:
                        raise ValueError(f"{motion_file}: {key} length {value.shape[0]} != {T}")

                contact_np = cg_contact.detach().cpu().numpy() > 0
                starts = np.flatnonzero(contact_np & np.concatenate(([True], ~contact_np[:-1])))
                valid_mask = flow_valid > 0
                if np.any(valid_mask):
                    valid_norm = np.linalg.norm(flow_dir[valid_mask], axis=1)
                    valid_numeric = (
                        np.all(np.isfinite(flow_dir[valid_mask]))
                        and np.all(np.isfinite(flow_distance[valid_mask]))
                        and np.all(np.isfinite(flow_duration[valid_mask]))
                        and np.all(flow_distance[valid_mask] > 0.0)
                        and np.all(flow_duration[valid_mask] > 0.0)
                        and np.all(flow_anchor[valid_mask] >= 0)
                        and np.allclose(valid_norm, 1.0, atol=1.0e-3)
                    )
                    if not valid_numeric:
                        raise ValueError(f"{motion_file}: invalid numeric values in dribble CG flow labels")
                if starts.size == 0 and np.any(valid_mask):
                    raise ValueError(f"{motion_file}: flow labels exist without CG contact anchors")
                if starts.size > 1:
                    for source_start, next_start in zip(starts[:-1], starts[1:]):
                        segment = slice(int(source_start), int(next_start))
                        if not np.all(valid_mask[segment]):
                            raise ValueError(f"{motion_file}: incomplete flow segment at frame {source_start}")
                        if not np.all(flow_anchor[segment] == source_start):
                            raise ValueError(
                                f"{motion_file}: flow anchor-frame ids disagree in segment {source_start}"
                            )
                if starts.size > 0 and np.any(valid_mask[: starts[0]]):
                    raise ValueError(f"{motion_file}: flow labels must start at the first contact anchor")
                if starts.size > 0 and np.any(valid_mask[starts[-1] :]):
                    raise ValueError(f"{motion_file}: final contact must not have an outgoing flow segment")

                cg_flow_dir_local[:] = torch.as_tensor(flow_dir, device=device, dtype=torch.float32)
                cg_flow_distance[:] = torch.as_tensor(flow_distance, device=device, dtype=torch.float32)
                cg_flow_duration[:] = torch.as_tensor(flow_duration, device=device, dtype=torch.float32)
                cg_flow_anchor_frame[:] = torch.as_tensor(flow_anchor, device=device, dtype=torch.long)
                cg_flow_valid[:] = torch.as_tensor(flow_valid, device=device, dtype=torch.int8)

            dribble_cg_contact_list.append(cg_contact)
            dribble_cg_foot_list.append(cg_foot)
            dribble_cg_surface_list.append(cg_surface)
            dribble_cg_foot_ball_dist_list.append(cg_foot_ball_dist)
            dribble_cg_dist_foot_list.append(cg_dist_foot)
            dribble_cg_flow_dir_local_list.append(cg_flow_dir_local)
            dribble_cg_flow_distance_list.append(cg_flow_distance)
            dribble_cg_flow_duration_list.append(cg_flow_duration)
            dribble_cg_flow_anchor_frame_list.append(cg_flow_anchor_frame)
            dribble_cg_flow_valid_list.append(cg_flow_valid)

            max_T = max(max_T, jp.shape[0])

        # Pad all files to max_T and stack into tensors.
        def pad_tensor_list(tensor_list, pad_value=0.0):
            padded = []
            for t in tensor_list:
                T, *rest = t.shape
                pad_size = [max_T - T] + rest
                pad_tensor = torch.cat([t, torch.full([*pad_size], pad_value, device=self.device)], dim=0)
                # pad_tensor = torch.cat([t, torch.full([*pad_size], pad_value, device=self.device, dtype=t.dtype)], dim=0)
                padded.append(pad_tensor)
            return torch.stack(padded, dim=0)  # shape: (num_files, max_T, ...)

        def pad_1d_int8(tensor_list: list[torch.Tensor], pad_value: int) -> torch.Tensor:
            padded = []
            for t in tensor_list:
                T = int(t.shape[0])
                pad_size = max_T - T
                pad_tensor = torch.cat(
                    [t, torch.full((pad_size,), pad_value, device=self.device, dtype=torch.int8)], dim=0
                )
                padded.append(pad_tensor)
            return torch.stack(padded, dim=0)

        def pad_1d_float(tensor_list: list[torch.Tensor], pad_value: float) -> torch.Tensor:
            padded = []
            for t in tensor_list:
                T = int(t.shape[0])
                pad_size = max_T - T
                pad_tensor = torch.cat(
                    [t, torch.full((pad_size,), pad_value, device=self.device, dtype=torch.float32)], dim=0
                )
                padded.append(pad_tensor)
            return torch.stack(padded, dim=0)

        def pad_1d_long(tensor_list: list[torch.Tensor], pad_value: int) -> torch.Tensor:
            padded = []
            for t in tensor_list:
                T = int(t.shape[0])
                pad_size = max_T - T
                pad_tensor = torch.cat(
                    [t, torch.full((pad_size,), pad_value, device=self.device, dtype=torch.long)], dim=0
                )
                padded.append(pad_tensor)
            return torch.stack(padded, dim=0)

        self.joint_pos = pad_tensor_list(joint_pos_list)
        self.joint_vel = pad_tensor_list(joint_vel_list)
        self._body_pos_w = pad_tensor_list(body_pos_w_list)
        self._body_quat_w = pad_tensor_list(body_quat_w_list)
        self._body_lin_vel_w = pad_tensor_list(body_lin_vel_w_list)
        self._body_ang_vel_w = pad_tensor_list(body_ang_vel_w_list)

        self.time_step_total = max_T  # Maximum frame count.
        self.file_lengths = torch.tensor([jp.shape[0] for jp in joint_pos_list],
                                         dtype=torch.long,
                                         device=self.device)
        self.fps = self.fps_list[0]  # Can be adjusted if needed.
        self._kick_leg_labels = tuple(kick_leg_labels)
        self._kick_frames = torch.tensor(kick_frame_list, dtype=torch.long, device=self.device)
        self._kick_end_frames = torch.tensor(kick_end_frame_list, dtype=torch.long, device=self.device)

        self._ball_pos_w = pad_tensor_list(ball_pos_w_list, pad_value=0.0)
        self._dribble_cg_contact = pad_1d_int8(dribble_cg_contact_list, pad_value=0)
        self._dribble_cg_foot = pad_1d_int8(dribble_cg_foot_list, pad_value=-1)
        self._dribble_cg_surface = pad_1d_int8(dribble_cg_surface_list, pad_value=-1)
        self._dribble_cg_foot_ball_dist = pad_1d_float(dribble_cg_foot_ball_dist_list, pad_value=-1.0)
        self._dribble_cg_dist_foot = pad_1d_int8(dribble_cg_dist_foot_list, pad_value=-1)
        self._dribble_cg_flow_dir_local = pad_tensor_list(dribble_cg_flow_dir_local_list, pad_value=0.0)
        self._dribble_cg_flow_distance = pad_1d_float(dribble_cg_flow_distance_list, pad_value=-1.0)
        self._dribble_cg_flow_duration = pad_1d_float(dribble_cg_flow_duration_list, pad_value=-1.0)
        self._dribble_cg_flow_anchor_frame = pad_1d_long(dribble_cg_flow_anchor_frame_list, pad_value=-1)
        self._dribble_cg_flow_valid = pad_1d_int8(dribble_cg_flow_valid_list, pad_value=0)
        self.motion_has_ball_demo = torch.tensor(motion_has_ball_demo_list, dtype=torch.bool, device=self.device)
        self.motion_has_dribble_cg = torch.any(self._dribble_cg_contact > 0, dim=1)
        # The first labelled contact is the reset-ball source for local
        # dribbling stages.  Keep ``-1`` for clips without a valid contact
        # rather than allowing ``argmax`` to silently return padded frame 0.
        first_contact = torch.argmax((self._dribble_cg_contact > 0).to(torch.int64), dim=1)
        self.first_dribble_cg_contact_frame = torch.where(
            self.motion_has_dribble_cg,
            first_contact,
            torch.full_like(first_contact, -1),
        )
        self.motion_has_dribble_cg_foot_ball_dist = torch.any(self._dribble_cg_foot_ball_dist >= 0.0, dim=1)
        self.motion_has_dribble_cg_flow = torch.any(self._dribble_cg_flow_valid > 0, dim=1)

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, :, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, :, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, :, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, :, self._body_indexes]

    @property
    def kick_leg_labels(self) -> tuple[str | None, ...]:
        return self._kick_leg_labels

    @property
    def kick_frames(self) -> torch.Tensor:
        """Per-motion kick start frame indices. -1 means not annotated."""
        return self._kick_frames

    @property
    def kick_end_frames(self) -> torch.Tensor:
        """Per-motion kick end frame indices. -1 means not annotated."""
        return self._kick_end_frames

    @property
    def ball_pos_w(self) -> torch.Tensor:
        """Demo ball positions from motion files ``[num_files, T, 3]`` (padded)."""
        return self._ball_pos_w

    @property
    def dribble_cg_contact(self) -> torch.Tensor:
        """Per-frame contact annotation ``[num_files, T]`` (0/1, padded with 0)."""
        return self._dribble_cg_contact

    @property
    def first_dribble_contact_frame(self) -> torch.Tensor:
        """First labelled contact frame per motion, or ``-1`` when unavailable."""
        return self.first_dribble_cg_contact_frame

    @property
    def dribble_cg_foot(self) -> torch.Tensor:
        """Per-frame foot id: -1 unknown/none, 0 left, 1 right (padded with -1)."""
        return self._dribble_cg_foot

    @property
    def dribble_cg_surface(self) -> torch.Tensor:
        """Per-frame contact surface: -1 unknown, 0 inside instep, 1 outside instep."""
        return self._dribble_cg_surface

    @property
    def dribble_cg_foot_ball_dist(self) -> torch.Tensor:
        """Per-frame demo foot–ball distance (m); ``-1`` = no label."""
        return self._dribble_cg_foot_ball_dist

    @property
    def dribble_cg_dist_foot(self) -> torch.Tensor:
        """Per-frame foot id used for distance reference: -1 none, 0 left, 1 right."""
        return self._dribble_cg_dist_foot

    @property
    def dribble_cg_flow_dir_local(self) -> torch.Tensor:
        """Outgoing unit XY direction in the source-touch pelvis-yaw frame."""
        return self._dribble_cg_flow_dir_local

    @property
    def dribble_cg_flow_distance(self) -> torch.Tensor:
        """Source-to-next-contact anchor distance in meters; ``-1`` if invalid."""
        return self._dribble_cg_flow_distance

    @property
    def dribble_cg_flow_duration(self) -> torch.Tensor:
        """Source-to-next-contact duration in seconds; ``-1`` if invalid."""
        return self._dribble_cg_flow_duration

    @property
    def dribble_cg_flow_anchor_frame(self) -> torch.Tensor:
        """Source contact frame held over each outgoing flow segment."""
        return self._dribble_cg_flow_anchor_frame

    @property
    def dribble_cg_flow_valid(self) -> torch.Tensor:
        """Whether each frame belongs to a non-final outgoing flow segment."""
        return self._dribble_cg_flow_valid

    def get_last_frame_anchor_pos(self, motion_idx: int, anchor_body_idx: int, motion_length: int) -> torch.Tensor:
        """Get the anchor position at the last frame of the specified motion."""
        last_frame_idx = motion_length - 1
        return self._body_pos_w[motion_idx, last_frame_idx, anchor_body_idx]

    def get_kick_frame_anchor_pos(self, motion_idx: int, anchor_body_idx: int) -> torch.Tensor | None:
        """Get the anchor position at the kick frame. Returns None if not annotated."""
        kf = int(self._kick_frames[motion_idx].item())
        if kf < 0:
            return None
        return self._body_pos_w[motion_idx, kf, anchor_body_idx]

    def get_first_frame_anchor_pos(self, motion_idx: int, anchor_body_idx: int) -> torch.Tensor:
        """Get the anchor position at the first frame of the specified motion."""
        return self._body_pos_w[motion_idx, 0, anchor_body_idx]

    def get_first_frame_anchor_quat(self, motion_idx: int, anchor_body_idx: int) -> torch.Tensor:
        """Get the anchor orientation at the first frame of the specified motion."""
        return self._body_quat_w[motion_idx, 0, anchor_body_idx]


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.soccer_ball: RigidObject | None = None
        # Try to get the soccer-ball object.
        if hasattr(env.scene, "__getitem__"):
            try:
                self.soccer_ball = env.scene["soccer_ball"]
            except KeyError:
                self.soccer_ball = None

        # Determine whether the motion sequence has ended.
        term_name = getattr(cfg, "term_name", None)
        if term_name is None:
            term_name = getattr(cfg, "name", None)
        if term_name is None:
            term_name = "motion"
            self._state_prefix = f"_{term_name}"
            self.kick_contact_tracker = KickContactTracker(env, self._state_prefix)

        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        self.motion = MultiMotionLoader(self.cfg.motion_files, self.body_indexes, device=self.device)
        kick_leg_to_id = {"left": 0, "right": 1}
        self._kick_leg_id_to_name = {v: k for k, v in kick_leg_to_id.items()}
        self._kick_leg_id_to_name[-1] = "unknown"
        self.motion_kick_leg = torch.full((self.motion.num_files,), -1, dtype=torch.int8, device=self.device)
        self.motion_kick_leg_names = []
        for idx, label in enumerate(self.motion.kick_leg_labels):
            normalized = label.lower() if isinstance(label, str) else None
            if normalized in kick_leg_to_id:
                self.motion_kick_leg[idx] = kick_leg_to_id[normalized]
                self.motion_kick_leg_names.append(normalized)
            else:
                self.motion_kick_leg_names.append("unknown")

        # ``time_steps`` remains a backwards-compatible alias used by legacy
        # rewards, terminations, and HUD code for the current demo frame.  The
        # episode clock stays in ``env.episode_length_buf``.
        self.style_phase_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.time_steps = self.style_phase_steps
        self.style_phase_wrap_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.motion_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.motion_length = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # True only for the final valid reference frame of a stage that uses
        # clip-end termination.  The termination manager consumes this flag
        # and the normal environment reset then resamples the next episode.
        self._motion_clip_finished = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Randomly assign initial motions (sequential playback starts at 0 and cycles).
        if self.motion.num_files > 1:
            if str(self.cfg.sampling_strategy).lower() == "sequential":
                self.motion_idx = torch.arange(self.num_envs, device=self.device) % self.motion.num_files
            else:
                self.motion_idx = torch.randint(0, self.motion.num_files, (self.num_envs,),
                                               dtype=torch.long, device=self.device)
        # Initialize per-environment motion lengths.
        self.motion_length[:] = self.motion.file_lengths[self.motion_idx]

        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        # Adaptive sampling settings.
        # Compute bin count: decimation * dt is one simulation step duration.
        # Thus each bin corresponds to ~1 second and bin_count is the total number of bins.
        self.bin_count = int(self.motion.time_step_total // (1 / (env.cfg.decimation * env.cfg.sim.dt))) + 1
        self.bin_failed_count = torch.zeros(
            (self.motion.num_files, self.bin_count), dtype=torch.float, device=self.device
        )
        self._current_bin_failed = torch.zeros_like(self.bin_failed_count)
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)

        # Target-point and soccer-ball generation logic.
        self.target_point_pos = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self.soccer_ball_pos = torch.zeros_like(self.target_point_pos)
        # Optional Stage-1 ghost-ball offset.  It is sampled independently at
        # episode reset and expressed in the actual pelvis-yaw frame.  Keeping
        # this state separate from ``soccer_ball_pos`` prevents a long,
        # translating motion clip from turning a bounded ball observation
        # into a multi-metre value merely because the kinematic ghost stayed
        # fixed in world space.
        self.soccer_ball_workspace_offset_pelvis_yaw = torch.zeros_like(self.target_point_pos)
        self.target_destination_pos = torch.zeros_like(self.target_point_pos)
        # Save initial target position at resample for kick-direction computation.
        self.initial_target_point_pos = torch.zeros_like(self.target_point_pos)
        
        # Blind-zone logic: ball is invisible when robot-ball (x, y) distance is out of range.
        self.blind_distance_min = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.blind_distance_max = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        # Target position at last visible frame (robot base frame).
        self.last_visible_target_point_base = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        # Whether currently in blind zone.
        self.is_in_blind_zone = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # Height for target_destination.
        self.destination_height = 0.11
        
        # target_destination generation parameters (world-frame based).
        self.destination_center = torch.tensor([0.0, -5.0, self.destination_height], device=self.device)  # Rectangle center (x, y, z).
        self.destination_length = 1.0  # Rectangle length (x-axis).
        self.destination_width = 0.5  # Rectangle width (y-axis).
        
        self.curve_radius_offset = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._steps_since_resample = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Locomotion velocity command (task / world-parallel +X/+Y/+Z). ``reference`` reads
        # demo anchor vel; ``manual`` uses ``locomotion_manual_*`` (editable at runtime).
        self._locomotion_command_mode = str(getattr(cfg, "locomotion_command_mode", "reference"))
        manual_lin = getattr(cfg, "locomotion_manual_lin_vel", (0.55, 0.0, 0.0))
        manual_ang = getattr(cfg, "locomotion_manual_ang_vel", (0.0, 0.0, 0.0))
        self.locomotion_manual_lin_vel = (
            torch.tensor(manual_lin, device=self.device, dtype=torch.float32)
            .unsqueeze(0)
            .expand(self.num_envs, -1)
            .clone()
        )
        self.locomotion_manual_ang_vel = (
            torch.tensor(manual_ang, device=self.device, dtype=torch.float32)
            .unsqueeze(0)
            .expand(self.num_envs, -1)
            .clone()
        )
        self._locomotion_cmd_steps_since_resample = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._locomotion_cmd_steps_since_change = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._locomotion_cmd_hold_steps_remaining = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.locomotion_cmd_speed = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.locomotion_cmd_heading = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        # These are requested endpoints.  The public command above remains the
        # effective command seen by the policy and reward after smoothing.
        self.locomotion_cmd_target_speed = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.locomotion_cmd_target_heading = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.locomotion_cmd_target_wz = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._locomotion_cmd_initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._locomotion_task_state_enabled = bool(getattr(cfg, "locomotion_task_state_enabled", False))
        sequence_cfg = getattr(cfg, "locomotion_task_state_sequence", ("dribble",))
        sequence = tuple(normalize_locomotion_task_state(state) for state in sequence_cfg)
        if not sequence:
            raise ValueError("locomotion_task_state_sequence must contain at least one state.")
        self._locomotion_task_state_sequence = torch.tensor(sequence, dtype=torch.long, device=self.device)
        default_task_state = TASK_STATE_IDLE if self._locomotion_task_state_enabled else TASK_STATE_DRIBBLE
        self.locomotion_task_state = torch.full(
            (self.num_envs,), default_task_state, dtype=torch.long, device=self.device
        )
        self._locomotion_task_state_sequence_idx = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._locomotion_segment_plans: list[list[tuple[float, float, float, float, int]]] = [
            [] for _ in range(self.num_envs)
        ]
        self._locomotion_segment_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._locomotion_segment_hold_last = True
        self._locomotion_segment_reset_on_end = False
        self._locomotion_sequence_finished = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self._locomotion_command_mode == "resampled":
            all_env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
            self._sample_locomotion_commands(all_env_ids, reset_task_sequence=True)
        self._radius_offset_min = None
        self._radius_offset_max = None
        curve_cfg = cfg.curve_offset_range or {}
        radius_range = curve_cfg.get("radius")
        if isinstance(radius_range, Sequence) and not isinstance(radius_range, (str, bytes)) and len(radius_range) >= 2:
            self._radius_offset_min = float(radius_range[0])
            self._radius_offset_max = float(radius_range[1])
        elif radius_range is not None:
            value = float(radius_range)
            self._radius_offset_min = value
            self._radius_offset_max = value
        self._target_lateral_spawn_jitter = float(
            curve_cfg.get("lateral_spawn_jitter", curve_cfg.get("lateral_spawn_max", 0.12))
        )
        # PAiD Stage I perturbs the motion-consistent terminal ball direction
        # by a small arc.  Newer +X spawn modes do not consume this value, but
        # initialize it unconditionally before the constructor's first ball
        # placement so the original path is safe during environment creation.
        self._target_arc_angle = float(curve_cfg.get("arc_angle", math.pi / 18.0))
        self._target_height = float(curve_cfg.get("height", 0.11))
        marker_cfg = cfg.target_point_marker_cfg
        self.target_point_marker = VisualizationMarkers(marker_cfg) if marker_cfg is not None else None
        dest_marker_cfg = getattr(cfg, "target_destination_marker_cfg", None)
        self.target_destination_marker = VisualizationMarkers(dest_marker_cfg) if dest_marker_cfg is not None else None

        all_env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._sample_soccer_offset(all_env_ids)
        self._compute_soccer_ball_positions(all_env_ids)
        self._update_soccer_ball(all_env_ids)
        self._update_target_points(all_env_ids)

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    def _cyclic_style_blend_frames(self) -> int:
        """Return the configured same-clip seam-bridge length in source frames."""
        if bool(getattr(self.cfg, "motion_clip_end_resample", True)):
            return 0
        return max(0, int(getattr(self.cfg, "motion_cyclic_blend_frames", 0)))

    def _style_cycle_lengths(self) -> torch.Tensor:
        """Per-environment virtual cycle lengths, accounting for a seam bridge."""
        raw_lengths = self.motion_length.clamp(min=1)
        bridge_frames = self._cyclic_style_blend_frames()
        if bridge_frames < 2:
            return raw_lengths
        # The loop contains source frames [K, L-K), then K blended frames
        # [L-K, L) -> [0, K).  It therefore has L-K virtual frames.
        valid = raw_lengths > 2 * bridge_frames
        return torch.where(valid, raw_lengths - bridge_frames, raw_lengths)

    def _style_motion_frame_indices(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return source-frame pairs and blend weight for the virtual style phase.

        For a bridge of ``K`` frames, the steady-state cycle is
        ``[K, ..., L-K-1]`` followed by a smooth blend of
        ``[L-K, ..., L-1]`` into ``[0, ..., K-1]``.  Its final sample is
        source frame ``K-1`` and its next sample is ``K``, removing the raw
        end-to-start discontinuity without another motion clip.
        """
        raw_lengths = self.motion_length.clamp(min=1)
        phase = self.style_phase_steps.clamp(min=0)
        bridge_frames = self._cyclic_style_blend_frames()
        if bridge_frames < 2:
            index = torch.minimum(phase, raw_lengths - 1)
            zeros = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
            return index, index, zeros, torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        valid = raw_lengths > 2 * bridge_frames
        regular_length = (raw_lengths - 2 * bridge_frames).clamp(min=1)
        in_bridge = valid & (phase >= regular_length)
        bridge_offset = (phase - regular_length).clamp(min=0, max=bridge_frames - 1)

        regular_index = torch.minimum(phase + bridge_frames, raw_lengths - 1)
        tail_index = raw_lengths - bridge_frames + bridge_offset
        head_index = bridge_offset
        first_index = torch.where(in_bridge, tail_index, torch.where(valid, regular_index, phase))
        first_index = torch.minimum(first_index, raw_lengths - 1)
        second_index = torch.where(in_bridge, head_index, first_index)

        blend = bridge_offset.to(torch.float32) / float(bridge_frames - 1)
        # Quintic smootherstep has zero first and second derivative at both
        # joins.  This keeps the synthesized reference velocity continuous.
        blend = blend * blend * blend * (blend * (blend * 6.0 - 15.0) + 10.0)
        blend = torch.where(in_bridge, blend, torch.zeros_like(blend))
        return first_index, second_index, blend, in_bridge

    @staticmethod
    def _slerp_quat(first: torch.Tensor, second: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Shortest-arc quaternion interpolation for tensors using wxyz convention."""
        view_shape = (blend.shape[0],) + (1,) * (first.ndim - 2) + (1,)
        blend_expanded = blend.reshape(view_shape)
        dot = torch.sum(first * second, dim=-1, keepdim=True)
        second = torch.where(dot < 0.0, -second, second)
        dot = torch.sum(first * second, dim=-1, keepdim=True).clamp(-1.0, 1.0)
        theta = torch.acos(dot)
        sin_theta = torch.sin(theta)
        linear = first + blend_expanded * (second - first)
        spherical = (
            torch.sin((1.0 - blend_expanded) * theta) / sin_theta.clamp_min(1.0e-8) * first
            + torch.sin(blend_expanded * theta) / sin_theta.clamp_min(1.0e-8) * second
        )
        result = torch.where(sin_theta.abs() > 1.0e-5, spherical, linear)
        return result / torch.linalg.vector_norm(result, dim=-1, keepdim=True).clamp_min(1.0e-8)

    def _style_motion_sample(self, values: torch.Tensor, *, quaternion: bool = False) -> torch.Tensor:
        """Sample a loaded motion tensor at the possibly blended virtual phase."""
        first_index, second_index, blend, in_bridge = self._style_motion_frame_indices()
        first = values[self.motion_idx, first_index]
        if not torch.any(in_bridge):
            return first
        second = values[self.motion_idx, second_index]
        if quaternion:
            return self._slerp_quat(first, second, blend)
        view_shape = (blend.shape[0],) + (1,) * (first.ndim - 1)
        return first + blend.reshape(view_shape) * (second - first)

    def _style_motion_blend_rate_s(self) -> torch.Tensor:
        """Time derivative of the quintic bridge weight in inverse seconds."""
        bridge_frames = self._cyclic_style_blend_frames()
        if bridge_frames < 2:
            return torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        raw_lengths = self.motion_length.clamp(min=1)
        phase = self.style_phase_steps.clamp(min=0)
        valid = raw_lengths > 2 * bridge_frames
        regular_length = (raw_lengths - 2 * bridge_frames).clamp(min=1)
        in_bridge = valid & (phase >= regular_length)
        offset = (phase - regular_length).clamp(min=0, max=bridge_frames - 1).to(torch.float32)
        u = offset / float(bridge_frames - 1)
        # d/du smootherstep(u) = 30 u^2 (u - 1)^2.
        source_fps = float(np.asarray(self.motion.fps).reshape(-1)[0])
        rate = 30.0 * u.square() * (u - 1.0).square() * source_fps / float(bridge_frames - 1)
        return torch.where(in_bridge, rate, torch.zeros_like(rate))

    def _style_motion_velocity_sample(self, positions: torch.Tensor, velocities: torch.Tensor) -> torch.Tensor:
        """Sample a velocity field consistent with a blended position target."""
        first_index, second_index, blend, in_bridge = self._style_motion_frame_indices()
        first_vel = velocities[self.motion_idx, first_index]
        if not torch.any(in_bridge):
            return first_vel
        second_vel = velocities[self.motion_idx, second_index]
        first_pos = positions[self.motion_idx, first_index]
        second_pos = positions[self.motion_idx, second_index]
        view_shape = (blend.shape[0],) + (1,) * (first_vel.ndim - 1)
        blended = first_vel + blend.reshape(view_shape) * (second_vel - first_vel)
        return blended + self._style_motion_blend_rate_s().reshape(view_shape) * (second_pos - first_pos)

    def _style_motion_labels(self, values: torch.Tensor, neutral_value: int | float) -> torch.Tensor:
        """Sample discrete labels, neutralizing the synthetic seam bridge."""
        first_index, _, _, in_bridge = self._style_motion_frame_indices()
        result = values[self.motion_idx, first_index]
        neutral = torch.as_tensor(neutral_value, device=self.device, dtype=result.dtype)
        condition = in_bridge.reshape((in_bridge.shape[0],) + (1,) * (result.ndim - 1))
        return torch.where(condition, neutral, result)

    @property
    def style_seam_bridge_mask(self) -> torch.Tensor:
        """Whether each environment currently consumes a synthesized seam frame."""
        return self._style_motion_frame_indices()[-1]

    def style_phase_reference_info(self) -> dict[str, torch.Tensor]:
        """Expose virtual-cycle/source-frame mapping for diagnostics and HUDs."""
        first, second, blend, in_bridge = self._style_motion_frame_indices()
        return {
            "cycle_length": self._style_cycle_lengths(),
            "source_first_frame": first,
            "source_second_frame": second,
            "seam_blend": blend,
            "in_seam_bridge": in_bridge,
        }

    @property
    def joint_pos(self) -> torch.Tensor:
        return self._style_motion_sample(self.motion.joint_pos)

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._style_motion_velocity_sample(self.motion.joint_pos, self.motion.joint_vel)

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._style_motion_sample(self.motion.body_pos_w) + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._style_motion_sample(self.motion.body_quat_w, quaternion=True)

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._style_motion_sample(self.motion.body_lin_vel_w)

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._style_motion_sample(self.motion.body_ang_vel_w)

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.body_pos_w[:, self.motion_anchor_body_index]

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.body_quat_w[:, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.body_lin_vel_w[:, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.body_ang_vel_w[:, self.motion_anchor_body_index]

    @property
    def locomotion_command_mode(self) -> str:
        return self._locomotion_command_mode

    def set_locomotion_command_mode(self, mode: str) -> None:
        if mode not in {"reference", "resampled", "manual"}:
            raise ValueError(
                f"locomotion_command_mode must be 'reference', 'resampled', or 'manual', got {mode!r}"
            )
        self._locomotion_command_mode = mode

    def _env_step_dt_s(self) -> float:
        return float(self._env.cfg.decimation) * float(self._env.cfg.sim.dt)

    def _duration_s_to_steps(self, duration_s: float | torch.Tensor) -> torch.Tensor:
        step_dt = max(self._env_step_dt_s(), 1e-6)
        if isinstance(duration_s, torch.Tensor):
            return torch.clamp((duration_s / step_dt).long(), min=1)
        return torch.tensor(max(1, int(round(float(duration_s) / step_dt))), device=self.device, dtype=torch.long)

    def _apply_polar_locomotion(
        self,
        env_ids: torch.Tensor,
        speed: torch.Tensor,
        heading: torch.Tensor,
        wz: torch.Tensor,
        task_state: torch.Tensor | None = None,
    ) -> None:
        """Set a requested polar command, smoothing later changes when configured."""
        if task_state is not None:
            self.locomotion_task_state[env_ids] = task_state.to(
                device=self.device, dtype=self.locomotion_task_state.dtype
            )
        self.locomotion_cmd_target_speed[env_ids] = speed
        self.locomotion_cmd_target_heading[env_ids] = heading
        self.locomotion_cmd_target_wz[env_ids] = wz
        self._locomotion_cmd_steps_since_change[env_ids] = 0

        # A reset must begin from a concrete command; only subsequent changes
        # are filtered.  Legacy tasks retain the original immediate behavior.
        smoothing_enabled = bool(getattr(self.cfg, "locomotion_cmd_smoothing_enabled", False))
        immediate = ~self._locomotion_cmd_initialized[env_ids]
        if not smoothing_enabled:
            immediate = torch.ones_like(immediate)
        if torch.any(immediate):
            ids = env_ids[immediate]
            self._write_effective_polar_locomotion(ids, speed[immediate], heading[immediate], wz[immediate])

    def _apply_local_twist_locomotion(
        self,
        env_ids: torch.Tensor,
        vx: torch.Tensor,
        vy: torch.Tensor,
        wz: torch.Tensor,
        task_state: torch.Tensor | None = None,
    ) -> None:
        """Set an immediate pelvis-local twist command.

        Local-twist tasks deliberately do not reuse the polar smoother: their
        command is a body-frame quantity, so interpolating a task/world-frame
        heading would reintroduce the coordinate-system ambiguity this mode is
        intended to remove.
        """
        if task_state is not None:
            self.locomotion_task_state[env_ids] = task_state.to(
                device=self.device, dtype=self.locomotion_task_state.dtype
            )
        speed = torch.sqrt(torch.square(vx) + torch.square(vy))
        heading = torch.atan2(vy, vx)
        prior_heading = self.locomotion_cmd_heading[env_ids]
        heading = torch.where(speed > 0.05, heading, prior_heading)
        self.locomotion_cmd_target_speed[env_ids] = speed
        self.locomotion_cmd_target_heading[env_ids] = heading
        self.locomotion_cmd_target_wz[env_ids] = wz
        self._locomotion_cmd_steps_since_change[env_ids] = 0
        self.locomotion_manual_lin_vel[env_ids, 0] = vx
        self.locomotion_manual_lin_vel[env_ids, 1] = vy
        self.locomotion_manual_lin_vel[env_ids, 2] = 0.0
        self.locomotion_manual_ang_vel[env_ids, 0] = 0.0
        self.locomotion_manual_ang_vel[env_ids, 1] = 0.0
        self.locomotion_manual_ang_vel[env_ids, 2] = wz
        self.locomotion_cmd_speed[env_ids] = speed
        self.locomotion_cmd_heading[env_ids] = heading
        self._locomotion_cmd_initialized[env_ids] = True

    def _write_effective_polar_locomotion(
        self,
        env_ids: torch.Tensor,
        speed: torch.Tensor,
        heading: torch.Tensor,
        wz: torch.Tensor,
    ) -> None:
        """Write the effective speed/heading into the velocity-command buffers."""
        vx = speed * torch.cos(heading)
        vy = speed * torch.sin(heading)
        self.locomotion_manual_lin_vel[env_ids, 0] = vx
        self.locomotion_manual_lin_vel[env_ids, 1] = vy
        self.locomotion_manual_lin_vel[env_ids, 2] = 0.0
        self.locomotion_manual_ang_vel[env_ids, 0] = 0.0
        self.locomotion_manual_ang_vel[env_ids, 1] = 0.0
        self.locomotion_manual_ang_vel[env_ids, 2] = wz
        self.locomotion_cmd_speed[env_ids] = speed
        self.locomotion_cmd_heading[env_ids] = heading
        self._locomotion_cmd_initialized[env_ids] = True

    def _update_smoothed_locomotion_commands(self) -> None:
        """Advance requested endpoints with bounded heading and speed rates."""
        if not bool(getattr(self.cfg, "locomotion_cmd_smoothing_enabled", False)):
            return
        if self._locomotion_command_mode not in {"resampled", "manual"}:
            return

        ids = self._locomotion_cmd_initialized.nonzero(as_tuple=False).flatten()
        if ids.numel() == 0:
            return

        dt = self._env_step_dt_s()
        heading_error = torch.atan2(
            torch.sin(self.locomotion_cmd_target_heading[ids] - self.locomotion_cmd_heading[ids]),
            torch.cos(self.locomotion_cmd_target_heading[ids] - self.locomotion_cmd_heading[ids]),
        )
        heading_rate = max(float(getattr(self.cfg, "locomotion_cmd_heading_rate_limit", 0.0)), 0.0)
        if heading_rate > 0.0:
            heading_step = torch.clamp(heading_error, min=-heading_rate * dt, max=heading_rate * dt)
        else:
            heading_step = heading_error
        heading = self.locomotion_cmd_heading[ids] + heading_step

        # Turning lowers the speed endpoint while the direction is changing;
        # the slew limit then applies a physically continuous brake/recovery.
        slow_angle = max(float(getattr(self.cfg, "locomotion_cmd_turn_slowdown_angle", 0.0)), 1.0e-6)
        min_speed_scale = min(max(float(getattr(self.cfg, "locomotion_cmd_turn_min_speed_scale", 1.0)), 0.0), 1.0)
        turn_fraction = torch.clamp(torch.abs(heading_error) / slow_angle, max=1.0)
        desired_speed = self.locomotion_cmd_target_speed[ids] * (
            1.0 - (1.0 - min_speed_scale) * turn_fraction
        )
        speed_delta = desired_speed - self.locomotion_cmd_speed[ids]
        accel_limit = max(float(getattr(self.cfg, "locomotion_cmd_accel_limit", 0.0)), 0.0)
        decel_limit = max(float(getattr(self.cfg, "locomotion_cmd_decel_limit", accel_limit)), 0.0)
        if accel_limit > 0.0 or decel_limit > 0.0:
            max_up = accel_limit * dt if accel_limit > 0.0 else float("inf")
            max_down = decel_limit * dt if decel_limit > 0.0 else float("inf")
            speed_delta = torch.clamp(speed_delta, min=-max_down, max=max_up)
        speed = torch.clamp(self.locomotion_cmd_speed[ids] + speed_delta, min=0.0)

        self._write_effective_polar_locomotion(ids, speed, heading, self.locomotion_cmd_target_wz[ids])

    def _task_state_ids(
        self,
        task_state: str | int | torch.Tensor | Sequence[str | int],
        count: int,
    ) -> torch.Tensor:
        """Return validated per-environment task-state ids for public command APIs."""
        if isinstance(task_state, torch.Tensor):
            states = task_state.to(device=self.device, dtype=torch.long).flatten()
        elif isinstance(task_state, str) or isinstance(task_state, int):
            states = torch.full(
                (count,), normalize_locomotion_task_state(task_state), device=self.device, dtype=torch.long
            )
        else:
            values = list(task_state)
            if len(values) != count:
                raise ValueError(f"Expected {count} task states, received {len(values)}.")
            states = torch.as_tensor(
                [normalize_locomotion_task_state(value) for value in values],
                device=self.device,
                dtype=torch.long,
            )
        if states.numel() == 1 and count != 1:
            states = states.expand(count)
        if states.numel() != count:
            raise ValueError(f"Expected {count} task states, received {states.numel()}.")
        if torch.any((states < TASK_STATE_IDLE) | (states > TASK_STATE_STOP)):
            raise ValueError("Task-state ids must be 0=idle, 1=dribble, or 2=stop.")
        return states

    def _infer_task_state_from_speed(self, speed: torch.Tensor) -> torch.Tensor:
        """Infer a safe manual state when an older caller supplies no explicit mode."""
        stationary_threshold = float(getattr(self.cfg, "locomotion_task_state_stationary_speed", 0.05))
        return torch.where(
            speed > stationary_threshold,
            torch.full_like(speed, TASK_STATE_DRIBBLE, dtype=torch.long),
            torch.full_like(speed, TASK_STATE_STOP, dtype=torch.long),
        )

    def set_locomotion_task_state(
        self,
        task_state: str | int | torch.Tensor | Sequence[str | int],
        env_ids: torch.Tensor | slice | None = None,
    ) -> None:
        """Set only the high-level IDLE/DRIBBLE/STOP input without changing velocity."""
        if env_ids is None:
            env_ids_t = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        elif isinstance(env_ids, slice):
            env_ids_t = torch.arange(self.num_envs, device=self.device, dtype=torch.long)[env_ids]
        else:
            env_ids_t = env_ids.to(device=self.device, dtype=torch.long)
        if env_ids_t.numel() > 0:
            self.locomotion_task_state[env_ids_t] = self._task_state_ids(task_state, env_ids_t.numel())

    def _sample_locomotion_commands(
        self,
        env_ids: torch.Tensor | Sequence[int],
        reset_task_sequence: bool = False,
    ) -> None:
        """Sample speed + heading (+ optional wz) and a random hold duration per env."""
        if isinstance(env_ids, Sequence) and not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(list(env_ids), dtype=torch.long, device=self.device)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return

        if str(getattr(self.cfg, "locomotion_command_frame", "world")) == "pelvis_local":
            lin_range = getattr(self.cfg, "locomotion_cmd_lin_vel_range", {})
            ang_range = getattr(self.cfg, "locomotion_cmd_ang_vel_range", {})
            duration_range = getattr(self.cfg, "locomotion_cmd_duration_range", (1.5, 3.0))
            n = env_ids.numel()
            vx_range = lin_range.get("x", (0.25, 0.65))
            vy_range = lin_range.get("y", (-0.25, 0.25))
            wz_range = ang_range.get("yaw", (-0.35, 0.35))
            vx = sample_uniform(vx_range[0], vx_range[1], (n,), device=self.device)
            vy = sample_uniform(vy_range[0], vy_range[1], (n,), device=self.device)
            wz = sample_uniform(wz_range[0], wz_range[1], (n,), device=self.device)
            stationary_probability = min(
                max(float(getattr(self.cfg, "locomotion_cmd_stationary_probability", 0.0)), 0.0), 1.0
            )
            if stationary_probability > 0.0:
                stationary = torch.rand(n, device=self.device) < stationary_probability
                vx = torch.where(stationary, torch.zeros_like(vx), vx)
                vy = torch.where(stationary, torch.zeros_like(vy), vy)
                wz = torch.where(stationary, torch.zeros_like(wz), wz)
            duration_s = sample_uniform(duration_range[0], duration_range[1], (n,), device=self.device)
            self._apply_local_twist_locomotion(env_ids, vx, vy, wz)
            self._locomotion_cmd_hold_steps_remaining[env_ids] = self._duration_s_to_steps(duration_s)
            return

        speed_range = getattr(self.cfg, "locomotion_cmd_speed_range", (0.25, 0.65))
        heading_range = getattr(self.cfg, "locomotion_cmd_heading_range", (-0.75, 0.75))
        duration_range = getattr(self.cfg, "locomotion_cmd_duration_range", (1.5, 3.0))
        wz_range = getattr(self.cfg, "locomotion_cmd_wz_range", (0.0, 0.0))

        n = env_ids.numel()
        heading_delta_range = getattr(self.cfg, "locomotion_cmd_heading_delta_range", None)
        initialized = self._locomotion_cmd_initialized[env_ids]
        if heading_delta_range is not None and torch.any(initialized):
            # The first command of an episode remains uniformly distributed.
            # Thereafter sample a bounded delta, explicitly training turning
            # transitions rather than only unrelated heading snapshots.
            heading = sample_uniform(heading_range[0], heading_range[1], (n,), device=self.device)
            delta = sample_uniform(heading_delta_range[0], heading_delta_range[1], (n,), device=self.device)
            transitioned_heading = torch.clamp(
                self.locomotion_cmd_heading[env_ids] + delta,
                min=heading_range[0],
                max=heading_range[1],
            )
            heading = torch.where(initialized, transitioned_heading, heading)
        else:
            heading = sample_uniform(heading_range[0], heading_range[1], (n,), device=self.device)
        if not self._locomotion_task_state_enabled:
            speed = sample_uniform(speed_range[0], speed_range[1], (n,), device=self.device)
            wz = sample_uniform(wz_range[0], wz_range[1], (n,), device=self.device)
            duration_s = sample_uniform(duration_range[0], duration_range[1], (n,), device=self.device)
            self._apply_polar_locomotion(env_ids, speed, heading, wz)
            self._locomotion_cmd_hold_steps_remaining[env_ids] = self._duration_s_to_steps(duration_s)
            return

        # Stateful control deliberately trains the complete interaction loop:
        # wait still (IDLE), carry the ball (DRIBBLE), then decelerate and
        # settle both robot and ball (STOP).  State changes remain immediate,
        # while the existing velocity smoother makes motion transitions safe.
        if reset_task_sequence:
            self._locomotion_task_state_sequence_idx[env_ids] = 0
        else:
            self._locomotion_task_state_sequence_idx[env_ids] = (
                self._locomotion_task_state_sequence_idx[env_ids] + 1
            ) % self._locomotion_task_state_sequence.numel()
        state_index = self._locomotion_task_state_sequence_idx[env_ids]
        task_state = self._locomotion_task_state_sequence[state_index]
        dribble_mask = task_state == TASK_STATE_DRIBBLE

        speed = torch.zeros(n, dtype=torch.float32, device=self.device)
        dribble_speed_range = getattr(self.cfg, "locomotion_cmd_dribble_speed_range", None) or speed_range
        if torch.any(dribble_mask):
            speed[dribble_mask] = sample_uniform(
                dribble_speed_range[0], dribble_speed_range[1], (int(dribble_mask.sum().item()),), device=self.device
            )
        # A zero-speed state retains the last heading.  This makes STOP's ball
        # corridor continuous with the preceding dribble instead of snapping
        # back to world +X at the instant braking begins.
        heading = torch.where(dribble_mask, heading, self.locomotion_cmd_heading[env_ids])
        wz = torch.zeros(n, dtype=torch.float32, device=self.device)
        duration_s = torch.empty(n, dtype=torch.float32, device=self.device)
        state_duration_ranges = {
            TASK_STATE_IDLE: getattr(self.cfg, "locomotion_task_idle_duration_range", duration_range),
            TASK_STATE_DRIBBLE: getattr(self.cfg, "locomotion_task_dribble_duration_range", duration_range),
            TASK_STATE_STOP: getattr(self.cfg, "locomotion_task_stop_duration_range", duration_range),
        }
        for state, state_range in state_duration_ranges.items():
            mask = task_state == state
            if torch.any(mask):
                duration_s[mask] = sample_uniform(
                    state_range[0], state_range[1], (int(mask.sum().item()),), device=self.device
                )

        self._apply_polar_locomotion(env_ids, speed, heading, wz, task_state=task_state)
        self._locomotion_cmd_hold_steps_remaining[env_ids] = self._duration_s_to_steps(duration_s)

    def _apply_locomotion_segment(self, env_id: int, seg_idx: int) -> None:
        plan = self._locomotion_segment_plans[env_id]
        if not plan or seg_idx < 0 or seg_idx >= len(plan):
            return
        speed, heading, duration_s, wz, task_state = plan[seg_idx]
        eid = torch.tensor([env_id], device=self.device, dtype=torch.long)
        self._apply_polar_locomotion(
            eid,
            torch.tensor([speed], device=self.device, dtype=torch.float32),
            torch.tensor([heading], device=self.device, dtype=torch.float32),
            torch.tensor([wz], device=self.device, dtype=torch.float32),
            task_state=torch.tensor([task_state], device=self.device, dtype=torch.long),
        )
        self._locomotion_cmd_hold_steps_remaining[env_id] = int(self._duration_s_to_steps(duration_s).item())
        self._locomotion_segment_idx[env_id] = seg_idx

    def _restart_locomotion_segment_plans(self, env_ids: torch.Tensor | Sequence[int]) -> None:
        """Restart manual command plans after an environment reset.

        A reset intentionally clears ``_locomotion_cmd_initialized`` so its
        first command is immediate.  Manual multi-segment playback must then
        write segment zero straight away; otherwise the global segment clock
        resumes in the middle of its old plan and the next transition is
        mistaken for a new episode's initial command.
        """
        env_ids_t = self._to_env_id_tensor(env_ids)
        for env_id in env_ids_t.tolist():
            if self._locomotion_segment_plans[env_id]:
                self._apply_locomotion_segment(env_id, 0)

    def _advance_locomotion_segments(self, env_ids: torch.Tensor) -> None:
        for env_id in env_ids.tolist():
            plan = self._locomotion_segment_plans[env_id]
            if not plan:
                continue
            cur = int(self._locomotion_segment_idx[env_id].item())
            next_idx = cur + 1
            if next_idx >= len(plan):
                if self._locomotion_segment_reset_on_end:
                    # The termination manager observes this flag in the same
                    # environment step, then resets robot, ball, and command
                    # back to segment zero.
                    self._locomotion_sequence_finished[env_id] = True
                    self._locomotion_cmd_hold_steps_remaining[env_id] = 1
                    continue
                if self._locomotion_segment_hold_last:
                    next_idx = len(plan) - 1
                else:
                    next_idx = 0
            self._apply_locomotion_segment(env_id, next_idx)

    def _update_locomotion_command_timers(self) -> None:
        self._locomotion_cmd_hold_steps_remaining -= 1
        due_mask = self._locomotion_cmd_hold_steps_remaining <= 0
        if not torch.any(due_mask):
            return
        env_ids = due_mask.nonzero(as_tuple=False).flatten()

        if self._locomotion_command_mode == "resampled":
            self._sample_locomotion_commands(env_ids)
            return

        if self._locomotion_command_mode == "manual":
            manual_advance = []
            for env_id in env_ids.tolist():
                if self._locomotion_segment_plans[env_id]:
                    manual_advance.append(env_id)
            if manual_advance:
                self._advance_locomotion_segments(
                    torch.tensor(manual_advance, device=self.device, dtype=torch.long)
                )

    def set_locomotion_polar_sequence(
        self,
        segments: Sequence[
            tuple[float, float, float]
            | tuple[float, float, float, float]
            | tuple[float, float, float, float, str | int]
        ],
        env_ids: torch.Tensor | Sequence[int] | None = None,
        hold_last: bool = True,
        reset_on_end: bool = False,
    ) -> None:
        """Queue polar commands as ``(speed, heading, duration_s[, wz, task_state])``.

        Omitted state is inferred from speed (positive=DRIBBLE, zero=STOP).
        Segments run in order. ``reset_on_end`` takes precedence at the final
        segment and requests an environment reset; otherwise ``hold_last``
        chooses whether the final segment repeats or the sequence loops.
        """
        if env_ids is None:
            env_id_list = list(range(self.num_envs))
        elif isinstance(env_ids, torch.Tensor):
            env_id_list = env_ids.tolist()
        else:
            env_id_list = list(env_ids)

        normalized: list[tuple[float, float, float, float, int]] = []
        for seg in segments:
            if len(seg) == 3:
                speed = float(seg[0])
                normalized.append(
                    (speed, float(seg[1]), float(seg[2]), 0.0, normalize_locomotion_task_state(
                        "dribble" if speed > 0.05 else "stop"
                    ))
                )
            elif len(seg) >= 4:
                speed = float(seg[0])
                state = (
                    normalize_locomotion_task_state(seg[4])
                    if len(seg) >= 5
                    else normalize_locomotion_task_state("dribble" if speed > 0.05 else "stop")
                )
                normalized.append((speed, float(seg[1]), float(seg[2]), float(seg[3]), state))
            else:
                raise ValueError(
                    f"Each segment needs (speed, heading, duration_s[, wz, task_state]); got {seg!r}"
                )

        if not normalized:
            raise ValueError("locomotion polar sequence must contain at least one segment")

        env_ids_t = self._to_env_id_tensor(env_id_list)
        self._locomotion_segment_hold_last = hold_last
        self._locomotion_segment_reset_on_end = bool(reset_on_end)
        self._locomotion_sequence_finished[env_ids_t] = False
        self._locomotion_cmd_initialized[env_ids_t] = False
        for env_id in env_ids_t.tolist():
            self._locomotion_segment_plans[env_id] = normalized
        self.set_locomotion_command_mode("manual")
        self._restart_locomotion_segment_plans(env_ids_t)

    def _update_locomotion_command_resample(self) -> None:
        """Backward-compatible alias."""
        self._update_locomotion_command_timers()

    def set_locomotion_polar_command(
        self,
        speed: float | torch.Tensor,
        heading: float | torch.Tensor,
        duration_s: float | torch.Tensor | None = None,
        wz: float | torch.Tensor = 0.0,
        task_state: str | int | torch.Tensor | Sequence[str | int] | None = None,
        env_ids: torch.Tensor | slice | None = None,
    ) -> None:
        """Set locomotion by speed (m/s) and heading (rad from task +X) for a hold duration.

        ``heading=0`` → +X forward; ``heading=+pi/2`` → +Y lateral. Independent of demo root vel.
        ``task_state`` is IDLE/DRIBBLE/STOP; if omitted, non-zero speed maps to
        DRIBBLE and zero speed maps to STOP.  ``duration_s=None`` holds until
        the next manual/resample update.
        """
        if env_ids is None:
            env_ids_t = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        elif isinstance(env_ids, slice):
            env_ids_t = torch.arange(self.num_envs, device=self.device, dtype=torch.long)[env_ids]
        else:
            env_ids_t = env_ids.to(device=self.device, dtype=torch.long)

        if env_ids_t.numel() == 0:
            return

        n = env_ids_t.numel()
        if not isinstance(speed, torch.Tensor):
            speed = torch.full((n,), float(speed), device=self.device, dtype=torch.float32)
        if not isinstance(heading, torch.Tensor):
            heading = torch.full((n,), float(heading), device=self.device, dtype=torch.float32)
        if not isinstance(wz, torch.Tensor):
            wz = torch.full((n,), float(wz), device=self.device, dtype=torch.float32)

        states = self._infer_task_state_from_speed(speed) if task_state is None else self._task_state_ids(task_state, n)
        self._apply_polar_locomotion(env_ids_t, speed, heading, wz, task_state=states)
        if duration_s is not None:
            if not isinstance(duration_s, torch.Tensor):
                hold_steps = self._duration_s_to_steps(float(duration_s))
                self._locomotion_cmd_hold_steps_remaining[env_ids_t] = hold_steps
            else:
                self._locomotion_cmd_hold_steps_remaining[env_ids_t] = self._duration_s_to_steps(duration_s)
        for env_id in env_ids_t.tolist():
            self._locomotion_segment_plans[env_id] = []
        self._locomotion_segment_reset_on_end = False
        self._locomotion_sequence_finished[env_ids_t] = False

    def set_locomotion_manual_command(
        self,
        lin_vel: Sequence[float] | torch.Tensor | None = None,
        ang_vel: Sequence[float] | torch.Tensor | None = None,
        task_state: str | int | torch.Tensor | Sequence[str | int] | None = None,
        env_ids: torch.Tensor | slice | None = None,
    ) -> None:
        """Set manual locomotion command in task frame (+X fwd, +Y lat, +Z up; rad/s for ang)."""
        if env_ids is None:
            env_ids = slice(None)

        if lin_vel is not None:
            if not isinstance(lin_vel, torch.Tensor):
                lin_vel = torch.tensor(lin_vel, device=self.device, dtype=torch.float32)
            if lin_vel.dim() == 1:
                self.locomotion_manual_lin_vel[env_ids] = lin_vel.unsqueeze(0).expand(
                    self.locomotion_manual_lin_vel[env_ids].shape[0], -1
                )
            else:
                self.locomotion_manual_lin_vel[env_ids] = lin_vel
            xy = self.locomotion_manual_lin_vel[env_ids, :2]
            speed = torch.norm(xy, dim=-1)
            heading = torch.atan2(xy[:, 1], xy[:, 0])
            self.locomotion_cmd_speed[env_ids] = speed
            # Keep a meaningful ball-control axis when a manual STOP command
            # has zero XY velocity instead of replacing it with atan2(0, 0).
            prior_heading = self.locomotion_cmd_heading[env_ids]
            self.locomotion_cmd_heading[env_ids] = torch.where(speed > 0.05, heading, prior_heading)
            self.locomotion_cmd_target_speed[env_ids] = self.locomotion_cmd_speed[env_ids]
            self.locomotion_cmd_target_heading[env_ids] = self.locomotion_cmd_heading[env_ids]
            self._locomotion_cmd_initialized[env_ids] = True
            states = self._infer_task_state_from_speed(speed) if task_state is None else self._task_state_ids(
                task_state, speed.numel()
            )
            self.locomotion_task_state[env_ids] = states

        if ang_vel is not None:
            if not isinstance(ang_vel, torch.Tensor):
                ang_vel = torch.tensor(ang_vel, device=self.device, dtype=torch.float32)
            if ang_vel.dim() == 1:
                self.locomotion_manual_ang_vel[env_ids] = ang_vel.unsqueeze(0).expand(
                    self.locomotion_manual_ang_vel[env_ids].shape[0], -1
                )
            else:
                self.locomotion_manual_ang_vel[env_ids] = ang_vel

    def set_locomotion_local_twist_command(
        self,
        vx: float | torch.Tensor,
        vy: float | torch.Tensor,
        wz: float | torch.Tensor,
        duration_s: float | torch.Tensor | None = None,
        env_ids: torch.Tensor | slice | None = None,
    ) -> None:
        """Set ``[vx, vy, wz]`` in the current pelvis yaw frame for local-twist tasks."""
        self.set_locomotion_command_mode("manual")
        if env_ids is None:
            env_ids_t = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        elif isinstance(env_ids, slice):
            env_ids_t = torch.arange(self.num_envs, device=self.device, dtype=torch.long)[env_ids]
        else:
            env_ids_t = env_ids.to(device=self.device, dtype=torch.long)
        n = env_ids_t.numel()
        if n == 0:
            return

        def _as_command_tensor(value: float | torch.Tensor) -> torch.Tensor:
            if isinstance(value, torch.Tensor):
                return value.to(device=self.device, dtype=torch.float32).flatten().expand(n)
            return torch.full((n,), float(value), device=self.device, dtype=torch.float32)

        self._apply_local_twist_locomotion(
            env_ids_t, _as_command_tensor(vx), _as_command_tensor(vy), _as_command_tensor(wz)
        )
        if duration_s is not None:
            self._locomotion_cmd_hold_steps_remaining[env_ids_t] = self._duration_s_to_steps(duration_s)
        for env_id in env_ids_t.tolist():
            self._locomotion_segment_plans[env_id] = []
        self._locomotion_segment_reset_on_end = False
        self._locomotion_sequence_finished[env_ids_t] = False

    def set_locomotion_local_twist_sequence(
        self,
        segments: Sequence[tuple[float, float, float, float]],
        env_ids: torch.Tensor | Sequence[int] | None = None,
        hold_last: bool = True,
        reset_on_end: bool = False,
    ) -> None:
        """Queue pelvis-local ``(vx, vy, wz, duration_s)`` playback segments.

        The shared sequence timer internally stores a speed/heading pair, but
        that conversion happens only at plan-load time.  The policy/rewards
        continue to receive the resulting Cartesian local twist and the
        compatibility task state remains DRIBBLE for every segment.
        """
        if not segments:
            raise ValueError("locomotion local-twist sequence must contain at least one segment")
        polar_segments: list[tuple[float, float, float, float, int]] = []
        for segment in segments:
            if len(segment) != 4:
                raise ValueError(
                    "Each local-twist segment needs (vx, vy, wz, duration_s); "
                    f"got {segment!r}"
                )
            vx, vy, wz, duration_s = (float(value) for value in segment)
            if duration_s <= 0.0:
                raise ValueError(f"Local-twist segment duration must be positive; got {duration_s}.")
            speed = math.hypot(vx, vy)
            heading = math.atan2(vy, vx) if speed > 1.0e-6 else 0.0
            polar_segments.append((speed, heading, duration_s, wz, TASK_STATE_DRIBBLE))
        self.set_locomotion_polar_sequence(
            polar_segments,
            env_ids=env_ids,
            hold_last=hold_last,
            reset_on_end=reset_on_end,
        )

    def reference_locomotion_lin_vel_w(self) -> torch.Tensor:
        """Demo anchor linear velocity in task-aligned world frame (mimic yaw delta)."""
        delta_ori_w = mimic_anchor_yaw_delta_quat(
            self.anchor_quat_w,
            self.robot_anchor_quat_w,
            align_task_frame=bool(getattr(self.cfg, "mimic_align_task_frame", False)),
        )
        return quat_apply(delta_ori_w, self.anchor_lin_vel_w)

    def reference_locomotion_ang_vel_w(self) -> torch.Tensor:
        """Demo anchor angular velocity in task-aligned world frame."""
        delta_ori_w = mimic_anchor_yaw_delta_quat(
            self.anchor_quat_w,
            self.robot_anchor_quat_w,
            align_task_frame=bool(getattr(self.cfg, "mimic_align_task_frame", False)),
        )
        return quat_apply(delta_ori_w, self.anchor_ang_vel_w)

    def reference_locomotion_twist_b(self) -> torch.Tensor:
        """Demo pelvis twist in the demo pelvis yaw frame: ``[vx, vy, wz]``.

        The reference coordinate system is reconstructed per frame rather than
        integrated in the world frame.  This is exactly the frame available on
        hardware from the current base orientation and IMU angular velocity.
        """
        ref_yaw_inv = quat_inv(yaw_quat(self.anchor_quat_w))
        ref_lin_b = quat_apply(ref_yaw_inv, self.anchor_lin_vel_w)
        ref_ang_b = quat_apply(ref_yaw_inv, self.anchor_ang_vel_w)
        return torch.stack((ref_lin_b[:, 0], ref_lin_b[:, 1], ref_ang_b[:, 2]), dim=-1)

    def locomotion_twist_blend_alpha(self) -> float:
        """Reference-to-task command interpolation factor for S3 warm start."""
        transition_steps = int(getattr(self.cfg, "locomotion_twist_reference_blend_env_steps", 0))
        if transition_steps <= 0 or self._locomotion_command_mode != "resampled":
            return 1.0
        elapsed_steps = int(getattr(self._env, "common_step_counter", 0))
        return min(max(elapsed_steps / transition_steps, 0.0), 1.0)

    def locomotion_twist_command_b(self) -> torch.Tensor:
        """Active local command, always ordered ``[vx, vy, wz]``."""
        reference_twist = self.reference_locomotion_twist_b()
        if self._locomotion_command_mode == "reference":
            return reference_twist
        task_twist = torch.stack(
            (
                self.locomotion_manual_lin_vel[:, 0],
                self.locomotion_manual_lin_vel[:, 1],
                self.locomotion_manual_ang_vel[:, 2],
            ),
            dim=-1,
        )
        alpha = self.locomotion_twist_blend_alpha()
        return (1.0 - alpha) * reference_twist + alpha * task_twist

    def locomotion_lin_vel_command_w(self) -> torch.Tensor:
        """Active locomotion linear-velocity command (reference, resampled, or manual)."""
        if str(getattr(self.cfg, "locomotion_command_frame", "world")) == "pelvis_local":
            local_twist = self.locomotion_twist_command_b()
            return quat_apply(yaw_quat(self.robot_anchor_quat_w), torch.nn.functional.pad(local_twist[:, :2], (0, 1)))
        if self._locomotion_command_mode in {"manual", "resampled"}:
            return self.locomotion_manual_lin_vel
        return self.reference_locomotion_lin_vel_w()

    def locomotion_ang_vel_command_w(self) -> torch.Tensor:
        """Active locomotion angular-velocity command (reference, resampled, or manual)."""
        if str(getattr(self.cfg, "locomotion_command_frame", "world")) == "pelvis_local":
            local_twist = self.locomotion_twist_command_b()
            return torch.stack(
                (torch.zeros_like(local_twist[:, 2]), torch.zeros_like(local_twist[:, 2]), local_twist[:, 2]), dim=-1
            )
        if self._locomotion_command_mode in {"manual", "resampled"}:
            return self.locomotion_manual_ang_vel
        return self.reference_locomotion_ang_vel_w()

    def _mimic_target_yaw_delta_quat(
        self,
        anchor_quat_w: torch.Tensor,
        robot_anchor_quat_w: torch.Tensor,
    ) -> torch.Tensor:
        """Map the style pose into the configured task or command-heading frame.

        Control locomotion headings are world-frame directions.  Keeping the
        reference upper body in fixed task ``+X`` while asking the pelvis to
        face an oblique command makes the arms (especially the wrist-yaw links)
        fight the turn.  Control opts into the command-heading frame; all
        legacy tasks retain their existing target construction.
        """
        delta = mimic_anchor_yaw_delta_quat(
            anchor_quat_w,
            robot_anchor_quat_w,
            align_task_frame=bool(getattr(self.cfg, "mimic_align_task_frame", False)),
        )
        if not bool(getattr(self.cfg, "mimic_align_locomotion_heading", False)):
            return delta

        heading = self.locomotion_cmd_heading
        # ``anchor_quat_w`` is [env, body, quat] during target construction;
        # make the per-env heading broadcast over its body dimension.
        while heading.ndim < anchor_quat_w.ndim - 1:
            heading = heading.unsqueeze(-1)
        heading_quat = quat_from_euler_xyz(
            torch.zeros_like(heading), torch.zeros_like(heading), heading
        )
        # IsaacLab's ``quat_mul`` requires identical shapes rather than
        # relying on PyTorch broadcasting.  Repeat the per-env heading over
        # the body dimension used by the mimic targets.
        heading_quat = heading_quat.expand_as(delta)
        # The task-frame delta strips demo yaw to +X.  Rotate that style pose
        # from +X into the active external-command heading.
        return quat_mul(heading_quat, delta)

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_pelvis_pos_w(self) -> torch.Tensor:
        pelvis_index = self.robot.body_names.index("pelvis")
        return self.robot.data.body_pos_w[:, pelvis_index]
    
    @property
    def robot_pelvis_quat_w(self) -> torch.Tensor:
        pelvis_index = self.robot.body_names.index("pelvis")
        return self.robot.data.body_quat_w[:, pelvis_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    @property
    def kick_leg(self) -> torch.Tensor:
        return self.motion_kick_leg[self.motion_idx]

    @property
    def kick_leg_name(self) -> list[str]:
        ids = self.motion_kick_leg[self.motion_idx].tolist()
        return [self._kick_leg_id_to_name.get(i, "unknown") for i in ids]

    @property
    def kick_frame(self) -> torch.Tensor:
        """Per-env kick start frame index. -1 means not annotated (no gating)."""
        return self.motion.kick_frames[self.motion_idx]

    @property
    def kick_start_frame(self) -> torch.Tensor:
        """Alias for kick_frame. Per-env kick start frame."""
        return self.kick_frame

    @property
    def kick_end_frame(self) -> torch.Tensor:
        """Per-env kick end frame index. -1 means not annotated."""
        return self.motion.kick_end_frames[self.motion_idx]

    @property
    def dribble_cg_contact_ref(self) -> torch.Tensor:
        """Annotated contact (0/1) at current motion time, shape ``(num_envs,)``."""
        return self._style_motion_labels(self.motion.dribble_cg_contact, 0).to(torch.bool)

    @property
    def dribble_cg_foot_ref(self) -> torch.Tensor:
        """Annotated foot id (-1 none, 0 left, 1 right), shape ``(num_envs,)``."""
        return self._style_motion_labels(self.motion.dribble_cg_foot, -1).to(torch.int64)

    @property
    def dribble_cg_surface_ref(self) -> torch.Tensor:
        """Contact surface id (-1 none, 0 inside instep, 1 outside instep)."""
        return self._style_motion_labels(self.motion.dribble_cg_surface, -1).to(torch.int64)

    @property
    def dribble_cg_foot_ball_dist_ref(self) -> torch.Tensor:
        """Demo foot–ball distance at current frame (m); ``-1`` if unlabeled."""
        return self._style_motion_labels(self.motion.dribble_cg_foot_ball_dist, -1.0)

    @property
    def dribble_cg_dist_foot_ref(self) -> torch.Tensor:
        """Foot id for distance reference at current frame; ``-1`` if unlabeled."""
        return self._style_motion_labels(self.motion.dribble_cg_dist_foot, -1).to(torch.int64)

    @property
    def motion_has_dribble_cg_foot_ball_dist_label(self) -> torch.Tensor:
        """Whether the motion clip has synthesized foot–ball distance labels."""
        return self.motion.motion_has_dribble_cg_foot_ball_dist[self.motion_idx]

    @property
    def dribble_cg_flow_dir_local_ref(self) -> torch.Tensor:
        """Outgoing unit XY direction in the reference source-touch pelvis frame."""
        return self._style_motion_labels(self.motion.dribble_cg_flow_dir_local, 0.0)

    @property
    def dribble_cg_flow_distance_ref(self) -> torch.Tensor:
        """Current contact-to-contact target distance in meters."""
        return self._style_motion_labels(self.motion.dribble_cg_flow_distance, -1.0)

    @property
    def dribble_cg_flow_duration_ref(self) -> torch.Tensor:
        """Current contact-to-contact target duration in seconds."""
        return self._style_motion_labels(self.motion.dribble_cg_flow_duration, -1.0)

    @property
    def dribble_cg_flow_anchor_frame_ref(self) -> torch.Tensor:
        """Current segment's source contact frame, or ``-1`` when invalid."""
        return self._style_motion_labels(self.motion.dribble_cg_flow_anchor_frame, -1).to(torch.int64)

    @property
    def dribble_cg_flow_valid_ref(self) -> torch.Tensor:
        """Whether the current reference frame has an outgoing flow target."""
        return self._style_motion_labels(self.motion.dribble_cg_flow_valid, 0).to(torch.bool)

    @property
    def motion_has_dribble_cg_flow_label(self) -> torch.Tensor:
        """Whether the selected motion contains any contact-to-contact flow label."""
        return self.motion.motion_has_dribble_cg_flow[self.motion_idx]

    @property
    def motion_has_dribble_cg_label(self) -> torch.Tensor:
        """Whether the loaded motion clip has any CG contact labels, shape ``(num_envs,)``."""
        return self.motion.motion_has_dribble_cg[self.motion_idx]

    def get_dribble_demo_ball_goal_world(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Optional demo ball goal in world frame for dribbling CG rewards.

        Returns ``(goal_pos_w, has_demo_mask)`` or ``(None, None)`` when not implemented.
        """
        return None, None

    def _to_env_id_tensor(self, env_ids: Sequence[int] | torch.Tensor) -> torch.Tensor:
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(self.device, dtype=torch.long)
        return torch.as_tensor(list(env_ids), dtype=torch.long, device=self.device)

    def _sample_soccer_offset(self, env_ids: Sequence[int] | torch.Tensor):
        ids = self._to_env_id_tensor(env_ids)
        if ids.numel() == 0:
            return
        if self._radius_offset_min is None or self._radius_offset_max is None:
            self.curve_radius_offset[ids] = 0.0
            return
        if abs(self._radius_offset_max - self._radius_offset_min) < 1e-6:
            self.curve_radius_offset[ids] = self._radius_offset_min
            return

        rand = torch.rand(ids.numel(), device=self.device)
        span = self._radius_offset_max - self._radius_offset_min
        self.curve_radius_offset[ids] = self._radius_offset_min + rand * span

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return

        episode_failed = self._env.termination_manager.terminated[env_ids]
        if isinstance(episode_failed, torch.Tensor):
            episode_failed = episode_failed.to(device=self.device, dtype=torch.bool)
        else:
            episode_failed = torch.tensor(episode_failed, dtype=torch.bool, device=self.device)
        # Clear failure histogram for the current update.
        self._current_bin_failed.zero_()
        # import ipdb; ipdb.set_trace()
        if torch.any(episode_failed):
            # import ipdb; ipdb.set_trace()
            # For failed environments, count the corresponding motion bins.
            failed_env_mask = episode_failed
            failed_motion_idx = self.motion_idx[env_ids][failed_env_mask]                       # [K]
            failed_lengths = self.motion_length[env_ids][failed_env_mask].clamp(min=1).float() # [K]
            failed_steps = self.time_steps[env_ids][failed_env_mask].float()                    # [K]
            # Map time_steps to normalized phase [0, 1], then to bins.
            failed_phase = failed_steps / (failed_lengths - 1.0 + 1e-6)
            failed_bins = torch.clamp((failed_phase * self.bin_count).long(), 0, self.bin_count - 1)  # [K]
            # Accumulate into a 2D histogram via flattened indices.
            flat_idx = failed_motion_idx * self.bin_count + failed_bins                          # [K]
            flat_size = int(self.motion.num_files * self.bin_count)

            # Accumulate safely on GPU to avoid CPU fallback and sync overhead.
            flat_counts = torch.zeros(flat_size, dtype=self._current_bin_failed.dtype, device=self.device)
            if flat_idx.numel() > 0:
                # Ensure indices are on the same device and in long dtype.
                flat_idx = flat_idx.to(self.device).long()
                ones = torch.ones_like(flat_idx, dtype=flat_counts.dtype, device=self.device)
                flat_counts.index_add_(0, flat_idx, ones)

            flat_counts = flat_counts.float()
            # In-place write to keep dtype/device stable.
            self._current_bin_failed[:] = flat_counts.view(self.motion.num_files, self.bin_count)

        # Probability: EMA failure counts plus a uniform prior.
        # Add self.cfg.adaptive_uniform_ratio / (M * B) per element to keep total mass consistent.
        M = max(1, int(self.motion.num_files))
        B = max(1, int(self.bin_count))
        uniform_per_pair = self.cfg.adaptive_uniform_ratio / float(M * B)
        probs = self.bin_failed_count + self._current_bin_failed + uniform_per_pair  # [M, B]
        # Non-causal padding + convolution to smooth along bins per motion.
        probs = torch.nn.functional.pad(
            probs.unsqueeze(1),  # [M, 1, B]
            (0, self.cfg.adaptive_kernel_size - 1),
            mode="replicate",
        )
        probs = torch.nn.functional.conv1d(probs, self.kernel.view(1, 1, -1)).squeeze(1)         # [M, B]

        # Flatten and sample from joint (motion, bin) distribution.
        probs = probs.view(-1)                                                                    # [M*B]
        probs = probs / (probs.sum() + 1e-12)

        sampled_flat = torch.multinomial(probs, len(env_ids), replacement=True)                   # [E]
        sampled_motion = sampled_flat // self.bin_count                                           # [E]
        sampled_bins = sampled_flat % self.bin_count                                              # [E]

        # Map sampled bins to per-motion time_steps with small random offsets.
        self.motion_idx[env_ids] = sampled_motion
        self.motion_length[env_ids] = self.motion.file_lengths[self.motion_idx[env_ids]]
        rand_offset = sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device).float()       # [E]
        sampled_phase = (sampled_bins.float() + rand_offset) / float(self.bin_count)              # [E]
        self.time_steps[env_ids] = (sampled_phase * (self.motion_length[env_ids].float() - 1)).long()

        # Metrics for the joint distribution.
        H = -(probs * (probs + 1e-12).log()).sum()
        denom = math.log(self.bin_count * max(1, int(self.motion.num_files)))
        H_norm = H / denom if denom > 1e-12 else torch.tensor(0.0, device=probs.device)
        pmax, imax = probs.max(dim=0)
        top1_motion = (imax // self.bin_count).float()
        top1_bin = (imax % self.bin_count).float() / self.bin_count
        # import ipdb; ipdb.set_trace()

        # Create metric entries only when needed.
        if "sampling_entropy" not in self.metrics or self.metrics["sampling_entropy"].shape[0] != self.num_envs:
            self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        if "sampling_top1_prob" not in self.metrics or self.metrics["sampling_top1_prob"].shape[0] != self.num_envs:
            self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        if "sampling_top1_bin" not in self.metrics or self.metrics["sampling_top1_bin"].shape[0] != self.num_envs:
            self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)
        if "sampling_top1_motion" not in self.metrics or self.metrics["sampling_top1_motion"].shape[0] != self.num_envs:
            self.metrics["sampling_top1_motion"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = top1_bin
        self.metrics["sampling_top1_motion"][:] = top1_motion

    def _uniform_sampling(self, env_ids: Sequence[int]):
        # Sample motion and time-step separately to avoid out-of-range issues.
        # First, sample motions.
        motion_indices = torch.randint(0, self.motion.num_files, (len(env_ids),), device=self.device)
        self.motion_idx[env_ids] = motion_indices
        self.motion_length[env_ids] = self.motion.file_lengths[motion_indices]
        
        # Then sample a time-step for each selected motion.
        # time_phase = sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device)
        # Start each selected motion from frame 0.
        time_phase = torch.zeros(len(env_ids), device=self.device)

        self.time_steps[env_ids] = (time_phase * (self.motion_length[env_ids].float() - 1)).long()

    def _sequential_sampling(self, env_ids: Sequence[int]):
        """Play the next reference clip in order (for evaluation videos)."""
        next_idx = (self.motion_idx[env_ids] + 1) % self.motion.num_files
        self.motion_idx[env_ids] = next_idx
        self.motion_length[env_ids] = self.motion.file_lengths[self.motion_idx[env_ids]]
        self.time_steps[env_ids] = 0

    def _wrap_style_phase(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        """Loop the current demo phase without resetting the task scene."""
        ids = self._to_env_id_tensor(env_ids)
        if ids.numel() == 0:
            return
        self.style_phase_steps[ids] = torch.remainder(
            self.style_phase_steps[ids], self._style_cycle_lengths()[ids]
        )
        self.style_phase_wrap_count[ids] += 1

    def _spawn_ball_at_motion_start(self, env_ids: torch.Tensor) -> None:
        """Place the ball a fixed distance ahead of the clip's frame-0 anchor (+X)."""
        lateral_jitter = float(self._target_lateral_spawn_jitter)
        distance = float(getattr(self.cfg, "soccer_ball_start_ahead_distance", 0.45))
        lateral_offset = float(getattr(self.cfg, "soccer_ball_start_ahead_lateral", 0.0))
        base_height = float(self._target_height)

        for env_id in env_ids:
            motion_idx = int(self.motion_idx[env_id].item())
            first_anchor = self.motion.get_first_frame_anchor_pos(motion_idx, self.motion_anchor_body_index)
            ball_pos = spawn_ball_ahead_env_local(
                first_anchor,
                distance + float(self.curve_radius_offset[env_id]),
                lateral_offset,
                base_height,
            )
            if lateral_jitter > 0.0:
                y_off = sample_uniform(-lateral_jitter, lateral_jitter, (1,), device=self.device).squeeze(0)
                ball_pos[1] = ball_pos[1] + y_off
            self.soccer_ball_pos[env_id] = ball_pos

    def _spawn_ball_at_paid_original_location(self, env_ids: torch.Tensor) -> None:
        """Reproduce PAiD Stage-I's motion-consistent fixed ball placement.

        The released PAiD command placed the ball at the frame-0 anchor plus
        the clip's complete planar anchor displacement.  Its direction was
        optionally perturbed by ``arc_angle``; ``radius`` supplied the radial
        offset.  Keep this path isolated from the newer task-+X spawn recipes
        so legacy reference-tracking configurations remain reproducible.
        """
        arc_limit = float(self._target_arc_angle)
        base_height = float(self._target_height)

        for env_id in env_ids:
            motion_idx = int(self.motion_idx[env_id].item())
            motion_len = max(1, int(self.motion_length[env_id].item()))

            first_anchor = self.motion.get_first_frame_anchor_pos(motion_idx, self.motion_anchor_body_index)
            last_anchor = self.motion.get_last_frame_anchor_pos(
                motion_idx, self.motion_anchor_body_index, motion_len
            )

            radius_vec = last_anchor[:2] - first_anchor[:2]
            radius_sq = torch.dot(radius_vec, radius_vec)
            if float(radius_sq) > 1e-12:
                radius = torch.sqrt(radius_sq)
                base_direction = radius_vec / radius
            else:
                radius = torch.tensor(0.0, device=self.device)
                base_direction = torch.tensor([1.0, 0.0], device=self.device)

            if arc_limit > 0.0 and float(radius_sq) > 1e-12:
                base_angle = torch.atan2(radius_vec[1], radius_vec[0])
                angle_offset = sample_uniform(-arc_limit, arc_limit, (1,), device=self.device).squeeze(0)
                new_angle = base_angle + angle_offset
                direction = torch.stack((torch.cos(new_angle), torch.sin(new_angle)))
            else:
                direction = base_direction

            radius = torch.clamp(radius + self.curve_radius_offset[env_id], min=0.0)
            ball_pos = self.soccer_ball_pos.new_empty(3)
            ball_pos[:2] = first_anchor[:2] + radius * direction
            ball_pos[2] = base_height
            self.soccer_ball_pos[env_id] = ball_pos

    def _compute_soccer_ball_positions(self, env_ids: Sequence[int] | torch.Tensor):
        if isinstance(env_ids, torch.Tensor):
            ids = env_ids.to(self.device, dtype=torch.long)
        else:
            ids = torch.as_tensor(list(env_ids), dtype=torch.long, device=self.device)

        if ids.numel() == 0:
            return

        spawn_mode = str(getattr(self.cfg, "soccer_ball_spawn_mode", "clip_displacement")).lower().strip()
        if spawn_mode in {"workspace", "stage1_workspace", "ghost_workspace"}:
            # Sample independently of the reference frame.  The first-frame
            # anchor supplies only the episode-local origin; x/y are sampled
            # in the fixed task heading frame, so ball observations cannot
            # become a phase or motion-id shortcut.
            x_range = getattr(self.cfg, "soccer_ball_workspace_x_range", (0.25, 0.80))
            y_range = getattr(self.cfg, "soccer_ball_workspace_y_range", (-0.35, 0.35))
            x = sample_uniform(float(x_range[0]), float(x_range[1]), (ids.numel(),), device=self.device)
            y = sample_uniform(float(y_range[0]), float(y_range[1]), (ids.numel(),), device=self.device)
            self.soccer_ball_workspace_offset_pelvis_yaw[ids, 0] = x
            self.soccer_ball_workspace_offset_pelvis_yaw[ids, 1] = y
            self.soccer_ball_workspace_offset_pelvis_yaw[ids, 2] = 0.0
            mi = self.motion_idx[ids]
            first_anchor = self.motion._body_pos_w[mi, 0, self.motion_anchor_body_index].clone()
            self.soccer_ball_pos[ids, 0] = first_anchor[:, 0] + x
            self.soccer_ball_pos[ids, 1] = first_anchor[:, 1] + y
            self.soccer_ball_pos[ids, 2] = float(self._target_height)
            return
        if spawn_mode in {"reference_first_contact", "first_contact", "contact"}:
            # Replay/diagnostic-only placement matching the annotated first
            # contact in files that contain ball_pos_w.  Training Flat keeps
            # its historical start-ahead randomization by default.
            mi = self.motion_idx[ids]
            contact_frames = self.motion.first_dribble_contact_frame[mi]
            valid = (contact_frames >= 0) & self.motion.motion_has_ball_demo[mi]
            safe_frames = contact_frames.clamp(min=0)
            first_anchor = self.motion._body_pos_w[mi, 0, self.motion_anchor_body_index]
            fallback = first_anchor.clone()
            fallback[:, 0] += float(getattr(self.cfg, "soccer_ball_start_ahead_distance", 0.45))
            fallback[:, 1] += float(getattr(self.cfg, "soccer_ball_start_ahead_lateral", 0.0))
            fallback[:, 2] = float(self._target_height)
            positions = self.motion.ball_pos_w[mi, safe_frames]
            self.soccer_ball_pos[ids] = torch.where(valid.unsqueeze(-1), positions, fallback)
            return
        if spawn_mode in {"start", "start_ahead", "motion_start"}:
            self._spawn_ball_at_motion_start(ids)
            return
        if spawn_mode in {"paid_original", "original"}:
            self._spawn_ball_at_paid_original_location(ids)
            return

        lateral_jitter = float(self._target_lateral_spawn_jitter)
        base_height = float(self._target_height)

        for env_id in ids:
            motion_idx = int(self.motion_idx[env_id].item())
            motion_len = max(1, int(self.motion_length[env_id].item()))

            first_anchor = self.motion.get_first_frame_anchor_pos(motion_idx, self.motion_anchor_body_index,)
            last_anchor = self.motion.get_last_frame_anchor_pos(motion_idx, self.motion_anchor_body_index, motion_len,)

            radius_vec = last_anchor[:2] - first_anchor[:2]
            radius_sq = torch.dot(radius_vec, radius_vec)
            radius = torch.sqrt(radius_sq) if float(radius_sq) > 1e-12 else torch.tensor(0.0, device=self.device)
            radius = torch.clamp(radius + self.curve_radius_offset[env_id], min=0.0)

            # Legacy: displacement magnitude along +X from frame 0 (≈ clip end X for forward clips).
            target_xy = first_anchor[:2].clone()
            target_xy[0] = target_xy[0] + radius
            if lateral_jitter > 0.0:
                y_off = sample_uniform(-lateral_jitter, lateral_jitter, (1,), device=self.device).squeeze(0)
                target_xy[1] = target_xy[1] + y_off

            ball_pos = self.soccer_ball_pos.new_empty(3)
            ball_pos[:2] = target_xy
            ball_pos[2] = base_height
            self.soccer_ball_pos[env_id] = ball_pos

    def _update_target_points(self, env_ids: Sequence[int] | torch.Tensor):
        ids = self._to_env_id_tensor(env_ids)
        if ids.numel() == 0:
            return

        self.target_point_pos[ids] = self.soccer_ball_pos[ids]
        # Also save initial target point for kick-direction computation.
        self.initial_target_point_pos[ids] = self.soccer_ball_pos[ids].clone()

        if self.target_point_marker is not None:
            env_origins = getattr(self._env.scene, "env_origins", None)
            if env_origins is not None:
                world_positions = self.target_point_pos + env_origins
            else:
                world_positions = self.target_point_pos
            self.target_point_marker.visualize(world_positions)

    def _update_target_points_from_sim(self):
        """Read soccer-ball position from simulation each step and update target_point_pos."""
        if self.soccer_ball is None:
            return
        if hasattr(self.soccer_ball, "is_initialized") and not self.soccer_ball.is_initialized:
            return
        
        env_origins = getattr(self._env.scene, "env_origins", None)
        if env_origins is None:
            return
        
        # Read world-space soccer-ball position from simulation.
        ball_world_pos = self.soccer_ball.data.root_pos_w  # [num_envs, 3]
        # Convert to local position relative to env origin.
        self.soccer_ball_pos = ball_world_pos - env_origins
        self.target_point_pos = self.soccer_ball_pos.clone()
        
        # Update visualization marker.
        if self.target_point_marker is not None:
            self.target_point_marker.visualize(ball_world_pos)



    def _update_destination_points(self, env_ids: Sequence[int] | torch.Tensor):
        ids = self._to_env_id_tensor(env_ids)
        if ids.numel() == 0:
            return
        
        # Generate target_destination in world coordinates.
        # Sample destination uniformly within the rectangle.
        rand_x = (torch.rand(ids.numel(), device=self.device) - 0.5) * self.destination_length
        rand_y = (torch.rand(ids.numel(), device=self.device) - 0.5) * self.destination_width
        destination = self.destination_center.expand(ids.numel(), -1) + torch.stack([rand_x, rand_y, torch.zeros_like(rand_x)], dim=1)
        self.target_destination_pos[ids] = destination

        if self.target_destination_marker is not None:
            env_origins = getattr(self._env.scene, "env_origins", None)
            if env_origins is not None:
                world_destination = self.target_destination_pos + env_origins
            else:
                world_destination = self.target_destination_pos
            self.target_destination_marker.visualize(world_destination)
        

    def _update_soccer_ball(self, env_ids: Sequence[int] | torch.Tensor):
        if self.soccer_ball is None or not hasattr(self.soccer_ball, "write_root_state_to_sim"):
            return
        if hasattr(self.soccer_ball, "is_initialized") and not self.soccer_ball.is_initialized:
            return
        ids = self._to_env_id_tensor(env_ids)
        if ids.numel() == 0:
            return
        env_origins = getattr(self._env.scene, "env_origins", None)
        if env_origins is None:
            return

        ball_pos = self.soccer_ball_pos[ids] + env_origins[ids]
        ball_quat = ball_pos.new_zeros((ids.numel(), 4))
        ball_quat[:, 0] = 1.0
        
        # Sample initial linear velocity based on config.
        if self.cfg.enable_soccer_ball_init_vel:
            lin_vel_range = self.cfg.soccer_ball_init_lin_vel_range or {}
            lin_vel_ranges = torch.tensor(
                [lin_vel_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]],
                device=self.device
            )  # [3, 2]
            ball_lin_vel = sample_uniform(
                lin_vel_ranges[:, 0], lin_vel_ranges[:, 1], (ids.numel(), 3), device=self.device
            )
        else:
            ball_lin_vel = ball_pos.new_zeros((ids.numel(), 3))
        
        # Set angular velocity to zero.
        ball_ang_vel = ball_pos.new_zeros((ids.numel(), 3))

        ball_state = torch.cat([ball_pos, ball_quat, ball_lin_vel, ball_ang_vel], dim=-1)
        self.soccer_ball.write_root_state_to_sim(ball_state, env_ids=ids)

    def _follow_workspace_ghost_with_actual_pelvis(self):
        """Keep the clean Stage-1 ghost at its sampled pelvis-yaw offset.

        The ghost stays on the ground plane and follows only pelvis planar
        translation and yaw.  Roll/pitch therefore remain observable through
        the full-pelvis-frame ball observation without making the visual ball
        float with the torso.  This path is explicitly opt-in and never runs
        for physical Stage-2 balls.
        """
        if not bool(getattr(self.cfg, "soccer_ball_follow_actual_pelvis_yaw", False)):
            return
        if self.soccer_ball is None:
            return

        ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        pelvis_pos_w = self.robot.data.body_pos_w[:, self.robot_anchor_body_index]
        pelvis_quat_w = self.robot.data.body_quat_w[:, self.robot_anchor_body_index]
        offset_w = quat_apply(
            yaw_quat(pelvis_quat_w), self.soccer_ball_workspace_offset_pelvis_yaw
        )
        env_origins = getattr(self._env.scene, "env_origins", None)
        if env_origins is None:
            return

        self.soccer_ball_pos[:, :2] = pelvis_pos_w[:, :2] + offset_w[:, :2] - env_origins[:, :2]
        self.soccer_ball_pos[:, 2] = float(self._target_height)
        self._update_soccer_ball(ids)

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return

        env_ids = self._to_env_id_tensor(env_ids)
        if env_ids.numel() == 0:
            return

        self._motion_clip_finished[env_ids] = False

        # Legacy manual diagnostics deliberately keep one global timeline
        # across ordinary failure resets.  There are two explicit exceptions:
        # stateful start/dribble/stop control restarts every reset, while a
        # manual sequence with ``reset_on_end`` restarts only after its final
        # segment deliberately requested this reset.
        manual_restart_env_ids = env_ids.new_empty((0,), dtype=torch.long)
        if self._locomotion_command_mode == "manual":
            restart_every_manual_reset = bool(
                getattr(self.cfg, "locomotion_task_state_restart_manual_sequence_on_reset", False)
            )
            sequence_finished = self._locomotion_sequence_finished[env_ids].clone()
            if restart_every_manual_reset:
                manual_restart_env_ids = env_ids
            elif self._locomotion_segment_reset_on_end:
                manual_restart_env_ids = env_ids[sequence_finished]

        if self._locomotion_command_mode != "manual":
            self._locomotion_cmd_initialized[env_ids] = False
        elif manual_restart_env_ids.numel() > 0:
            self._locomotion_cmd_initialized[manual_restart_env_ids] = False
        self._locomotion_sequence_finished[env_ids] = False

        # In style-looping control this method is reached only for a real
        # episode reset, so restart the per-episode diagnostic counter.
        if not bool(getattr(self.cfg, "motion_clip_end_resample", True)):
            self.style_phase_wrap_count[env_ids] = 0

        self._sample_soccer_offset(env_ids)
        sampling_strategy = str(self.cfg.sampling_strategy).lower()
        if sampling_strategy == "adaptive":
            self._adaptive_sampling(env_ids)
        elif sampling_strategy == "uniform":
            self._uniform_sampling(env_ids)
        elif sampling_strategy == "sequential":
            self._sequential_sampling(env_ids)
        else:
            raise ValueError(f"Unsupported sampling_strategy: {self.cfg.sampling_strategy}")
        self._compute_soccer_ball_positions(env_ids)
        self._update_soccer_ball(env_ids)
        self._update_target_points(env_ids)
        if bool(getattr(self.cfg, "soccer_ball_generate_destination", True)):
            self._update_destination_points(env_ids)
        
        # Sample blind-zone min/max thresholds and reset blind-zone state.
        blind_min_low, blind_min_high = self.cfg.blind_distance_min_range
        blind_max_low, blind_max_high = self.cfg.blind_distance_max_range
        self.blind_distance_min[env_ids] = blind_min_low + torch.rand(env_ids.numel(), device=self.device) * (blind_min_high - blind_min_low)
        self.blind_distance_max[env_ids] = blind_max_low + torch.rand(env_ids.numel(), device=self.device) * (blind_max_high - blind_max_low)
        self.is_in_blind_zone[env_ids] = False
        self.last_visible_target_point_base[env_ids] = 0.0

        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        if bool(getattr(self.cfg, "reset_face_task_forward", False)):
            root_ori[env_ids] = align_body_quat_yaw_to_task_forward(root_ori[env_ids])

        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        if bool(getattr(self.cfg, "reset_zero_velocity", False)):
            root_lin_vel[env_ids] = 0.0
            root_ang_vel[env_ids] = 0.0
            joint_vel[env_ids] = 0.0

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )

        # Set resample flag so env can refresh observations on next step.
        flag_name = f"{self._state_prefix}_motion_resampled"
        resample_flags = getattr(self._env, flag_name, None)
        if resample_flags is None or resample_flags.shape[0] != self.num_envs:
            resample_flags = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        else:
            resample_flags = resample_flags.to(device=self.device, dtype=torch.bool)
        resample_flags[env_ids] = True
        setattr(self._env, flag_name, resample_flags)
        self._steps_since_resample[env_ids] = 0
        if self._locomotion_command_mode == "resampled":
            self._sample_locomotion_commands(env_ids, reset_task_sequence=True)
        elif manual_restart_env_ids.numel() > 0:
            self._restart_locomotion_segment_plans(manual_restart_env_ids)

    # Called every step in the IsaacLab main loop.
    def _update_command(self):
        self.kick_contact_tracker.begin_step(self)
        self._update_locomotion_command_timers()
        self._update_smoothed_locomotion_commands()
        self._steps_since_resample += 1
        # Advance the demo style phase.  Legacy tasks retain full clip-end
        # resampling; control opts into a style-only wrap so its task scene and
        # external locomotion command remain continuous.
        self.style_phase_steps += 1
        self._locomotion_cmd_steps_since_change += 1
        if bool(getattr(self.cfg, "motion_clip_end_terminate", False)):
            # Hold the final valid frame until the termination manager emits
            # done.  Do not call _resample_command here: doing so would write
            # the next episode's state while the recurrent policy still owns
            # the previous episode's hidden state.
            last_frame = (self.motion_length - 1).clamp(min=0)
            env_ids = torch.where(self.style_phase_steps >= last_frame)[0]
            if env_ids.numel() > 0:
                self.style_phase_steps[env_ids] = last_frame[env_ids]
                self._motion_clip_finished[env_ids] = True
        else:
            env_ids = torch.where(self.style_phase_steps >= self._style_cycle_lengths())[0]
            if bool(getattr(self.cfg, "motion_clip_end_resample", True)):
                self._resample_command(env_ids)
            else:
                self._wrap_style_phase(env_ids)
        
        # Clean Stage-1 only: preserve the episode-random, bounded ghost-ball
        # observation while the reference motion translates through world.
        self._follow_workspace_ghost_with_actual_pelvis()

        # Update target point each step using current ball position.
        self._update_target_points_from_sim()

        # Continuously refresh pre-kick target until contact occurs; then keep it frozen.
        if hasattr(self, "kick_contact_tracker"):
            contact_awarded = self.kick_contact_tracker.get_contact_awarded()
            no_contact_mask = ~contact_awarded
            if torch.any(no_contact_mask):
                self.initial_target_point_pos[no_contact_mask] = self.target_point_pos[no_contact_mask]

        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = self._mimic_target_yaw_delta_quat(
            anchor_quat_w_repeat,
            robot_anchor_quat_w_repeat,
        )

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING

    #motion_file: str = MISSING
    motion_files: list[str] = MISSING

    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING
    # Strip demo anchor yaw to task +X for all mimic targets (pos/ori/vel), not only pelvis.
    mimic_align_task_frame: bool = False
    # Control-only opt-in: rotate task-frame style targets from +X into the
    # active locomotion heading.  This avoids an upper-body pose reward that
    # fights command-frame turning.  Old tasks retain task-frame +X targets.
    mimic_align_locomotion_heading: bool = False
    # Legacy behavior resamples the full command and scene when a demo clip
    # ends.  Control disables this so the clip becomes a looping style phase.
    motion_clip_end_resample: bool = True
    # Strict-reference stages may turn the last valid source frame into a
    # genuine time-limit termination.  This keeps recurrent memory aligned
    # with episode resets instead of relying on a hidden mid-episode teleport.
    motion_clip_end_terminate: bool = False
    # Optional same-clip seam bridge for non-resampling style loops.  The
    # final K source frames smoothly cross-fade into the first K frames.
    motion_cyclic_blend_frames: int = 0

    # Locomotion velocity command for follow / control Stage-2 envs.
    # ``reference``: per-frame demo anchor root vel (follow).
    # ``resampled``: random speed/heading/duration — independent of demo root vel (control).
    # ``manual``: fixed polar/xy command via ``set_locomotion_polar_command`` (play/debug).
    locomotion_command_mode: str = "reference"
    # ``world`` preserves legacy polar/task-frame behavior.  ``pelvis_local``
    # makes manual/resampled commands and the policy input a current-pelvis
    # twist ``[vx, vy, wz]``; it is the sim2real-safe command convention.
    locomotion_command_frame: str = "world"
    # During the first N environment steps of a local resampled task, blend
    # its sampled twist from the per-frame demo twist to reduce S2 -> S3 shift.
    locomotion_twist_reference_blend_env_steps: int = 0
    locomotion_manual_lin_vel: tuple[float, float, float] = (0.55, 0.0, 0.0)
    locomotion_manual_ang_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # High-level task input.  Disabled by default so legacy follow/forward
    # training remains unchanged.  Stateful control cycles IDLE -> DRIBBLE ->
    # STOP while using zero velocity for both stationary modes.
    locomotion_task_state_enabled: bool = False
    locomotion_task_state_sequence: tuple[str | int, ...] = ("dribble",)
    locomotion_task_state_stationary_speed: float = 0.05
    locomotion_task_state_restart_manual_sequence_on_reset: bool = False
    locomotion_task_idle_duration_range: tuple[float, float] = (1.0, 2.0)
    locomotion_task_dribble_duration_range: tuple[float, float] = (1.5, 3.0)
    locomotion_task_stop_duration_range: tuple[float, float] = (1.0, 2.0)
    locomotion_resample_interval_s: float = 2.0  # legacy fallback if duration_range unset
    locomotion_cmd_speed_range: tuple[float, float] = (0.25, 0.65)
    # Optional moving-only range used by stateful control.  Generic speed range
    # may include zero so the policy's complete command distribution does too.
    locomotion_cmd_dribble_speed_range: tuple[float, float] | None = None
    locomotion_cmd_heading_range: tuple[float, float] = (-0.75, 0.75)
    locomotion_cmd_duration_range: tuple[float, float] = (1.5, 3.0)
    locomotion_cmd_wz_range: tuple[float, float] = (0.0, 0.0)
    # Optional control transition filter.  Keeping this disabled preserves
    # legacy forward/follow tasks and their instantaneous command semantics.
    locomotion_cmd_smoothing_enabled: bool = False
    locomotion_cmd_heading_rate_limit: float = 0.0
    locomotion_cmd_accel_limit: float = 0.0
    locomotion_cmd_decel_limit: float = 0.0
    locomotion_cmd_turn_slowdown_angle: float = 0.6
    locomotion_cmd_turn_min_speed_scale: float = 1.0
    locomotion_cmd_heading_delta_range: tuple[float, float] | None = None
    locomotion_cmd_lin_vel_range: dict[str, tuple[float, float]] = {
        "x": (0.25, 0.65),
        "y": (-0.25, 0.25),
        "z": (0.0, 0.0),
    }
    locomotion_cmd_ang_vel_range: dict[str, tuple[float, float]] = {
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (-0.35, 0.35),
    }
    # Local resampled command probability for an exact standing command
    # ``[0, 0, 0]``. Disabled by default to preserve all legacy tasks.
    locomotion_cmd_stationary_probability: float = 0.0

    # Soccer ball spawn on motion resample (Stage-1 motion pretrain).
    # ``paid_original``: released PAiD Stage-I placement along the clip's planar displacement.
    # ``clip_displacement`` (legacy dribble): frame-0 anchor + ||last-first|| along +X.
    # ``start_ahead``: fixed distance along +X from frame-0 anchor.
    # ``workspace``: sample an episode-independent local ball position around
    # the frame-0 pelvis.  This is used by clean Stage-1 only so ball position
    # cannot encode reference phase.
    soccer_ball_spawn_mode: str = "clip_displacement"
    soccer_ball_start_ahead_distance: float = 0.45
    soccer_ball_start_ahead_lateral: float = 0.0
    soccer_ball_workspace_x_range: tuple[float, float] = (0.25, 0.80)
    soccer_ball_workspace_y_range: tuple[float, float] = (-0.35, 0.35)
    # Clean Stage-1 opt-in.  A kinematic, collision-free workspace ghost keeps
    # the sampled planar offset relative to the *actual* pelvis yaw.  Physical
    # soccer tasks must leave this false.
    soccer_ball_follow_actual_pelvis_yaw: bool = False
    soccer_ball_generate_destination: bool = True

    # Reset pose: face task +X and drop reference velocities (reduces frame-0 ee_body_pos fails).
    reset_face_task_forward: bool = False
    reset_zero_velocity: bool = False

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)
    sampling_strategy: str = "uniform"

    adaptive_kernel_size: int = 3
    adaptive_lambda: float = 0.1
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.4

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)

    # Target-point marker config; typically overridden in subclasses.
    target_point_marker_cfg: VisualizationMarkersCfg | None = None
    target_destination_marker_cfg: VisualizationMarkersCfg | None = None
    # Offset configuration for arc distribution and destination height.
    curve_offset_range: dict[str, float | tuple[float, float]] | None = None
    
    # Initial soccer-ball velocity configuration.
    enable_soccer_ball_init_vel: bool = False
    soccer_ball_init_lin_vel_range: dict[str, tuple[float, float]] | None = None
    
    # Blind-zone config: ball is invisible when robot-ball (x, y) distance is outside [min, max].
    blind_distance_min_range: tuple[float, float] = (0.3, 0.5)  # Minimum distance sampling range.
    blind_distance_max_range: tuple[float, float] = (1.5, 2.0)  # Maximum distance sampling range.
