"""Which robot body touches the physical ball.

The ball's contact sensor is filtered against every G1 link by name, feet first (``BALL_FILTER_BODY_NAMES``), so
``force_matrix_w`` is ``[N, 1, 30, 3]`` with column 0 = left foot, 1 = right foot, 2.. = all other links; the history
(one entry per physics substep) is used so that short touches inside a control step are kept.  The touching
body is read directly, without a nearest-body heuristic.  (Filtering the robot's 30-body sensor against the single
ball is not supported by PhysX: it needs one filter entry per sensor body.)

Legal dribbling surfaces are the two ankle-roll links (the foot); any other link touching the ball (shin, knee,
hands, torso) is a non-foot contact.
"""

from __future__ import annotations

import torch

FOOT_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")
_OTHER_BODY_NAMES = (
    "pelvis",
    "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link", "left_knee_link", "left_ankle_pitch_link",
    "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link", "right_knee_link", "right_ankle_pitch_link",
    "waist_yaw_link", "waist_roll_link", "torso_link",
    "left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link", "left_elbow_link",
    "left_wrist_roll_link", "left_wrist_pitch_link", "left_wrist_yaw_link",
    "right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link", "right_elbow_link",
    "right_wrist_roll_link", "right_wrist_pitch_link", "right_wrist_yaw_link",
)
BALL_FILTER_BODY_NAMES = FOOT_BODY_NAMES + _OTHER_BODY_NAMES


def ball_filter_prim_paths() -> list[str]:
    """``filter_prim_paths_expr`` for the ball contact sensor (order defines the force-matrix columns)."""
    return ["{ENV_REGEX_NS}/Robot/" + name for name in BALL_FILTER_BODY_NAMES]


def ball_force_matrix_steps(env, sensor_name: str = "soccer_ball_contact") -> torch.Tensor:
    """``[N, H, 30, 3]`` force on the ball per link for each physics substep of the last control step.

    The sensor must keep ``history_length >= decimation`` (see ``ball_dribble_env_cfg``): Isaac Lab then refreshes it
    on every physics substep, and a touch that starts and ends inside one control step is not missed.  With only
    ``force_matrix_w`` (the last substep) about a third of the touches were missed (velocity-jump cross-check).
    """
    data = env.scene.sensors[sensor_name].data
    fm = data.force_matrix_w_history
    if fm is None:
        fm = data.force_matrix_w.unsqueeze(1)
    if fm.shape[3] != len(BALL_FILTER_BODY_NAMES):
        raise RuntimeError(f"{sensor_name}: expected filters {len(BALL_FILTER_BODY_NAMES)}, got {tuple(fm.shape)}; "
                           "use ball_filter_prim_paths()")
    return fm[:, :, 0]


def ball_force_by_body(env, sensor_name: str = "soccer_ball_contact") -> torch.Tensor:
    """``[N, 30]`` max over the control step's substeps of the ball contact force magnitude per link."""
    return torch.linalg.vector_norm(ball_force_matrix_steps(env, sensor_name), dim=-1).amax(dim=1)


def ball_foot_contact(env, sensor_name: str = "soccer_ball_contact", threshold: float = 1.0) -> torch.Tensor:
    """``[N, 2]`` bool: left / right foot touches the ball."""
    return ball_force_by_body(env, sensor_name)[:, :2] > threshold


def nonfoot_ball_contact(env, sensor_name: str = "soccer_ball_contact", threshold: float = 1.0) -> torch.Tensor:
    """Termination: any non-foot link touches the ball."""
    return (ball_force_by_body(env, sensor_name)[:, 2:] > threshold).any(dim=-1)
