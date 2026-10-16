"""Contact events, soft phase, and gated ball observation (interface v1).

Spec: docs/milestones/SKILL_INTERFACE_V1_PLAN_20261016_CN.md, section 2. Shared by the shoot, pass
and dribble lines; torch only, no Isaac imports. All functions are batched over environments.

Student observation layout (`OBS_DIM` = 134):
    proprio 93 | ball 10 | body command 3 | current event 13 | soft phase 4 | next-event preview 11

Frames: "foot" is the `*_ankle_roll_link` body frame (x forward, y left, z up). "heading" is the
robot's yaw-only frame. The foot plate spans x -0.064..0.142 m, |y| <= 0.036 m, z about -0.035..-0.015 m.
"""

from __future__ import annotations

import math

import torch

# --- layout -------------------------------------------------------------------------------------

# Bumped by the main line on every interface change; "draft" until sync point 1 freezes v1.0
# (docs/milestones/MAINLINE_DRIBBLE_COLLABORATION_20261016_CN.md).
INTERFACE_VERSION = "1.0-draft-20261016"

REGIONS = ("front", "front_inner", "inner", "front_outer", "outer", "instep", "sole")
NUM_REGIONS = len(REGIONS)
REGION_INDEX = {name: i for i, name in enumerate(REGIONS)}
UNSPECIFIED = -1

STAGES = ("approach", "contact", "follow")
APPROACH, CONTACT, FOLLOW = 0, 1, 2

MODE_ARRIVE, MODE_MAX = 0.0, 1.0
TARGET_SCALE = 0.1  # metres -> observation units

PROPRIO_DIM = 93
BALL_DIM = 10
BODY_CMD_DIM = 3
EVENT_DIM = 1 + NUM_REGIONS + 1 + 2 + 1 + 1  # foot, region, contact height, target xy, mode, lifted
PHASE_DIM = len(STAGES) + 1
NEXT_DIM = 1 + 1 + NUM_REGIONS + 2  # has-next, foot, region, target xy
OBS_DIM = PROPRIO_DIM + BALL_DIM + BODY_CMD_DIM + EVENT_DIM + PHASE_DIM + NEXT_DIM

# Encoder reference window (training only): future joint positions at these offsets plus the current
# reference joint velocity (29) and reference angular velocity (3).
REF_OFFSETS = (0, 5, 10, 20, 30)
REF_DIM = 29 * len(REF_OFFSETS) + 29 + 3

# --- foot-plate geometry (right foot; the left foot mirrors y) ------------------------------------

_REGION_CENTER_RIGHT = torch.tensor(
    [
        [0.140, 0.000, -0.025],  # front
        [0.100, 0.036, -0.025],  # front_inner
        [0.030, 0.036, -0.025],  # inner
        [0.100, -0.036, -0.025],  # front_outer
        [0.030, -0.036, -0.025],  # outer
        [0.090, 0.000, -0.015],  # instep (top of the fore plate; moves up once an instep collider exists)
        [0.040, 0.000, -0.035],  # sole
    ]
)
_S = math.sqrt(0.5)
_REGION_NORMAL_RIGHT = torch.tensor(
    [
        [1.0, 0.0, 0.0],
        [_S, _S, 0.0],
        [0.0, 1.0, 0.0],
        [_S, -_S, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ]
)
FORE_X, HEEL_X = 0.070, -0.040


def _mirror(v: torch.Tensor, is_right: torch.Tensor) -> torch.Tensor:
    sign = torch.where(is_right.bool(), 1.0, -1.0).to(v.dtype)
    return torch.stack([v[..., 0], v[..., 1] * sign, v[..., 2]], dim=-1)


def region_center(region: torch.Tensor, is_right: torch.Tensor) -> torch.Tensor:
    """Region centre in the foot frame, shape (N, 3). Unspecified regions map to the front."""
    idx = region.clamp(min=0).long()
    return _mirror(_REGION_CENTER_RIGHT.to(is_right.device)[idx], is_right)


def region_normal(region: torch.Tensor, is_right: torch.Tensor) -> torch.Tensor:
    """Outward normal of the region in the foot frame, shape (N, 3)."""
    idx = region.clamp(min=0).long()
    return _mirror(_REGION_NORMAL_RIGHT.to(is_right.device)[idx], is_right)


def classify_contact(point_foot: torch.Tensor, force_dir_foot: torch.Tensor, is_right: torch.Tensor) -> torch.Tensor:
    """Region index of a contact from its point and the direction of the force on the ball (foot frame).

    The face is the dominant axis of the force on the ball: +x front, +z top (instep), -z sole, and
    inner/outer by the sign of y (+y is inner for the right foot). The zone along the plate (fore
    x >= 0.07 m, heel x < -0.04 m) splits front/inner/outer. Heel contacts return UNSPECIFIED.
    Must match `scripts/motion_gen/audit_s0_contact_regions.py`, which imports it.
    """
    inner_sign = torch.where(is_right.bool(), 1.0, -1.0).to(force_dir_foot.dtype)
    faces = torch.stack(
        [
            force_dir_foot[..., 0],  # front
            inner_sign * force_dir_foot[..., 1],  # inner
            -inner_sign * force_dir_foot[..., 1],  # outer
            force_dir_foot[..., 2],  # top
            -force_dir_foot[..., 2],  # sole
        ],
        dim=-1,
    ).argmax(-1)
    x = point_foot[..., 0]
    inner_side = inner_sign * point_foot[..., 1] > 0
    fore, heel = x >= FORE_X, x < HEEL_X
    out = torch.full_like(faces, UNSPECIFIED)
    out = torch.where((faces == 0) & fore, REGION_INDEX["front"], out)
    out = torch.where((faces == 0) & ~fore & ~heel, torch.where(inner_side, REGION_INDEX["inner"], REGION_INDEX["outer"]), out)
    out = torch.where((faces == 1) & fore, REGION_INDEX["front_inner"], out)
    out = torch.where((faces == 1) & ~fore & ~heel, REGION_INDEX["inner"], out)
    out = torch.where((faces == 2) & fore, REGION_INDEX["front_outer"], out)
    out = torch.where((faces == 2) & ~fore & ~heel, REGION_INDEX["outer"], out)
    out = torch.where((faces == 3) & ~heel, REGION_INDEX["instep"], out)
    out = torch.where(faces == 4, REGION_INDEX["sole"], out)
    return out


def dribble_surface_to_region(surface: torch.Tensor) -> torch.Tensor:
    """`dribble_cg_surface` labels (0 inner, 1 outer, -1 none) to region indices."""
    out = torch.full_like(surface.long(), UNSPECIFIED)
    out = torch.where(surface == 0, REGION_INDEX["inner"], out)
    return torch.where(surface == 1, REGION_INDEX["outer"], out)


# --- encoders -----------------------------------------------------------------------------------


def _region_one_hot(region: torch.Tensor) -> torch.Tensor:
    oh = torch.zeros(*region.shape, NUM_REGIONS, device=region.device)
    valid = region >= 0
    oh[valid, region[valid].long()] = 1.0
    return oh


def encode_event(
    is_right: torch.Tensor,
    region: torch.Tensor,
    contact_height: torch.Tensor,
    target_xy: torch.Tensor,
    mode: torch.Tensor,
    lifted: torch.Tensor,
) -> torch.Tensor:
    """Current contact event, (N, 13).

    contact_height: -1 below the ball centre, 0 centre, +1 above. target_xy: where the ball should go,
    relative to the ball, heading frame, metres. mode: MODE_ARRIVE or MODE_MAX. region: index or
    UNSPECIFIED (all-zero one-hot).
    """
    return torch.cat(
        [
            is_right.float().unsqueeze(-1),
            _region_one_hot(region),
            contact_height.float().unsqueeze(-1),
            target_xy.float() * TARGET_SCALE,
            mode.float().unsqueeze(-1),
            lifted.float().unsqueeze(-1),
        ],
        dim=-1,
    )


def encode_next(has_next: torch.Tensor, is_right: torch.Tensor, region: torch.Tensor, target_xy: torch.Tensor) -> torch.Tensor:
    """Next-event preview, (N, 11); all zeros when there is no next event."""
    h = has_next.float().unsqueeze(-1)
    return h * torch.cat(
        [torch.ones_like(h), is_right.float().unsqueeze(-1), _region_one_hot(region), target_xy.float() * TARGET_SCALE],
        dim=-1,
    )


def encode_phase(stage: torch.Tensor, progress: torch.Tensor) -> torch.Tensor:
    """Soft phase, (N, 4): stage one-hot and approach progress in [0, 1]."""
    oh = torch.nn.functional.one_hot(stage.long(), len(STAGES)).float()
    return torch.cat([oh, progress.float().clamp(0.0, 1.0).unsqueeze(-1)], dim=-1)


def ball_obs(
    ball_pos_b: torch.Tensor, ball_vel_b: torch.Tensor, ball_from_region_b: torch.Tensor, possession: torch.Tensor
) -> torch.Tensor:
    """Gated ball observation, (N, 10): possession x [pelvis-frame position, velocity, offset from the
    current event's foot region], then the possession flag. Zero after release."""
    p = possession.float().unsqueeze(-1)
    return torch.cat([p * ball_pos_b, p * ball_vel_b, p * ball_from_region_b, p], dim=-1)


# --- soft phase state machine -------------------------------------------------------------------


class SoftPhase:
    """Event-driven phase per environment (no frame clock).

    approach -> contact when the ball comes within `reach` of the event's foot region;
    contact -> follow when a touch is detected (or the ball leaves reach again after contact).
    `reach` is drawn per event from reach * [1 - jitter, 1 + jitter], which is how training perturbs
    the switching time instead of shifting a clock.
    """

    def __init__(self, num_envs: int, device, reach: float = 0.25, jitter: float = 0.3):
        self.reach_nominal, self.jitter = reach, jitter
        self.stage = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.start_dist = torch.ones(num_envs, device=device)
        self.reach = torch.full((num_envs,), reach, device=device)
        self.progress = torch.zeros(num_envs, device=device)

    def start_event(self, mask: torch.Tensor, dist: torch.Tensor) -> None:
        """Begin a new event for envs in `mask` (episode reset or the planner issued the next event)."""
        m = mask.bool()
        self.stage[m] = APPROACH
        self.start_dist[m] = dist[m].clamp(min=1e-3)
        u = torch.rand(int(m.sum()), device=dist.device) * 2.0 - 1.0
        self.reach[m] = self.reach_nominal * (1.0 + self.jitter * u)
        self.progress[m] = 0.0

    def update(self, dist: torch.Tensor, touched: torch.Tensor) -> torch.Tensor:
        """dist: ball distance to the event's region centre; touched: ball-foot contact this step."""
        touched = touched.bool()
        approach = self.stage == APPROACH
        was_contact = self.stage == CONTACT
        self.progress = torch.where(approach, 1.0 - dist / self.start_dist, torch.ones_like(dist)).clamp(0.0, 1.0)
        enter = approach & ((dist <= self.reach) | touched)
        self.stage = torch.where(enter, torch.full_like(self.stage, CONTACT), self.stage)
        # Leave on the detected touch, or when the ball moves back out of reach without one (a miss).
        leave = (self.stage == CONTACT) & (touched | (was_contact & (dist > 1.5 * self.reach)))
        self.stage = torch.where(leave, torch.full_like(self.stage, FOLLOW), self.stage)
        return encode_phase(self.stage, self.progress)


# --- rule planner -------------------------------------------------------------------------------


def shoot_or_pass_target(goal_xy_h: torch.Tensor, ball_xy_h: torch.Tensor) -> torch.Tensor:
    """Outcome target for a single-touch event: goal (shoot) or teammate (pass) relative to the ball."""
    return goal_xy_h - ball_xy_h


def dribble_target(cmd_vel_xy_h: torch.Tensor, touch_interval_s: float | torch.Tensor, lead_m: float = 0.0) -> torch.Tensor:
    """Outcome target for the next dribble touch ("pass to your future self"): the ball should travel
    roughly as far as the body will in one touch interval, plus an optional lead, along the command."""
    speed = cmd_vel_xy_h.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return cmd_vel_xy_h * touch_interval_s + lead_m * cmd_vel_xy_h / speed


def possession_after(stage: torch.Tensor, mode: torch.Tensor, has_next: torch.Tensor) -> torch.Tensor:
    """Possession flag: released (0) once a single-touch event reaches follow-through with no next
    event. Dribble always has a next event, so it never releases."""
    released = (stage == FOLLOW) & ~has_next.bool()
    return (~released).float()
