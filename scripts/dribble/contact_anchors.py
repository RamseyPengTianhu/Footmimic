"""Contact anchors from an annotated motion npz, written to the main line's agreed interface.

Mirrors the interface of the main line's (not yet committed) ``scripts/motion_gen/contact_anchors.py`` so the two
can be swapped.  Dribble clip: one anchor per contiguous ``dribble_cg_contact`` segment, foot from
``dribble_cg_foot`` (1 = right), direction/distance/duration from the first ``dribble_cg_flow_valid`` frame
inside the segment.  Kick clip: a single anchor from ``kick_frame`` to ``kick_end_frame``; direction and strength
are left empty (they are task commands, not clip properties).

  load_contact_anchors(npz_path) -> list[ContactAnchor]   # sorted by frame
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ContactAnchor:
    frame: int                                     # first frame of the contact window
    end_frame: int                                 # last frame of the contact window
    foot: str                                      # "left" / "right"
    source: str                                    # "kick_frame" or "dribble_cg"
    surface: int | None                            # 0 = inside, 1 = outside (dribble_cg_surface)
    direction_local: tuple[float, float] | None    # dribble_cg_flow_dir_local
    distance_m: float | None                       # dribble_cg_flow_distance
    duration_s: float | None                       # dribble_cg_flow_duration


def _foot_name(value: int) -> str:
    return "right" if int(value) == 1 else "left"


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask).reshape(-1).astype(bool)
    starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    ends = np.flatnonzero(mask & ~np.r_[mask[1:], False])
    return [(int(a), int(b)) for a, b in zip(starts, ends)]


def load_contact_anchors(npz_path: str) -> list[ContactAnchor]:
    z = np.load(npz_path, allow_pickle=True)
    files = set(z.files)
    anchors: list[ContactAnchor] = []

    if "dribble_cg_contact" in files and (np.asarray(z["dribble_cg_contact"]) > 0).any():
        foot = np.asarray(z["dribble_cg_foot"])
        surface = np.asarray(z["dribble_cg_surface"]) if "dribble_cg_surface" in files else None
        has_flow = {"dribble_cg_flow_valid", "dribble_cg_flow_dir_local", "dribble_cg_flow_distance", "dribble_cg_flow_duration"} <= files
        for a, b in _segments(np.asarray(z["dribble_cg_contact"]) > 0):
            direction = distance = duration = None
            if has_flow:
                valid = np.flatnonzero(np.asarray(z["dribble_cg_flow_valid"])[a:b + 1] > 0)
                if valid.size:
                    i = a + int(valid[0])
                    d = np.asarray(z["dribble_cg_flow_dir_local"])[i]
                    direction = (float(d[0]), float(d[1]))
                    distance = float(np.asarray(z["dribble_cg_flow_distance"])[i])
                    duration = float(np.asarray(z["dribble_cg_flow_duration"])[i])
            s = None
            if surface is not None and int(surface[a]) in (0, 1):
                s = int(surface[a])
            anchors.append(ContactAnchor(a, b, _foot_name(foot[a]), "dribble_cg", s, direction, distance, duration))
    elif "kick_frame" in files and int(np.asarray(z["kick_frame"])) >= 0:
        kick = int(np.asarray(z["kick_frame"]))
        end = int(np.asarray(z["kick_end_frame"])) if "kick_end_frame" in files else kick
        leg = str(np.asarray(z["kick_leg"])) if "kick_leg" in files else "right"
        anchors.append(ContactAnchor(kick, max(end, kick), "left" if leg.startswith("l") else "right", "kick_frame", None, None, None, None))

    return sorted(anchors, key=lambda x: x.frame)
