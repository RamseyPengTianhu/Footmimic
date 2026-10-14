# Local-twist dribbling curriculum

The active three-stage path uses one 163-D actor interface and one 29-D action
interface.  Its locomotion command is always the current-pelvis local twist:
`[vx_local, vy_local, wz]`.  There is no required world heading or accumulated
yaw target, which matches real-robot deployment.

Train the complete path with:

```bash
bash shell/progressive_dribbling_train.sh my_run
```

| Stage | Task ID | Objective |
| --- | --- | --- |
| S1 | `Tracking-CG-G1-Motion-RNN-unified-s1-local-strict` | Strict relative motion/gait imitation and exact reference local twist; no ball objective. |
| S2 | `Tracking-CG-G1-Dribbling-RNN-unified-s2-local-reference` | One task whose Isaac Lab curriculum automatically advances through frame-0 prefixes of two/four/eight/full contacts. |
| S3 | `Tracking-CG-G1-Dribbling-RNN-unified-s3-local-task` | Sampled local task twist with generic physical `instep` contacts; no reference ball position, contact time, contact foot, or foot--ball distance reward. |

S1 preserves the raw clip and emits a time-limit `done` on its final frame.
S2 starts at frame 0 and automatically advances through five levels in the same
run. Partial levels end after their second, fourth, or eighth contact window;
the last two levels run the complete clip. Every reset therefore gives the
robot, physical ball, and LSTM the same causal prefix. S3 instead loops the same master clip through a
25-frame (0.5 s at 50 Hz) quintic bridge, blending its tail into its start
without resetting the ongoing task scene.

The S2 levels are the exact first two contacts, exact first four, exact first
eight, exact full clip, and disturbed full clip. No curriculum level samples an
isolated or interior contact. Missed events are recorded but never end an
episode, so every rollout preserves the physical transition into later contact
attempts. The curriculum evaluates non-overlapping windows of at least 1024
eligible prefix episodes and requires two consecutive passing windows. Levels
0--2 require at least 60% completion of the two/four/eight prefix; level 3
requires at least 60% completion of every selected contact in the full clip
and at most 5% falls. Promotion is monotonic; there is no automatic demotion or
manual task switch. New checkpoints save the active level, pass streak,
evaluation-window accumulators, level-entry history, and a signature of the
level schedule. Resume and playback restore that state before resampling the
first episode. A checkpoint from any former schedule, including the v5.17
random-window sequence schedule, retains its policy and optimizer but resets
the incompatible curriculum state to the new level 0. Playback accepts
`--s2_curriculum_level 0..4` to inspect another
distribution explicitly.

At reset, all three stages retain the selected reference pelvis pose, yaw, and
velocity; they do not canonicalize the robot to a simulation/world `+X` axis.
For S2 the physical ball is initialized at the reference's first contact target.
Levels 0--3 use the exact reset; only level 4 adds its configured one-time planar
perturbation. Internally this point is stored as a pelvis-local offset, so the
same reset geometry remains valid under any simulator yaw. If a motion lacks
either a contact label or
`ball_pos_w`, the explicit fallback is 0.45 m ahead in the frame-0 pelvis
local frame. S1 applies no ball reward; it retains this identical spawn only
because the unified 163-D observation contains the ball-relative input.

S3 samples `vx∈[0,1.50] m/s`, `vy∈[-0.50,0.50] m/s`, and
`wz∈[-0.80,0.80] rad/s`, with 10% exact `[0,0,0]` standing commands. For its
first 24,000 environment steps (1,000 PPO updates at 24 rollout steps), the
effective command blends from the S2 reference twist to the sampled task
twist. All root-velocity rewards and policy inputs use that same effective
command.

Only the three local S1/S2/S3 task IDs in the table above are registered.

## Interface and deployment

The final policy input is the 3-D local twist.  A retained 3-D DRIBBLE one-hot
field is only an interface-compatibility placeholder, so the actor remains
163-D; it is not a second command.  The actor does not receive simulated local
linear velocity.  The critic receives that privileged pelvis-local velocity
for value learning.

For one manual command, use `--locomotion_cmd_vx`, `--locomotion_cmd_vy`, and
`--locomotion_cmd_wz`; on the local tasks these are in the current pelvis frame.
For repeatable multi-segment playback, pass
`--locomotion_cmd_plan motions/command_plan_dribble_left_right_local.json`.
The JSON plan uses direct pelvis-local `vx`, `vy`, `wz`, and `duration_s` per
segment; it never exposes a polar command or IDLE/STOP state to the policy.

## Contact semantics

S2 keeps each source label explicitly as `0=inside_instep` or
`1=outside_instep`; conversion to the selected foot's medial axis happens only
inside the geometry calculation. Contact proximity uses only the physical ball
relative to the actual specified foot: distance beyond the 0.25 m contact reach and the labelled lateral
half-space beyond a 4 cm dead zone. A coarse one-sided front gate ramps from
zero at -0.05 m to full at +0.05 m in the foot-yaw frame. It has no forward
target coordinate or upper forward bound; it exists only to suppress dense
reward from keeping the ball behind the foot. Starting 0.30 s before the event,
contact proximity ramps from 0.2 to full strength and stops accumulating after
the first valid touch. During the final 0.3 s before an event, the contact foot's
imitation weight ramps from 1.0 to 0.1 and stays at 0.1 after contact until the
window closes; the support foot always keeps its full reference target. A
correct-side touch is decided from the physical ball's current collision-frame
lateral sign and dead zone, without a forward-position condition or lateral
upper bound. The S2 ball terms remain
contact proximity, one new correct-foot touch, correct-side bonus, a continuous
impact penalty starting at 80 N, and undesired-contact penalty. The touch remains
valid through the 150 N hard cap; premature touches reuse the existing
undesired-contact term at half strength. Between events no reference ball
trajectory or velocity is supervised.

When S2 loads an S1 checkpoint, policy/LSTM weights and observation normalizers
are retained, but the S1 optimizer state is not loaded and Gaussian exploration
is reset to standard deviation 0.40. Resuming an S2 checkpoint retains its
optimizer, learned exploration, and curriculum state.

The checked-in `master-aligned` clip is regenerated with the current contact
geometry and intentionally labels the right foot only:
S2 learns that foot's inside/outside contacts, not left/right-foot alternation.

Use `--show_s2_contact_regions` during playback to display the frozen reference
foot frame, coarse front ramp, and expected lateral half-space. Combine it with
`--diagnostic --diagnostic_stride 1` to save event id/frame, expected foot,
expected source surface and derived local side, reference foot pose, ball offset, physical ball-foot distance,
front-gate value, and proximity error. The `dribble-v8` diagnostic also records
the checkpoint iteration and `s2_curriculum_level_entry_iteration`: the exact
RSL-RL learning iteration at which every future level first became active.
Unknown history from an older checkpoint remains `-1`. The trace retains the
restored curriculum level, uniform-audit flag, sampled episode contact count,
and cumulative premature-contact count.

### Removed S2 ablations

The four former S2 ablations were cumulative diagnostic environments, not four
extra reward terms. They have now been removed together with the old task IDs:

| Label | Contact supervision that was enabled |
|---|---|
| `motion` | Motion/reference-twist objectives only; all S2 contact-event rewards and contact-related terminations were disabled. |
| `time` | Added reference-window touch timing, but accepted either ankle and ignored the labelled instep side. |
| `foot` | Added the expected left/right contact foot, but still ignored inside/outside instep side. |
| `side` | Added the labelled inside/outside instep side, making it equivalent to the complete S2 contact objective. |

All four used a fixed two-contact initialization distribution so that contact
supervision was the only independent variable.

The S2 contact-curriculum task requires valid `dribble_cg_contact`,
`dribble_cg_foot`, `dribble_cg_surface`, and `ball_pos_w` arrays. At this
revision the compatible checked-in bank is `motions/master-aligned`; passing a
legacy mixed bank fails during environment creation instead of training with
unknown side labels.

`instep` means a dorsal contact region on either foot. S3 does not force either
side or a reference left/right alternation, keeping recovery and consecutive
same-side touches feasible when the task requires them.

## Diagnostic

Run playback with `--diagnostic --diagnostic_stride 1`.  The `dribble-v8`
archive records `reference_twist_local`, `active_twist_local`,
`actual_twist_local`, `twist_local_error`, and `twist_blend_alpha` alongside
contact and action traces. It also records the virtual style phase, its source
frame pair, and the seam blend factor; these seam fields are active only in
S3. `ball_spawn_source`, `ball_spawn_reference_contact_frame`, and
`ball_spawn_reference_local` record whether reset used the labelled first
contact point or the deliberate local-front fallback. For S1/S2, check local twist error and reference
foot/contact agreement.  For S3, judge local twist tracking, possession,
generic legal instep touches, and ball progress; do not score CG-foot match as
an S3 requirement.
