# S2 outside/inside transition analysis

Date: 2026-08-13

## Scope

This note records why the current single-motion S2 tracker reliably completes
the first two right-foot touches but does not complete the third touch, which
is the first `outside -> inside` transition.  Contact frame, contact point and
inside/outside labels remain reward-side supervision; they are not proposed as
actor observations here.

## Reference geometry

The first three contact anchors are:

| Event | Frame | Surface | Ball position relative to right ankle yaw frame |
| --- | ---: | --- | --- |
| 0 | 35 | outside | `[+0.070, -0.139] m` |
| 1 | 47 | outside | `[+0.078, -0.142] m` |
| 2 | 67 | inside | `[+0.082, +0.143] m` |

The important outgoing segments are:

| Segment | Duration | Anchor distance | Source-foot-local displacement | Nominal speed |
| --- | ---: | ---: | --- | ---: |
| 35 -> 47, outside -> outside | 0.24 s | 0.268 m | `[+0.185, -0.193] m` | 1.11 m/s |
| 47 -> 67, outside -> inside | 0.40 s | 0.415 m | `[+0.396, +0.126] m` | 1.04 m/s |

Although the second segment's anchor displacement has only `+0.126 m` lateral
motion in the source-foot frame, the ball must change from `-0.142 m` outside
the moving right foot to `+0.143 m` inside it.  Therefore the second outside
touch must control both forward release and the future body-relative lateral
relationship; merely making the ball move forward is insufficient.

## Training evidence

Before adding full-segment anchor progress:

- `s2_complete_2`: approximately `0.97`
- `s2_complete_3`: exactly `0.0`
- `s2_first_release_quality`: approximately `1.0`
- `s2_target_region_distance`: approximately `0.22--0.24 m`

After adding `s2_next_anchor_progress`:

- episode reward for the new term increased from approximately `0.13` to
  `0.165`
- `s2_target_region_distance` decreased to approximately `0.17 m`
- `s2_complete_2` recovered to approximately `0.93--0.94`
- `s2_complete_3` remained exactly `0.0`
- robot anchor-position error increased to approximately `0.15 m`

The new term changes behavior but does not solve the transition.  The policy
learns partial approach rather than arrival.

## Root causes

### 1. Release target and contact-success target use different coordinates

At event 1, `s2_next_touch_target_w()` maps event 2's reference anchor using
the robot/reference alignment at the physical event-1 touch.  The reward then
latches that world point for the entire segment.

At event 2, `s2_contact_anchor_target_w()` maps the same reference anchor again
using the *current* robot/reference alignment.  Thus contact success is checked
against a live body-relative target, not the latched release target.

When the robot accumulates position/yaw tracking error between events, the two
world targets separate.  The observed approximately `0.15 m` robot anchor error
is the same order as the `0.14 m` hard contact-anchor gate.  A ball can improve
the old latched-target reward and still fail the live event-2 gate.

This inconsistency is especially damaging for outside/inside switches because
success depends on the ball being on the opposite side of the moving foot.

### 2. The current progress reward can be farmed without net arrival

The implementation rewards

```text
max(previous_distance - current_distance, 0)
```

and then always replaces `previous_distance` with `current_distance`.  Moving
away is free; approaching again is rewarded again.  Oscillation can therefore
earn unbounded repeated progress without reducing final distance.  The logged
`s2_next_anchor_progress` is not evidence of net arrival.

### 3. There is no deadline objective

The eight-step release reward scores initial direction/speed.  The later
progress reward scores partial distance reduction.  Neither requires the ball
to be inside the event-2 anchor tolerance at the event-2 deadline.  The policy
can keep both rewards while stopping around `0.17 m`, just outside the `0.14 m`
success gate.

### 4. Current aggregate metrics cannot identify the exact third-touch failure

`s2_target_region_distance` averages the foot-side proximity geometry across
all active event windows.  It is not the third event's contact-anchor distance.
The final printed `s2_next_anchor_progress` is an instantaneous value and is
normally zero after the segment ends.  Current logs do not separate:

- no physical third contact;
- physical third contact on the wrong side;
- correct-side contact rejected only by the anchor-distance gate;
- distance to latched target versus distance to live target.

### 5. Observation is not the first blocker for the single-motion tracker

The actor receives the current reference joint position and velocity.  For one
fixed motion this already identifies the reference phase, and the required
outside/inside sequence is deterministic.  Adding an explicit contact frame,
surface label or future anchor to the actor would change the deployment
contract and is not justified until the reward target is internally
consistent.  It may be revisited later for a multi-motion/general controller.

## Proposed reward design

### A. One live body-relative target for both transition shaping and success

For source event `k`, evaluate the next event `k+1` target every step using the
same mapping used by `s2_contact_anchor_target_w(k+1)`.  Compute the error in the
current robot pelvis-yaw frame:

```text
e_k(t) = ball_relative_to_robot(t) - next_anchor_reference_relative_to_robot(t)
d_k(t) = ||e_k(t)||_xy
```

The event-`k+1` contact gate must use this same `d_k(t)`.  Do not shape toward a
latched world point and later validate against a moving body-relative point.

The outgoing direction used for the short impact/release score can still be
latched at contact, because it describes the initial physical impulse.  Only
the longer arrival objective must use the live body-relative target.

### B. Bounded monotonic progress

Store the best (minimum) distance reached during the segment:

```text
best_new = min(best_previous, d_k(t))
r_progress = (best_previous - best_new) / dt
```

This reward cannot be farmed by moving away and approaching again.  Its total
undiscounted return is bounded by the initial distance reduction.  Normalize by
the initial distance if equal reward magnitude across short and long segments
is desired:

```text
r_progress_normalized = r_progress / max(d_initial, eps)
```

Recommended initial weight: `2.0--3.0` after normalization, or retain weight
`4.0` without normalization.

### C. Deadline arrival reward and miss penalty

At the next event's contact window, use the same live distance:

```text
r_arrival = exp(-(d_k / 0.10)^2)
```

At the event center or event completion, add one pulse:

```text
r_deadline_success = 1[d_k <= 0.14]
r_deadline_miss = -clip((d_k - 0.14) / 0.14, 0, 2)
```

Suggested weights:

- continuous final-window arrival: `2.0`
- deadline success pulse: `+4.0`
- deadline miss pulse: `-2.0`

The existing valid-contact bonus remains separate: arrival means the ball was
placed correctly; contact means the expected foot actually touched it at the
correct time and side.

### D. Keep the short release-quality reward

Retain the first eight simulation steps of direction, speed-band and separation
reward.  It teaches how to create the impulse; the bounded progress and
deadline terms teach where that impulse must end.  Do not extend a constant
target-speed reward over the entire 0.40 s because rolling friction naturally
changes ball speed.

### E. Optional switch weighting

After the generic version is verified, transitions whose next surface differs
from the current surface may multiply the bounded progress and deadline terms
by `1.25--1.5`.  This remains reward-side supervision.  It should not be used
to hide coordinate inconsistencies or replace transition-specific metrics.

## Required diagnostics before the next long run

Add the following per-event or per-transition metrics:

- event-0, event-1 and event-2 valid-contact rates;
- event-2 physical-contact rate;
- event-2 correct-side rate before the anchor gate;
- event-2 live anchor distance at window start, event center and completion;
- minimum live anchor distance during 47 -> 67;
- distance between the old latched world target and the live target;
- explicit rejection counts for wrong foot, wrong side and anchor distance.

The next experiment is successful only if event-2 minimum/deadline distance
falls and `s2_complete_3` becomes nonzero.  An increase in progress reward alone
is not a success criterion.

## Curriculum support

If the corrected reward still produces no event-2 successes, add a
transition-focused teacher-reset training pool starting shortly before event 1
with the robot and ball in the reference event-1 state.  Keep full frame-0
three-touch episodes as the only promotion audit.  This shortens credit
assignment for learning 47 -> 67 without changing the final controller or
allowing teacher resets to promote the curriculum.
