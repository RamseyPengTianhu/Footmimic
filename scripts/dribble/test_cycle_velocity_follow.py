"""Tests for the stateful cycle-averaged velocity-follow reward (pure torch, no Isaac Lab).

  python scripts/dribble/test_cycle_velocity_follow.py
Replays the references frame by frame (boundaries from the contact clock) and checks the pulses against the offline
numbers from offline_cycle_velocity_ceiling.py, then checks the failure cases the reward exists to reject.
"""
import importlib.util
import os

import numpy as np
import torch

here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("brm", os.path.join(here, "..", "..", "source", "whole_body_tracking", "soccer", "tasks", "tracking", "mdp", "ball_reward_math.py"))
brm = importlib.util.module_from_spec(spec); spec.loader.exec_module(brm)

fails = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        fails.append(name)


def replay(path, L, ball_override=None, cmd_scale=1.0):
    z = np.load(os.path.join(here, "..", "..", path), allow_pickle=True)
    dt = 1.0 / int(np.asarray(z["fps"]).reshape(-1)[0])
    c = z["dribble_cg_contact"] > 0
    ev = np.flatnonzero(c & ~np.r_[False, c[:-1]])
    bounds = set(int(ev[i]) for i in range(0, len(ev), L))
    ball = z["ball_pos_w"][:, :2] if ball_override is None else ball_override
    cmd = z["body_lin_vel_w"][:, 0, :2] * cmd_scale
    st = brm.CycleVelocityFollow(1, "cpu", std=0.5)
    out = []
    for f in range(len(ball)):
        s = st.step(torch.tensor(ball[f:f + 1], dtype=torch.float32), torch.tensor(cmd[f:f + 1], dtype=torch.float32),
                    torch.tensor([f in bounds]), dt)
        if f in bounds:
            out.append(float(s))
    return np.array(out), ev


REF1 = "motions/master-aligned/master_001675_001884_unitree_g1.npz"
REF2 = "motions/master-annotated-003178-grounded/master_003178_003326_unitree_g1.npz"

p1, ev1 = replay(REF1, 4)
check("ref1: first boundary only starts a cycle (no pulse)", p1[0] == 0.0)
check("ref1: 3 full cycles after the first boundary", len(p1) == 4 and (p1[1:] > 0).all(), f"(pulses {np.round(p1,2).tolist()})")
check("ref1: mean cycle pulse ~0.89 (offline ceiling)", abs(p1[1:].mean() - 0.89) < 0.12, f"({p1[1:].mean():.2f})")
p2, _ = replay(REF2, 2)
check("ref2: mean cycle pulse ~0.78 (offline ceiling)", abs(p2[1:].mean() - 0.78) < 0.12, f"({p2[1:].mean():.2f})")

z = np.load(os.path.join(here, "..", "..", REF1), allow_pickle=True)
ball = z["ball_pos_w"][:, :2]
still, _ = replay(REF1, 4, ball_override=np.repeat(ball[:1], len(ball), axis=0))
check("stationary ball scores far below the reference", still[1:].mean() < 0.3 * p1[1:].mean(), f"({still[1:].mean():.3f} vs {p1[1:].mean():.2f})")
wrong, _ = replay(REF1, 4, cmd_scale=-1.0)
check("ball moving against the command is rejected", wrong[1:].mean() < 0.3 * p1[1:].mean(), f"({wrong[1:].mean():.3f})")
fast, _ = replay(REF1, 4, cmd_scale=2.0)
check("command twice as fast as the ball is penalised", fast[1:].mean() < 0.6 * p1[1:].mean(), f"({fast[1:].mean():.3f})")

# reset behaviour and batching
st = brm.CycleVelocityFollow(3, "cpu")
z2 = torch.zeros(3, 2)
st.step(z2, z2, torch.tensor([True, True, True]), 0.02)
st.reset(torch.tensor([1]))
check("reset deactivates only the chosen env", st.active.tolist() == [True, False, True])
s = st.step(z2, z2, torch.tensor([True, True, True]), 0.02)
check("after reset the next boundary only restarts (env 1: no pulse; others pulse)", s.tolist() == [1.0, 0.0, 1.0], f"({s.tolist()})")
st = brm.CycleVelocityFollow(2, "cpu")
check("boundary False everywhere -> zero reward", st.step(z2[:2], z2[:2], torch.tensor([False, False]), 0.02).abs().sum() == 0)

print("\nALL PASSED" if not fails else f"\nFAILED: {fails}")
raise SystemExit(1 if fails else 0)
