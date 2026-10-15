"""Unit tests for the command-defined ball rewards (no Isaac Lab needed).

  python scripts/dribble/test_ball_rewards.py
Scenarios use a 1.28 m/s forward command (the Reference-1 body speed).
"""
import importlib.util
import os

import torch

spec = importlib.util.spec_from_file_location(
    "ball_reward_math",
    os.path.join(os.path.dirname(__file__), "..", "..", "source", "whole_body_tracking", "soccer", "tasks", "tracking", "mdp", "ball_reward_math.py"),
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

T = lambda *a: torch.tensor(a, dtype=torch.float32)  # noqa: E731
cmd = T(1.28, 0.0)
fails = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        fails.append(name)


# front zone
inside = m.front_zone_reward(T(0.4, 0.0))
check("zone: inside box -> 1", abs(float(inside) - 1.0) < 1e-6, f"({float(inside):.3f})")
check("zone: edge of box -> 1", abs(float(m.front_zone_reward(T(0.70, 0.40))) - 1.0) < 1e-6)
behind = float(m.front_zone_reward(T(-1.0, 0.0)))
check("zone: 1 m behind -> small", behind < 0.01, f"({behind:.4f})")
check("zone: monotone falloff behind", float(m.front_zone_reward(T(-0.2, 0))) > float(m.front_zone_reward(T(-0.6, 0))) > behind)
check("zone: far ahead penalised", float(m.front_zone_reward(T(2.0, 0.0))) < 0.01)
check("zone: sideways penalised", float(m.front_zone_reward(T(0.4, 1.4))) < 0.01)
check("zone: symmetric in y", abs(float(m.front_zone_reward(T(0.4, 0.8))) - float(m.front_zone_reward(T(0.4, -0.8)))) < 1e-6)

# velocity follow
ideal = float(m.velocity_follow_reward(T(1.28, 0.0), cmd))
still = float(m.velocity_follow_reward(T(0.0, 0.0), cmd))
back = float(m.velocity_follow_reward(T(-1.0, 0.0), cmd))
side = float(m.velocity_follow_reward(T(0.0, 1.28), cmd))
slow = float(m.velocity_follow_reward(T(0.64, 0.0), cmd))
check("vel: matches command -> 1", abs(ideal - 1.0) < 1e-6)
check("vel: stationary ball is low", still < 0.15, f"({still:.3f})")
check("vel: ordering ideal > slow > still > backwards", ideal > slow > still > back, f"({ideal:.2f},{slow:.2f},{still:.2f},{back:.2f})")
check("vel: sideways kick worse than slow forward", side < slow, f"({side:.3f} < {slow:.3f})")
check("vel: stationary ball still has a gradient", still > 0.05)
check("vel: zero command -> stationary ball scores 1", abs(float(m.velocity_follow_reward(T(0.0, 0.0), T(0.0, 0.0))) - 1.0) < 1e-6)
check("vel: batched shapes", m.velocity_follow_reward(torch.zeros(5, 2), torch.ones(5, 2)).shape == (5,))
try:
    m.velocity_follow_reward(T(0, 0), cmd, shape="nope")
    raised = False
except ValueError:
    raised = True
check("vel: unknown shape raises", raised)

# the V1-a hack: stationary ball + robot standing next to it must not be a good policy
hack = float(m.front_zone_reward(T(0.35, -0.1))) * float(m.velocity_follow_reward(T(0.0, 0.0), cmd))
good = float(m.front_zone_reward(T(0.35, -0.1))) * float(m.velocity_follow_reward(T(1.2, 0.05), cmd))
check("hack: ball parked next to robot loses to a ball moving with the body", good > 5 * hack, f"(good {good:.2f} vs hack {hack:.2f})")

print("\nALL PASSED" if not fails else f"\nFAILED: {fails}")
raise SystemExit(1 if fails else 0)
