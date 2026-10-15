"""Measure the simulated ball's free-roll deceleration on the training ground.

Gives the ball an initial horizontal speed and records speed vs time with no robot
contact.  The robot is parked far away.  Used to calibrate the rolling-decay model
for the predicted-arrival reward.  Example:
  python scripts/dribble/measure_ball_roll.py --output out.npz --headless
"""
import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--speeds", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0, 3.0])
parser.add_argument("--seconds", type=float, default=3.0)
parser.add_argument("--linear_damping", type=float, default=None)
parser.add_argument("--const_mu", type=float, default=None, help="constant-deceleration rolling resistance coefficient (mu_r)")
parser.add_argument("--force_scale", type=float, default=1.0)
parser.add_argument("--velocity_decel", type=float, default=None, help="engine-independent constant deceleration [m/s^2] applied to the velocity every physics step")
parser.add_argument("--update_every", type=int, default=1, help="physics steps between force updates (4 = policy rate)")
parser.add_argument("--rolling_start", action="store_true", help="start with the spin of pure rolling instead of sliding")
parser.add_argument("--angular_damping", type=float, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = False
simulation_app = AppLauncher(args_cli).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402

from soccer.robots.g1 import G1_CYLINDER_CFG  # noqa: E402
from soccer.tasks.tracking.config.g1.soccer_flat_env_cfg import G1FlatSoccerSceneCfg, apply_ball_damping  # noqa: E402
from soccer.tasks.tracking.mdp.ball_physics import apply_ball_rolling_resistance, apply_ball_velocity_level_deceleration  # noqa: E402


def main():
    dt = 0.005
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args_cli.device, dt=dt))
    n = len(args_cli.speeds)
    cfg = G1FlatSoccerSceneCfg(num_envs=n, env_spacing=6.0)
    apply_ball_damping(cfg, args_cli.linear_damping, args_cli.angular_damping)
    cfg.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    scene = InteractiveScene(cfg)
    sim.reset()
    dev = sim.device
    ball, robot = scene["soccer_ball"], scene["robot"]
    origins = scene.env_origins
    # park the robot far from the ball
    rs = robot.data.default_root_state.clone()
    rs[:, :3] += origins
    rs[:, 0] += 500.0
    robot.write_root_state_to_sim(rs)
    bs = ball.data.default_root_state.clone()
    bs[:, :3] = origins + torch.tensor([0.0, 0.0, 0.11], device=dev)
    bs[:, 7:] = 0.0
    bs[:, 7] = torch.tensor(args_cli.speeds, device=dev)
    if args_cli.rolling_start:
        bs[:, 11] = bs[:, 7] / 0.11  # omega_y = v / R for rolling along +x
    ball.write_root_state_to_sim(bs)
    steps = int(args_cli.seconds / dt)
    t_axis, speeds, angs, xs = [], [], [], []
    for k in range(steps):
        if args_cli.const_mu is not None and k % args_cli.update_every == 0:
            apply_ball_rolling_resistance(ball, args_cli.const_mu, hold_dt=args_cli.update_every * dt, force_scale=args_cli.force_scale)
        if args_cli.velocity_decel is not None:
            apply_ball_velocity_level_deceleration(ball, args_cli.velocity_decel, dt=dt)
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(dt)
        if k % 4 == 3:  # every 0.02 s
            v = ball.data.root_lin_vel_w[:, :2]
            speeds.append(torch.linalg.vector_norm(v, dim=-1).cpu().numpy())
            angs.append(torch.linalg.vector_norm(ball.data.root_ang_vel_w, dim=-1).cpu().numpy())
            xs.append((ball.data.root_pos_w[:, 0] - origins[:, 0]).cpu().numpy())
            t_axis.append((k + 1) * dt)
    np.savez(args_cli.output, t=np.array(t_axis), speed=np.array(speeds), ang=np.array(angs), x=np.array(xs), v0=np.array(args_cli.speeds))
    print("[ROLL] saved", args_cli.output)
    os._exit(0)


if __name__ == "__main__":
    main()
