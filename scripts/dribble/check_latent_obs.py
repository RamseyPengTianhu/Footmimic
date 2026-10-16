"""D-a step-2 check: build the interface-v1 environment, verify observation dimensions and layout, trace the event /
soft-phase signals on a dribble reference, and run the shared LatentSkillNet forward in posterior and prior mode on
the real observations.  No learning.

  python scripts/dribble/check_latent_obs.py --task Tracking-Flat-G1-LatentDribble-Da \
      --motion_path motions/latent_da_mix --num_envs 96 --steps 160 --headless
"""

import argparse
import glob
import os
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rsl_rl"))
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", required=True)
parser.add_argument("--motion_path", required=True)
parser.add_argument("--num_envs", type=int, default=96)
parser.add_argument("--steps", type=int, default=160)
parser.add_argument("--trace_motion", type=str, default="", help="substring of the motion file to trace")
parser.add_argument("--no_terminations", action="store_true", help="keep episodes running to trace the reference clock (phase only uses reference geometry)")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import soccer.tasks  # noqa: E402,F401
from soccer.latent import skill_event as se  # noqa: E402
from soccer.latent.latent_skill_net import LatentSkillNet, kl_gauss  # noqa: E402

LAYOUT = [("proprio", se.PROPRIO_DIM), ("ball", se.BALL_DIM), ("body_cmd", se.BODY_CMD_DIM),
          ("event", se.EVENT_DIM), ("phase", se.PHASE_DIM), ("next", se.NEXT_DIM)]


def slices():
    out, a = {}, 0
    for name, d in LAYOUT:
        out[name] = slice(a, a + d)
        a += d
    return out


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.commands.motion.motion_files = sorted(glob.glob(os.path.join(args_cli.motion_path, "*.npz")))
    if args_cli.no_terminations:
        for name in list(vars(env_cfg.terminations)):
            if name != "time_out" and not name.startswith("_"):
                setattr(env_cfg.terminations, name, None)
        env_cfg.commands.motion.motion_clip_end_resample = False
    env = gym.make(args_cli.task, cfg=env_cfg)
    base = env.unwrapped
    obs, _ = env.reset()
    ok = True
    print("[CHECK] groups:", {k: tuple(v.shape) for k, v in obs.items()})
    full = obs["policy"]
    pol, ref = full[:, : se.OBS_DIM], full[:, se.OBS_DIM :]
    for name, got, want in (("policy (student + ref)", full.shape[-1], se.OBS_DIM + se.REF_DIM), ("student", pol.shape[-1], se.OBS_DIM), ("ref", ref.shape[-1], se.REF_DIM)):
        print(f"[CHECK] {name} dim {got} (expected {want}) -> {'OK' if got == want else 'MISMATCH'}")
        ok &= got == want
    S = slices()
    cmd = base.command_manager.get_term("motion")
    cache = cmd._latent_v1_cache
    has_ev = cache.ev_count[cmd.motion_idx.long()] > 0
    print(f"[CHECK] clips with events: {int((cache.ev_count > 0).sum())}/{cache.ev_count.numel()}, total events {cache.num_events_total}; "
          f"envs on a dribble clip at reset: {int(has_ev.sum())}/{base.num_envs}")
    print(f"[CHECK] possession column all 1: {bool(torch.all(pol[:, S['ball']][:, -1] == 1.0))}")
    pick = has_ev.clone()
    if args_cli.trace_motion:
        names = [os.path.basename(f) for f in cmd.cfg.motion_files]
        want = torch.tensor([args_cli.trace_motion in names[int(m)] for m in cmd.motion_idx], device=has_ev.device)
        pick &= want
    trace_env = int(torch.nonzero(pick).flatten()[0]) if pick.any() else 0
    net = LatentSkillNet(obs_dim=se.OBS_DIM, ref_dim=se.REF_DIM).to(base.device)
    hidden = net.init_hidden(base.num_envs, base.device)
    rows = []
    finite = True
    ball_pos_hist, ball_vel_hist = [], []
    for t in range(args_cli.steps):
        with torch.no_grad():
            act_post, st, hidden_post = net.step(pol, hidden, ref=ref, use_prior=False)
            act_prior, _, _ = net.step(pol, hidden, ref=None, use_prior=True)
            kl = kl_gauss(st["mu_q"], st["lv_q"], st["mu_p"], st["lv_p"])
        hidden = hidden_post
        if t == 0:
            print(f"[CHECK] net: posterior action {tuple(act_post.shape)}, prior action {tuple(act_prior.shape)}, KL mean {float(kl.mean()):.3f}, finite {bool(torch.isfinite(act_post).all() and torch.isfinite(act_prior).all())}")
        e = pol[trace_env]
        ph = e[S["phase"]]; ev = e[S["event"]]; nx = e[S["next"]]
        f = int(cmd.time_steps[trace_env]); mi = int(cmd.motion_idx[trace_env])
        stage = int(torch.argmax(ph[:3])) if ph[:3].sum() > 0 else -1
        region = int(torch.argmax(ev[1:1 + se.NUM_REGIONS])) if ev[1:1 + se.NUM_REGIONS].sum() > 0 else -1
        rows.append((t, mi, f, stage, float(ph[3]), int(ev[0]), region, float(ev[9] / se.TARGET_SCALE), float(ev[10] / se.TARGET_SCALE), int(nx[0])))
        finite &= bool(torch.isfinite(pol).all() and torch.isfinite(ref).all())
        bslice = pol[:, S["ball"]]
        ball_pos_hist.append(bslice[:, 0:3].detach().cpu()); ball_vel_hist.append(bslice[:, 3:6].detach().cpu())
        obs, _, term, trunc, _ = env.step(torch.zeros(base.num_envs, base.action_manager.total_action_dim, device=base.device))
        pol, ref = obs["policy"][:, : se.OBS_DIM], obs["policy"][:, se.OBS_DIM :]
        hidden = net.reset_hidden(hidden, term | trunc)
    print(f"[CHECK] all observations finite over {args_cli.steps} steps: {finite}")
    bp = torch.cat(ball_pos_hist); bv = torch.cat(ball_vel_hist)
    q = torch.tensor([0.01, 0.25, 0.5, 0.75, 0.99])
    print(f"[CHECK] ball pos (pelvis frame) x quantiles {[round(v, 2) for v in torch.quantile(bp[:, 0], q).tolist()]}, "
          f"y {[round(v, 2) for v in torch.quantile(bp[:, 1], q).tolist()]}; behind the pelvis (x<0): {float((bp[:, 0] < 0).float().mean()):.2f}")
    sp = torch.linalg.vector_norm(bv[:, :2], dim=-1)
    print(f"[CHECK] ball speed (pelvis frame) quantiles {[round(v, 2) for v in torch.quantile(sp, q).tolist()]}")
    names = ["approach", "contact", "follow"]
    print(f"[CHECK] trace env {trace_env} (motion {os.path.basename(cmd.cfg.motion_files[rows[0][1]])}): t frame stage progress right region target_xy has_next")
    last = None
    for r in rows:
        key = (r[1], r[3], r[6], r[9])
        if key != last or r[0] % 20 == 0:
            st = names[r[3]] if r[3] >= 0 else "-"
            rg = se.REGIONS[r[6]] if r[6] >= 0 else "-"
            print(f"   t={r[0]:3d} m={r[1]:2d} f={r[2]:3d} {st:8s} p={r[4]:.2f} right={r[5]} {rg:6s} target=({r[7]:+.2f},{r[8]:+.2f}) next={r[9]}")
        last = key
    print(f"[CHECK] RESULT: {'PASS' if ok and finite else 'FAIL'}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
