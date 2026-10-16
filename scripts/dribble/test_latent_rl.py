"""D-a step 3 acceptance test for the latent PPO wrapper (no Isaac; synthetic rollouts).

  python scripts/dribble/test_latent_rl.py
Checks: one full recurrent PPO iteration (rollout -> storage -> recurrent mini-batches -> update) runs with finite
losses and a KL term; gradients reach the encoder, prior, tracker and critic; prior mode runs; a state_dict round trip
reproduces the outputs exactly.
"""

import importlib.util
import os
import sys
import types

import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "source", "whole_body_tracking", "soccer")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Stub the `soccer` package so its Isaac-importing __init__ is skipped; load only torch-level modules.
for pkg, sub in (("soccer", ""), ("soccer.latent", "latent"), ("soccer.utils", "utils")):
    m = types.ModuleType(pkg)
    m.__path__ = [os.path.join(ROOT, sub)]
    sys.modules[pkg] = m
se = _load("soccer.latent.skill_event", os.path.join(ROOT, "latent", "skill_event.py"))
_load("soccer.latent.latent_skill_net", os.path.join(ROOT, "latent", "latent_skill_net.py"))
lr = _load("soccer.utils.latent_rl", os.path.join(ROOT, "utils", "latent_rl.py"))

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
N, T, A, C = 16, 24, 29, 50
OBS = se.OBS_DIM + se.REF_DIM
fails = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        fails.append(name)


policy = lr.LatentActorCritic(OBS, C, A).to(dev)
alg = lr.LatentPPO(policy, kl_coef=0.01, num_learning_epochs=2, num_mini_batches=2, learning_rate=1e-3,
                   entropy_coef=0.005, schedule="adaptive", desired_kl=0.01, device=dev)
alg.init_storage("rl", N, T, [OBS], [C], [A])
check("runner can resolve class names", hasattr(sys.modules.get("rsl_rl.runners.on_policy_runner"), "LatentActorCritic"))

before = {k: v.detach().clone() for k, v in policy.named_parameters()}
obs, cobs = torch.randn(N, OBS, device=dev), torch.randn(N, C, device=dev)
policy.train()
with torch.inference_mode():
    for t in range(T):
        actions = alg.act(obs, cobs)
        if t == 0:
            check("rollout action shape", tuple(actions.shape) == (N, A), str(tuple(actions.shape)))
        dones = (torch.rand(N, device=dev) < 0.1).float()
        alg.process_env_step(torch.randn(N, device=dev), dones, {})
        obs, cobs = torch.randn(N, OBS, device=dev), torch.randn(N, C, device=dev)
    alg.compute_returns(cobs)
hid_a, hid_c = policy.get_hidden_states()
check("actor hidden packed as (h, c) of prior(1) + tracker(2) layers", [tuple(h.shape) for h in hid_a] == [(3, N, 256), (3, N, 256)],
      str([tuple(h.shape) for h in hid_a]))
check("storage keeps actor and critic hidden pairs", len(alg.storage.saved_hidden_states_a) == 2 and len(alg.storage.saved_hidden_states_c) == 2)
losses = alg.update()
check("update returns finite losses incl. latent KL", "latent_kl" in losses and all(torch.isfinite(torch.tensor(v)) for v in losses.values()),
      str({k: round(v, 4) for k, v in losses.items()}))
changed = {k: not torch.equal(before[k], v.detach()) for k, v in policy.named_parameters()}
for part in ("net.encoder", "net.prior_rnn", "net.prior_head", "net.dec_rnn", "net.dec_head", "memory_c", "critic", "std"):
    check(f"gradient reached {part}", any(c for k, c in changed.items() if k.startswith(part)))

# prior mode (runtime: no reference window)
policy.eval()
policy.reset()
policy.use_prior = True
with torch.no_grad():
    a_prior = policy.act_inference(obs)
check("prior mode forward", tuple(a_prior.shape) == (N, A) and bool(torch.isfinite(a_prior).all()))
policy.use_prior = False

# state_dict round trip reproduces outputs exactly
clone = lr.LatentActorCritic(OBS, C, A).to(dev)
clone.load_state_dict(policy.state_dict())
policy.eval(); clone.eval(); policy.reset(); clone.reset()
x = torch.randn(N, OBS, device=dev)
with torch.no_grad():
    y1, y2 = policy.act_inference(x), clone.act_inference(x)
check("checkpoint round trip identical", torch.equal(y1, y2), f"max diff {float((y1 - y2).abs().max()):.2e}")
check("normalizer buffers saved", any("stud_norm" in k for k in policy.state_dict()) and any("ref_norm" in k for k in policy.state_dict()))

# KL warm-up schedule and free bits
sched = lr.LatentPPO(lr.LatentActorCritic(OBS, C, A).to(dev), kl_coef=0.02, kl_free_bits=0.5, kl_warmup_iters=2, kl_ramp_iters=4, device=dev)
coefs = []
for k in range(8):
    sched.num_updates = k
    coefs.append(round(sched.current_kl_coef(), 4))
check("KL warm-up then linear ramp", coefs == [0.0, 0.0, 0.0, 0.005, 0.01, 0.015, 0.02, 0.02], str(coefs))
small = torch.full((4, 3, 8), 0.1, device=dev, requires_grad=True)
sched.policy.kl_batch_dims = small
loss_small = sched.kl_loss(); loss_small.backward()
check("free bits: dims below the floor cost a constant and give no gradient", abs(float(loss_small) - 8 * 0.5) < 1e-6 and float(small.grad.abs().sum()) == 0.0)
big = torch.full((4, 3, 8), 1.0, device=dev, requires_grad=True)
sched.policy.kl_batch_dims = big
loss_big = sched.kl_loss(); loss_big.backward()
check("free bits: dims above the floor are penalised", abs(float(loss_big) - 8.0) < 1e-5 and float(big.grad.abs().sum()) > 0.0)

print("\nALL PASSED" if not fails else f"\nFAILED: {fails}")
raise SystemExit(1 if fails else 0)
