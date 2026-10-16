"""Tests for the interface-v1 shared modules (torch only).

Run: python -m pytest scripts/rsl_rl/test_skill_interface.py
The modules are loaded by file path so the Isaac-dependent `soccer` package is not imported.
"""

import importlib.util
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parents[2] / "source/whole_body_tracking/soccer/latent"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


se = _load("skill_event")
net_mod = _load("latent_skill_net")


def _obs(n):
    return torch.cat(
        [
            torch.randn(n, se.PROPRIO_DIM),
            se.ball_obs(torch.randn(n, 3), torch.randn(n, 3), torch.randn(n, 3), torch.ones(n)),
            torch.randn(n, se.BODY_CMD_DIM),
            se.encode_event(torch.ones(n), torch.full((n,), se.REGION_INDEX["front_inner"]), torch.full((n,), -0.5),
                            torch.randn(n, 2), torch.full((n,), se.MODE_MAX), torch.zeros(n)),
            se.encode_phase(torch.zeros(n, dtype=torch.long), torch.rand(n)),
            se.encode_next(torch.zeros(n), torch.ones(n), torch.full((n,), se.UNSPECIFIED), torch.zeros(n, 2)),
        ],
        dim=-1,
    )


def test_layout_dims():
    assert se.EVENT_DIM == 13 and se.PHASE_DIM == 4 and se.NEXT_DIM == 11 and se.BALL_DIM == 10
    assert _obs(4).shape == (4, se.OBS_DIM) == (4, 134)


def test_ball_gating_zeroes_after_release():
    b = se.ball_obs(torch.ones(2, 3), torch.ones(2, 3), torch.ones(2, 3), torch.tensor([1.0, 0.0]))
    assert torch.all(b[0] == 1.0) and torch.all(b[1] == 0.0)


def test_unspecified_region_is_all_zero():
    e = se.encode_event(torch.ones(1), torch.tensor([se.UNSPECIFIED]), torch.zeros(1), torch.zeros(1, 2),
                        torch.zeros(1), torch.zeros(1))
    assert torch.all(e[0, 1:1 + se.NUM_REGIONS] == 0)


def test_classify_matches_plate_faces_and_mirrors_left_foot():
    right, left = torch.ones(1), torch.zeros(1)
    fore_point = torch.tensor([[0.12, 0.0, -0.025]])
    push_left = torch.tensor([[0.0, 1.0, 0.0]])  # force on ball along foot +y
    assert se.classify_contact(fore_point, push_left, right).item() == se.REGION_INDEX["front_inner"]
    assert se.classify_contact(fore_point, push_left, left).item() == se.REGION_INDEX["front_outer"]
    mid = torch.tensor([[0.02, -0.03, -0.025]])
    assert se.classify_contact(mid, torch.tensor([[0.0, -1.0, 0.0]]), right).item() == se.REGION_INDEX["outer"]
    assert se.classify_contact(fore_point, torch.tensor([[0.3, 0.2, 0.9]]), right).item() == se.REGION_INDEX["instep"]
    assert se.classify_contact(mid, torch.tensor([[0.0, 0.0, -1.0]]), right).item() == se.REGION_INDEX["sole"]
    heel = torch.tensor([[-0.06, 0.0, -0.025]])
    assert se.classify_contact(heel, torch.tensor([[1.0, 0.0, 0.0]]), right).item() == se.UNSPECIFIED


def test_region_geometry_mirrors():
    r = torch.tensor([se.REGION_INDEX["inner"]] * 2)
    c = se.region_center(r, torch.tensor([1.0, 0.0]))
    n = se.region_normal(r, torch.tensor([1.0, 0.0]))
    assert c[0, 1] > 0 and c[1, 1] < 0 and n[0, 1] == 1.0 and n[1, 1] == -1.0


def test_dribble_surface_labels():
    out = se.dribble_surface_to_region(torch.tensor([0, 1, -1]))
    assert out.tolist() == [se.REGION_INDEX["inner"], se.REGION_INDEX["outer"], se.UNSPECIFIED]


def test_soft_phase_is_event_driven():
    ph = se.SoftPhase(3, "cpu", reach=0.2, jitter=0.0)
    ph.start_event(torch.ones(3, dtype=torch.bool), torch.tensor([1.0, 1.0, 1.0]))
    ph.update(torch.tensor([0.5, 0.15, 0.15]), torch.tensor([False, False, True]))
    assert ph.stage.tolist() == [se.APPROACH, se.CONTACT, se.FOLLOW]
    assert abs(ph.progress[0].item() - 0.5) < 1e-6
    ph.update(torch.tensor([0.5, 0.1, 0.5]), torch.tensor([False, True, False]))
    assert ph.stage.tolist() == [se.APPROACH, se.FOLLOW, se.FOLLOW]
    ph.start_event(torch.tensor([False, True, False]), torch.tensor([0.0, 0.8, 0.0]))
    ph.update(torch.tensor([0.5, 0.9, 0.5]), torch.zeros(3, dtype=torch.bool))
    assert ph.stage.tolist() == [se.APPROACH, se.APPROACH, se.FOLLOW]


def test_soft_phase_miss_goes_to_follow():
    ph = se.SoftPhase(1, "cpu", reach=0.2, jitter=0.0)
    ph.start_event(torch.ones(1, dtype=torch.bool), torch.tensor([1.0]))
    ph.update(torch.tensor([0.1]), torch.tensor([False]))
    ph.update(torch.tensor([0.4]), torch.tensor([False]))
    assert ph.stage.item() == se.FOLLOW


def test_possession_releases_only_single_touch():
    stage = torch.tensor([se.FOLLOW, se.FOLLOW, se.APPROACH])
    has_next = torch.tensor([False, True, False])
    assert se.possession_after(stage, torch.zeros(3), has_next).tolist() == [0.0, 1.0, 1.0]


def test_targets():
    t = se.dribble_target(torch.tensor([[1.0, 0.0]]), 0.6, lead_m=0.2)
    assert torch.allclose(t, torch.tensor([[0.8, 0.0]]))
    assert torch.allclose(se.shoot_or_pass_target(torch.tensor([[5.0, 1.0]]), torch.tensor([[1.0, 0.0]])),
                          torch.tensor([[4.0, 1.0]]))


def test_net_step_modes_and_reset():
    torch.manual_seed(0)
    n = 6
    net = net_mod.LatentSkillNet(se.OBS_DIM, se.REF_DIM)
    h = net.init_hidden(n, "cpu")
    obs, ref = _obs(n), torch.randn(n, se.REF_DIM)
    net.update_norms(obs, ref)
    a, stats, h2 = net.step(obs, h, ref=ref, use_prior=torch.tensor([0, 1, 0, 1, 0, 1]))
    assert a.shape == (n, 29) and stats["z"].shape == (n, 8)
    kl = net_mod.kl_gauss(stats["mu_q"], stats["lv_q"], stats["mu_p"], stats["lv_p"])
    assert kl.shape == (n,) and torch.all(kl >= -1e-5)
    a_prior, _, _ = net.step(obs, h, ref=None, use_prior=True, deterministic=True)
    assert a_prior.shape == (n, 29)
    h3 = net.reset_hidden(h2, torch.tensor([1, 0, 0, 0, 0, 0]))
    assert torch.all(h3["tracker"][0][:, 0] == 0) and torch.any(h3["tracker"][0][:, 1] != 0)


def test_loads_a1_state_dict_names():
    net = net_mod.LatentSkillNet(obs_dim=104, ref_dim=29 * 5 + 61)
    names = {k.split(".")[0] for k in net.state_dict()}
    assert names == {"stud_norm", "ref_norm", "encoder", "prior_rnn", "prior_head", "dec_rnn", "dec_head"}
