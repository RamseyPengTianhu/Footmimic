"""Runner for the latent dribble tasks (interface v1).

RSL-RL's ``OnPolicyRunner`` only treats an algorithm called exactly ``"PPO"`` as RL training, and builds it with
``eval(class_name)`` in its own module.  This runner keeps ``class_name = "PPO"`` for that check while pointing the
name ``PPO`` at ``LatentPPO`` only for the duration of construction, then restores it, so other tasks in the same
process are unaffected.  Select it when ``algorithm.class_name == "LatentPPO"`` (see scripts/rsl_rl/train_multi.py).
"""

import rsl_rl.runners.on_policy_runner as runner_module

from soccer.utils.latent_rl import LatentPPO
from soccer.utils.my_on_policy_runner import MotionOnPolicyRunner


class LatentOnPolicyRunner(MotionOnPolicyRunner):
    def __init__(self, env, train_cfg: dict, log_dir=None, device="cpu", registry_name=None):
        cfg = dict(train_cfg)
        cfg["algorithm"] = dict(cfg["algorithm"])
        if cfg["algorithm"].get("class_name") != "LatentPPO":
            raise ValueError("LatentOnPolicyRunner expects algorithm.class_name == 'LatentPPO'")
        cfg["algorithm"]["class_name"] = "PPO"
        original = runner_module.PPO
        runner_module.PPO = LatentPPO
        try:
            super().__init__(env, cfg, log_dir=log_dir, device=device, registry_name=registry_name)
        finally:
            runner_module.PPO = original
        if not isinstance(self.alg, LatentPPO):
            raise RuntimeError("expected a LatentPPO algorithm")
