"""Deployment-oriented G1 dribbling curriculum.

The clean route has two external training stages: a motion prior and one
dribbling task.  C2/T2/C3/C4/T4/Repeat are monotonic curriculum levels inside
that single Stage-2 task.  Per-level registrations are retained only as
diagnostic fixtures; they are not formal checkpoint boundaries.
"""

import gymnasium as gym

from . import agents, soccer_dribbling_env_cfg, soccer_flat_env_cfg


def _register(env_id: str, env_cfg_entry_point, runner_cfg: str) -> None:
    gym.register(
        id=env_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg_entry_point,
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:{runner_cfg}",
        },
    )


# Historical flat Stage1 baseline, retained for checkpoint replay/diagnostics.
_register(
    "Tracking-Flat-G1-Motion-RNN-v0",
    soccer_flat_env_cfg.G1FlatMotionPretrainEnvCfg,
    "G1FlatRecurrentPPORunnerCfg",
)
_register(
    "Tracking-Flat-G1-Motion-RNN-reference-twist-v1",
    soccer_dribbling_env_cfg.G1FlatMotionPretrainReferenceTwistEnvCfg,
    "G1FlatRecurrentPPORunnerCfg",
)
_register(
    "Tracking-Flat-G1-CleanV0-Stage1-RNN",
    soccer_dribbling_env_cfg.G1CleanV0Stage1EnvCfg,
    "G1CleanV0Stage1RecurrentPPORunnerCfg",
)

# Historical unified stages share the deployment-oriented 163-D observation and
# 29-D action contract.  The clean-v0 registrations below intentionally use a
# fresh 160-D Actor contract (Flat154 + current ball xyz + reference twist3)
# with the same 29-D action interface; they must therefore start from a clean
# Stage-1 checkpoint rather than warm-starting a destination-bearing model.
_register(
    "Tracking-CG-G1-Motion-RNN-unified-s1-local-strict",
    soccer_dribbling_env_cfg.G1FlatMotionCGPretrainUnifiedS1LocalStrictEnvCfg,
    "G1DribblingRecurrentPPORunnerCfg",
)
_register(
    "Tracking-CG-G1-Dribbling-RNN-unified-s2-local-reference",
    soccer_dribbling_env_cfg.G1FlatCGDribblingUnifiedS2LocalReferenceEnvCfg,
    "G1DribblingS2RecurrentPPORunnerCfg",
)
_register(
    "Tracking-CG-G1-Dribbling-RNN-unified-s2-t2-local-reference",
    soccer_dribbling_env_cfg.G1FlatCGDribblingUnifiedS2T2LocalReferenceEnvCfg,
    "G1DribblingS2RecurrentPPORunnerCfg",
)
_register(
    "Tracking-Flat-G1-Dribbling-RNN-reference-twist-s2",
    soccer_dribbling_env_cfg.G1FlatCGDribblingReferenceTwistS2EnvCfg,
    "G1DribblingS2RecurrentPPORunnerCfg",
)
_register(
    "Tracking-Flat-G1-CleanV0-Stage2-C2-RNN",
    soccer_dribbling_env_cfg.G1CleanV0Stage2C2EnvCfg,
    "G1DribblingS2RecurrentPPORunnerCfg",
)
_register(
    "Tracking-Flat-G1-CleanV0-Stage2-T2-RNN",
    soccer_dribbling_env_cfg.G1CleanV0Stage2T2EnvCfg,
    "G1DribblingS2RecurrentPPORunnerCfg",
)
_register(
    "Tracking-Flat-G1-CleanV0-Stage2-C3-RNN",
    soccer_dribbling_env_cfg.G1CleanV0Stage2C3EnvCfg,
    "G1DribblingS2RecurrentPPORunnerCfg",
)
_register(
    "Tracking-Flat-G1-CleanV0-Stage2-C4-RNN",
    soccer_dribbling_env_cfg.G1CleanV0Stage2C4EnvCfg,
    "G1DribblingS2RecurrentPPORunnerCfg",
)
_register(
    "Tracking-Flat-G1-CleanV0-Stage2-T4-RNN",
    soccer_dribbling_env_cfg.G1CleanV0Stage2T4EnvCfg,
    "G1DribblingS2RecurrentPPORunnerCfg",
)
_register(
    "Tracking-Flat-G1-CleanV0-Stage2-RNN",
    soccer_dribbling_env_cfg.G1CleanV0Stage2EnvCfg,
    "G1DribblingS2RecurrentPPORunnerCfg",
)
_register(
    "Tracking-Flat-G1-Dribbling-RNN-reference-twist-s2-l1-window-outcome-control",
    soccer_dribbling_env_cfg.G1FlatCGDribblingReferenceTwistS2Level1WindowOutcomeControlCfg,
    "G1DribblingS2RecurrentPPORunnerCfg",
)
_register(
    "Tracking-Flat-G1-Dribbling-RNN-reference-twist-s2-l1-window-outcome3",
    soccer_dribbling_env_cfg.G1FlatCGDribblingReferenceTwistS2Level1WindowOutcome3Cfg,
    "G1DribblingS2RecurrentPPORunnerCfg",
)
_register(
    "Tracking-Flat-G1-Dribbling-RNN-reference-twist-s2-l1-delayed-settlement",
    soccer_dribbling_env_cfg.G1FlatCGDribblingReferenceTwistS2Level1DelayedSettlementCfg,
    "G1DribblingS2RecurrentPPORunnerCfg",
)
_register(
    "Tracking-Flat-G1-Dribbling-RNN-reference-twist-s2-l1-fixed-target-audit",
    soccer_dribbling_env_cfg.G1FlatCGDribblingReferenceTwistS2Level1FixedTargetAuditCfg,
    "G1DribblingS2RecurrentPPORunnerCfg",
)
_register(
    "Tracking-Flat-G1-Dribbling-RNN-reference-twist-s2-l1-fixed-target-progress-probe",
    soccer_dribbling_env_cfg.G1FlatCGDribblingReferenceTwistS2Level1FixedTargetProgressProbeCfg,
    "G1DribblingS2RecurrentPPORunnerCfg",
)
_register(
    "Tracking-CG-G1-Dribbling-RNN-unified-s3-local-task",
    soccer_dribbling_env_cfg.G1FlatCGDribblingUnifiedS3LocalTaskEnvCfg,
    "G1DribblingRecurrentPPORunnerCfg",
)
_register(
    "Tracking-Flat-G1-CleanV1-BallTrack-RNN",
    soccer_dribbling_env_cfg.G1CleanV1BallTrackEnvCfg,
    "G1CleanV1RecurrentPPORunnerCfg",
)
