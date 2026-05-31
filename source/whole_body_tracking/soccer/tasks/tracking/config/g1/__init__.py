import gymnasium as gym

from . import agents, flat_env_cfg
from . import soccer_flat_env_cfg
from . import soccer_dribbling_env_cfg
from . import soccer_anchor_env_cfg
from . import soccer_v35_env_cfg
from . import soccer_v36_env_cfg
from . import soccer_v36_env_cfg
from . import soccer_v36b_env_cfg
from . import soccer_v3_cvae_env_cfg
from . import soccer_latent_prior_env_cfg

##
# Register Gym environments.
##

## Motion tracking environments
gym.register(
    id="Tracking-Flat-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1FlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-G1-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1FlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatRecurrentPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-G1-Wo-State-Estimation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1FlatWoStateEstimationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)


gym.register(
    id="Tracking-Flat-G1-Low-Freq-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1FlatLowFreqEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatLowFreqPPORunnerCfg",
    },
)


## Soccer environments
###  Stage 1
# Terrain
gym.register(
    id="Tracking-Terrain-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_flat_env_cfg.G1TerrainMotionEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Terrain-G1-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_flat_env_cfg.G1TerrainMotionEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatRecurrentPPORunnerCfg",
    },
)
# Flat
gym.register(
    id="Tracking-Flat-G1-Motion-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_flat_env_cfg.G1FlatMotionEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatRecurrentPPORunnerCfg",
    },
)


###  Stage 2
gym.register(
    id="Tracking-Flat-G1-SoccerDestination-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_flat_env_cfg.G1FlatKickEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-G1-SoccerDestination-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_flat_env_cfg.G1FlatKickEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatRecurrentPPORunnerCfg",
    },
)


gym.register(
    id="Tracking-Flat-G1-SoccerMoving-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_flat_env_cfg.G1FlatKickMovingEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatRecurrentPPORunnerCfg",
    },
)




## Advanced Soccer environments

# Only-vision
gym.register(
    id="Tracking-Flat-G1-SoccerBlind-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_flat_env_cfg.G1FlatSoccerBlindEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)


gym.register(
    id="Tracking-Flat-G1-SoccerBlind-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_flat_env_cfg.G1FlatSoccerBlindEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatRecurrentPPORunnerCfg",
    },
)


gym.register(
    id="Tracking-Flat-G1-SuperSoccer-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_flat_env_cfg.G1FlatSuperSoccerEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)


gym.register(
    id="Tracking-Flat-G1-Soccer-Distillation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_flat_env_cfg.G1FlatSoccerStudentEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatStudentTeacherPPORunnerCfg",
    },
)


## Dribbling environments
gym.register(
    id="Tracking-Flat-G1-Dribbling-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_dribbling_env_cfg.G1FlatDribblingEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-G1-Dribbling-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_dribbling_env_cfg.G1FlatDribblingEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatRecurrentPPORunnerCfg",
    },
)

# Dribbling Stage 1: ankle disturbance mode
gym.register(
    id="Tracking-Flat-G1-Dribbling-AnkleDisturb-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_dribbling_env_cfg.G1TerrainDribblingAnkleDisturbEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-G1-Dribbling-AnkleDisturb-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_dribbling_env_cfg.G1TerrainDribblingAnkleDisturbEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatRecurrentPPORunnerCfg",
    },
)


## Anchor-based kick environments (Sprint 2 — isolated from baseline)
# Stage 1: egocentric observation + velocity downweight
gym.register(
    id="Anchor-Kick-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_anchor_env_cfg.G1AnchorTrackingEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Anchor-Kick-G1-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_anchor_env_cfg.G1AnchorTrackingEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatRecurrentPPORunnerCfg",
    },
)

# Stage 2: egocentric obs + ankle masking + kick rewards
gym.register(
    id="Anchor-Kick-G1-Soccer-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_anchor_env_cfg.G1AnchorKickEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Anchor-Kick-G1-Soccer-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_anchor_env_cfg.G1AnchorKickEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatRecurrentPPORunnerCfg",
    },
)

# Stage 2 CG: Soft Contact Graph kick (Sprint 4)
gym.register(
    id="Anchor-CG-Kick-G1-Soccer-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_anchor_env_cfg.G1AnchorCGKickEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Anchor-CG-Kick-G1-Soccer-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_anchor_env_cfg.G1AnchorCGKickEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatRecurrentPPORunnerCfg",
    },
)

# Stage C: VQ Student Kick (Reference-Free, Interaction-First)
gym.register(
    id="VQ-Student-Kick-G1-Soccer-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_anchor_env_cfg.G1VQStudentKickEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="VQ-Student-Kick-G1-Soccer-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_anchor_env_cfg.G1VQStudentKickEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatRecurrentPPORunnerCfg",
    },
)

# Pure V3/CG reward family with physical terminations for action-CVAE latent
# residual policies. This is the clean counterpart to the V3 teacher, without
# the experimental content-CVAE motion prior.
gym.register(
    id="Anchor-CG-Latent-Kick-G1-Soccer-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_v3_cvae_env_cfg.G1AnchorCGLatentKickEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Anchor-CG-Latent-Kick-G1-Soccer-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_v3_cvae_env_cfg.G1AnchorCGLatentKickEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatRecurrentPPORunnerCfg",
    },
)

# Stage 2 CG + weak Content-CVAE prior. Keeps V3 behavior isolated from v36.
gym.register(
    id="Anchor-V3CVAE-Kick-G1-Soccer-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_v3_cvae_env_cfg.G1AnchorCGContentCVAEPriorKickEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Anchor-V3CVAE-Kick-G1-Soccer-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_v3_cvae_env_cfg.G1AnchorCGContentCVAEPriorKickEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatRecurrentPPORunnerCfg",
    },
)

# Same reward family as V3CVAE, but with physical terminations for latent
# residual policies.  Avoids killing trajectories solely because a distilled
# latent action deviates from exact reference end-effector height.
gym.register(
    id="Anchor-V3CVAE-Latent-Kick-G1-Soccer-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_v3_cvae_env_cfg.G1AnchorCGContentCVAELatentKickEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Anchor-V3CVAE-Latent-Kick-G1-Soccer-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_v3_cvae_env_cfg.G1AnchorCGContentCVAELatentKickEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatRecurrentPPORunnerCfg",
    },
)

# ===========================================================================
# Latent-prior stage1: CVAE condition + latent command, compact prior tracking
# ===========================================================================
gym.register(
    id="LatentPrior-Stage1-G1-Soccer-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_latent_prior_env_cfg.G1LatentPriorStage1EnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="LatentPrior-Stage1-G1-Soccer-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_latent_prior_env_cfg.G1LatentPriorStage1EnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatRecurrentPPORunnerCfg",
    },
)

# Stage 2 v9a: Adaptive kick (ball position randomization, same obs)
gym.register(
    id="Anchor-Adaptive-Kick-G1-Soccer-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_anchor_env_cfg.G1AnchorAdaptiveKickEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Anchor-Adaptive-Kick-G1-Soccer-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_anchor_env_cfg.G1AnchorAdaptiveKickEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatRecurrentPPORunnerCfg",
    },
)

# Stage 2 SM: state-machine kick (APPROACH/STRIKE distance trigger)
gym.register(
    id="Anchor-SM-Kick-G1-Soccer-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_anchor_env_cfg.G1AnchorStateMachineKickEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Anchor-SM-Kick-G1-Soccer-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_anchor_env_cfg.G1AnchorStateMachineKickEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatRecurrentPPORunnerCfg",
    },
)

## Ablation tests (隔离测试)
# Test 2: xyz obs + velocity/ankle changes
gym.register(
    id="Ablation-Xyz-Kick-G1-Soccer-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_anchor_env_cfg.G1AblationXyzKickEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatRecurrentPPORunnerCfg",
    },
)

# Test 3: polar obs only, keep velocity/ankle as baseline
gym.register(
    id="Ablation-Polar-Kick-G1-Soccer-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_anchor_env_cfg.G1AblationPolarOnlyKickEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatRecurrentPPORunnerCfg",
    },
)

# ===========================================================================
# v10: Event-Conditioned Kick (MLP + History)
# ===========================================================================
gym.register(
    id="Event-Conditioned-Kick-G1-Soccer-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_anchor_env_cfg.G1EventConditionedKickEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

# ===========================================================================
# v3.5: Strike Discriminator Gated Kick (phase-free timing + strike constraint)
# ===========================================================================
gym.register(
    id="Strike-Gated-Kick-G1-Soccer-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_v35_env_cfg.G1StrikeGatedKickEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Strike-Gated-Kick-G1-Soccer-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_v35_env_cfg.G1StrikeGatedKickEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatRecurrentPPORunnerCfg",
    },
)

# ===========================================================================
# v3.6: Semantic Motion Prior (phase-dependent tracking + phase-free contact)
# ===========================================================================
gym.register(
    id="Strike-Quality-Kick-G1-Soccer-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_v36_env_cfg.G1SoccerV36EnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Strike-Quality-Kick-G1-Soccer-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_v36_env_cfg.G1SoccerV36EnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatRecurrentPPORunnerCfg",
    },
)

# ===========================================================================
# v3.6: Semantic Motion Prior Kick (phase-modulated tracking + contact quality)
# ===========================================================================
gym.register(
    id="Anchor-V36-Kick-G1-Soccer-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_v36_env_cfg.G1SemanticPriorKickEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Anchor-V36-Kick-G1-Soccer-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_v36_env_cfg.G1SemanticPriorKickEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatRecurrentPPORunnerCfg",
    },
)

# ===========================================================================
# v3.6b: Ball-Ready Semantic Prior Kick
# ===========================================================================
gym.register(
    id="Anchor-V36B-Kick-G1-Soccer-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_v36b_env_cfg.G1BallReadySemanticPriorKickEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Anchor-V36B-Kick-G1-Soccer-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_v36b_env_cfg.G1BallReadySemanticPriorKickEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatRecurrentPPORunnerCfg",
    },
)

# ===========================================================================
# v3.6b2: Ball-Ready Power-Balanced Kick
# ===========================================================================
gym.register(
    id="Anchor-V36B2-Kick-G1-Soccer-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_v36b_env_cfg.G1BallReadyPowerBalanceKickEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Anchor-V36B2-Kick-G1-Soccer-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_v36b_env_cfg.G1BallReadyPowerBalanceKickEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatRecurrentPPORunnerCfg",
    },
)

# ===========================================================================
# v3.6b3: Ball-Ready Power Balance + Empty-Swing Guard
# ===========================================================================
gym.register(
    id="Anchor-V36B3-Kick-G1-Soccer-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_v36b_env_cfg.G1BallReadyEmptySwingGuardKickEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Anchor-V36B3-Kick-G1-Soccer-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_v36b_env_cfg.G1BallReadyEmptySwingGuardKickEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatRecurrentPPORunnerCfg",
    },
)

# ===========================================================================
# v3.6b4: Ball-Ready Attempt Window
# ===========================================================================
gym.register(
    id="Anchor-V36B4-Kick-G1-Soccer-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_v36b_env_cfg.G1BallReadyAttemptWindowKickEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Anchor-V36B4-Kick-G1-Soccer-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_v36b_env_cfg.G1BallReadyAttemptWindowKickEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatRecurrentPPORunnerCfg",
    },
)

# ===========================================================================
# v3.6c: Content-conditioned CVAE Motion Prior
# ===========================================================================
gym.register(
    id="Anchor-V36C-Kick-G1-Soccer-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_v36b_env_cfg.G1ContentCVAEPriorKickEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Anchor-V36C-Kick-G1-Soccer-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_v36b_env_cfg.G1ContentCVAEPriorKickEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatRecurrentPPORunnerCfg",
    },
)

gym.register(
    id="Anchor-CVAE-Kick-G1-Soccer-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_v36b_env_cfg.G1ContentCVAEPriorKickEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Anchor-CVAE-Kick-G1-Soccer-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_v36b_env_cfg.G1ContentCVAEPriorKickEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatRecurrentPPORunnerCfg",
    },
)
