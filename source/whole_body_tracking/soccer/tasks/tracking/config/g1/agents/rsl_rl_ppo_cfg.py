from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from isaaclab_rl.rsl_rl import (
    RslRlDistillationStudentTeacherCfg,
    RslRlDistillationStudentTeacherRecurrentCfg,
    RslRlPpoActorCriticRecurrentCfg,
    RslRlDistillationAlgorithmCfg
)

@configclass
class G1FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 100000
    save_interval = 1000
    experiment_name = "g1_flat"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class G1FlatRecurrentPPORunnerCfg(G1FlatPPORunnerCfg):
    """RNN-enabled PPO configuration mirroring the feed-forward defaults."""

    def __post_init__(self):
        super().__post_init__()
        self.policy = RslRlPpoActorCriticRecurrentCfg(
            init_noise_std=1.0,
            actor_hidden_dims=[128, 64, 32],
            critic_hidden_dims=[128, 64, 32],
            activation="elu",

            rnn_type="lstm",
            rnn_hidden_dim=128,
            rnn_num_layers=2,
        )


@configclass
class G1DribblingPPORunnerCfg(G1FlatPPORunnerCfg):
    """Same as G1 flat PPO but logs under ``g1_dribbling`` (isolated from kick / generic flat runs)."""

    experiment_name = "g1_dribbling"


@configclass
class G1DribblingRecurrentPPORunnerCfg(G1FlatRecurrentPPORunnerCfg):
    """RNN PPO with dribbling log root; matches ``shell/progressive_dribbling_train.sh`` defaults."""

    experiment_name = "g1_dribbling"


@configclass
class G1CleanV0Stage1RecurrentPPORunnerCfg(G1FlatRecurrentPPORunnerCfg):
    """Fresh clean-v0 Stage-1 prior with the same 500-iteration checkpoint cadence."""

    experiment_name = "g1_dribbling"
    save_interval = 500


@configclass
class G1DribblingS2RecurrentPPORunnerCfg(G1DribblingRecurrentPPORunnerCfg):
    """S2 PPO settings for precise, bounded contact exploration."""

    save_interval = 500
    policy_std_min: float = 0.10
    policy_std_max: float = 1.50
    s2_warm_start_reset_optimizer: bool = True
    s2_warm_start_policy_std: float = 0.40

    def __post_init__(self):
        super().__post_init__()
        self.policy.init_noise_std = 0.40
        self.algorithm.entropy_coef = 0.001
        # A cross-stage warm start creates a fresh Adam optimizer. Starting it
        # at the generic 1e-3 rate destroyed the loaded S1 actor in the first
        # PPO update (surrogate 0.316, then ~100% falls), before any strict O1
        # contact existed. 1e-5 is the rate the adaptive schedule selected
        # immediately after that destructive update; start there instead.
        self.algorithm.learning_rate = 1.0e-5


LOW_FREQ_SCALE = 0.5


@configclass
class G1FlatStudentTeacherPPORunnerCfg(G1FlatPPORunnerCfg):
    """RNN-enabled Student-Teacher configuration mirroring the feed-forward defaults."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "g1_flat"
        self.policy = RslRlDistillationStudentTeacherCfg(
            init_noise_std=1.0,
            student_hidden_dims=[512, 256, 128],
            teacher_hidden_dims=[512, 256, 128],
            activation="elu",
        )
        self.algorithm = RslRlDistillationAlgorithmCfg(
            num_learning_epochs=5,
            learning_rate=1.0e-3,
            gradient_length=24,
            max_grad_norm=1.0,
        )



@configclass
class G1FlatLowFreqPPORunnerCfg(G1FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.num_steps_per_env = round(self.num_steps_per_env * LOW_FREQ_SCALE)
        self.algorithm.gamma = self.algorithm.gamma ** (1 / LOW_FREQ_SCALE)
        self.algorithm.lam = self.algorithm.lam ** (1 / LOW_FREQ_SCALE)


@configclass
class G1CleanV1RecurrentPPORunnerCfg(G1DribblingS2RecurrentPPORunnerCfg):
    """Clean-v1: keep the exploration std near the Stage-1 converged value (~0.2).

    The S2 default of 0.40 doubles the converged Stage-1 noise on a warm start.
    """

    s2_warm_start_policy_std: float = 0.20

    def __post_init__(self):
        super().__post_init__()
        self.policy.init_noise_std = 0.20
