"""Random-v0 sweep task with an episode-initial target observation."""

from isaaclab.utils import configclass
from isaaclab.utils.noise import (
    ConstantNoiseCfg,
    NoiseModelWithAdditiveBiasCfg,
    UniformNoiseCfg,
)

from ... import mdp
from .joint_pos_random_sweep_env_cfg import UR5eSweepEnvCfg


@configclass
class UR5eSweepRandomNonTrackingEnvCfg(UR5eSweepEnvCfg):
    """Random-v0 whose policy receives only the initial target position."""

    def __post_init__(self):
        super().__post_init__()

        # The command term snapshots the target during reset and never tracks
        # the live object afterward.  Keep Random-v0's observation-error range,
        # but sample it once per reset as a fixed bias so the policy value is
        # genuinely constant throughout the episode.
        self.observations.policy.target_obs_state.func = (
            mdp.InitialTargetPositionInRRF
        )
        self.observations.policy.target_obs_state.params = {
            "command_name": "target_goal_pos"
        }
        self.observations.policy.target_obs_state.noise = (
            NoiseModelWithAdditiveBiasCfg(
                noise_cfg=ConstantNoiseCfg(bias=0.0),
                bias_noise_cfg=UniformNoiseCfg(
                    n_min=-0.01,
                    n_max=0.01,
                    operation="abs",
                ),
                sample_bias_per_component=True,
            )
        )


@configclass
class UR5eSweepRandomNonTrackingEnvCfg_PLAY(UR5eSweepRandomNonTrackingEnvCfg):
    """Small visualization configuration for Random--Non-Tracking-v0."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
