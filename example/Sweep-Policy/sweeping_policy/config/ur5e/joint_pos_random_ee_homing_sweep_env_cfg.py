"""Random-v0 sweep task with Cartesian end-effector homing."""

from isaaclab.utils import configclass

from ... import mdp
from .joint_pos_random_sweep_env_cfg import UR5eSweepEnvCfg


@configclass
class UR5eSweepEEHomingEnvCfg(UR5eSweepEnvCfg):
    """Random-v0 with only the homing reward changed to EE position error."""

    def __post_init__(self):
        super().__post_init__()

        self.rewards.homing_after_sweep.func = (
            mdp.reward_random_sweep.EndEffectorHomingReward
        )


@configclass
class UR5eSweepEEHomingEnvCfg_PLAY(UR5eSweepEEHomingEnvCfg):
    """Small visualization configuration for the EE-homing task."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
