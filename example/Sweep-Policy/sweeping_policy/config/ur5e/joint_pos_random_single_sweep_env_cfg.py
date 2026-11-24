"""Random-v0 sweep task with only the selected target on the shelf."""

from isaaclab.utils import configclass

from ... import mdp
from .joint_pos_random_sweep_env_cfg import UR5eSweepEnvCfg


@configclass
class UR5eSweepRandomSingleEnvCfg(UR5eSweepEnvCfg):
    """Random-v0 with the target-only scene randomization event."""

    def __post_init__(self):
        super().__post_init__()

        # Keep every Random-v0 setting and replace only scene randomization.
        self.events.object_spawn.func = mdp.randomize_single_object_scene


@configclass
class UR5eSweepRandomSingleEnvCfg_PLAY(UR5eSweepRandomSingleEnvCfg):
    """Small visualization configuration for the target-only task."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
