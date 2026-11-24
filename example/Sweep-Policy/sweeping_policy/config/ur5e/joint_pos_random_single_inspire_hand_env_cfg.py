"""Inspire Open-v1: v0 sweep settings with directly controlled 2D hand action."""

from isaaclab.utils import configclass

from ...src.ur5e_inspire_hand import InspireHandOpennessActionCfg, make_movable_inspire_cfg
from .joint_pos_random_single_inspire_open_env_cfg import UR5eSweepRandomSingleInspireOpenEnvCfg


@configclass
class UR5eSweepRandomSingleInspireHandEnvCfg(UR5eSweepRandomSingleInspireOpenEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = make_movable_inspire_cfg().replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.gripper_action = InspireHandOpennessActionCfg()


@configclass
class UR5eSweepRandomSingleInspireHandEnvCfg_PLAY(UR5eSweepRandomSingleInspireHandEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = False
