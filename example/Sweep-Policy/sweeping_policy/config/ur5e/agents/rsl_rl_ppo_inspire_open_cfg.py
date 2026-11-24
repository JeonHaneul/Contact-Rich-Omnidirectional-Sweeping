"""Original Random-v0 PPO parameters with a separate experiment directory."""

from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg_02 import UR5eSweepPPORunnerCfg


@configclass
class UR5eSweepInspireOpenPPORunnerCfg(UR5eSweepPPORunnerCfg):
    experiment_name = "UR5e_shelf_sweep_random_single_inspire_open"
