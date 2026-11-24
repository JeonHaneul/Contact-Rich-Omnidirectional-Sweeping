"""Same PPO settings as Inspire Open-v0, with an independent v1 experiment."""

from isaaclab.utils import configclass

from .rsl_rl_ppo_inspire_open_cfg import UR5eSweepInspireOpenPPORunnerCfg


@configclass
class UR5eSweepInspireHandPPORunnerCfg(UR5eSweepInspireOpenPPORunnerCfg):
    experiment_name = "UR5e_shelf_sweep_random_single_inspire_hand_v1"
