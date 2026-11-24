"""v1 PPO settings with a separate sensor-observation experiment."""

from isaaclab.utils import configclass

from .rsl_rl_ppo_inspire_hand_cfg import UR5eSweepInspireHandPPORunnerCfg


@configclass
class UR5eSweepInspireSensorsPPORunnerCfg(UR5eSweepInspireHandPPORunnerCfg):
    experiment_name = "UR5e_shelf_sweep_random_single_inspire_sensors_v2"
