"""Original v1 PPO settings with an independent mass-cylinder experiment."""

from isaaclab.utils import configclass

from .rsl_rl_ppo_inspire_hand_cfg import UR5eSweepInspireHandPPORunnerCfg


@configclass
class UR5eSweepInspireHandMassPPORunnerCfg(UR5eSweepInspireHandPPORunnerCfg):
    experiment_name = "UR5e_shelf_sweep_random_single_inspire_hand_v1_mass"


@configclass
class UR5eSweepInspireHandMassLimitPPORunnerCfg(UR5eSweepInspireHandMassPPORunnerCfg):
    experiment_name = "UR5e_shelf_sweep_random_single_inspire_hand_v1_mass_limit"
