"""Independent logs for sensor and reaching-start mass experiments."""

from isaaclab.utils import configclass
from .rsl_rl_ppo_inspire_hand_mass_cfg import UR5eSweepInspireHandMassPPORunnerCfg


@configclass
class UR5eSweepInspireMassSensorsPPORunnerCfg(UR5eSweepInspireHandMassPPORunnerCfg):
    experiment_name = "UR5e_shelf_sweep_random_single_inspire_hand_v2_mass"


@configclass
class UR5eSweepInspireMassReachingPPORunnerCfg(UR5eSweepInspireMassSensorsPPORunnerCfg):
    experiment_name = "UR5e_shelf_sweep_random_single_inspire_hand_v3_mass"
