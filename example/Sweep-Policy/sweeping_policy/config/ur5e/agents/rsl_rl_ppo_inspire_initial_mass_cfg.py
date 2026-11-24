"""Same v2-mass PPO hyperparameters, with explicit actor/critic observation routes."""

from isaaclab.utils import configclass
from .rsl_rl_ppo_inspire_mass_sensors_cfg import UR5eSweepInspireMassSensorsPPORunnerCfg


@configclass
class UR5eSweepInspireInitialMassPPORunnerCfg(UR5eSweepInspireMassSensorsPPORunnerCfg):
    experiment_name = "UR5e_shelf_sweep_random_single_inspire_initial_v3_mass"
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}


@configclass
class UR5eSweepInspirePrivilegedMassPPORunnerCfg(UR5eSweepInspireInitialMassPPORunnerCfg):
    experiment_name = "UR5e_shelf_sweep_random_single_inspire_privileged_v4_mass"
    obs_groups = {"actor": ["policy"], "critic": ["policy", "privileged"]}
