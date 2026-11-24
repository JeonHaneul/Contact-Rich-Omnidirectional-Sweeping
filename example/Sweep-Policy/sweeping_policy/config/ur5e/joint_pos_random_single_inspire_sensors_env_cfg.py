"""Inspire Open-v2: v1 plus cylindrical F/T housing and tactile observations."""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from ...mdp import inspire_sensor_observations as sensor_obs
from ...shelf_sweep_random_env_cfg import ObservationsCfg
from ...src.ur5e_inspire_sensors import PALM_CHANNEL_NAMES, make_sensor_inspire_cfg
from .joint_pos_random_single_inspire_hand_env_cfg import UR5eSweepRandomSingleInspireHandEnvCfg


@configclass
class InspireSensorObservationsCfg(ObservationsCfg):
    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        # Append after all inherited terms; the v1 observation prefix stays intact.
        tactile = ObsTerm(func=sensor_obs.palm_tactile_bits)
        wrist_wrench = ObsTerm(func=sensor_obs.wrist_wrench_eef)

    policy: PolicyCfg = PolicyCfg()


@configclass
class UR5eSweepRandomSingleInspireSensorsEnvCfg(UR5eSweepRandomSingleInspireHandEnvCfg):
    observations: InspireSensorObservationsCfg = InspireSensorObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = make_sensor_inspire_cfg().replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.palm_tactile = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/(" + "|".join(PALM_CHANNEL_NAMES) + ")",
            update_period=0.0, history_length=1, force_threshold=0.05, debug_vis=False,
        )


@configclass
class UR5eSweepRandomSingleInspireSensorsEnvCfg_PLAY(UR5eSweepRandomSingleInspireSensorsEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = False
