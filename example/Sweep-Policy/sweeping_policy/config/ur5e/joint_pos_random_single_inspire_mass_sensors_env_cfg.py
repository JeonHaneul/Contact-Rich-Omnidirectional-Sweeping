"""v2-mass: bottom-heavy target, 2D hand, palm-only tactile and wrist F/T."""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from ...mdp import inspire_sensor_observations as sensor_obs
from ...mdp import inspire_mass_rewards as mass_rewards
from ...shelf_sweep_random_env_cfg import ObservationsCfg
from ...src.ur5e_inspire_sensors import make_sensor_inspire_cfg
from .joint_pos_random_single_inspire_open_env_cfg import _tool_frame
from .joint_pos_random_single_inspire_hand_mass_env_cfg import UR5eSweepRandomSingleInspireHandMassEnvCfg

TARGET_Z_OFFSET = 0.04


@configclass
class MassSensorObservationsCfg(ObservationsCfg):
    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        tactile = ObsTerm(func=sensor_obs.palm_contact_bit)
        wrist_wrench = ObsTerm(func=sensor_obs.wrist_wrench_eef)

    policy: PolicyCfg = PolicyCfg()


@configclass
class UR5eSweepRandomSingleInspireMassSensorsEnvCfg(UR5eSweepRandomSingleInspireHandMassEnvCfg):
    observations: MassSensorObservationsCfg = MassSensorObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = make_sensor_inspire_cfg().replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.palm_tactile = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/inspire_palm_force_sensor",
            update_period=0.0, history_length=1, force_threshold=0.05, debug_vis=False,
        )
        # The spawner physically moves the hand by sensor_height along local Z.
        # Reference the housing center so the TCP is mount Z=height+0.13,
        # still hand Z=0.13. Adding height again to a hand-relative frame would
        # count the extension twice. Housing and hand use the same local axes.
        height = self.scene.robot.spawn.sensor_height
        self.scene.ee_frame = _tool_frame(
            (("end_effector", (0.0, 0.0, 0.13 + height / 2)),), debug_vis=True,
        )
        self.scene.ee_frame.target_frames[0].prim_path = "{ENV_REGEX_NS}/Robot/axia80_link"
        self.rewards.reaching.func = mass_rewards.reaching_at_object_height
        self.rewards.reaching.params = {"reaching_z_offset": TARGET_Z_OFFSET}
        self.rewards.orientation.func = mass_rewards.align_eef_y_with_shelf_z
        self.rewards.sweeping_object.params["pushing_z_offset"] = TARGET_Z_OFFSET


@configclass
class UR5eSweepRandomSingleInspireMassSensorsEnvCfg_PLAY(UR5eSweepRandomSingleInspireMassSensorsEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = False
