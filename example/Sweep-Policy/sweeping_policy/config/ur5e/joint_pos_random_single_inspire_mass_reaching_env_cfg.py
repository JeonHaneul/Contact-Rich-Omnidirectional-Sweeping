"""v3-mass: v2-mass starting around the reaching pose, with XY variation only."""

from isaaclab.managers import EventTermCfg as EventTerm, SceneEntityCfg
from isaaclab.utils import configclass

from ... import mdp
from ...shelf_sweep_random_env_cfg import EventCfg as RandomSweepEventCfg
from ...src.ur5e_inspire_open import ARM_JOINT_NAMES
from .joint_pos_random_single_inspire_open_env_cfg import TOOL_ROTATION_H
from .joint_pos_random_single_inspire_mass_sensors_env_cfg import (
    TARGET_Z_OFFSET, UR5eSweepRandomSingleInspireMassSensorsEnvCfg,
)


@configclass
class MassReachingEventsCfg(RandomSweepEventCfg):
    # Run after inherited reset_all and single-object spawn. GT is used for IK.
    initialize_reaching_pose = EventTerm(
        func=mdp.initialize_robot_at_reaching_pose,
        mode="reset",
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "shelf_cfg": SceneEntityCfg("shelf"),
            "object_collection_cfg": SceneEntityCfg("object_collection"),
            "arm_joint_names": ARM_JOINT_NAMES,
            "eef_body_name": "axia80_link",
            "eef_offset": (0.0, 0.0, 0.1427),
            "eef_rotation": TOOL_ROTATION_H,
            "reaching_x_offset": -0.02,
            "reaching_z_offset": TARGET_Z_OFFSET,
            "use_column_axes": True,
            # Trial distribution, not a measured hardware error specification.
            "position_noise": (0.01, 0.01, 0.0),
            "max_iterations": 80,
            "damping": 0.05,
            "step_size": 0.5,
            "position_tolerance": 0.003,
            "orientation_tolerance": 0.01,
            "joint_seed_offsets": (
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                (0.15, -0.10, 0.10, 0.0, 0.0, 0.0),
                (-0.15, 0.10, -0.10, 0.0, 0.0, 0.0),
            ),
        },
    )


@configclass
class UR5eSweepRandomSingleInspireMassReachingEnvCfg(UR5eSweepRandomSingleInspireMassSensorsEnvCfg):
    events: MassReachingEventsCfg = MassReachingEventsCfg()

    def __post_init__(self):
        super().__post_init__()
        frame = self.scene.ee_frame.target_frames[0]
        self.events.initialize_reaching_pose.params["eef_offset"] = frame.offset.pos
        self.events.initialize_reaching_pose.params["eef_rotation"] = frame.offset.rot
        self.rewards.reaching = None


@configclass
class UR5eSweepRandomSingleInspireMassReachingEnvCfg_PLAY(UR5eSweepRandomSingleInspireMassReachingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = False
