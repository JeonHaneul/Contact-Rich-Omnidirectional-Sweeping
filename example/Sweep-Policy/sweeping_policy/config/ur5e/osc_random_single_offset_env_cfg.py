"""Single-object OSC sweep task with expanded random TCP initialization."""

import math

from isaaclab.controllers import OperationalSpaceControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import OperationalSpaceControllerActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ... import mdp
from ...shelf_sweep_random_env_cfg import EventCfg as RandomSweepEventCfg
from .joint_pos_random_sweep_env_cfg import UR5eSweepEnvCfg


ARM_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


@configclass
class EventCfg(RandomSweepEventCfg):
    """Spawn one object, then solve an expanded collision-safe TCP pose."""

    initialize_random_shelf_pose = EventTerm(
        func=mdp.initialize_robot_in_random_shelf_pose_with_target_offset,
        mode="reset",
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "object_collection_cfg": SceneEntityCfg("object_collection"),
            "arm_joint_names": ARM_JOINT_NAMES,
            "locked_joint_name": "wrist_3_joint",
            "eef_body_name": "robotiq_base_link",
            "eef_offset": (0.13, 0.0, 0.0),
            "tool_heading_axis": (1.0, 0.0, 0.0),
            # The inherited region covers the free base-to-shelf gap and the
            # shelf interior in front of the object.
            "base_x_clearance": 0.38,
            "object_x_clearance": 0.15,
            "eef_y_range": (-0.25, 0.25),
            "eef_z_range": (1.15, 1.32),
            # One third of samples try a pose aligned with the target in shelf
            # depth and offset opposite the commanded left/right sweep.  The
            # sampler falls back to the inherited region when an edge object
            # has insufficient lateral room.
            "target_side_probability": 1.0 / 3.0,
            "target_side_offset_range": (0.14, 0.22),
            "target_side_clearance": 0.07,
            "target_side_x_tolerance": 0.10,
            "max_heading_angle": math.radians(18.0),
            "heading_limit": math.radians(20.0),
            "max_iterations": 120,
            "damping": 0.05,
            "step_size": 0.5,
            "position_tolerance": 0.005,
            "heading_tolerance": math.radians(0.5),
            "acceptance_position_tolerance": 0.08,
            "acceptance_heading_tolerance": math.radians(2.0),
            "max_pose_attempts": 8,
            "pose_bank_fallback_attempts": 8,
            "pose_bank_capacity_per_cell": 32,
            "joint_seed_offsets": (
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                (0.25, -0.15, 0.15, 0.0, 0.0, 0.0),
                (-0.25, 0.15, -0.15, 0.0, 0.0, 0.0),
                (0.45, -0.30, 0.25, 0.15, -0.10, 0.0),
                (-0.45, 0.30, -0.25, -0.15, 0.10, 0.0),
            ),
        },
    )


@configclass
class UR5eSweepRandomSingleOSCOffsetEnvCfg(UR5eSweepEnvCfg):
    """Random-v0 with one object, expanded TCP reset poses, and relative OSC."""

    events: EventCfg = EventCfg()

    def __post_init__(self):
        super().__post_init__()

        self.events.object_spawn.func = mdp.randomize_single_object_scene
        self.rewards.homing_after_sweep = None

        self.scene.robot.actuators["arm"].stiffness = 0.0
        self.scene.robot.actuators["arm"].damping = 0.0

        self.actions.arm_action = OperationalSpaceControllerActionCfg(
            asset_name="robot",
            joint_names=ARM_JOINT_NAMES,
            body_name="robotiq_base_link",
            body_offset=OperationalSpaceControllerActionCfg.OffsetCfg(
                # Match the inherited end-effector observation/reward frame.
                pos=(0.13, 0.0, 0.0),
            ),
            controller_cfg=OperationalSpaceControllerCfg(
                # Fixed impedance means the policy supplies only the target
                # pose; stiffness and damping are not part of the action.
                target_types=["pose_rel"],
                impedance_mode="fixed",
                motion_control_axes_task=(1, 1, 1, 1, 1, 1),
                contact_wrench_control_axes_task=(0, 0, 0, 0, 0, 0),
                inertial_dynamics_decoupling=True,
                partial_inertial_dynamics_decoupling=False,
                gravity_compensation=False,
                # Match the workspace's fixed UR5e OSC baseline in
                # src/trash/test/rtde_osc_velocity.py.
                motion_stiffness_task=(100.0, 100.0, 100.0, 100.0, 100.0, 100.0),
                motion_damping_ratio_task=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
                nullspace_control="none",
            ),
            nullspace_joint_pos_target="none",
            position_scale=1.0,
            orientation_scale=1.0,
        )


@configclass
class UR5eSweepRandomSingleOSCOffsetEnvCfg_PLAY(
    UR5eSweepRandomSingleOSCOffsetEnvCfg
):
    """Small visualization configuration for the expanded-reset OSC task."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
