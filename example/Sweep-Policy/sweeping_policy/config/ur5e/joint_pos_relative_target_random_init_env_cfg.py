"""Joint-space relative-target sweep task with reset-only randomization."""

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ... import mdp
from ...shelf_sweep_random_env_cfg import EventCfg as RandomSweepEventCfg
from .joint_pos_random_sweep_env_cfg import UR5eSweepRelativeTargetEnvCfg


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
    """Inherited resets followed by a fixed-base, random arm-pose reset."""

    initialize_random_shelf_pose = EventTerm(
        func=mdp.initialize_robot_in_random_shelf_pose,
        mode="reset",
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "object_collection_cfg": SceneEntityCfg("object_collection"),
            "arm_joint_names": ARM_JOINT_NAMES,
            "locked_joint_name": "wrist_3_joint",
            "eef_body_name": "robotiq_base_link",
            "eef_offset": (0.13, 0.0, 0.0),
            # The TCP offset is along local +x, so +x is the tool heading.
            "tool_heading_axis": (1.0, 0.0, 0.0),
            # Along world x, sample strictly between the fixed base and target.
            # These margins avoid the folded-arm singular region and contact
            # with the target before the episode begins.
            "base_x_clearance": 0.38,
            "object_x_clearance": 0.15,
            # The active shelf board spans y=[-0.50, 0.50] and the next board
            # is at z=1.50.  Insets leave room for the gripper geometry.
            "eef_y_range": (-0.25, 0.25),
            "eef_z_range": (1.15, 1.32),
            # A two-degree guard keeps achieved headings strictly below the
            # requested 20-degree limit after numerical IK validation.
            "max_heading_angle": math.radians(18.0),
            "heading_limit": math.radians(20.0),
            "max_iterations": 120,
            "damping": 0.05,
            "step_size": 0.5,
            "position_tolerance": 0.005,
            "heading_tolerance": math.radians(0.5),
            # A sampled pose is an IK attractor, not part of the MDP.  Accept
            # a nearby result only when the final TCP still satisfies every
            # requested workspace/heading/wrist constraint.
            "acceptance_position_tolerance": 0.08,
            "acceptance_heading_tolerance": math.radians(2.0),
            # A numerical failure changes only the sampled robot pose.  The
            # object remains fixed and every retry uses the same complete
            # workspace distribution, so difficult regions are not removed.
            "max_pose_attempts": 8,
            # Last-resort poses are sampled from successful randomized poses
            # stratified over X/Y/Z/heading, never from one fixed home pose.
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
class UR5eSweepRelativeTargetRandomInitEnvCfg(
    UR5eSweepRelativeTargetEnvCfg
):
    """Reference joint-space task with only reset events replaced/extended."""

    events: EventCfg = EventCfg()

    def __post_init__(self):
        super().__post_init__()

        # Preserve the complete inherited MDP and change only object/robot
        # reset behavior.
        self.events.object_spawn.func = mdp.randomize_single_object_in_work_area


@configclass
class UR5eSweepRelativeTargetRandomInitEnvCfg_PLAY(
    UR5eSweepRelativeTargetRandomInitEnvCfg
):
    """Small visualization configuration for the random-initial-state task."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
