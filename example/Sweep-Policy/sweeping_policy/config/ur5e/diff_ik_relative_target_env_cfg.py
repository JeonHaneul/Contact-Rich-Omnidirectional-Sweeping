"""Differential-IK random sweep task with EEF-relative task observations."""

from isaaclab.utils import configclass
from dataclasses import MISSING

from ... import mdp
from .diff_ik_random_sweep_env_cfg import UR5eSweepDiffIKEnvCfg
from ...shelf_sweep_random_env_cfg import EventCfg as RandomSweepEventCfg

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

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
    """Random-v0 resets followed by privileged reaching-pose IK."""

    # Inherited reset_all and object_spawn terms run first.  This term reads
    # their reset-time target state, then initializes only the arm joints.
    initialize_reaching_pose = EventTerm(
        func=mdp.initialize_robot_at_reaching_pose,
        mode="reset",
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "shelf_cfg": SceneEntityCfg("shelf"),
            "object_collection_cfg": SceneEntityCfg("object_collection"),
            "arm_joint_names": ARM_JOINT_NAMES,
            "eef_body_name": "robotiq_base_link",
            "eef_offset": (0.13, 0.0, 0.0),
            "reaching_x_offset": -0.02,
            "reaching_z_offset": 0.09,
            "position_noise": 0.01,
            "max_iterations": 40,
            "damping": 0.05,
            "step_size": 0.5,
            "position_tolerance": 0.003,
            "orientation_tolerance": 0.001,
            "joint_seed_offsets": (
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                (0.15, -0.10, 0.10, 0.0, 0.0, 0.0),
                (-0.15, 0.10, -0.10, 0.0, 0.0, 0.0),
            ),
        },
    )


@configclass
class ObservationswoEEFCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos = ObsTerm(func=mdp.MA_joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.MA_joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        target_obs_state = ObsTerm(
            func=mdp.MA_object_position_in_RRF,
            params={},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        target_obj_width = ObsTerm(
            func=mdp.MA_object_width, noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        # ee_pose = ObsTerm(func=mdp.ee_pos_r)
        goal_pos = ObsTerm(
            func=mdp.MA_target_goal_command, params={"command_name": MISSING}
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class UR5eSweepDiffIKRelativeTargetEnvCfg(UR5eSweepDiffIKEnvCfg):
    """Random-v1 with target and goal positions expressed in the EEF frame."""

    # events: EventCfg = EventCfg()

    def __post_init__(self):
        super().__post_init__()

        # self.decimation = 10
        # self.events.object_spawn.func = mdp.randomize_single_object_scene

        # Keep every inherited observation setting (including params, noise,
        # and order) and replace only the target and goal coordinate frames.
        self.observations.policy.target_obs_state.func = mdp.MA_object_position_in_EEF
        self.observations.policy.goal_pos.func = mdp.MA_target_goal_command_in_EEF


@configclass
class UR5eSweepDiffIK5RelativeTargetEnvCfg(UR5eSweepDiffIKEnvCfg):
    """Random-v1 with target and goal positions expressed in the EEF frame."""

    def __post_init__(self):
        super().__post_init__()

        self.actions.arm_action.scale = 0.5

        self.observations.policy.target_obs_state.func = mdp.MA_object_position_in_EEF
        self.observations.policy.goal_pos.func = mdp.MA_target_goal_command_in_EEF


@configclass
class UR5eSweepDiffIK6RelativeTargetEnvCfg(UR5eSweepDiffIKEnvCfg):
    """Random-v1 with target and goal positions expressed in the EEF frame."""

    def __post_init__(self):
        super().__post_init__()

        self.actions.arm_action.scale = 0.6

        self.observations.policy.target_obs_state.func = mdp.MA_object_position_in_EEF
        self.observations.policy.goal_pos.func = mdp.MA_target_goal_command_in_EEF


@configclass
class UR5eSweepDiffIK7RelativeTargetEnvCfg(UR5eSweepDiffIKEnvCfg):
    """Random-v1 with target and goal positions expressed in the EEF frame."""

    def __post_init__(self):
        super().__post_init__()

        self.actions.arm_action.scale = 0.7

        self.observations.policy.target_obs_state.func = mdp.MA_object_position_in_EEF
        self.observations.policy.goal_pos.func = mdp.MA_target_goal_command_in_EEF


@configclass
class UR5eSweepDiffIKRelativeTargetwoEEFEnvCfg(UR5eSweepDiffIKEnvCfg):
    """Random-v1 with target and goal positions expressed in the EEF frame."""

    # events: EventCfg = EventCfg()
    observations: ObservationswoEEFCfg = ObservationswoEEFCfg()

    def __post_init__(self):
        super().__post_init__()

        # self.decimation = 10
        # self.events.object_spawn.func = mdp.randomize_single_object_scene

        # Keep every inherited observation setting (including params, noise,
        # and order) and replace only the target and goal coordinate frames.
        self.observations.policy.target_obs_state.func = mdp.MA_object_position_in_EEF
        self.observations.policy.goal_pos.func = mdp.MA_target_goal_command_in_EEF
