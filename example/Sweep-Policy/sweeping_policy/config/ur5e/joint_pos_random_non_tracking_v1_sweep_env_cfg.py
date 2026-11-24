"""Random-v0 without target observations and with reset-time reaching IK."""

# Load the base environment before importing the manager package directly.
# This preserves Isaac Lab's required env/mdp initialization order.
from .joint_pos_random_sweep_env_cfg import UR5eSweepEnvCfg
from ...shelf_sweep_random_env_cfg import EventCfg as RandomSweepEventCfg
from ... import mdp

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
class UR5eSweepRandomNonTrackingV1EnvCfg(UR5eSweepEnvCfg):
    """Random-v0 with privileged reset placement and no target policy state."""

    events: EventCfg = EventCfg()

    def __post_init__(self):
        super().__post_init__()

        # target identity, initial position, and width remain available to
        # commands, events, rewards, and terminations, but not to the policy.
        self.observations.policy.target_obs_state = None
        self.observations.policy.target_obj_width = None

        # The arm already starts beside the target, so reaching shaping is no
        # longer needed.  All other reward terms retain their live state.
        self.rewards.reaching.weight = 0.0


@configclass
class UR5eSweepRandomNonTrackingV1EnvCfg_PLAY(
    UR5eSweepRandomNonTrackingV1EnvCfg
):
    """Small visualization configuration for Random--Non-Tracking-v1."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
