"""Fixed-impedance operational-space-control variant of the random sweep task."""

from dataclasses import MISSING

from isaaclab.controllers import OperationalSpaceControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import OperationalSpaceControllerActionCfg
from isaaclab.utils import configclass

from ... import mdp


from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from .joint_pos_random_sweep_env_cfg import UR5eSweepEnvCfg

ARM_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

"""
Isaac-Sweep-Object-UR5e-Random-v2 의 엔트리 포인트입니다.
"""


@configclass
class RelativeObservationsCfg:
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
        ee_pose = ObsTerm(func=mdp.ee_pos_r)
        goal_pos = ObsTerm(
            func=mdp.MA_target_goal_command, params={"command_name": MISSING}
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class UR5eSweepOSCEnvCfg(UR5eSweepEnvCfg):
    """Random-v0 with a relative-pose, motion-only OSC arm action.

    Scene contents, observations, commands, rewards, events, terminations, and
    the binary gripper action are inherited unchanged from
    :class:`UR5eSweepEnvCfg`.  Apart from the arm action, only the arm actuator
    gains are changed as required for pure torque control.  The arm action is an
    relative pose delta ``[dx, dy, dz, rx, ry, rz]`` in the robot-root frame.
    Fixed OSC gains turn that target into joint torques, which the action term
    writes directly as arm effort targets.
    """

    def __post_init__(self):
        super().__post_init__()

        # The OSC action writes joint-effort targets.  Removing the implicit
        # arm drive gains prevents position/velocity drive torques from being
        # added to the OSC result.  The separate gripper actuator is untouched.
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
class UR5eSweepOSCRelativeObsEnvCfg(UR5eSweepOSCEnvCfg):

    observations: RelativeObservationsCfg = RelativeObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.observations.policy.target_obs_state.func = mdp.MA_object_position_in_EEF
        self.observations.policy.goal_pos.func = mdp.MA_target_goal_command_in_EEF


@configclass
class UR5eSweepOSCRelativeObsEEHomingEnvCfg(UR5eSweepOSCRelativeObsEnvCfg):
    """Relative-observation OSC task using the second EE homing reward."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot.actuators["arm"].stiffness = 0.0
        self.scene.robot.actuators["arm"].damping = 0.0

        self.actions.arm_action = OperationalSpaceControllerActionCfg(
            asset_name="robot",
            joint_names=ARM_JOINT_NAMES,
            body_name="robotiq_base_link",
            body_offset=OperationalSpaceControllerActionCfg.OffsetCfg(
                pos=(0.13, 0.0, 0.0),
            ),
            controller_cfg=OperationalSpaceControllerCfg(
                target_types=["pose_rel"],
                impedance_mode="fixed",
                motion_control_axes_task=(1, 1, 1, 1, 1, 1),
                contact_wrench_control_axes_task=(0, 0, 0, 0, 0, 0),
                inertial_dynamics_decoupling=True,
                partial_inertial_dynamics_decoupling=False,
                gravity_compensation=False,
                motion_stiffness_task=(
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                ),
                motion_damping_ratio_task=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
                nullspace_control="none",
            ),
            nullspace_joint_pos_target="none",
            position_scale=1.0,
            orientation_scale=1.0,
        )

        self.rewards.homing_after_sweep.func = (
            mdp.reward_random_sweep.EndEffectorHomingReward2
        )


@configclass
class UR5eSweepKpOSCRelativeObsEnvCfg(UR5eSweepOSCEnvCfg):

    observations: RelativeObservationsCfg = RelativeObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

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
                impedance_mode="variable_kp",
                motion_control_axes_task=(1, 1, 1, 1, 1, 1),
                contact_wrench_control_axes_task=(0, 0, 0, 0, 0, 0),
                inertial_dynamics_decoupling=True,
                partial_inertial_dynamics_decoupling=False,
                gravity_compensation=False,
                # Match the workspace's fixed UR5e OSC baseline in
                # src/trash/test/rtde_osc_velocity.py.
                motion_stiffness_task=100.0,
                motion_damping_ratio_task=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
                motion_stiffness_limits_task=(50.0, 200.0),
                nullspace_control="none",
            ),
            nullspace_joint_pos_target="none",
            position_scale=1.0,
            orientation_scale=1.0,
            stiffness_scale=100.0,
        )

        self.observations.policy.target_obs_state.func = mdp.MA_object_position_in_EEF
        self.observations.policy.goal_pos.func = mdp.MA_target_goal_command_in_EEF

        # self.rewards.homing_after_sweep.func = (
        #     mdp.reward_random_sweep.EndEffectorHomingReward
        # )


@configclass
class UR5eSweepOSCHomingRewardCartesianEnvCfg(UR5eSweepOSCRelativeObsEnvCfg):

    observations: RelativeObservationsCfg = RelativeObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

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

        self.observations.policy.target_obs_state.func = mdp.MA_object_position_in_EEF
        self.observations.policy.goal_pos.func = mdp.MA_target_goal_command_in_EEF

        self.rewards.homing_after_sweep.func = (
            mdp.reward_random_sweep.EndEffectorHomingReward
        )


@configclass
class UR5eSweepOSCEnvCfg_PLAY(UR5eSweepOSCEnvCfg):
    """Small, deterministic visualization configuration for the OSC task."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
