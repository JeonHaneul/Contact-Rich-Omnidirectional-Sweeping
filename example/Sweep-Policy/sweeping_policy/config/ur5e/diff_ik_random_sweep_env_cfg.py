"""Relative Cartesian Differential-IK variant of the random sweep task."""

from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.utils import configclass

from ... import mdp
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
Isaac-Sweep-Object-UR5e-Random-v1 의 엔트리 포인트입니다.
"""


@configclass
class UR5eSweepDiffIKEnvCfg(UR5eSweepEnvCfg):
    """v0 random sweep task with a 6D Cartesian arm action.

    The arm action is ``[dx, dy, dz, rx, ry, rz]``.  Translation is in the
    robot-root frame and rotation is an axis-angle rotation vector.  Since the
    command is relative, six values are sufficient and the policy never has to
    emit or normalize a quaternion.  Isaac Lab's action term solves DLS
    Differential IK internally and writes joint-position targets to the UR5e.

    The inherited one-dimensional binary gripper action is intentionally kept,
    so only the arm command representation changes for a controlled joint-vs-
    Cartesian convergence comparison.  Consequently the full policy action is
    7D (Cartesian arm 6D + gripper 1D), just as v0 is numerically 7D.
    """

    def __post_init__(self):
        super().__post_init__()

        self.actions.arm_action = mdp.DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=ARM_JOINT_NAMES,
            body_name="robotiq_base_link",
            body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(
                # Keep the IK-controlled point identical to the inherited
                # ee_frame/end_effector observation and reward point.
                pos=(0.13, 0.0, 0.0),
            ),
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=True,
                ik_method="dls",
                ik_params={"lambda_val": 0.01},
            ),
            # Apply the same scalar to translation and rotation deltas.
            # This action term does not configure a separate clipping bound.
            scale=0.5,
        )


@configclass
class UR5eSweepDiffIKEnvCfg_PLAY(UR5eSweepDiffIKEnvCfg):
    """Small, deterministic visualization configuration for v1."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
