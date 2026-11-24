"""Random-v0, one active shelf object, and an unactuated open Inspire hand."""

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from ... import mdp
from ...src.ur5e_inspire_open import ARM_JOINT_NAMES, make_fixed_open_inspire_cfg
from .joint_pos_random_single_sweep_env_cfg import UR5eSweepRandomSingleEnvCfg


# Express the reference tool axes in H: tool X -> H Z, tool Y -> H X.
# The original +Y/up alignment reward can therefore be used unchanged.
TOOL_ROTATION_H = (0.5, -0.5, -0.5, -0.5)


def _tool_frame(targets, debug_vis=False):
    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
    marker_cfg.prim_path = f"/Visuals/Inspire/{targets[0][0]}"
    return FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link", debug_vis=debug_vis,
        visualizer_cfg=marker_cfg,
        target_frames=[FrameTransformerCfg.FrameCfg(
            prim_path="{ENV_REGEX_NS}/Robot/inspire_base_link", name=name,
            offset=OffsetCfg(pos=position, rot=TOOL_ROTATION_H),
        ) for name, position in targets],
    )


@configclass
class UR5eSweepRandomSingleInspireOpenEnvCfg(UR5eSweepRandomSingleEnvCfg):
    """Keep original Random-v0 rewards and JointPositionAction; remove grip action."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = make_fixed_open_inspire_cfg().replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.gripper_action = None
        self.rewards.action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.03)
        self.rewards.joint_vel = RewTerm(
            func=mdp.reward_random_sweep.joint_vel_l2,
            weight=-0.03,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        # Match the source's 13 cm forward EEF/finger offsets along Inspire H +Z.
        self.scene.ee_frame = _tool_frame(
            (("end_effector", (0.0, 0.0, 0.13)),), debug_vis=True,
        )
        self.scene.finger_frame = _tool_frame((
            ("l_finger", (0.07, 0.0, 0.13)), ("r_finger", (-0.07, 0.0, 0.13)),
        ))
        self.scene.wrist_frame = _tool_frame((("wrist", (0.0, 0.0, -0.17)),))
        # Only the Arm remains movable. Resolve names instead of old gripper slices.
        arm = SceneEntityCfg("robot", joint_names=list(ARM_JOINT_NAMES), preserve_order=True)
        self.observations.policy.joint_pos.func = mdp.joint_pos_rel
        self.observations.policy.joint_pos.params = {"asset_cfg": arm}
        self.observations.policy.joint_vel.func = mdp.joint_vel_rel
        self.observations.policy.joint_vel.params = {"asset_cfg": arm}
        # Relax the source's 1 rad/s cutoff while still ending very fast Arm motions.
        self.terminations.hand_velocity = DoneTerm(
            func=mdp.hand_velocity_termination,
            params={"asset_cfg": arm, "threshold": 3.0},
            time_out=False,
        )


@configclass
class UR5eSweepRandomSingleInspireOpenEnvCfg_PLAY(UR5eSweepRandomSingleInspireOpenEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = False
