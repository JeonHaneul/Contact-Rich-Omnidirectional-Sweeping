"""Direct UR5e/Inspire mount with mechanically fixed, fully extended fingers."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from pxr import Gf, Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.sim.utils import clone, get_current_stage
from isaaclab.utils import configclass

from .ur5e import UR5e_CFG


ARM_JOINT_NAMES = (
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
)
HAND_JOINT_NAMES = (
    *(f"inspire_left_thumb_{i}_joint" for i in range(1, 5)),
    *(f"inspire_left_{finger}_{i}_joint"
      for finger in ("index", "middle", "ring", "little") for i in (1, 2)),
)


def _freeze_inspire_hand_open(stage, robot_path):
    """Freeze the source USD's zero-angle, extended hand before PhysX loads it.

    Recompute the child joint frame from the authored body transforms so the
    fixed constraints agree with the rendered open pose without a settling step.
    The six UR5e revolute joints and the direct wrist mount remain untouched.
    """
    root = stage.GetPrimAtPath(robot_path)
    if not root.IsValid():
        raise RuntimeError(f"Robot prim does not exist: {robot_path}")
    prims = list(Usd.PrimRange(root))
    if any(prim.GetName() == "axia80_link" for prim in prims):
        raise ValueError("The fixed-open Inspire task requires a direct mount without Axia80")
    joints = {prim.GetName(): prim for prim in prims
              if prim.IsA(UsdPhysics.RevoluteJoint) and prim.GetName() in HAND_JOINT_NAMES}
    if set(joints) != set(HAND_JOINT_NAMES):
        raise ValueError(f"Missing Inspire revolute joints: {set(HAND_JOINT_NAMES) - set(joints)}")
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    for name in HAND_JOINT_NAMES:
        prim = joints[name]
        joint = UsdPhysics.RevoluteJoint(prim)
        parent = stage.GetPrimAtPath(joint.GetBody0Rel().GetTargets()[0])
        child = stage.GetPrimAtPath(joint.GetBody1Rel().GetTargets()[0])
        parent_frame = Gf.Matrix4d(1.0)
        parent_frame.SetRotate(Gf.Quatd(joint.GetLocalRot0Attr().Get()))
        parent_frame.SetTranslateOnly(Gf.Vec3d(joint.GetLocalPos0Attr().Get()))
        joint_world = parent_frame * cache.GetLocalToWorldTransform(parent)
        child_frame = Gf.Transform(joint_world * cache.GetLocalToWorldTransform(child).GetInverse())
        fixed = UsdPhysics.FixedJoint.Define(stage, prim.GetPath())
        fixed.CreateLocalPos1Attr().Set(Gf.Vec3f(child_frame.GetTranslation()))
        fixed.CreateLocalRot1Attr().Set(Gf.Quatf(child_frame.GetRotation().GetQuat()))
        fixed.CreateCollisionEnabledAttr().Set(False)
        if prim.HasAPI(UsdPhysics.DriveAPI, "angular"):
            prim.RemoveAPI(UsdPhysics.DriveAPI, "angular")


@clone
def spawn_fixed_open_inspire(prim_path, cfg, translation=None, orientation=None, **kwargs):
    prim = sim_utils.spawn_from_usd(
        prim_path, cfg, translation=translation, orientation=orientation, **kwargs
    )
    _freeze_inspire_hand_open(get_current_stage(), prim_path)
    return prim


@configclass
class FixedOpenInspireUsdCfg(sim_utils.UsdFileCfg):
    func: Callable = spawn_fixed_open_inspire


def make_fixed_open_inspire_cfg():
    """Keep Random-v0 Arm PD gains, limits, root pose, and initial joints."""
    default = (Path(__file__).resolve().parents[4] / "src" / "trash" / "hand_manipulation_rl"
               / "hand_manipulation_rl" / "assets" / "data" / "ur5e_inspire_usd" / "ur5e_inspire.usd")
    asset = Path(os.environ.get("SWEEP_POLICY_INSPIRE_ROBOT_USD_PATH", str(default))).expanduser().resolve()
    if not asset.is_file():
        raise FileNotFoundError(f"Direct-mount Inspire USD does not exist: {asset}")
    cfg = UR5e_CFG.copy()
    cfg.spawn = FixedOpenInspireUsdCfg(
        usd_path=str(asset), rigid_props=UR5e_CFG.spawn.rigid_props.copy(),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False),
    )
    # Fixed fingers have no DOFs, actuators, commands, or policy observations.
    cfg.actuators = {"arm": UR5e_CFG.actuators["arm"].copy()}
    cfg.init_state.joint_pos = {name: UR5e_CFG.init_state.joint_pos[name] for name in ARM_JOINT_NAMES}
    cfg.init_state.joint_vel = {name: 0.0 for name in ARM_JOINT_NAMES}
    return cfg
