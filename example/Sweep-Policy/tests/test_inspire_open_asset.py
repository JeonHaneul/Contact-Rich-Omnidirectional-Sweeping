"""Offline USD checks for a fixed-open hand, without starting Isaac Sim."""

import ast
import math
from pathlib import Path

import pytest

pytest.importorskip("pxr")
from pxr import Gf, Usd, UsdGeom, UsdPhysics


ROOT = Path(__file__).resolve().parents[3]
ASSET = ROOT / "src/trash/hand_manipulation_rl/hand_manipulation_rl/assets/data/ur5e_inspire_usd/ur5e_inspire.usd"
SOURCE = Path(__file__).resolve().parents[1] / "sweeping_policy/src/ur5e_inspire_open.py"
tree = ast.parse(SOURCE.read_text())
nodes = [node for node in tree.body if
         (isinstance(node, ast.FunctionDef) and node.name == "_freeze_inspire_hand_open")
         or (isinstance(node, ast.Assign) and any(
             isinstance(target, ast.Name) and target.id == "HAND_JOINT_NAMES" for target in node.targets))]
namespace = {"Gf": Gf, "Usd": Usd, "UsdGeom": UsdGeom, "UsdPhysics": UsdPhysics}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)


def _stage():
    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/Robot", "Xform").GetReferences().AddReference(str(ASSET))
    return stage


def test_fixed_hand_removes_only_finger_dofs_and_preserves_open_body_poses():
    stage = _stage()
    cache = UsdGeom.XformCache()
    before = {p.GetPath(): cache.GetLocalToWorldTransform(p)
              for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    arm_before = {p.GetName() for p in stage.Traverse()
                  if p.IsA(UsdPhysics.RevoluteJoint) and p.GetName() not in namespace["HAND_JOINT_NAMES"]}
    namespace["_freeze_inspire_hand_open"](stage, "/Robot")
    assert len(arm_before) == 6
    assert {p.GetName() for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)} == arm_before
    assert not any(p.GetName() == "axia80_link" for p in stage.Traverse())
    cache = UsdGeom.XformCache()
    for path, matrix in before.items():
        assert Gf.IsClose(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path)), matrix, 1e-10)
    for name in namespace["HAND_JOINT_NAMES"]:
        prim = stage.GetPrimAtPath(f"/Robot/joints/{name}")
        assert prim.IsA(UsdPhysics.FixedJoint)
        assert not prim.HasAPI(UsdPhysics.DriveAPI, "angular")
        joint = UsdPhysics.FixedJoint(prim)
        frames = []
        for side in (0, 1):
            body = stage.GetPrimAtPath(getattr(joint, f"GetBody{side}Rel")().GetTargets()[0])
            local = Gf.Matrix4d(1.0)
            local.SetRotate(Gf.Quatd(getattr(joint, f"GetLocalRot{side}Attr")().Get()))
            local.SetTranslateOnly(Gf.Vec3d(getattr(joint, f"GetLocalPos{side}Attr")().Get()))
            frames.append(local * cache.GetLocalToWorldTransform(body))
        assert Gf.IsClose(frames[0], frames[1], 1e-5)


def test_axia_asset_is_rejected_before_finger_joints_are_changed():
    stage = _stage()
    stage.DefinePrim("/Robot/axia80_link", "Xform")
    with pytest.raises(ValueError, match="without Axia80"):
        namespace["_freeze_inspire_hand_open"](stage, "/Robot")
    assert sum(p.IsA(UsdPhysics.RevoluteJoint) for p in stage.Traverse()) == 18


def test_v1_raw_hand_is_movable_and_command_endpoints_fit_usd_limits():
    source = SOURCE.with_name("ur5e_inspire_hand.py")
    tree = ast.parse(source.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id in ("HAND_OPEN_POSITIONS", "HAND_CLOSED_POSITIONS")
        for target in node.targets)]
    values = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), values)
    stage = _stage()
    assert sum(p.IsA(UsdPhysics.RevoluteJoint) for p in stage.Traverse()) == 18
    assert not any(p.GetName() == "axia80_link" for p in stage.Traverse())
    for index, name in enumerate(namespace["HAND_JOINT_NAMES"]):
        joint = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(f"/Robot/joints/{name}"))
        for key in ("HAND_OPEN_POSITIONS", "HAND_CLOSED_POSITIONS"):
            angle = math.degrees(values[key][index])
            assert joint.GetLowerLimitAttr().Get() - 1e-5 <= angle
            assert angle <= joint.GetUpperLimitAttr().Get() + 1e-5
