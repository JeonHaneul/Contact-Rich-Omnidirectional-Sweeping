"""Validate the reused Axia/Inspire assembly topology offline with USD."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("pxr")
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics


REPO = Path(__file__).resolve().parents[3]
ASSET_ROOT = REPO / "src/trash/hand_manipulation_rl/hand_manipulation_rl/assets"


def test_sensor_assembly_preserves_movable_hand_and_inserts_two_fixed_joints():
    stage = Usd.Stage.CreateInMemory()

    class Cfg(SimpleNamespace):
        pass

    def spawn_usd(path, cfg, translation, orientation):
        prim = stage.DefinePrim(path, "Xform")
        prim.GetReferences().AddReference(cfg.usd_path)
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        transform = Gf.Matrix4d(1.)
        transform.SetRotate(Gf.Quatd(orientation[0], Gf.Vec3d(*orientation[1:])))
        transform.SetTranslateOnly(Gf.Vec3d(*translation))
        xform.AddTransformOp(UsdGeom.XformOp.PrecisionDouble).Set(transform)

    class UsdCfg(Cfg):
        func = staticmethod(spawn_usd)

    def spawn_cylinder(path, cfg, translation, orientation):
        cylinder = UsdGeom.Cylinder.Define(stage, path)
        cylinder.CreateRadiusAttr(cfg.radius)
        cylinder.CreateHeightAttr(cfg.height)
        cylinder.CreateAxisAttr(cfg.axis)
        xform = UsdGeom.Xformable(cylinder)
        xform.AddTranslateOp().Set(Gf.Vec3d(*translation))
        xform.AddOrientOp().Set(Gf.Quatf(orientation[0], Gf.Vec3f(*orientation[1:])))
        UsdPhysics.RigidBodyAPI.Apply(cylinder.GetPrim())
        UsdPhysics.MassAPI.Apply(cylinder.GetPrim()).CreateMassAttr(cfg.mass_props.mass)

    class CylinderCfg(Cfg):
        func = staticmethod(spawn_cylinder)

    ns = {"Path": Path, "Gf": Gf, "Sdf": Sdf, "Usd": Usd,
          "UsdGeom": UsdGeom, "UsdPhysics": UsdPhysics,
          "get_current_stage": lambda: stage,
          "HAND_BASE_BODY_NAME": "inspire_base_link", "WRIST_BODY_NAME": "wrist_3_link",
          "FT_SENSOR_BODY_NAME": "axia80_link", "ORIGINAL_MOUNT_JOINT_NAME": "inspire_mount",
          "sim_utils": SimpleNamespace(UsdFileCfg=UsdCfg, CylinderCfg=CylinderCfg,
                                       RigidBodyPropertiesCfg=Cfg, MassPropertiesCfg=Cfg,
                                       CollisionPropertiesCfg=Cfg, PreviewSurfaceCfg=Cfg)}
    tree = ast.parse((ASSET_ROOT / "robot.py").read_text())
    names = {"_world_transform", "_offset_matrix", "_matrix_to_pose", "_set_world_transform",
             "_create_fixed_joint", "_move_hand_rigid_bodies", "spawn_ur5e_axia80_inspire"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    for node in nodes:
        node.decorator_list = []  # the test assembles one stage, no Isaac Sim cloning
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "axia_spawner", "exec"), ns)

    asset = str(ASSET_ROOT / "data/ur5e_inspire_usd/ur5e_inspire.usd")
    pose = ((.3, -.2, .8), (.5, .5, .5, .5))
    spawn_usd("/Reference", UsdCfg(usd_path=asset), *pose)
    reference = ns["_world_transform"](stage.GetPrimAtPath("/Reference/inspire_base_link"))
    cfg = Cfg(usd_path=asset, rigid_props=None, articulation_props=None,
              sensor_height=.0254, sensor_radius=.041, sensor_mass=.001)
    ns["spawn_ur5e_axia80_inspire"]("/Robot", cfg, translation=pose[0], orientation=pose[1])

    bodies = [p for p in Usd.PrimRange(stage.GetPrimAtPath("/Robot"))
              if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    assert sum(p.IsA(UsdPhysics.RevoluteJoint) for p in Usd.PrimRange(stage.GetPrimAtPath("/Robot"))) == 18
    assert not stage.GetPrimAtPath("/Robot/joints/inspire_mount").IsActive()
    assert len([p for p in bodies if "force_sensor" in p.GetName()]) == 17
    offset = ns["_offset_matrix"]((0., 0., .0254))
    for body in bodies:
        if not body.GetName().startswith("inspire_"):
            continue
        before = ns["_world_transform"](stage.GetPrimAtPath(f"/Reference/{body.GetName()}"))
        expected = before * reference.GetInverse() * offset * reference
        assert Gf.IsClose(ns["_world_transform"](body), expected, 1e-8)
    # Arm geometry is not translated with the sensor housing.
    for name in ("forearm_link", "wrist_3_link"):
        assert Gf.IsClose(ns["_world_transform"](stage.GetPrimAtPath(f"/Robot/{name}")),
                         ns["_world_transform"](stage.GetPrimAtPath(f"/Reference/{name}")), 1e-8)
    cylinder = UsdGeom.Cylinder(stage.GetPrimAtPath("/Robot/axia80_link"))
    assert cylinder.GetHeightAttr().Get() == pytest.approx(.0254)
    assert cylinder.GetRadiusAttr().Get() == pytest.approx(.041)
    for name, parent, child in (
        ("ur5e_to_axia80_joint", "wrist_3_link", "axia80_link"),
        ("axia80_measurement_joint", "axia80_link", "inspire_base_link"),
    ):
        joint = UsdPhysics.FixedJoint(stage.GetPrimAtPath(f"/Robot/joints/{name}"))
        assert str(joint.GetBody0Rel().GetTargets()[0]) == f"/Robot/{parent}"
        assert str(joint.GetBody1Rel().GetTargets()[0]) == f"/Robot/{child}"
        # Both fixed-joint frames coincide; insertion introduces no initial strain.
        frames = []
        for side in (0, 1):
            body = stage.GetPrimAtPath(getattr(joint, f"GetBody{side}Rel")().GetTargets()[0])
            local = Gf.Matrix4d(1.)
            local.SetRotate(Gf.Quatd(getattr(joint, f"GetLocalRot{side}Attr")().Get()))
            local.SetTranslateOnly(Gf.Vec3d(getattr(joint, f"GetLocalPos{side}Attr")().Get()))
            frames.append(local * ns["_world_transform"](body))
        assert Gf.IsClose(frames[0], frames[1], 1e-6)
