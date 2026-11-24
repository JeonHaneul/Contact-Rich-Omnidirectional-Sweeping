"""Offline USD checks for bottom origin, collision shape and authored mass data."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("pxr")
from pxr import Gf, Usd, UsdGeom, UsdPhysics


def test_cylinder_spawner_authors_bottom_heavy_mass_without_moving_body_origin():
    stage = Usd.Stage.CreateInMemory()

    def spawn_cylinder(path, cfg, translation=None, orientation=None, **kwargs):
        prim = stage.DefinePrim(path, "Xform")
        UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(*translation))
        UsdGeom.Xformable(prim).AddOrientOp().Set(Gf.Quatf(orientation[0], Gf.Vec3f(*orientation[1:])))
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(cfg.mass_props.mass)
        cylinder = UsdGeom.Cylinder.Define(stage, f"{path}/geometry/mesh")
        cylinder.CreateAxisAttr(cfg.axis)
        cylinder.CreateRadiusAttr(cfg.radius)
        cylinder.CreateHeightAttr(cfg.height)
        UsdPhysics.CollisionAPI.Apply(cylinder.GetPrim())
        return prim

    source = Path(__file__).resolve().parents[1] / "sweeping_policy/src/mass_cylinder.py"
    tree = ast.parse(source.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    for node in nodes:
        node.decorator_list = []
    ns = {"Gf": Gf, "UsdGeom": UsdGeom, "UsdPhysics": UsdPhysics,
          "sim_utils": SimpleNamespace(spawn_cylinder=spawn_cylinder)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), ns)
    cfg = SimpleNamespace(radius=.04, height=.10, axis="Z", mass_props=SimpleNamespace(mass=1.))
    prim = ns["spawn_mass_cylinder"]("/Cylinder", cfg, translation=(-.75, 0., 1.05),
                                     orientation=(1., 0., 0., 0.))
    mass = UsdPhysics.MassAPI(prim)
    assert mass.GetMassAttr().Get() == pytest.approx(1.)
    assert tuple(mass.GetCenterOfMassAttr().Get()) == pytest.approx((0., 0., .02))
    assert tuple(mass.GetDiagonalInertiaAttr().Get()) == pytest.approx((.0009333333333, .0009333333333, .0008))
    assert mass.GetPrincipalAxesAttr().Get() == Gf.Quatf(1.)
    cache = UsdGeom.XformCache()
    assert tuple(cache.GetLocalToWorldTransform(prim).ExtractTranslation()) == pytest.approx((-.75, 0., 1.05))
    box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"]).ComputeWorldBound(prim).ComputeAlignedRange()
    assert box.GetMin()[2] == pytest.approx(1.05)
    assert box.GetMax()[2] == pytest.approx(1.15)
    assert tuple(box.GetSize()) == pytest.approx((.08, .08, .10))
    assert stage.GetPrimAtPath("/Cylinder/geometry/mesh").HasAPI(UsdPhysics.CollisionAPI)
    assert len([p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]) == 1
