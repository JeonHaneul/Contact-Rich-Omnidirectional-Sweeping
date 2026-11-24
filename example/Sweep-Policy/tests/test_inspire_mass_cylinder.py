"""Pure contract checks for the mass model and single-target catalog hooks."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _execute(path, namespace, classes=(), functions=()):
    tree = ast.parse((ROOT / path).read_text())
    nodes = [node for node in tree.body if
             isinstance(node, ast.ClassDef) and node.name in classes
             or isinstance(node, ast.FunctionDef) and node.name in functions]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), namespace)
    return namespace


def test_bottom_heavy_com_and_inertia_are_consistent_with_two_mass_layers():
    ns = _execute("sweeping_policy/src/mass_cylinder.py", {},
                  functions=("bottom_heavy_mass_properties",))
    com, inertia = ns["bottom_heavy_mass_properties"](.04, .10, 1.)
    assert com == pytest.approx(.02)
    assert inertia == pytest.approx((.0009333333333333333, .0009333333333333333, .0008))
    assert all(value > 0 for value in inertia)
    # At twice the mass the COM is unchanged and all inertias double.
    com2, inertia2 = ns["bottom_heavy_mass_properties"](.04, .10, 2.)
    assert com2 == pytest.approx(com)
    assert inertia2 == pytest.approx(tuple(2 * value for value in inertia))


def test_catalog_contains_only_one_cylinder_and_matches_its_radius():
    class Cfg(SimpleNamespace):
        pass

    class RigidCfg(Cfg):
        InitialStateCfg = Cfg

    class CylinderCfg(Cfg):
        pass

    sim = SimpleNamespace(CylinderCfg=CylinderCfg, MassPropertiesCfg=Cfg,
                          RigidBodyPropertiesCfg=Cfg, CollisionPropertiesCfg=Cfg, PreviewSurfaceCfg=Cfg)
    ns = {"sim_utils": sim, "configclass": lambda cls: cls, "Callable": object,
          "spawn_mass_cylinder": lambda *a: None}
    _execute("sweeping_policy/src/mass_cylinder.py", ns, classes=("MassCylinderCfg",))
    ns.update(RigidObjectCfg=RigidCfg, UR5eSweepRandomSingleInspireHandEnvCfg=object)
    _execute("sweeping_policy/config/ur5e/joint_pos_random_single_inspire_hand_mass_env_cfg.py", ns,
             classes=("UR5eSweepRandomSingleInspireHandMassEnvCfg",))
    cfg = ns["UR5eSweepRandomSingleInspireHandMassEnvCfg"]()
    catalog = cfg._load_object_catalog()
    assert set(catalog["objects"]) == {"mass_cylinder"}
    assert catalog["id"] == {"mass_cylinder": 0}
    obj = cfg._make_object_cfg("mass_cylinder", None, catalog["pose"]["mass_cylinder"])
    assert obj.spawn.radius == catalog["width"]["mass_cylinder"] == .04
    assert obj.spawn.height == .10
    assert obj.spawn.mass_props.mass == 1.
    assert obj.init_state.pos == [-.75, 0., 1.05]
    assert obj.spawn.rigid_props.disable_gravity is False
    assert obj.spawn.collision_props.collision_enabled is True


def test_default_object_factory_and_catalog_loading_remain_unchanged():
    class Cfg(SimpleNamespace):
        pass

    class RigidCfg(Cfg):
        InitialStateCfg = Cfg

    sentinel = object()
    ns = {"ShelfSweepRandomEnvCfg": object, "configclass": lambda cls: cls,
          "RigidObjectCfg": RigidCfg, "UsdFileCfg": Cfg, "RigidBodyPropertiesCfg": Cfg,
          "MassPropertiesCfg": Cfg, "ENVIRONMENT_YAML_PATH": "original.yaml",
          "load_yaml_config": lambda yaml_path: sentinel}
    _execute("sweeping_policy/config/ur5e/joint_pos_random_sweep_env_cfg.py", ns,
             classes=("UR5eSweepEnvCfg",))
    cfg = ns["UR5eSweepEnvCfg"]()
    assert cfg._load_object_catalog() is sentinel
    obj = cfg._make_object_cfg("cup", "cup.usd", [-.75, 0., 1.05, 1., 0., 0., 0.])
    assert obj.spawn.usd_path == "cup.usd"
    assert obj.spawn.mass_props.mass == 1.
    assert obj.spawn.scale == (1., 1., 1.)
    assert obj.spawn.rigid_props.solver_position_iteration_count == 16
    assert obj.spawn.rigid_props.solver_velocity_iteration_count == 1
