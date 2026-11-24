"""Exercise actual v1 mapping/reset code without starting Isaac Sim."""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


class Config(SimpleNamespace):
    def copy(self):
        return copy.deepcopy(self)


class ActionBase:
    def __init__(self, cfg, env):
        self.device = "cpu"
        self.num_envs = env.num_envs
        self._asset = env.scene[cfg.asset_name]


def _implementation():
    tree = ast.parse((ROOT / "sweeping_policy/src/ur5e_inspire_hand.py").read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.Assign, ast.FunctionDef))
             or isinstance(node, ast.ClassDef) and node.name == "InspireHandOpennessAction"]
    names = ("inspire_left_thumb_1_joint", "inspire_left_thumb_2_joint",
             "inspire_left_thumb_3_joint", "inspire_left_thumb_4_joint",
             *(f"inspire_left_{finger}_{index}_joint"
               for finger in ("index", "middle", "ring", "little") for index in (1, 2)))
    ns = {"torch": torch, "ActionTerm": ActionBase, "HAND_JOINT_NAMES": names}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "hand_action", "exec"), ns)
    return ns


class Robot:
    def __init__(self, names):
        # Deliberately reverse hand DOF order to exercise name-based mapping.
        self.names = [f"arm_{index}" for index in range(6)] + list(reversed(names))
        self.targets = torch.zeros(3, 18)

    def find_joints(self, names, preserve_order):
        assert preserve_order
        return [self.names.index(name) for name in names], names

    def set_joint_position_target(self, targets, joint_ids, env_ids=None):
        rows = torch.arange(3) if env_ids is None or isinstance(env_ids, slice) else torch.as_tensor(env_ids)
        self.targets[rows[:, None], torch.tensor(joint_ids)[None, :]] = targets


@pytest.fixture
def action():
    ns = _implementation()
    robot = Robot(ns["HAND_JOINT_NAMES"])
    env = SimpleNamespace(num_envs=3, scene={"robot": robot})
    return ns["InspireHandOpennessAction"](SimpleNamespace(asset_name="robot"), env)


def test_two_channels_are_independent_and_open_is_one(action):
    action.process_actions(torch.tensor([[1., 1.], [0., 1.], [1., 0.]]))
    targets = action._joint_targets
    assert torch.equal(targets[0], torch.zeros(12))
    assert targets[1, 0] == 0.0
    assert torch.all(targets[1, 1:] > 0.0)
    assert targets[2, 0] == pytest.approx(1.1641)
    assert torch.equal(targets[2, 1:], torch.zeros(11))


def test_direct_mapping_has_only_bounds_and_no_rate_limit(action):
    inputs = torch.tensor([[0., 0.], [-3., 5.], [.25, .75]])
    action.process_actions(inputs)
    assert torch.equal(action.raw_actions, inputs)
    assert torch.equal(action.processed_actions, inputs.clamp(0, 1))
    assert torch.equal(action._joint_targets[0], action._closed_positions)
    assert torch.allclose(action._joint_targets[2, 1:], action._closed_positions[1:] * .75)
    assert action._joint_targets[2, 0] == pytest.approx(1.1641 * .25)


def test_only_named_hand_joints_receive_targets(action):
    action.process_actions(torch.zeros(3, 2))
    action.apply_actions()
    assert torch.equal(action._asset.targets[:, :6], torch.zeros(3, 6))
    for index, name in enumerate(_implementation()["HAND_JOINT_NAMES"]):
        assert torch.allclose(action._asset.targets[:, action._asset.names.index(name)],
                              action._closed_positions[index].expand(3))


def test_partial_reset_restores_one_and_open_targets_only_for_reset_env(action):
    action.process_actions(torch.zeros(3, 2))
    action.apply_actions()
    before = action._asset.targets.clone()
    action.reset(torch.tensor([1]))
    assert torch.equal(action.raw_actions, torch.tensor([[0., 0.], [1., 1.], [0., 0.]]))
    assert torch.equal(action._asset.targets[1], torch.zeros(18))
    assert torch.equal(action._asset.targets[[0, 2]], before[[0, 2]])


def test_reset_history_is_one_for_hand_and_preserves_arm_and_other_envs():
    tree = ast.parse((ROOT / "sweeping_policy/env_inspire_hand.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    ns = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "history", "exec"), ns)
    manager = SimpleNamespace(_action=torch.zeros(3, 8), _prev_action=torch.zeros(3, 8))
    ns["initialize_hand_action_history"](manager, torch.tensor([0, 2]))
    expected = torch.zeros(3, 8)
    expected[[0, 2], -2:] = 1.0
    assert torch.equal(manager._action, expected)
    assert torch.equal(manager._prev_action, expected)


def test_factory_keeps_arm_settings_and_uses_raw_usd_instead_of_fixed_spawner():
    ns = _implementation()
    original = Config(
        spawn=Config(usd_path="direct_hand.usd", func="freeze_hand",
                     rigid_props=Config(disable_gravity=False),
                     articulation_props=Config(enabled_self_collisions=False)),
        init_state=Config(joint_pos={"arm_joint": 1.25}, joint_vel={"arm_joint": 0.}),
        actuators={"arm": Config(stiffness=3328.7, damping=368.8)},
    )
    ns.update(make_fixed_open_inspire_cfg=lambda: original.copy(),
              sim_utils=SimpleNamespace(UsdFileCfg=Config), ImplicitActuatorCfg=Config)
    cfg = ns["make_movable_inspire_cfg"]()
    assert not hasattr(cfg.spawn, "func")
    assert cfg.spawn.usd_path == original.spawn.usd_path
    assert cfg.actuators["arm"] == original.actuators["arm"]
    assert cfg.init_state.joint_pos["arm_joint"] == 1.25
    assert len(cfg.actuators["hand"].joint_names_expr) == 12
    assert all(cfg.init_state.joint_pos[name] == 0 for name in ns["HAND_JOINT_NAMES"])
    assert original.spawn.func == "freeze_hand"
    assert "hand" not in original.actuators
