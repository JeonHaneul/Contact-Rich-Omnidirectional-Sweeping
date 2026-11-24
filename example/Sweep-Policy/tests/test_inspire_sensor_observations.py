"""Exercise physical sensor observation/reset code without launching Isaac Sim."""

import ast
import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO / "src/trash/hand_manipulation_rl"))
from hand_manipulation_rl.sensors import (
    FixedJointWrenchReader, PALM_CHANNEL_NAMES, contact_magnitudes_and_bits,
)

# Use the actual Isaac Lab quaternion utilities, which are simulator independent.
spec = importlib.util.spec_from_file_location(
    "inspire_test_math", REPO / "IsaacLab/source/isaaclab/isaaclab/utils/math.py",
)
math_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(math_utils)


def _functions(path, namespace):
    tree = ast.parse((ROOT / path).read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), namespace)
    return namespace


@pytest.fixture
def observations():
    return _functions("sweeping_policy/mdp/inspire_sensor_observations.py", {
        "torch": torch, "math_utils": math_utils,
        "FixedJointWrenchReader": FixedJointWrenchReader,
        "PALM_CHANNEL_NAMES": PALM_CHANNEL_NAMES,
        "contact_magnitudes_and_bits": contact_magnitudes_and_bits,
    })


def test_tactile_restores_canonical_order_and_masks_reset_rows(observations):
    names = list(reversed(PALM_CHANNEL_NAMES))
    forces = torch.zeros(3, 17, 3)
    for index, name in enumerate(names):
        forces[[0, 2], index, 0] = 0.1 if PALM_CHANNEL_NAMES.index(name) % 2 else 0.0
    forces[1] = float("nan")  # reset row is stale, not a live sample
    env = SimpleNamespace(num_envs=3, device="cpu", sensor_data_fresh=torch.tensor([True, False, True]),
                          scene={"palm_tactile": SimpleNamespace(body_names=names,
                                 data=SimpleNamespace(net_forces_w=forces))})
    result = observations["palm_tactile_bits"](env)
    assert result.shape == (3, 17)
    assert result.dtype == torch.float32
    assert result[0].tolist() == [index % 2 for index in range(17)]
    assert result[1].tolist() == [0] * 17
    assert torch.equal(result[0], result[2])
    forces[0, 0, 0] = float("nan")  # live invalid readings must not silently disappear
    with pytest.raises(RuntimeError, match="non-finite"):
        observations["palm_tactile_bits"](env)


def test_tactile_rejects_wrong_body_set(observations):
    env = SimpleNamespace(num_envs=1, device="cpu", sensor_data_fresh=torch.tensor([True]),
                          scene={"palm_tactile": SimpleNamespace(body_names=["wrong_pad"] * 17)})
    with pytest.raises(RuntimeError, match="17 physical"):
        observations["palm_tactile_bits"](env)


def test_initial_observation_does_not_read_uninitialized_physics(observations):
    env = SimpleNamespace(num_envs=2, device="cpu", sensor_data_fresh=torch.zeros(2, dtype=torch.bool))
    assert torch.equal(observations["palm_tactile_bits"](env), torch.zeros(2, 17))
    assert torch.equal(observations["wrist_wrench_eef"](env), torch.zeros(2, 6))


def test_wrench_selects_child_body_and_transports_to_rotated_eef(observations):
    names = ["wrist_3_link", "inspire_base_link", "axia80_link"]
    reaction = torch.zeros(2, 3, 6)
    reaction[:, 0] = 999.0  # unrelated arm joint must not leak into F/T
    reaction[:, 1, 0] = 1.0
    robot = SimpleNamespace(body_names=names, data=SimpleNamespace(
        body_incoming_joint_wrench_b=reaction,
        body_pos_w=torch.zeros(2, 3, 3),
        body_quat_w=torch.tensor([1., 0., 0., 0.]).expand(2, 3, 4),
    ))
    robot.find_bodies = lambda name: ([names.index(name)], [name])
    # EEF is at world +Y 10 cm, rotated +90 degrees about world Z.
    quat = torch.tensor([2 ** -0.5, 0., 0., 2 ** -0.5])
    frame = SimpleNamespace(data=SimpleNamespace(
        target_pos_w=torch.tensor([0., .1, 0.]).expand(2, 1, 3),
        target_quat_w=quat.expand(2, 1, 4),
    ))
    env = SimpleNamespace(num_envs=2, device="cpu", sensor_data_fresh=torch.tensor([True, False]),
                          scene={"robot": robot, "ee_frame": frame})
    actual = observations["wrist_wrench_eef"](env)
    torch.testing.assert_close(actual[0], torch.tensor([0., -1., 0., 0., 0., .1]), atol=1e-6, rtol=1e-5)
    assert torch.equal(actual[1], torch.zeros(6))
    # Uncancelled vertical force remains present; no self-weight compensation.
    reaction[:, 1, 2] = 9.81
    assert observations["wrist_wrench_eef"](env)[0, 2].item() == pytest.approx(9.81)


def test_v2_partial_reset_keeps_v1_hand_history_and_marks_only_reset_rows_stale():
    class BaseEnv:
        def __init__(self, cfg, **kwargs):
            self.action_manager = SimpleNamespace(_action=torch.zeros(3, 8), _prev_action=torch.zeros(3, 8))

        def _reset_idx(self, env_ids):
            self.action_manager._action[env_ids] = 0.0
            self.action_manager._prev_action[env_ids] = 0.0

        def step(self, action):
            self._reset_idx(torch.tensor([1]))
            return self.sensor_data_fresh.clone()

    ns = _functions("sweeping_policy/env_inspire_hand.py", {"ShelfSweepEnv": BaseEnv})
    ns["torch"] = torch
    _functions("sweeping_policy/env_inspire_sensors.py", ns)
    cfg = SimpleNamespace(scene=SimpleNamespace(num_envs=3), sim=SimpleNamespace(device="cpu"))
    env = ns["ShelfSweepInspireSensorsEnv"](cfg)
    assert not env.sensor_data_fresh.any()
    assert env.step(torch.zeros(3, 8)).tolist() == [True, False, True]
    assert env.action_manager._action[1].tolist() == [0.] * 6 + [1., 1.]
    assert env.action_manager._prev_action[1].tolist() == [0.] * 6 + [1., 1.]
    assert torch.equal(env.action_manager._action[[0, 2]], torch.zeros(2, 8))


def test_sensor_factory_preserves_arm_hand_initial_state_and_spawn_properties():
    class Config(SimpleNamespace):
        def copy(self):
            return copy.deepcopy(self)

    original = Config(spawn=Config(usd_path="source.usd", rigid_props=Config(disable_gravity=False),
                                  articulation_props=Config(solver_position_iteration_count=8)),
                      init_state=Config(joint_pos={"arm": 1.2, "hand": 0.}),
                      actuators={"arm": Config(stiffness=3328.7), "hand": Config(stiffness=1.2)})
    ns = _functions("sweeping_policy/src/ur5e_inspire_sensors.py", {
        "make_movable_inspire_cfg": original.copy, "Ur5eAxia80InspireSpawnerCfg": Config,
    })
    cfg = ns["make_sensor_inspire_cfg"]()
    assert cfg.actuators == original.actuators
    assert cfg.init_state == original.init_state
    assert cfg.spawn == original.spawn
    assert cfg.spawn is not original.spawn


def test_config_inherits_observation_prefix_without_mutating_v1(observations):
    # Run Isaac Lab's actual dataclass decorator without importing Sim-dependent
    # utils/__init__.py. Its dictionary conversion methods are unused here.
    source = REPO / "IsaacLab/source/isaaclab/isaaclab/utils/configclass.py"
    tree = ast.parse(source.read_text())
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.ImportFrom) and node.level
    )]
    ns = {"__name__": __name__, "class_to_dict": lambda obj: vars(obj),
          "update_class_from_dict": lambda obj, values: None}
    exec(compile(tree, str(source), "exec"), ns)
    configclass = ns["configclass"]

    @configclass
    class Group:
        enable_corruption: bool = False
        concatenate_terms: bool = True

    @configclass
    class Term:
        func: object = None
        params: dict = {}
        noise: object = None

    ns = {"__name__": __name__, "configclass": configclass, "ObsGroup": Group, "ObsTerm": Term,
          "MISSING": "unset",
          "Unoise": lambda **kw: SimpleNamespace(**kw),
          "mdp": SimpleNamespace(**{name: lambda env: None for name in (
              "MA_joint_pos_rel", "MA_joint_vel_rel", "last_action", "MA_object_position_in_RRF",
              "MA_object_width", "ee_pos_r", "MA_target_goal_command",
          )})}
    original = ast.parse((ROOT / "sweeping_policy/shelf_sweep_random_env_cfg.py").read_text())
    node = next(n for n in original.body if isinstance(n, ast.ClassDef) and n.name == "ObservationsCfg")
    exec(compile(ast.Module(body=[node], type_ignores=[]), "original_observations", "exec"), ns)

    @configclass
    class ParentCfg:
        observations = ns["ObservationsCfg"]()
        scene = SimpleNamespace()
        rewards = SimpleNamespace(reaching_weight=3., sweeping_weight=6.)

        def __post_init__(self):
            self.observations.policy.goal_pos.params["command_name"] = "target_goal_pos"

    ns.update(UR5eSweepRandomSingleInspireHandEnvCfg=ParentCfg,
              PALM_CHANNEL_NAMES=PALM_CHANNEL_NAMES,
              sensor_obs=SimpleNamespace(**observations),
              ContactSensorCfg=SimpleNamespace,
              make_sensor_inspire_cfg=lambda: SimpleNamespace(replace=lambda **kw: "sensor_robot"))
    _functions("sweeping_policy/config/ur5e/joint_pos_random_single_inspire_sensors_env_cfg.py", ns)
    v1 = ParentCfg()
    v2 = ns["UR5eSweepRandomSingleInspireSensorsEnvCfg"]()
    prefix = [name for name, value in vars(v1.observations.policy).items() if isinstance(value, Term)]
    actual = [name for name, value in vars(v2.observations.policy).items() if isinstance(value, Term)]
    assert actual == prefix + ["tactile", "wrist_wrench"]
    assert v2.rewards == v1.rewards
    assert v2.observations.policy.goal_pos.params == {"command_name": "target_goal_pos"}
    assert v2.scene.robot == "sensor_robot"
    assert not hasattr(v1.scene, "palm_tactile")
    assert not hasattr(v1.observations.policy, "tactile")
