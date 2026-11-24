"""Contracts for the target-only Random-v0 environment."""

from __future__ import annotations

import ast
import unittest
from dataclasses import MISSING
from pathlib import Path
from types import SimpleNamespace

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = (
    REPOSITORY_ROOT
    / "sweeping_policy"
    / "config"
    / "ur5e"
    / "joint_pos_random_single_sweep_env_cfg.py"
)
V1_CFG_PATH = (
    REPOSITORY_ROOT
    / "sweeping_policy"
    / "config"
    / "ur5e"
    / "joint_pos_random_single_v1_sweep_env_cfg.py"
)
EVENTS_PATH = REPOSITORY_ROOT / "sweeping_policy" / "mdp" / "events.py"


def _class_node(tree: ast.Module, name: str) -> ast.ClassDef:
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == name
    )


def _function_node(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _keyword(call: ast.Call, name: str) -> ast.expr:
    return next(keyword.value for keyword in call.keywords if keyword.arg == name)


class _SceneEntityCfg:
    def __init__(self, name: str):
        self.name = name


class _RootPhysxView:
    def __init__(self, count: int):
        self.count = count
        self.gravity_disabled = torch.zeros((count, 1), dtype=torch.uint8)
        self.indices = torch.empty(0, dtype=torch.long)

    def get_disable_gravities(self):
        return self.gravity_disabled

    def set_disable_gravities(self, values, *, indices):
        if values.shape != (self.count, 1):
            raise AssertionError("PhysX gravity data must cover the full view")
        self.gravity_disabled[indices] = values[indices]
        self.indices = indices.clone()


class _ObjectCollection:
    def __init__(self, asset_names: tuple[str, ...]):
        self._object_ids = {name: index for index, name in enumerate(asset_names)}
        self._last_object_name = ""
        self.states: dict[str, torch.Tensor] = {}
        self.num_objects = len(asset_names)
        self.num_instances = 4
        self.root_physx_view = _RootPhysxView(
            self.num_objects * self.num_instances
        )

    def find_objects(self, name_keys: str):
        self._last_object_name = name_keys
        return torch.tensor([self._object_ids[name_keys]]), [name_keys]

    def write_object_link_state_to_sim(self, state, *, env_ids, object_ids):
        del env_ids, object_ids
        self.states[self._last_object_name] = state[:, 0].clone()


class _Scene:
    def __init__(self, object_collection: _ObjectCollection):
        self.env_origins = torch.zeros((object_collection.num_instances, 3))
        self._object_collection = object_collection

    def __getitem__(self, name: str):
        if name != "object_collection":
            raise KeyError(name)
        return self._object_collection


class _RobotScene:
    def __init__(self, root_pos_w: torch.Tensor, env_origins: torch.Tensor):
        self.env_origins = env_origins
        self._robot = SimpleNamespace(
            data=SimpleNamespace(root_pos_w=root_pos_w)
        )

    def __getitem__(self, name: str):
        if name != "robot":
            raise KeyError(name)
        return self._robot


class RandomSingleConfigurationContractTests(unittest.TestCase):
    def test_config_inherits_random_v0_and_only_replaces_the_event_function(self):
        tree = ast.parse(CFG_PATH.read_text(encoding="utf-8"), filename=str(CFG_PATH))
        env_class = _class_node(tree, "UR5eSweepRandomSingleEnvCfg")
        self.assertEqual(
            [ast.unparse(base) for base in env_class.bases],
            ["UR5eSweepEnvCfg"],
        )

        post_init = next(
            node
            for node in env_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "__post_init__"
        )
        self.assertEqual(ast.unparse(post_init.body[0]), "super().__post_init__()")
        self.assertEqual(
            [ast.unparse(node) for node in post_init.body[1:]],
            [
                "self.events.object_spawn.func = "
                "mdp.randomize_single_object_scene"
            ],
        )

    def test_target_only_event_keeps_only_the_target_active_on_the_shelf(self):
        tree = ast.parse(
            EVENTS_PATH.read_text(encoding="utf-8"), filename=str(EVENTS_PATH)
        )
        selected_functions = [
            _function_node(tree, "randomize_scene"),
            _function_node(tree, "randomize_single_object_scene"),
        ]
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__",
                    names=[ast.alias(name="annotations")],
                    level=0,
                ),
                *selected_functions,
            ],
            type_ignores=[],
        )
        ast.fix_missing_locations(module)

        def choose(options):
            return "cup_1" if options and isinstance(options[0], str) else -1.0

        namespace = {
            "MISSING": MISSING,
            "SceneEntityCfg": _SceneEntityCfg,
            "choice": choose,
            "random_yaw_orientation": lambda num, device: torch.tensor(
                [[1.0, 0.0, 0.0, 0.0]], device=device
            ).repeat(num, 1),
            "shuffle": lambda values: None,
            "torch": torch,
        }
        exec(compile(module, str(EVENTS_PATH), "exec"), namespace)

        asset_names = (
            "bottle_1",
            "cup_1",
            "cup_2",
            "mug_1",
            "mug_2",
            "can_1",
        )
        object_collection = _ObjectCollection(asset_names)
        env = SimpleNamespace(
            device="cpu",
            scene=_Scene(object_collection),
            sweep_dir=torch.zeros((4, 3)),
            target_id=torch.zeros((4, 1), dtype=torch.long),
            target_width=torch.zeros((4, 1)),
        )
        env_ids = torch.tensor([1, 3], dtype=torch.long)
        poses = (
            (
                (
                    (-0.6, -0.2, 1.05, 1.0, 0.0, 0.0, 0.0),
                    (-0.6, 0.0, 1.05, 1.0, 0.0, 0.0, 0.0),
                    (-0.6, 0.2, 1.05, 1.0, 0.0, 0.0, 0.0),
                ),
                (
                    (-0.75, -0.2, 1.05, 1.0, 0.0, 0.0, 0.0),
                    (-0.75, 0.0, 1.05, 1.0, 0.0, 0.0, 0.0),
                    (-0.75, 0.2, 1.05, 1.0, 0.0, 0.0, 0.0),
                ),
            ),
        )

        namespace["randomize_single_object_scene"](
            env=env,
            env_ids=env_ids,
            pose_array=poses,
            asset_dict={name: object() for name in asset_names},
            object_width_dict={name: 0.06 for name in asset_names},
            ceiling_height=1.8,
            sweep_dir_options=True,
            task_mode="sweeping_right",
        )

        self.assertEqual(set(object_collection.states), set(asset_names))
        self.assertTrue(
            torch.allclose(
                object_collection.states["cup_1"][:, 2], torch.tensor([1.05, 1.05])
            )
        )
        for name in asset_names:
            self.assertTrue(
                torch.equal(
                    object_collection.states[name][:, 7:], torch.zeros((2, 6))
                )
            )
            if name != "cup_1":
                self.assertTrue(
                    torch.allclose(
                        object_collection.states[name][:, 2],
                        torch.tensor([1.8, 1.8]),
                    ),
                    msg=f"{name} was left in the active shelf area",
                )
        for object_id, name in enumerate(asset_names):
            for env_id in env_ids.tolist():
                view_id = object_id * object_collection.num_instances + env_id
                self.assertEqual(
                    bool(
                        object_collection.root_physx_view.gravity_disabled[
                            view_id, 0
                        ].item()
                    ),
                    name != "cup_1",
                    msg=f"incorrect gravity state for {name} in env {env_id}",
                )
        self.assertTrue(
            torch.equal(env.target_id[env_ids, 0], torch.tensor([1, 1]))
        )
        self.assertTrue(
            torch.allclose(
                env.target_width[env_ids, 0], torch.tensor([0.06, 0.06])
            )
        )
        self.assertTrue(
            torch.allclose(env.sweep_dir[env_ids, 1], torch.tensor([-0.18, -0.18]))
        )


class RandomSingleV1ConfigurationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(
            V1_CFG_PATH.read_text(encoding="utf-8"), filename=str(V1_CFG_PATH)
        )

    def test_v1_directly_inherits_v0_and_adds_only_base_randomization_observation(self):
        env_class = _class_node(self.tree, "UR5eSweepRandomSingleV1EnvCfg")
        self.assertEqual(
            [ast.unparse(base) for base in env_class.bases],
            ["UR5eSweepRandomSingleEnvCfg"],
        )

        config_assignments = {
            ast.unparse(node.target): node
            for node in env_class.body
            if isinstance(node, ast.AnnAssign)
        }
        self.assertEqual(set(config_assignments), {"observations", "events"})
        observations_assignment = config_assignments["observations"]
        self.assertEqual(
            ast.unparse(observations_assignment.annotation), "ObservationsCfg"
        )
        self.assertEqual(
            ast.unparse(observations_assignment.value), "ObservationsCfg()"
        )
        events_assignment = config_assignments["events"]
        self.assertEqual(ast.unparse(events_assignment.annotation), "EventCfg")
        self.assertEqual(ast.unparse(events_assignment.value), "EventCfg()")

        post_init = next(
            node
            for node in env_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "__post_init__"
        )
        self.assertEqual(
            [ast.unparse(node) for node in post_init.body],
            ["super().__post_init__()"],
        )

    def test_v1_declares_base_position_as_the_only_new_policy_term(self):
        observations_class = _class_node(self.tree, "ObservationsCfg")
        self.assertEqual(
            [ast.unparse(base) for base in observations_class.bases],
            ["RandomSweepObservationsCfg"],
        )
        policy_class = _class_node(observations_class, "PolicyCfg")
        self.assertEqual(
            [ast.unparse(base) for base in policy_class.bases],
            ["RandomSweepObservationsCfg.PolicyCfg"],
        )
        policy_terms = [
            node
            for node in policy_class.body
            if isinstance(node, ast.Assign)
        ]
        self.assertEqual(len(policy_terms), 1)
        self.assertEqual(ast.unparse(policy_terms[0].targets[0]), "base_pos")
        self.assertEqual(
            ast.unparse(policy_terms[0].value),
            "ObsTerm(func=robot_base_xy_in_env_frame)",
        )

        policy_assignment = next(
            node
            for node in observations_class.body
            if isinstance(node, ast.AnnAssign)
            and ast.unparse(node.target) == "policy"
        )
        self.assertEqual(ast.unparse(policy_assignment.annotation), "PolicyCfg")
        self.assertEqual(ast.unparse(policy_assignment.value), "PolicyCfg()")

    def test_robot_spawn_randomizes_only_xy_at_the_configured_range(self):
        events_class = _class_node(self.tree, "EventCfg")
        self.assertEqual(
            [ast.unparse(base) for base in events_class.bases],
            ["RandomSweepEventCfg"],
        )
        assignment = next(
            node
            for node in events_class.body
            if isinstance(node, ast.Assign)
            and ast.unparse(node.targets[0]) == "robot_spawn"
        )
        self.assertIsInstance(assignment.value, ast.Call)
        event_call = assignment.value
        assert isinstance(event_call, ast.Call)
        self.assertEqual(ast.unparse(event_call.func), "EventTerm")
        self.assertEqual(
            ast.unparse(_keyword(event_call, "func")),
            "mdp.reset_root_state_uniform",
        )
        self.assertEqual(ast.literal_eval(_keyword(event_call, "mode")), "reset")

        params = _keyword(event_call, "params")
        self.assertIsInstance(params, ast.Dict)
        assert isinstance(params, ast.Dict)
        param_values = {
            ast.literal_eval(key): value
            for key, value in zip(params.keys, params.values)
        }
        pose_range = ast.literal_eval(param_values["pose_range"])
        self.assertEqual(
            pose_range,
            {"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
        )
        self.assertEqual(ast.literal_eval(param_values["velocity_range"]), {})
        self.assertEqual(
            ast.unparse(param_values["asset_cfg"]),
            "SceneEntityCfg('robot')",
        )

    def test_base_observation_is_xy_relative_to_each_environment_origin(self):
        observation_function = _function_node(
            self.tree, "robot_base_xy_in_env_frame"
        )
        module = ast.Module(body=[observation_function], type_ignores=[])
        ast.fix_missing_locations(module)
        namespace = {"SceneEntityCfg": _SceneEntityCfg}
        exec(compile(module, str(V1_CFG_PATH), "exec"), namespace)

        env_origins = torch.tensor(
            [[1.0, 2.0, 0.0], [-4.0, 3.0, 0.0]]
        )
        root_pos_w = torch.tensor(
            [[0.97, 2.01, 0.79505], [-3.95, 2.98, 1.25]]
        )
        env = SimpleNamespace(scene=_RobotScene(root_pos_w, env_origins))

        observation = namespace["robot_base_xy_in_env_frame"](env)

        self.assertEqual(observation.shape, (2, 2))
        self.assertTrue(
            torch.allclose(
                observation,
                torch.tensor([[-0.03, 0.01], [0.05, -0.02]]),
            )
        )


if __name__ == "__main__":
    unittest.main()
