"""Contracts for the joint-space relative-target random-init environment."""

from __future__ import annotations

import ast
import math
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
    / "joint_pos_relative_target_random_init_env_cfg.py"
)
EVENTS_PATH = REPOSITORY_ROOT / "sweeping_policy" / "mdp" / "events.py"


def _top_level_node(tree: ast.Module, name: str):
    return next(node for node in tree.body if getattr(node, "name", None) == name)


class _SceneEntityCfg:
    def __init__(self, name: str):
        self.name = name


class _RootPhysxView:
    def __init__(self, count: int):
        self.gravity_disabled = torch.zeros((count, 1), dtype=torch.uint8)

    def get_disable_gravities(self):
        return self.gravity_disabled

    def set_disable_gravities(self, values, *, indices):
        self.gravity_disabled[indices] = values[indices]


class _ObjectCollection:
    def __init__(self, names: tuple[str, ...], num_envs: int):
        self._ids = {name: index for index, name in enumerate(names)}
        self.num_objects = len(names)
        self.num_instances = num_envs
        default_state = torch.zeros((num_envs, len(names), 13))
        default_state[..., 3] = 1.0
        self.data = SimpleNamespace(default_object_state=default_state)
        self.root_physx_view = _RootPhysxView(num_envs * len(names))
        self.states = torch.empty(0)

    def find_objects(self, name_keys: str):
        return torch.tensor([self._ids[name_keys]]), [name_keys]

    def write_object_link_state_to_sim(self, states, *, env_ids):
        del env_ids
        self.states = states.clone()


class _Scene:
    def __init__(self, origins: torch.Tensor, objects: _ObjectCollection):
        self.env_origins = origins
        self._objects = objects

    def __getitem__(self, name: str):
        if name != "object_collection":
            raise KeyError(name)
        return self._objects


class RelativeTargetRandomInitContractTests(unittest.TestCase):
    def test_config_inherits_reference_and_changes_only_events(self):
        tree = ast.parse(CFG_PATH.read_text(encoding="utf-8"), filename=str(CFG_PATH))
        env_class = _top_level_node(
            tree, "UR5eSweepRelativeTargetRandomInitEnvCfg"
        )
        self.assertEqual(
            [ast.unparse(base) for base in env_class.bases],
            ["UR5eSweepRelativeTargetEnvCfg"],
        )
        assignments = {
            ast.unparse(node.target): ast.unparse(node.value)
            for node in env_class.body
            if isinstance(node, ast.AnnAssign)
        }
        self.assertEqual(assignments, {"events": "EventCfg()"})

        post_init = next(
            node
            for node in env_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "__post_init__"
        )
        self.assertEqual(
            [ast.unparse(node) for node in post_init.body],
            [
                "super().__post_init__()",
                "self.events.object_spawn.func = "
                "mdp.randomize_single_object_in_work_area",
            ],
        )

    def test_event_bounds_and_fixed_wrist_contract(self):
        source = CFG_PATH.read_text(encoding="utf-8")
        self.assertIn('"base_x_clearance": 0.38', source)
        self.assertIn('"object_x_clearance": 0.15', source)
        self.assertIn('"eef_y_range": (-0.25, 0.25)', source)
        self.assertIn('"eef_z_range": (1.15, 1.32)', source)
        self.assertIn('"max_heading_angle": math.radians(18.0)', source)
        self.assertIn('"heading_limit": math.radians(20.0)', source)
        self.assertIn('"max_pose_attempts": 8', source)
        self.assertIn('"pose_bank_fallback_attempts": 8', source)
        self.assertIn('"pose_bank_capacity_per_cell": 32', source)
        self.assertIn('"locked_joint_name": "wrist_3_joint"', source)
        self.assertNotIn("reset_root_state", source)

        events_source = EVENTS_PATH.read_text(encoding="utf-8")
        self.assertIn(
            "joint_id != self.locked_joint_id",
            events_source,
        )
        self.assertIn(
            "solve_env_ids, self.locked_joint_id",
            events_source,
        )

    def test_failed_ik_resamples_pose_without_fixed_fallback(self):
        source = EVENTS_PATH.read_text(encoding="utf-8")
        event_class = ast.get_source_segment(
            source,
            _top_level_node(
                ast.parse(source), "initialize_robot_in_random_shelf_pose"
            ),
        )
        self.assertIn("for _ in range(max_pose_attempts):", event_class)
        self.assertIn("pose_rows = torch.where(~success)[0]", event_class)
        self.assertIn(
            "desired_heading[pose_rows] = _sample_heading_in_cone(",
            event_class,
        )
        self.assertIn("self._pose_bank: dict[int, torch.Tensor] = {}", event_class)
        self.assertIn("for cell_id in range(16):", event_class)
        self.assertIn("bank_state[\"hard_valid\"]", event_class)
        self.assertNotIn("fallback_joint_pos", event_class)

    def test_convergence_requires_all_hard_constraints(self):
        source = EVENTS_PATH.read_text(encoding="utf-8")
        event_class = ast.get_source_segment(
            source,
            _top_level_node(
                ast.parse(source), "initialize_robot_in_random_shelf_pose"
            ),
        )
        self.assertIn('& state["hard_valid"]', event_class)
        self.assertIn('seed_state["hard_valid"]', event_class)
        for constraint in (
            "target_position_accuracy",
            "target_heading_accuracy",
            "base_clearance",
            "object_clearance",
            "eef_y_range",
            "eef_z_range",
            "heading_limit",
            "wrist_3",
        ):
            self.assertIn(f'\"{constraint}\"', event_class)

    def test_heading_sampler_stays_strictly_inside_the_cone(self):
        tree = ast.parse(EVENTS_PATH.read_text(encoding="utf-8"))
        function = _top_level_node(tree, "_sample_heading_in_cone")
        module = ast.Module(body=[function], type_ignores=[])
        ast.fix_missing_locations(module)
        namespace = {"math": math, "torch": torch}
        exec(compile(module, str(EVENTS_PATH), "exec"), namespace)

        reference = torch.tensor([[1.0, 0.0, 0.0]]).repeat(4096, 1)
        sampled = namespace["_sample_heading_in_cone"](
            reference, math.radians(18.0)
        )
        angles = torch.acos(torch.clamp((reference * sampled).sum(-1), -1.0, 1.0))
        self.assertTrue(torch.all(angles < math.radians(20.0)))
        self.assertTrue(torch.allclose(torch.linalg.norm(sampled, dim=-1), torch.ones(4096)))

    def test_object_event_uses_continuous_legacy_bounds_and_one_target(self):
        tree = ast.parse(EVENTS_PATH.read_text(encoding="utf-8"))
        function = _top_level_node(tree, "randomize_single_object_in_work_area")
        module = ast.Module(body=[function], type_ignores=[])
        ast.fix_missing_locations(module)
        namespace = {
            "MISSING": MISSING,
            "ManagerBasedRLEnv": object,
            "RigidObjectCollection": object,
            "SceneEntityCfg": _SceneEntityCfg,
            "random_yaw_orientation": lambda num, device: torch.tensor(
                [[1.0, 0.0, 0.0, 0.0]], device=device
            ).repeat(num, 1),
            "torch": torch,
        }
        exec(compile(module, str(EVENTS_PATH), "exec"), namespace)

        names = ("bottle", "cup_1", "cup_2", "mug_1", "mug_2", "can")
        num_envs = 512
        objects = _ObjectCollection(names, num_envs)
        origins = torch.zeros((num_envs, 3))
        origins[:, 0] = torch.arange(num_envs) * 2.5
        env = SimpleNamespace(
            device="cpu",
            scene=_Scene(origins, objects),
            target_id=torch.zeros((num_envs, 1), dtype=torch.long),
            target_width=torch.zeros((num_envs, 1)),
            sweep_dir=torch.zeros((num_envs, 3)),
        )
        poses = (
            (
                (
                    (-0.75, -0.20, 1.05, 1.0, 0.0, 0.0, 0.0),
                    (-0.75, 0.00, 1.05, 1.0, 0.0, 0.0, 0.0),
                    (-0.75, 0.20, 1.05, 1.0, 0.0, 0.0, 0.0),
                ),
                (
                    (-0.60, -0.20, 1.05, 1.0, 0.0, 0.0, 0.0),
                    (-0.60, 0.00, 1.05, 1.0, 0.0, 0.0, 0.0),
                    (-0.60, 0.20, 1.05, 1.0, 0.0, 0.0, 0.0),
                ),
            ),
        )
        widths = {name: 0.05 + 0.01 * index for index, name in enumerate(names)}
        torch.manual_seed(7)
        namespace["randomize_single_object_in_work_area"](
            env=env,
            env_ids=torch.arange(num_envs),
            pose_array=poses,
            asset_dict={name: object() for name in names},
            object_width_dict=widths,
            ceiling_height=1.8,
            sweep_dir_options=True,
            task_mode="sweeping_right",
        )

        local_positions = objects.states[..., :3] - origins.unsqueeze(1)
        rows = torch.arange(num_envs)
        target_ids = env.target_id[:, 0]
        target_pos = local_positions[rows, target_ids]
        self.assertTrue(torch.all((-0.77 <= target_pos[:, 0]) & (target_pos[:, 0] <= -0.58)))
        self.assertTrue(torch.all((-0.22 <= target_pos[:, 1]) & (target_pos[:, 1] <= 0.22)))
        self.assertTrue(torch.allclose(target_pos[:, 2], torch.full((num_envs,), 1.05)))
        self.assertTrue(torch.all((local_positions[..., 2] == 1.05).sum(dim=1) == 1))
        self.assertTrue(torch.all((local_positions[..., 2] == 1.8).sum(dim=1) == 5))

        all_object_ids = torch.arange(len(names)).unsqueeze(1)
        selected = all_object_ids == target_ids.unsqueeze(0)
        gravity = objects.root_physx_view.gravity_disabled.reshape(len(names), num_envs)
        self.assertTrue(torch.equal(gravity.bool(), ~selected))
        expected_widths = torch.tensor([widths[name] for name in names])
        self.assertTrue(torch.allclose(env.target_width[:, 0], expected_widths[target_ids]))
        self.assertTrue(torch.all(torch.abs(env.sweep_dir[:, 1]) == 0.18))


if __name__ == "__main__":
    unittest.main()
