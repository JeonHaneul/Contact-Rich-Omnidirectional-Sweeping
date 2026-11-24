"""Contracts for the single-object, expanded-reset OSC environment."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = (
    REPOSITORY_ROOT
    / "sweeping_policy"
    / "config"
    / "ur5e"
    / "osc_random_single_offset_env_cfg.py"
)
EVENTS_PATH = REPOSITORY_ROOT / "sweeping_policy" / "mdp" / "events.py"


def _class_node(tree: ast.Module, name: str) -> ast.ClassDef:
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == name
    )


class RandomSingleOSCOffsetContractTests(unittest.TestCase):
    def test_environment_inherits_random_v0_and_applies_requested_overrides(self):
        source = CFG_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(CFG_PATH))
        env_class = _class_node(tree, "UR5eSweepRandomSingleOSCOffsetEnvCfg")
        self.assertEqual(
            [ast.unparse(base) for base in env_class.bases], ["UR5eSweepEnvCfg"]
        )
        post_init = next(
            node
            for node in env_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "__post_init__"
        )
        body = [ast.unparse(node) for node in post_init.body]
        self.assertEqual(body[0], "super().__post_init__()")
        self.assertIn(
            "self.events.object_spawn.func = mdp.randomize_single_object_scene",
            body,
        )
        self.assertIn("self.rewards.homing_after_sweep = None", body)
        self.assertIn('target_types=["pose_rel"]', source)
        self.assertIn("impedance_mode=\"fixed\"", source)
        self.assertIn("gravity_compensation=False", source)
        self.assertIn(
            'self.scene.robot.actuators["arm"].stiffness = 0.0', source
        )
        self.assertIn(
            'self.scene.robot.actuators["arm"].damping = 0.0', source
        )

    def test_event_uses_the_inherited_initializer_subclass(self):
        cfg_source = CFG_PATH.read_text(encoding="utf-8")
        self.assertIn(
            "func=mdp.initialize_robot_in_random_shelf_pose_with_target_offset",
            cfg_source,
        )

        events_tree = ast.parse(EVENTS_PATH.read_text(encoding="utf-8"))
        event_class = _class_node(
            events_tree,
            "initialize_robot_in_random_shelf_pose_with_target_offset",
        )
        self.assertEqual(
            [ast.unparse(base) for base in event_class.bases],
            ["initialize_robot_in_random_shelf_pose"],
        )

    def test_target_side_sampler_is_opposite_sweep_and_respects_shelf_bounds(self):
        tree = ast.parse(EVENTS_PATH.read_text(encoding="utf-8"))
        selected = [
            _class_node(tree, "initialize_robot_in_random_shelf_pose"),
            _class_node(
                tree,
                "initialize_robot_in_random_shelf_pose_with_target_offset",
            ),
        ]
        module = ast.Module(body=selected, type_ignores=[])
        ast.fix_missing_locations(module)
        namespace = {
            "ManagerTermBase": object,
            "ManagerBasedRLEnv": object,
            "EventTermCfg": object,
            "SceneEntityCfg": object,
            "Articulation": object,
            "RigidObjectCollection": object,
            "torch": torch,
        }
        exec(compile(module, str(EVENTS_PATH), "exec"), namespace)
        initializer_class = namespace[
            "initialize_robot_in_random_shelf_pose_with_target_offset"
        ]
        initializer = initializer_class.__new__(initializer_class)
        initializer._target_side_probability = 1.0
        initializer._target_side_offset_range = (0.14, 0.22)
        initializer._target_side_clearance = 0.07
        initializer._target_side_x_tolerance = 0.10

        count = 256
        env_ids = torch.arange(count)
        pose_rows = torch.arange(count)
        target_pos = torch.zeros((count, 3))
        target_pos[:, 0] = -0.65
        target_pos[:, 2] = 1.05
        base_pos = torch.zeros((count, 3))
        x_delta = target_pos[:, 0] - base_pos[:, 0]
        origins = torch.zeros((count, 3))
        sweep_sign = torch.where(env_ids % 2 == 0, 1.0, -1.0)
        env = SimpleNamespace(
            device="cpu",
            sweep_dir=torch.stack(
                (torch.zeros(count), 0.18 * sweep_sign, torch.zeros(count)), dim=1
            ),
            target_width=torch.full((count, 1), 0.06),
        )
        desired = torch.zeros((count, 3))

        torch.manual_seed(11)
        initializer._sample_desired_eef_positions(
            env=env,
            env_ids=env_ids,
            pose_rows=pose_rows,
            desired_eef_pos=desired,
            base_pos_w=base_pos,
            target_pos_w=target_pos,
            x_delta=x_delta,
            available_x=torch.full((count,), 0.12),
            env_origins=origins,
            base_x_clearance=0.38,
            eef_y_range=(-0.25, 0.25),
            eef_z_range=(1.15, 1.32),
        )

        opposite_distance = (target_pos[:, 1] - desired[:, 1]) * sweep_sign
        self.assertTrue(torch.allclose(desired[:, 0], target_pos[:, 0]))
        self.assertTrue(torch.all(opposite_distance >= 0.14))
        self.assertTrue(torch.all(opposite_distance <= 0.22))
        self.assertTrue(torch.all(desired[:, 1] >= -0.25))
        self.assertTrue(torch.all(desired[:, 1] <= 0.25))

        x_distance = torch.abs(x_delta)
        base_distance = (desired[:, 0] - base_pos[:, 0]) * torch.sign(x_delta)
        safe = initializer._is_object_clearance_safe(
            env=env,
            env_ids=env_ids,
            rows=pose_rows,
            current_eef_pos=desired,
            target_pos_w=target_pos,
            base_distance=base_distance,
            x_distance=x_distance,
            x_delta=x_delta,
            object_x_clearance=0.15,
        )
        self.assertTrue(torch.all(safe))

        wrong_side = desired.clone()
        wrong_side[:, 1] = target_pos[:, 1] + sweep_sign * 0.18
        unsafe = initializer._is_object_clearance_safe(
            env=env,
            env_ids=env_ids,
            rows=pose_rows,
            current_eef_pos=wrong_side,
            target_pos_w=target_pos,
            base_distance=base_distance,
            x_distance=x_distance,
            x_delta=x_delta,
            object_x_clearance=0.15,
        )
        self.assertTrue(torch.all(~unsafe))


if __name__ == "__main__":
    unittest.main()
