"""Contracts for the privileged-reset Random--Non-Tracking-v1 task."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = (
    REPOSITORY_ROOT
    / "sweeping_policy"
    / "config"
    / "ur5e"
    / "joint_pos_random_non_tracking_v1_sweep_env_cfg.py"
)
EVENTS_PATH = REPOSITORY_ROOT / "sweeping_policy" / "mdp" / "events.py"
REWARDS_PATH = (
    REPOSITORY_ROOT / "sweeping_policy" / "mdp" / "reward_random_sweep.py"
)


def _class_node(tree: ast.Module | ast.ClassDef, name: str) -> ast.ClassDef:
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


class NonTrackingV1ConfigurationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg_tree = ast.parse(
            CFG_PATH.read_text(encoding="utf-8"), filename=str(CFG_PATH)
        )

    def test_environment_directly_inherits_random_v0_and_removes_only_target_terms(self):
        env_class = _class_node(
            self.cfg_tree, "UR5eSweepRandomNonTrackingV1EnvCfg"
        )
        self.assertEqual(
            [ast.unparse(base) for base in env_class.bases],
            ["UR5eSweepEnvCfg"],
        )

        config_assignments = {
            ast.unparse(node.target): ast.unparse(node.value)
            for node in env_class.body
            if isinstance(node, ast.AnnAssign)
        }
        self.assertEqual(config_assignments, {"events": "EventCfg()"})

        post_init = next(
            node
            for node in env_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "__post_init__"
        )
        self.assertEqual(
            [ast.unparse(node) for node in post_init.body],
            [
                "super().__post_init__()",
                "self.observations.policy.target_obs_state = None",
                "self.observations.policy.target_obj_width = None",
                "self.rewards.reaching.weight = 0.0",
            ],
        )

    def test_reaching_ik_is_the_last_declared_reset_event_with_reference_parameters(self):
        event_class = _class_node(self.cfg_tree, "EventCfg")
        self.assertEqual(
            [ast.unparse(base) for base in event_class.bases],
            ["RandomSweepEventCfg"],
        )
        assignment = next(
            node
            for node in event_class.body
            if isinstance(node, ast.Assign)
            and ast.unparse(node.targets[0]) == "initialize_reaching_pose"
        )
        self.assertIsInstance(assignment.value, ast.Call)
        event_call = assignment.value
        assert isinstance(event_call, ast.Call)
        self.assertEqual(ast.unparse(event_call.func), "EventTerm")
        self.assertEqual(
            ast.unparse(_keyword(event_call, "func")),
            "mdp.initialize_robot_at_reaching_pose",
        )
        self.assertEqual(ast.literal_eval(_keyword(event_call, "mode")), "reset")

        params = _keyword(event_call, "params")
        self.assertIsInstance(params, ast.Dict)
        assert isinstance(params, ast.Dict)
        values = {
            ast.literal_eval(key): value
            for key, value in zip(params.keys, params.values)
        }
        self.assertEqual(
            ast.unparse(values["object_collection_cfg"]),
            "SceneEntityCfg('object_collection')",
        )
        self.assertEqual(ast.literal_eval(values["eef_body_name"]), "robotiq_base_link")
        self.assertEqual(ast.literal_eval(values["eef_offset"]), (0.13, 0.0, 0.0))
        self.assertEqual(ast.literal_eval(values["reaching_x_offset"]), -0.02)
        self.assertEqual(ast.literal_eval(values["reaching_z_offset"]), 0.09)
        self.assertEqual(ast.literal_eval(values["position_noise"]), 0.01)
        self.assertEqual(ast.literal_eval(values["max_iterations"]), 40)
        self.assertEqual(ast.literal_eval(values["damping"]), 0.05)
        self.assertEqual(ast.literal_eval(values["step_size"]), 0.5)
        self.assertEqual(ast.literal_eval(values["position_tolerance"]), 0.003)
        self.assertEqual(ast.literal_eval(values["orientation_tolerance"]), 0.001)
        self.assertEqual(
            ast.literal_eval(values["joint_seed_offsets"]),
            (
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                (0.15, -0.10, 0.10, 0.0, 0.0, 0.0),
                (-0.15, 0.10, -0.10, 0.0, 0.0, 0.0),
            ),
        )

    def test_reset_event_uses_privileged_initial_target_and_width_not_policy_observations(self):
        tree = ast.parse(
            EVENTS_PATH.read_text(encoding="utf-8"), filename=str(EVENTS_PATH)
        )
        event_class = _class_node(tree, "initialize_robot_at_reaching_pose")
        source = ast.unparse(event_class)

        self.assertIn(
            "self.object_collection.data.object_link_state_w[env_ids, target_ids, :3].clone()",
            source,
        )
        self.assertIn("env.target_width[env_ids, 0] * direction", source)
        self.assertIn("torch.sign(env.sweep_dir[env_ids, 1])", source)
        self.assertIn("desired_eef_pos[:, 0] += reaching_x_offset", source)
        self.assertIn("desired_eef_pos[:, 2] += reaching_z_offset", source)
        self.assertIn(
            "tcp_pos_error = desired_eef_pos[active_rows] - current_eef_pos",
            source,
        )
        self.assertIn(
            "torch.linalg.norm(tcp_pos_error, dim=-1) < position_tolerance",
            source,
        )
        self.assertNotIn("observation_manager", source)
        self.assertNotIn("command_manager", source)

    def test_non_reaching_rewards_continue_to_read_live_privileged_state(self):
        tree = ast.parse(
            REWARDS_PATH.read_text(encoding="utf-8"), filename=str(REWARDS_PATH)
        )
        pushing = ast.unparse(_function_node(tree, "pushing_target"))
        homing = ast.unparse(_function_node(tree, "homing_reward"))
        collision = ast.unparse(_function_node(tree, "object_collision"))

        self.assertIn("object_collection.data.object_pos_w", pushing)
        self.assertIn("object_collection.data.object_lin_vel_w", pushing)
        self.assertIn("env.target_width", pushing)
        self.assertIn("object_collection.data.object_pos_w", homing)
        self.assertIn("object_collection.data.object_lin_vel_w", collision)
        for source in (pushing, homing, collision):
            self.assertNotIn("observation_manager", source)


if __name__ == "__main__":
    unittest.main()
