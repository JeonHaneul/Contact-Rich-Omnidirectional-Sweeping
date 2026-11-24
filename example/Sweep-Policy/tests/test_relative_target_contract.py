"""Static contracts for the EEF-relative target/goal observation environment."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class RelativeTargetConfigurationContractTests(unittest.TestCase):
    def test_environment_inherits_v1_and_replaces_target_and_goal_funcs(self):
        cfg_path = (
            REPOSITORY_ROOT
            / "sweeping_policy"
            / "config"
            / "ur5e"
            / "diff_ik_relative_target_env_cfg.py"
        )
        source = cfg_path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        config_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "UR5eSweepDiffIKRelativeTargetEnvCfg"
        )
        self.assertEqual(
            [base.id for base in config_class.bases if isinstance(base, ast.Name)],
            ["UR5eSweepDiffIKEnvCfg"],
        )

        post_init = next(
            node
            for node in config_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "__post_init__"
        )
        self.assertIn(
            "super().__post_init__()",
            [
                ast.unparse(node)
                for node in ast.walk(post_init)
                if isinstance(node, ast.Call)
            ],
        )
        assignments = [
            node for node in ast.walk(post_init) if isinstance(node, ast.Assign)
        ]
        self.assertEqual(
            {
                ast.unparse(assignment.targets[0]): ast.unparse(assignment.value)
                for assignment in assignments
            },
            {
                "self.observations.policy.target_obs_state.func": (
                    "mdp.MA_object_position_in_EEF"
                ),
                "self.observations.policy.goal_pos.func": (
                    "mdp.MA_target_goal_command_in_EEF"
                ),
            },
        )

    def test_observation_uses_eef_world_pose_as_the_reference_transform(self):
        observations_path = (
            REPOSITORY_ROOT / "sweeping_policy" / "mdp" / "observations.py"
        )
        tree = ast.parse(observations_path.read_text(encoding="utf-8"))
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "MA_object_position_in_EEF"
        )
        call = next(
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "subtract_frame_transforms"
        )
        self.assertEqual(
            [ast.unparse(argument) for argument in call.args],
            [
                "ee_frame.data.target_pos_w[..., 0, :]",
                "ee_frame.data.target_quat_w[..., 0, :]",
                "target_pos_w",
            ],
        )

    def test_goal_observation_uses_eef_world_pose_as_the_reference_transform(self):
        observations_path = (
            REPOSITORY_ROOT / "sweeping_policy" / "mdp" / "observations.py"
        )
        tree = ast.parse(observations_path.read_text(encoding="utf-8"))
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "MA_target_goal_command_in_EEF"
        )
        call = next(
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "subtract_frame_transforms"
        )
        self.assertEqual(
            [ast.unparse(argument) for argument in call.args],
            [
                "ee_frame.data.target_pos_w[..., 0, :]",
                "ee_frame.data.target_quat_w[..., 0, :]",
                "goal_pos_w",
            ],
        )
        self.assertIn(
            "env.command_manager.get_command(command_name)",
            [
                ast.unparse(node)
                for node in ast.walk(function)
                if isinstance(node, ast.Call)
            ],
        )

    def test_inherited_eef_is_the_offset_robotiq_frame(self):
        scene_cfg_path = (
            REPOSITORY_ROOT
            / "sweeping_policy"
            / "config"
            / "ur5e"
            / "joint_pos_random_sweep_env_cfg.py"
        )
        tree = ast.parse(scene_cfg_path.read_text(encoding="utf-8"))
        end_effector_frame = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "FrameCfg"
            and any(
                keyword.arg == "name"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value == "end_effector"
                for keyword in node.keywords
            )
        )
        keywords = {keyword.arg: keyword.value for keyword in end_effector_frame.keywords}
        self.assertEqual(
            ast.literal_eval(keywords["prim_path"]),
            "{ENV_REGEX_NS}/Robot/robotiq_base_link",
        )
        offset_call = keywords["offset"]
        self.assertIsInstance(offset_call, ast.Call)
        offset_keywords = {
            keyword.arg: keyword.value for keyword in offset_call.keywords
        }
        self.assertEqual(ast.literal_eval(offset_keywords["pos"]), [0.13, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
