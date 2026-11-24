"""Static contracts for the relative-observation OSC EE-homing environment."""

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
    / "osc_random_sweep_env_cfg.py"
)
PACKAGE_PATH = REPOSITORY_ROOT / "sweeping_policy" / "__init__.py"
CLASS_NAME = "UR5eSweepOSCRelativeObsEEHomingEnvCfg"
TASK_ID_NAME = "RANDOM_OSC_RELATIVE_OBS_EE_HOMING_TASK_ID"
TASK_ID = "Isaac-Sweep-Object-UR5e-Random-Relative-Obs-EE-Homing-v0"


def _class_node(tree: ast.Module, name: str) -> ast.ClassDef:
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == name
    )


def _keyword(call: ast.Call, name: str) -> ast.expr:
    return next(keyword.value for keyword in call.keywords if keyword.arg == name)


class OSCRelativeObsEEHomingContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg_tree = ast.parse(CFG_PATH.read_text(encoding="utf-8"))
        cls.env_class = _class_node(cls.cfg_tree, CLASS_NAME)
        cls.post_init = next(
            node
            for node in cls.env_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "__post_init__"
        )

    def test_inherits_relative_observation_environment(self):
        self.assertEqual(
            [ast.unparse(base) for base in self.env_class.bases],
            ["UR5eSweepOSCRelativeObsEnvCfg"],
        )
        self.assertEqual(ast.unparse(self.post_init.body[0]), "super().__post_init__()")

    def test_declares_fixed_pose_relative_osc_with_stiffness_100(self):
        arm_assignment = next(
            node
            for node in self.post_init.body
            if isinstance(node, ast.Assign)
            and ast.unparse(node.targets[0]) == "self.actions.arm_action"
        )
        arm_call = arm_assignment.value
        self.assertIsInstance(arm_call, ast.Call)
        assert isinstance(arm_call, ast.Call)

        controller_call = _keyword(arm_call, "controller_cfg")
        self.assertIsInstance(controller_call, ast.Call)
        assert isinstance(controller_call, ast.Call)
        self.assertEqual(
            ast.literal_eval(_keyword(controller_call, "target_types")),
            ["pose_rel"],
        )
        self.assertEqual(
            ast.literal_eval(_keyword(controller_call, "impedance_mode")),
            "fixed",
        )
        self.assertEqual(
            ast.literal_eval(_keyword(controller_call, "motion_stiffness_task")),
            (100.0, 100.0, 100.0, 100.0, 100.0, 100.0),
        )

    def test_uses_end_effector_homing_reward_2(self):
        reward_assignment = next(
            node
            for node in self.post_init.body
            if isinstance(node, ast.Assign)
            and ast.unparse(node.targets[0])
            == "self.rewards.homing_after_sweep.func"
        )
        self.assertEqual(
            ast.unparse(reward_assignment.value),
            "mdp.reward_random_sweep.EndEffectorHomingReward2",
        )

    def test_registers_new_task_id_and_config(self):
        package_tree = ast.parse(PACKAGE_PATH.read_text(encoding="utf-8"))
        assignment = next(
            node
            for node in package_tree.body
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == TASK_ID_NAME
        )
        self.assertEqual(ast.literal_eval(assignment.value), TASK_ID)

        registration = next(
            node
            for node in package_tree.body
            if isinstance(node, ast.If)
            and TASK_ID_NAME in ast.unparse(node.test)
        )
        register_call = next(
            node.value
            for node in registration.body
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        )
        kwargs = next(
            keyword.value for keyword in register_call.keywords if keyword.arg == "kwargs"
        )
        values = {
            ast.literal_eval(key): ast.literal_eval(value)
            for key, value in zip(kwargs.keys, kwargs.values)
        }
        self.assertEqual(
            values["env_cfg_entry_point"],
            "sweeping_policy.config.ur5e.osc_random_sweep_env_cfg:"
            + CLASS_NAME,
        )


if __name__ == "__main__":
    unittest.main()
