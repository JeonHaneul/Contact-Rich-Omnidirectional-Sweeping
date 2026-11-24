"""Static contracts for the multi-object fixed-impedance OSC environment."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = (
    REPOSITORY_ROOT
    / "sweeping_policy"
    / "config"
    / "ur5e"
    / "osc_random_sweep_env_cfg.py"
)


def _literal(node: ast.AST) -> Any:
    return ast.literal_eval(node)


def _keyword(call: ast.Call, name: str) -> ast.expr:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    raise AssertionError(f"Missing keyword {name!r}: {ast.unparse(call)}")


def _class_node(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"Class {name!r} was not found")


class V2OSCConfigurationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tree = ast.parse(CFG_PATH.read_text(encoding="utf-8"), filename=str(CFG_PATH))
        cls.env_class = _class_node(cls.tree, "UR5eSweepOSCEnvCfg")
        cls.post_init = next(
            node
            for node in cls.env_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "__post_init__"
        )

    def test_v2_inherits_random_v0_and_replaces_only_arm_control(self):
        self.assertEqual([ast.unparse(base) for base in self.env_class.bases], ["UR5eSweepEnvCfg"])
        self.assertEqual(
            ast.unparse(self.post_init.body[0]),
            "super().__post_init__()",
        )

        assignment_targets = {
            ast.unparse(node.targets[0]): node.value
            for node in self.post_init.body
            if isinstance(node, ast.Assign) and len(node.targets) == 1
        }
        self.assertEqual(
            set(assignment_targets),
            {
                "self.scene.robot.actuators['arm'].stiffness",
                "self.scene.robot.actuators['arm'].damping",
                "self.actions.arm_action",
            },
        )
        self.assertEqual(
            _literal(assignment_targets['self.scene.robot.actuators[\'arm\'].stiffness']),
            0.0,
        )
        self.assertEqual(
            _literal(assignment_targets['self.scene.robot.actuators[\'arm\'].damping']),
            0.0,
        )

    def test_action_is_fixed_absolute_pose_motion_only_osc(self):
        arm_assignment = next(
            node
            for node in self.post_init.body
            if isinstance(node, ast.Assign)
            and ast.unparse(node.targets[0]) == "self.actions.arm_action"
        )
        self.assertIsInstance(arm_assignment.value, ast.Call)
        arm_call = arm_assignment.value
        assert isinstance(arm_call, ast.Call)
        self.assertEqual(ast.unparse(arm_call.func), "OperationalSpaceControllerActionCfg")
        self.assertEqual(_literal(_keyword(arm_call, "asset_name")), "robot")
        self.assertEqual(ast.unparse(_keyword(arm_call, "joint_names")), "ARM_JOINT_NAMES")
        self.assertEqual(_literal(_keyword(arm_call, "body_name")), "robotiq_base_link")
        self.assertEqual(_literal(_keyword(arm_call, "nullspace_joint_pos_target")), "none")
        self.assertEqual(_literal(_keyword(arm_call, "position_scale")), 1.0)
        self.assertEqual(_literal(_keyword(arm_call, "orientation_scale")), 1.0)

        offset_call = _keyword(arm_call, "body_offset")
        self.assertIsInstance(offset_call, ast.Call)
        assert isinstance(offset_call, ast.Call)
        self.assertEqual(_literal(_keyword(offset_call, "pos")), (0.13, 0.0, 0.0))

        controller_call = _keyword(arm_call, "controller_cfg")
        self.assertIsInstance(controller_call, ast.Call)
        assert isinstance(controller_call, ast.Call)
        self.assertEqual(ast.unparse(controller_call.func), "OperationalSpaceControllerCfg")
        expected = {
            "target_types": ["pose_abs"],
            "impedance_mode": "fixed",
            "motion_control_axes_task": (1, 1, 1, 1, 1, 1),
            "contact_wrench_control_axes_task": (0, 0, 0, 0, 0, 0),
            "inertial_dynamics_decoupling": True,
            "partial_inertial_dynamics_decoupling": False,
            "gravity_compensation": True,
            "motion_stiffness_task": (80.0, 80.0, 80.0, 40.0, 40.0, 40.0),
            "motion_damping_ratio_task": (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            "nullspace_control": "none",
        }
        actual = {
            keyword.arg: _literal(keyword.value)
            for keyword in controller_call.keywords
            if keyword.arg in expected
        }
        self.assertEqual(actual, expected)

    def test_arm_joint_list_contains_only_the_six_ur5e_arm_joints(self):
        assignment = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and ast.unparse(node.targets[0]) == "ARM_JOINT_NAMES"
        )
        self.assertEqual(
            _literal(assignment.value),
            [
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
        )


if __name__ == "__main__":
    unittest.main()
