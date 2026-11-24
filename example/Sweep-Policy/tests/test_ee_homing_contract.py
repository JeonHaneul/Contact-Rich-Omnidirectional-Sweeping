"""Contracts for the Random-EE-Homing-v0 environment."""

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
    / "joint_pos_random_ee_homing_sweep_env_cfg.py"
)
REWARD_PATH = REPOSITORY_ROOT / "sweeping_policy" / "mdp" / "reward_random_sweep.py"


def _class_node(tree: ast.Module, name: str) -> ast.ClassDef:
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == name
    )


class _SceneEntityCfg:
    def __init__(self, name: str):
        self.name = name


class _ManagerTermBase:
    def __init__(self, cfg, env):
        self.cfg = cfg
        self._env = env


class _Scene(dict):
    def __init__(self, num_envs: int, **entities):
        super().__init__(entities)
        self.num_envs = num_envs


class EndEffectorHomingContractTests(unittest.TestCase):
    def test_config_inherits_random_v0_and_only_replaces_homing_function(self):
        tree = ast.parse(CFG_PATH.read_text(encoding="utf-8"), filename=str(CFG_PATH))
        config_class = _class_node(tree, "UR5eSweepEEHomingEnvCfg")
        self.assertEqual(
            [ast.unparse(base) for base in config_class.bases], ["UR5eSweepEnvCfg"]
        )

        post_init = next(
            node
            for node in config_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "__post_init__"
        )
        statements = [ast.unparse(node) for node in post_init.body]
        self.assertEqual(statements[0], "super().__post_init__()")
        self.assertEqual(len(statements), 2)
        self.assertEqual(
            statements[1],
            "self.rewards.homing_after_sweep.func = "
            "mdp.reward_random_sweep.EndEffectorHomingReward",
        )

    def test_reward_uses_ee_position_l2_error_and_preserves_smooth_gate(self):
        tree = ast.parse(
            REWARD_PATH.read_text(encoding="utf-8"), filename=str(REWARD_PATH)
        )
        reward_class = _class_node(tree, "EndEffectorHomingReward")
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__",
                    names=[ast.alias(name="annotations")],
                    level=0,
                ),
                reward_class,
            ],
            type_ignores=[],
        )
        ast.fix_missing_locations(module)

        namespace = {
            "Articulation": object,
            "FrameTransformer": object,
            "ManagerBasedRLEnv": object,
            "ManagerTermBase": _ManagerTermBase,
            "RewTerm": object,
            "RigidObjectCollection": object,
            "SceneEntityCfg": _SceneEntityCfg,
            "torch": torch,
        }
        exec(compile(module, str(REWARD_PATH), "exec"), namespace)

        initial_ee_pos = torch.tensor(
            [[[0.0, 0.0, 0.0]], [[1.0, -1.0, 0.5]]], dtype=torch.float32
        )
        ee = SimpleNamespace(
            data=SimpleNamespace(target_pos_w=initial_ee_pos.clone())
        )
        object_collection = SimpleNamespace(
            data=SimpleNamespace(
                object_pos_w=torch.tensor(
                    [[[0.0, 0.02, 0.0]], [[0.0, 0.10, 0.0]]],
                    dtype=torch.float32,
                )
            )
        )
        command = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32
        )
        env = SimpleNamespace(
            scene=_Scene(2, ee_frame=ee, object_collection=object_collection),
            target_id=torch.zeros((2, 1), dtype=torch.long),
            command_manager=SimpleNamespace(get_command=lambda name: command),
        )
        cfg = SimpleNamespace(params={"command_name": "target_goal_pos"})

        reward_term = namespace["EndEffectorHomingReward"](cfg, env)
        ee.data.target_pos_w += torch.tensor(
            [[[0.03, 0.04, 0.0]], [[0.0, 0.0, 0.12]]]
        )
        actual = reward_term(env, command_name="target_goal_pos")

        ee_error = torch.tensor([0.05, 0.12])
        gate_distance = torch.tensor([0.02, 0.10])
        gate = 0.5 * (1.0 - torch.tanh(100.0 * (gate_distance - 0.03)))
        expected = torch.exp(-0.5 * ee_error) * gate
        self.assertTrue(torch.allclose(actual, expected))


if __name__ == "__main__":
    unittest.main()
