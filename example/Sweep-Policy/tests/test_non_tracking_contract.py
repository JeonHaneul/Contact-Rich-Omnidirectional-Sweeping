"""Contracts for the Random--Non-Tracking-v0 environment."""

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
    / "joint_pos_random_non_tracking_sweep_env_cfg.py"
)
OBSERVATIONS_PATH = (
    REPOSITORY_ROOT / "sweeping_policy" / "mdp" / "observations.py"
)


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


class _SceneEntityCfg:
    def __init__(self, name: str):
        self.name = name


class NonTrackingConfigurationContractTests(unittest.TestCase):
    def test_config_directly_inherits_random_v0_and_only_changes_target_observation(self):
        tree = ast.parse(CFG_PATH.read_text(encoding="utf-8"), filename=str(CFG_PATH))
        config_class = _class_node(tree, "UR5eSweepRandomNonTrackingEnvCfg")
        self.assertEqual(
            [ast.unparse(base) for base in config_class.bases],
            ["UR5eSweepEnvCfg"],
        )

        post_init = next(
            node
            for node in config_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "__post_init__"
        )
        statements = [ast.unparse(node) for node in post_init.body]
        self.assertEqual(statements[0], "super().__post_init__()")
        self.assertIn(
            "self.observations.policy.target_obs_state.func = "
            "mdp.InitialTargetPositionInRRF",
            statements,
        )
        self.assertIn(
            "self.observations.policy.target_obs_state.params = "
            "{'command_name': 'target_goal_pos'}",
            statements,
        )
        self.assertFalse(
            any(
                "observations.policy.goal_pos" in statement
                for statement in statements[1:]
            ),
            msg="goal_pos should remain the inherited reset-time command",
        )
        self.assertFalse(
            any("rewards" in statement for statement in statements[1:]),
            msg="RewardsCfg must remain inherited and use live simulator state",
        )

    def test_target_noise_is_sampled_once_per_reset_as_a_fixed_bias(self):
        tree = ast.parse(CFG_PATH.read_text(encoding="utf-8"), filename=str(CFG_PATH))
        config_class = _class_node(tree, "UR5eSweepRandomNonTrackingEnvCfg")
        post_init = next(
            node
            for node in config_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "__post_init__"
        )
        noise_assignment = next(
            node
            for node in post_init.body
            if isinstance(node, ast.Assign)
            and ast.unparse(node.targets[0])
            == "self.observations.policy.target_obs_state.noise"
        )
        self.assertEqual(
            ast.unparse(noise_assignment.value),
            "NoiseModelWithAdditiveBiasCfg(noise_cfg=ConstantNoiseCfg(bias=0.0), "
            "bias_noise_cfg=UniformNoiseCfg(n_min=-0.01, n_max=0.01, "
            "operation='abs'), "
            "sample_bias_per_component=True)",
        )

    def test_initial_target_function_reads_command_snapshot_not_live_object_state(self):
        tree = ast.parse(
            OBSERVATIONS_PATH.read_text(encoding="utf-8"),
            filename=str(OBSERVATIONS_PATH),
        )
        function = _function_node(tree, "MA_initial_object_position_in_RRF")
        latch_class = _class_node(tree, "InitialTargetPositionInRRF")
        function_source = ast.unparse(function)
        self.assertIn(
            "env.command_manager.get_term(command_name)", function_source
        )
        self.assertIn("command_term.target_init_state_w[..., :3]", function_source)
        self.assertNotIn("object_state_w", function_source)
        self.assertNotIn("object_pos_w", function_source)

        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__",
                    names=[ast.alias(name="annotations")],
                    level=0,
                ),
                function,
                latch_class,
            ],
            type_ignores=[],
        )
        ast.fix_missing_locations(module)

        def subtract_frame_transforms(root_pos, root_quat, target_pos):
            del root_quat
            return target_pos - root_pos, None

        class ManagerTermBase:
            def __init__(self, cfg, env):
                self.cfg = cfg
                self._env = env

        namespace = {
            "MISSING": MISSING,
            "ManagerBasedRLEnv": object,
            "ManagerTermBase": ManagerTermBase,
            "ObsTerm": object,
            "RigidObject": object,
            "SceneEntityCfg": _SceneEntityCfg,
            "subtract_frame_transforms": subtract_frame_transforms,
            "torch": torch,
        }
        exec(compile(module, str(OBSERVATIONS_PATH), "exec"), namespace)

        initial_target_w = torch.tensor(
            [[1.0, 2.0, 3.0], [-3.0, 4.5, 2.0]]
        )
        root_state_w = torch.tensor(
            [
                [0.5, 1.0, 1.5, 1.0, 0.0, 0.0, 0.0],
                [-4.0, 4.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            ]
        )
        command_term = SimpleNamespace(target_init_state_w=initial_target_w)
        command_manager = SimpleNamespace(
            get_term=lambda name: command_term
            if name == "target_goal_pos"
            else None
        )
        robot = SimpleNamespace(data=SimpleNamespace(root_state_w=root_state_w))
        scene = {"robot": robot}
        env = SimpleNamespace(
            scene=scene,
            command_manager=command_manager,
            device="cpu",
            num_envs=2,
        )

        observation = namespace["MA_initial_object_position_in_RRF"](
            env, command_name="target_goal_pos"
        )
        expected = initial_target_w - root_state_w[:, :3]
        self.assertTrue(torch.equal(observation, expected))

        latch = namespace["InitialTargetPositionInRRF"](object(), env)
        first_latched = latch(env, command_name="target_goal_pos")

        moved_target_w = initial_target_w + torch.tensor(
            [[0.0, 0.2, 0.0], [0.0, -0.3, 0.0]]
        )
        command_term.target_init_state_w = moved_target_w
        robot.data.root_state_w[:, :3] += 0.25
        second_latched = latch(env, command_name="target_goal_pos")
        self.assertTrue(torch.equal(second_latched, first_latched))

        latch.reset(torch.tensor([1]))
        after_partial_reset = latch(env, command_name="target_goal_pos")
        self.assertTrue(torch.equal(after_partial_reset[0], first_latched[0]))
        self.assertTrue(
            torch.equal(
                after_partial_reset[1],
                moved_target_w[1] - robot.data.root_state_w[1, :3],
            )
        )


if __name__ == "__main__":
    unittest.main()
