"""Static contracts for the independent single-object sweep task.

These tests intentionally do not import Isaac Lab.  They validate the source
tree, registration metadata, portable asset resolution, and user-facing
documentation with the standard Python runtime only.
"""

from __future__ import annotations

import ast
import builtins
import importlib
import importlib.util
import re
import sys
import unittest
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPOSITORY_ROOT / "single_object_sweep"
TASK_ID = "Isaac-Sweep-Single-Object-UR5e-Random-v0"
DIFF_IK_TASK_ID = "Isaac-Sweep-Single-Object-UR5e-Random-v1"
DIFF_IK_V2_TASK_ID = "Isaac-Sweep-Single-Object-UR5e-Random-v2"
LEGACY_PACKAGE = "sweeping_policy"


def _python_files() -> tuple[Path, ...]:
    return tuple(sorted(PACKAGE_ROOT.rglob("*.py")))


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _static_value(node: ast.AST, symbols: dict[str, Any] | None = None) -> Any:
    """Evaluate the small literal subset used by task registration/configs."""

    symbols = symbols or {}
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id == "__name__":
            return "single_object_sweep"
        if node.id in symbols:
            return symbols[node.id]
        raise ValueError(node.id)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_static_value(node.operand, symbols)
    if isinstance(node, (ast.Tuple, ast.List)):
        values = [_static_value(value, symbols) for value in node.elts]
        return tuple(values) if isinstance(node, ast.Tuple) else values
    if isinstance(node, ast.Dict):
        return {
            _static_value(key, symbols): _static_value(value, symbols)
            for key, value in zip(node.keys, node.values)
            if key is not None
        }
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant):
                parts.append(str(value.value))
            elif isinstance(value, ast.FormattedValue):
                parts.append(str(_static_value(value.value, symbols)))
            else:  # pragma: no cover - defensive guard for future AST nodes
                raise ValueError(ast.dump(value))
        return "".join(parts)
    raise ValueError(ast.dump(node))


def _module_assignments(path: Path) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for node in _tree(path).body:
        target: ast.expr | None = None
        value: ast.expr | None = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        if isinstance(target, ast.Name) and value is not None:
            try:
                values[target.id] = _static_value(value, values)
            except ValueError:
                pass
    return values


def _class_node(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"Class {name!r} was not found")


def _class_calls(path: Path, class_name: str) -> dict[str, ast.Call]:
    result: dict[str, ast.Call] = {}
    class_node = _class_node(_tree(path), class_name)
    for node in class_node.body:
        target: ast.expr | None = None
        value: ast.expr | None = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        if isinstance(target, ast.Name) and isinstance(value, ast.Call):
            result[target.id] = value
    return result


def _keyword(call: ast.Call, name: str) -> ast.expr:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    raise AssertionError(f"Keyword {name!r} is missing from {ast.unparse(call)}")


def _load_assets_module():
    module_path = PACKAGE_ROOT / "assets.py"
    spec = importlib.util.spec_from_file_location(
        "single_object_sweep_assets_contract", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class IndependenceContractTests(unittest.TestCase):
    def test_package_is_a_complete_python_tree(self):
        expected = {
            "__init__.py",
            "agents/__init__.py",
            "agents/rsl_rl_ppo_cfg.py",
            "assets.py",
            "diff_ik_env_cfg.py",
            "diff_ik_v2_env_cfg.py",
            "env.py",
            "env_cfg.py",
            "mdp/__init__.py",
            "mdp/actions.py",
            "mdp/events.py",
            "mdp/observations.py",
            "mdp/rewards.py",
            "mdp/terminations.py",
            "mdp/commands/__init__.py",
            "mdp/commands/commands_cfg.py",
            "mdp/commands/target_goal_command.py",
            "ur5e.py",
        }
        actual = {
            path.relative_to(PACKAGE_ROOT).as_posix() for path in _python_files()
        }
        self.assertTrue(expected.issubset(actual), expected - actual)

    def test_no_source_imports_or_inherits_from_legacy_package(self):
        violations: list[str] = []
        for path in _python_files():
            tree = _tree(path)
            for node in ast.walk(tree):
                imported: list[str] = []
                if isinstance(node, ast.Import):
                    imported = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported = [node.module]
                if any(
                    name == LEGACY_PACKAGE or name.startswith(f"{LEGACY_PACKAGE}.")
                    for name in imported
                ):
                    violations.append(
                        f"{path.relative_to(REPOSITORY_ROOT)}:{node.lineno} import"
                    )
                if isinstance(node, ast.ClassDef):
                    for base in node.bases:
                        if LEGACY_PACKAGE in ast.unparse(base):
                            violations.append(
                                f"{path.relative_to(REPOSITORY_ROOT)}:{node.lineno} base"
                            )
        self.assertEqual(violations, [])

    def test_registration_import_succeeds_when_legacy_package_is_blocked(self):
        original_import = builtins.__import__
        legacy_modules = {
            name: module
            for name, module in tuple(sys.modules.items())
            if name == LEGACY_PACKAGE or name.startswith(f"{LEGACY_PACKAGE}.")
        }

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == LEGACY_PACKAGE or name.startswith(f"{LEGACY_PACKAGE}."):
                raise AssertionError(f"legacy import attempted: {name}")
            return original_import(name, globals, locals, fromlist, level)

        for name in legacy_modules:
            sys.modules.pop(name, None)
        sys.modules.pop("single_object_sweep", None)
        builtins.__import__ = guarded_import
        try:
            package = importlib.import_module("single_object_sweep")
            legacy_loaded = any(
                name == LEGACY_PACKAGE or name.startswith(f"{LEGACY_PACKAGE}.")
                for name in sys.modules
            )
        finally:
            builtins.__import__ = original_import
            sys.modules.update(legacy_modules)

        self.assertEqual(package.TASK_ID, TASK_ID)
        self.assertEqual(package.DIFF_IK_TASK_ID, DIFF_IK_TASK_ID)
        self.assertEqual(package.DIFF_IK_V2_TASK_ID, DIFF_IK_V2_TASK_ID)
        self.assertFalse(legacy_loaded)

    def test_multi_object_and_non_target_collision_symbols_are_absent(self):
        forbidden = {
            "RigidObjectCollection",
            "RigidObjectCollectionCfg",
            "object_collection",
            "object_collision",
            "obj_collision",
            "target_id",
        }
        violations: list[str] = []
        for path in _python_files():
            for node in ast.walk(_tree(path)):
                values: tuple[str, ...] = ()
                if isinstance(node, ast.Name):
                    values = (node.id,)
                elif isinstance(node, ast.Attribute):
                    values = (node.attr,)
                elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    values = (node.name,)
                elif isinstance(node, ast.arg):
                    values = (node.arg,)
                elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                    values = tuple(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", node.value))
                for value in values:
                    if value in forbidden:
                        violations.append(
                            f"{path.relative_to(REPOSITORY_ROOT)}:{getattr(node, 'lineno', 0)}:{value}"
                        )
        self.assertEqual(violations, [])

    def test_dedicated_training_launcher_has_no_legacy_import(self):
        launcher = REPOSITORY_ROOT / "train_single_object.py"
        imports: list[str] = []
        for node in ast.walk(_tree(launcher)):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        self.assertIn("single_object_sweep", imports)
        self.assertFalse(
            any(
                name == LEGACY_PACKAGE or name.startswith(f"{LEGACY_PACKAGE}.")
                for name in imports
            )
        )


class RegistrationContractTests(unittest.TestCase):
    def test_public_api_and_gym_entry_points(self):
        init_path = PACKAGE_ROOT / "__init__.py"
        tree = _tree(init_path)
        symbols = _module_assignments(init_path)
        self.assertEqual(symbols["TASK_ID"], TASK_ID)
        self.assertEqual(symbols["DIFF_IK_TASK_ID"], DIFF_IK_TASK_ID)
        self.assertEqual(symbols["DIFF_IK_V2_TASK_ID"], DIFF_IK_V2_TASK_ID)

        registrations = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "register"
        ]
        self.assertEqual(len(registrations), 3)
        registration_values = {
            values["id"]: values
            for registration in registrations
            for values in [
                {
                    keyword.arg: _static_value(keyword.value, symbols)
                    for keyword in registration.keywords
                    if keyword.arg is not None
                }
            ]
        }
        self.assertEqual(
            set(registration_values),
            {TASK_ID, DIFF_IK_TASK_ID, DIFF_IK_V2_TASK_ID},
        )

        values = registration_values[TASK_ID]
        self.assertEqual(values["id"], TASK_ID)
        self.assertEqual(values["entry_point"], "single_object_sweep.env:SingleObjectSweepEnv")
        self.assertTrue(values["disable_env_checker"])
        self.assertEqual(
            values["kwargs"]["env_cfg_entry_point"],
            "single_object_sweep.env_cfg:UR5eSingleObjectSweepEnvCfg",
        )
        self.assertEqual(
            values["kwargs"]["rsl_rl_cfg_entry_point"],
            "single_object_sweep.agents.rsl_rl_ppo_cfg:UR5eSingleObjectSweepPPORunnerCfg",
        )

        diff_ik_values = registration_values[DIFF_IK_TASK_ID]
        self.assertEqual(
            diff_ik_values["entry_point"],
            "single_object_sweep.env:SingleObjectSweepEnv",
        )
        self.assertTrue(diff_ik_values["disable_env_checker"])
        self.assertEqual(
            diff_ik_values["kwargs"]["env_cfg_entry_point"],
            "single_object_sweep.diff_ik_env_cfg:"
            "UR5eSingleObjectSweepDiffIKEnvCfg",
        )
        self.assertEqual(
            diff_ik_values["kwargs"]["rsl_rl_cfg_entry_point"],
            "single_object_sweep.agents.rsl_rl_ppo_cfg:"
            "UR5eSingleObjectSweepDiffIKPPORunnerCfg",
        )

        diff_ik_v2_values = registration_values[DIFF_IK_V2_TASK_ID]
        self.assertEqual(
            diff_ik_v2_values["entry_point"],
            "single_object_sweep.env:SingleObjectSweepEnv",
        )
        self.assertTrue(diff_ik_v2_values["disable_env_checker"])
        self.assertEqual(
            diff_ik_v2_values["kwargs"]["env_cfg_entry_point"],
            "single_object_sweep.diff_ik_v2_env_cfg:"
            "UR5eSingleObjectSweepDiffIKV2EnvCfg",
        )
        self.assertEqual(
            diff_ik_v2_values["kwargs"]["rsl_rl_cfg_entry_point"],
            "single_object_sweep.agents.rsl_rl_ppo_cfg:"
            "UR5eSingleObjectSweepDiffIKV2PPORunnerCfg",
        )

        exported = set(symbols.get("__all__", ()))
        self.assertTrue(
            {
                "TASK_ID",
                "DIFF_IK_TASK_ID",
                "DIFF_IK_V2_TASK_ID",
                "SingleObjectSweepEnv",
                "UR5eSingleObjectSweepEnvCfg",
                "UR5eSingleObjectSweepDiffIKEnvCfg",
                "UR5eSingleObjectSweepDiffIKV2EnvCfg",
            }
            <= exported
        )


class DifferentialIKV1ContractTests(unittest.TestCase):
    def test_v1_inherits_v0_and_replaces_only_actions(self):
        cfg_path = PACKAGE_ROOT / "diff_ik_env_cfg.py"
        tree = _tree(cfg_path)
        env_class = _class_node(tree, "UR5eSingleObjectSweepDiffIKEnvCfg")

        self.assertEqual(
            [ast.unparse(base) for base in env_class.bases],
            ["UR5eSingleObjectSweepEnvCfg"],
        )
        assignments = [
            node
            for node in env_class.body
            if isinstance(node, (ast.Assign, ast.AnnAssign))
        ]
        self.assertEqual(len(assignments), 1)
        assignment = assignments[0]
        target = (
            assignment.target
            if isinstance(assignment, ast.AnnAssign)
            else assignment.targets[0]
        )
        self.assertIsInstance(target, ast.Name)
        self.assertEqual(target.id, "actions")

    def test_v1_uses_default_pose_absolute_differential_ik(self):
        cfg_source = (PACKAGE_ROOT / "diff_ik_env_cfg.py").read_text(
            encoding="utf-8"
        )
        action_source = (PACKAGE_ROOT / "mdp" / "actions.py").read_text(
            encoding="utf-8"
        )

        required_cfg_fragments = (
            "class DifferentialIKActionsCfg(ActionsCfg)",
            "DefaultPoseDifferentialIKActionCfg(",
            'body_name="robotiq_base_link"',
            "pos=(0.13, 0.0, 0.0)",
            'command_type="pose"',
            "use_relative_mode=False",
            'ik_method="dls"',
            'ik_params={"lambda_val": 0.01}',
            "scale=(0.05, 0.05, 0.05, 0.25, 0.25, 0.25)",
        )
        for fragment in required_cfg_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, cfg_source)

        required_action_fragments = (
            "return 6",
            "self._default_ee_pos = default_pos.clone()",
            "self._default_ee_quat = default_quat.clone()",
            "math_utils.apply_delta_pose(",
            "self._ik_controller.set_command(",
        )
        for fragment in required_action_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, action_source)

    def test_v1_runner_has_an_isolated_experiment_name(self):
        source = (PACKAGE_ROOT / "agents" / "rsl_rl_ppo_cfg.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "class UR5eSingleObjectSweepDiffIKPPORunnerCfg(", source
        )
        self.assertIn(
            'experiment_name = "UR5e_single_object_shelf_sweep_diff_ik_abs"',
            source,
        )


class DifferentialIKV2ContractTests(unittest.TestCase):
    def test_v2_directly_inherits_v0_and_replaces_only_actions(self):
        cfg_path = PACKAGE_ROOT / "diff_ik_v2_env_cfg.py"
        tree = _tree(cfg_path)
        env_class = _class_node(tree, "UR5eSingleObjectSweepDiffIKV2EnvCfg")

        self.assertEqual(
            [ast.unparse(base) for base in env_class.bases],
            ["UR5eSingleObjectSweepEnvCfg"],
        )
        assignments = [
            node
            for node in env_class.body
            if isinstance(node, (ast.Assign, ast.AnnAssign))
        ]
        self.assertEqual(len(assignments), 1)
        assignment = assignments[0]
        target = (
            assignment.target
            if isinstance(assignment, ast.AnnAssign)
            else assignment.targets[0]
        )
        value = assignment.value
        self.assertIsInstance(target, ast.Name)
        self.assertEqual(target.id, "actions")
        self.assertIsInstance(assignment, ast.AnnAssign)
        self.assertEqual(ast.unparse(assignment.annotation), "DifferentialIKActionsCfg")
        self.assertIsInstance(value, ast.Call)
        self.assertEqual(ast.unparse(value.func), "DifferentialIKActionsCfg")

    def test_v2_reuses_the_v1_action_configuration(self):
        cfg_path = PACKAGE_ROOT / "diff_ik_v2_env_cfg.py"
        tree = _tree(cfg_path)
        imports = [
            node
            for node in tree.body
            if isinstance(node, ast.ImportFrom)
            and node.module == "diff_ik_env_cfg"
        ]
        self.assertEqual(len(imports), 1)
        self.assertEqual(
            [alias.name for alias in imports[0].names],
            ["DifferentialIKActionsCfg"],
        )

    def test_v2_runner_has_an_isolated_experiment_name(self):
        source = (PACKAGE_ROOT / "agents" / "rsl_rl_ppo_cfg.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "class UR5eSingleObjectSweepDiffIKV2PPORunnerCfg(", source
        )
        self.assertIn(
            'experiment_name = "UR5e_single_object_shelf_sweep_diff_ik_abs_v2"',
            source,
        )


class AssetAndConfigurationContractTests(unittest.TestCase):
    def test_target_asset_resolves_to_the_shared_external_usd(self):
        assets = _load_assets_module()
        expected_root = REPOSITORY_ROOT.parents[1] / "asset" / "Shelf_USD"
        expected_cup = expected_root / "Objects" / "Collected_Cup_1" / "Cup_1.usd"
        self.assertEqual(assets.ASSET_ROOT, expected_root)
        self.assertEqual(Path(assets.CUP_1_USD_PATH), expected_cup)
        self.assertEqual(
            Path(assets.object_usd_path("Objects/Collected_Cup_1/Cup_1.usd")),
            expected_cup,
        )
        self.assertTrue(expected_cup.is_file())
        self.assertTrue(Path(assets.ROBOT_USD_PATH).is_file())
        self.assertTrue(Path(assets.SHELF_USD_PATH).is_file())

    def test_spawn_and_task_constants_are_exact(self):
        cfg_path = PACKAGE_ROOT / "env_cfg.py"
        constants = _module_assignments(cfg_path)
        self.assertEqual(constants["TARGET_OBJECT_NAME"], "cup_1")
        self.assertEqual(constants["TARGET_OBJECT_WIDTH"], 0.06)
        self.assertEqual(constants["SPAWN_X_RANGE"], (-0.75, -0.60))
        self.assertEqual(constants["SPAWN_Y_RANGE"], (-0.20, 0.20))
        self.assertEqual(constants["SPAWN_Z"], 1.05)
        self.assertEqual(constants["SWEEP_DISTANCE"], 0.18)

    def test_scene_action_observation_and_simulation_source_contract(self):
        cfg_path = PACKAGE_ROOT / "env_cfg.py"
        source = cfg_path.read_text(encoding="utf-8")
        compact = re.sub(r"\s+", "", source)
        tree = _tree(cfg_path)
        _class_node(tree, "UR5eSingleObjectSweepEnvCfg")

        constants = _module_assignments(cfg_path)
        self.assertEqual(len(constants["ARM_JOINT_NAMES"]), 6)
        self.assertEqual(len(constants["GRIPPER_JOINT_NAMES"]), 8)
        self.assertEqual(len(constants["ARM_JOINT_NAMES"]) + 1, 7)

        observations_class = _class_node(tree, "ObservationsCfg")
        policy_class = next(
            node
            for node in observations_class.body
            if isinstance(node, ast.ClassDef) and node.name == "PolicyCfg"
        )
        observation_terms = [
            node.targets[0].id
            for node in policy_class.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Call)
        ]
        self.assertEqual(
            observation_terms,
            [
                "joint_pos",
                "joint_vel",
                "actions",
                "target_obs_state",
                "target_obj_width",
                "ee_pose",
                "goal_pos",
            ],
        )
        self.assertEqual(sum((8, 6, 7, 3, 1, 7, 3)), 35)

        required_fragments = (
            "target_object:RigidObjectCfg",
            'prim_path="{ENV_REGEX_NS}/TargetObject"',
            "usd_path=str(CUP_1_USD_PATH)",
            "num_envs=4096",
            "env_spacing=2.5",
            "scale=0.5",
            "use_default_offset=True",
            "self.decimation=2",
            "self.episode_length_s=10.0",
            "self.sim.dt=0.01",
        )
        for fragment in required_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, compact)

        for function_name in (
            "joint_pos_rel_8",
            "joint_vel_rel_6",
            "last_action",
            "target_position_in_robot_root_frame",
            "target_width",
            "ee_pose_in_robot_root_frame",
            "target_goal_command",
        ):
            self.assertIn(function_name, source)

    def test_reward_weights_and_termination_thresholds(self):
        cfg_path = PACKAGE_ROOT / "env_cfg.py"
        rewards = _class_calls(cfg_path, "RewardsCfg")
        expected_weights = {
            "action_rate": -0.03,
            "joint_vel": -0.03,
            "shelf_collision": -0.5,
            "reaching": 3.0,
            "orientation": 2.0,
            "sweeping_object": 6.0,
            "homing_after_sweep": 9.0,
        }
        self.assertEqual(set(rewards), set(expected_weights))
        for name, expected in expected_weights.items():
            self.assertEqual(_static_value(_keyword(rewards[name], "weight")), expected)

        terminations = _class_calls(cfg_path, "TerminationsCfg")
        self.assertEqual(
            _static_value(_keyword(terminations["object_drop"], "params")),
            {"height_threshold": 1.04, "rotation_threshold": 0.9},
        )
        self.assertEqual(
            _static_value(_keyword(terminations["push_fast"], "params")),
            {"speed_threshold": 0.3},
        )
        self.assertEqual(
            _static_value(_keyword(terminations["shelf_collision"], "params")),
            {"threshold": 0.1},
        )
        self.assertEqual(
            _static_value(_keyword(terminations["hand_velocity"], "params")),
            {"threshold": 1.0},
        )

        compact = re.sub(r"\s+", "", cfg_path.read_text(encoding="utf-8"))
        self.assertRegex(compact, r"curriculum(?::[^=]+)?=None")

    def test_ppo_entry_point_keeps_random_v0_hyperparameters(self):
        ppo_path = PACKAGE_ROOT / "agents" / "rsl_rl_ppo_cfg.py"
        source = ppo_path.read_text(encoding="utf-8")
        compact = re.sub(r"\s+", "", source)
        _class_node(_tree(ppo_path), "UR5eSingleObjectSweepPPORunnerCfg")
        for fragment in (
            "num_steps_per_env=36",
            "max_iterations=90000",
            "save_interval=50",
            "actor_hidden_dims=[256,128,64]",
            "critic_hidden_dims=[256,128,64]",
            "num_learning_epochs=8",
            "num_mini_batches=4",
            "gamma=0.98",
            "lam=0.95",
            "desired_kl=0.02",
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, compact)
        self.assertRegex(source, r'experiment_name\s*=\s*["\'][^"\']*single_object[^"\']*["\']')


class DocumentationContractTests(unittest.TestCase):
    def test_korean_environment_document_covers_the_public_contract(self):
        path = REPOSITORY_ROOT / "docs" / "single_object_sweep_environment.md"
        document = path.read_text(encoding="utf-8")
        required_text = (
            TASK_ID,
            "single_object_sweep",
            "TargetObject",
            "cup_1",
            "35D",
            "7D",
            "4096",
            "0.01",
            "10초",
            "-0.75",
            "-0.60",
            "-0.20",
            "0.20",
            "1.05",
            "±0.18",
            "object_collision",
            "train_single_object.py",
            "--full-env",
        )
        for text in required_text:
            with self.subTest(text=text):
                self.assertIn(text, document)

    def test_v2_document_covers_inheritance_and_action_contract(self):
        path = REPOSITORY_ROOT / "docs" / "single_object_sweep" / "10_diff_ik_v2.md"
        document = path.read_text(encoding="utf-8")
        required_text = (
            DIFF_IK_V2_TASK_ID,
            "UR5eSingleObjectSweepEnvCfg",
            "DifferentialIKActionsCfg",
            "actions",
            "default EE pose",
            "7D",
            "DLS",
            "lambda=0.01",
        )
        for text in required_text:
            with self.subTest(text=text):
                self.assertIn(text, document)


if __name__ == "__main__":
    unittest.main()
