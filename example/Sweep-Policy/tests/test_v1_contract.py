"""Static tests that do not require launching Isaac Sim."""

from __future__ import annotations

import ast
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ASSET_ENVIRONMENT_VARIABLES = (
    "SWEEP_POLICY_ASSET_ROOT",
    "INSPIRE_SWEEP_ASSET_ROOT",
    "SWEEP_POLICY_ROBOT_USD_PATH",
    "SWEEP_POLICY_SHELF_USD_PATH",
)


def _load_assets_module(**environment_overrides):
    module_path = REPOSITORY_ROOT / "sweeping_policy" / "src" / "assets.py"
    spec = importlib.util.spec_from_file_location("sweep_policy_assets_contract", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(os.environ, clear=False):
        for name in ASSET_ENVIRONMENT_VARIABLES:
            os.environ.pop(name, None)
        os.environ.update(environment_overrides)
        spec.loader.exec_module(module)
    return module


class AssetPathContractTests(unittest.TestCase):
    def test_default_asset_root_is_the_workspace_shelf(self):
        assets = _load_assets_module()
        expected_root = str(REPOSITORY_ROOT.parents[1] / "assets" / "shelf")
        self.assertEqual(assets.ASSET_ROOT, expected_root)
        self.assertEqual(
            assets.ROBOT_USD_PATH,
            f"{expected_root}/Robots/UR5e/Collected_UR5e_v4/UR5e_v4.usd",
        )
        self.assertEqual(
            assets.SHELF_USD_PATH,
            f"{expected_root}/Arena/Collected_speedrack_shape/speedrack_shape.usd",
        )

    def test_all_object_usds_resolve_below_the_asset_root(self):
        relative_paths = (
            "Objects/Bottle_6/Bottle_6.usd",
            "Objects/Collected_Cup_1/Cup_1.usd",
            "Objects/Collected_Cup_4/Cup_4.usd",
            "Objects/Collected_Mug_2/Mug_2.usd",
            "Objects/Collected_Mug_3/Mug_3.usd",
            "Objects/Can_6/Can_6.usd",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            assets = _load_assets_module(SWEEP_POLICY_ASSET_ROOT=str(root))
            for suffix in relative_paths:
                expected = root / suffix
                with self.assertRaises(FileNotFoundError):
                    assets.object_usd_path(suffix)
                expected.parent.mkdir(parents=True, exist_ok=True)
                expected.touch()
                self.assertEqual(assets.object_usd_path(suffix), str(expected))

    def test_nucleus_environment_overrides_preserve_uri_separators(self):
        assets = _load_assets_module(
            SWEEP_POLICY_ASSET_ROOT="omniverse://nucleus.example/Custom/Shelf/",
            SWEEP_POLICY_ROBOT_USD_PATH="omniverse://nucleus.example/Robots/custom.usd",
        )
        self.assertEqual(assets.ASSET_ROOT, "omniverse://nucleus.example/Custom/Shelf")
        self.assertEqual(
            assets.ROBOT_USD_PATH,
            "omniverse://nucleus.example/Robots/custom.usd",
        )
        self.assertEqual(
            assets.object_usd_path("Objects/item.usd"),
            "omniverse://nucleus.example/Custom/Shelf/Objects/item.usd",
        )


class V1ConfigurationContractTests(unittest.TestCase):
    def test_v1_is_registered_and_uses_relative_6d_pose_ik(self):
        init_source = (
            REPOSITORY_ROOT / "sweeping_policy" / "__init__.py"
        ).read_text(encoding="utf-8")
        cfg_path = (
            REPOSITORY_ROOT
            / "sweeping_policy"
            / "config"
            / "ur5e"
            / "diff_ik_random_sweep_env_cfg.py"
        )
        cfg_source = cfg_path.read_text(encoding="utf-8")
        tree = ast.parse(cfg_source)

        self.assertIn('"Isaac-Sweep-Object-UR5e-Random-v1"', init_source)
        self.assertIn("class UR5eSweepDiffIKEnvCfg(UR5eSweepEnvCfg)", cfg_source)
        self.assertIn('command_type="pose"', cfg_source)
        self.assertIn("use_relative_mode=True", cfg_source)
        self.assertIn('ik_method="dls"', cfg_source)
        action = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                      and isinstance(node.func, ast.Attribute)
                      and node.func.attr == "DifferentialInverseKinematicsActionCfg")
        scale = next(keyword.value for keyword in action.keywords if keyword.arg == "scale")
        self.assertEqual(ast.literal_eval(scale), 0.5)

    def test_v1_preserves_materials_from_the_imported_usds(self):
        cfg_path = (
            REPOSITORY_ROOT
            / "sweeping_policy"
            / "config"
            / "ur5e"
            / "diff_ik_random_sweep_env_cfg.py"
        )
        cfg_source = cfg_path.read_text(encoding="utf-8")

        self.assertNotIn("PreviewSurfaceCfg", cfg_source)
        self.assertNotIn("OBJECT_COLORS", cfg_source)
        self.assertNotIn("visual_material", cfg_source)
        self.assertNotIn(".spawn.visible", cfg_source)


if __name__ == "__main__":
    unittest.main()
