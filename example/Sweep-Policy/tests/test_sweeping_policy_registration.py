"""Registration contracts for the multi-object sweep environments."""

from __future__ import annotations

import builtins
import importlib
import sys
import unittest

import gymnasium as gym


TASK_SPECS = {
    "Isaac-Sweep-Object-UR5e-Random-Single-Inspire-Open-v0": {
        "env_cfg_entry_point": (
            "sweeping_policy.config.ur5e.joint_pos_random_single_inspire_open_env_cfg:"
            "UR5eSweepRandomSingleInspireOpenEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "sweeping_policy.config.ur5e.agents.rsl_rl_ppo_inspire_open_cfg:"
            "UR5eSweepInspireOpenPPORunnerCfg"
        ),
    },
    "Isaac-Sweep-Object-UR5e-v0": {
        "env_cfg_entry_point": (
            "sweeping_policy.config.ur5e.joint_pos_env_cfg:UR5eShelfEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "sweeping_policy.config.ur5e.agents.rsl_rl_ppo_cfg:"
            "UR5eSweepPPORunnerCfg"
        ),
    },
    "Isaac-Sweep-Object-UR5e-Random-v0": {
        "env_cfg_entry_point": (
            "sweeping_policy.config.ur5e.joint_pos_random_sweep_env_cfg:"
            "UR5eSweepEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "sweeping_policy.config.ur5e.agents.rsl_rl_ppo_cfg_02:"
            "UR5eSweepPPORunnerCfg"
        ),
    },
    "Isaac-Sweep-Object-UR5e-Random-EE-Homing-v0": {
        "env_cfg_entry_point": (
            "sweeping_policy.config.ur5e."
            "joint_pos_random_ee_homing_sweep_env_cfg:"
            "UR5eSweepEEHomingEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "sweeping_policy.config.ur5e.agents.rsl_rl_ppo_cfg_02:"
            "UR5eSweepPPORunnerCfg"
        ),
    },
    "Isaac-Sweep-Object-UR5e-Random--Non-Tracking-v0": {
        "env_cfg_entry_point": (
            "sweeping_policy.config.ur5e."
            "joint_pos_random_non_tracking_sweep_env_cfg:"
            "UR5eSweepRandomNonTrackingEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "sweeping_policy.config.ur5e.agents.rsl_rl_ppo_cfg_02:"
            "UR5eSweepPPORunnerCfg"
        ),
    },
    "Isaac-Sweep-Object-UR5e-Random--Non-Tracking-v1": {
        "env_cfg_entry_point": (
            "sweeping_policy.config.ur5e."
            "joint_pos_random_non_tracking_v1_sweep_env_cfg:"
            "UR5eSweepRandomNonTrackingV1EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "sweeping_policy.config.ur5e.agents.rsl_rl_ppo_cfg_02:"
            "UR5eSweepPPORunnerCfg"
        ),
    },
    "Isaac-Sweep-Object-UR5e-Random-Single-v0": {
        "env_cfg_entry_point": (
            "sweeping_policy.config.ur5e."
            "joint_pos_random_single_sweep_env_cfg:"
            "UR5eSweepRandomSingleEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "sweeping_policy.config.ur5e.agents.rsl_rl_ppo_cfg_02:"
            "UR5eSweepPPORunnerCfg"
        ),
    },
    "Isaac-Sweep-Object-UR5e-Random-Single-v1": {
        "env_cfg_entry_point": (
            "sweeping_policy.config.ur5e."
            "joint_pos_random_single_v1_sweep_env_cfg:"
            "UR5eSweepRandomSingleV1EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "sweeping_policy.config.ur5e.agents.rsl_rl_ppo_cfg_02:"
            "UR5eSweepPPORunnerCfg"
        ),
    },
    "Isaac-Sweep-Object-UR5e-Random-Single-OSC-Offset-Init-v0": {
        "env_cfg_entry_point": (
            "sweeping_policy.config.ur5e."
            "osc_random_single_offset_env_cfg:"
            "UR5eSweepRandomSingleOSCOffsetEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "sweeping_policy.config.ur5e.agents.rsl_rl_ppo_cfg_02:"
            "UR5eSweepPPORunnerCfg"
        ),
    },
    "Isaac-Sweep-Object-UR5e-Random-v1": {
        "env_cfg_entry_point": (
            "sweeping_policy.config.ur5e.diff_ik_random_sweep_env_cfg:"
            "UR5eSweepDiffIKEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "sweeping_policy.config.ur5e.agents.rsl_rl_ppo_cfg_02:"
            "UR5eSweepPPORunnerCfg"
        ),
    },
    "Isaac-Sweep-Object-UR5e-Random-v2": {
        "env_cfg_entry_point": (
            "sweeping_policy.config.ur5e.osc_random_sweep_env_cfg:"
            "UR5eSweepOSCEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "sweeping_policy.config.ur5e.agents.rsl_rl_ppo_cfg_02:"
            "UR5eSweepPPORunnerCfg"
        ),
    },
    "Isaac-Sweep-Object-UR5e-Random-Relative-Target-v0": {
        "env_cfg_entry_point": (
            "sweeping_policy.config.ur5e.diff_ik_relative_target_env_cfg:"
            "UR5eSweepDiffIKRelativeTargetEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "sweeping_policy.config.ur5e.agents.rsl_rl_ppo_cfg_02:"
            "UR5eSweepPPORunnerCfg"
        ),
    },
    "Isaac-Sweep-Object-UR5e-Random-Relative-Target-Joint-Space-v0": {
        "env_cfg_entry_point": (
            "sweeping_policy.config.ur5e.joint_pos_random_sweep_env_cfg:"
            "UR5eSweepRelativeTargetEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "sweeping_policy.config.ur5e.agents.rsl_rl_ppo_cfg_02:"
            "UR5eSweepPPORunnerCfg"
        ),
    },
    "Isaac-Sweep-Object-UR5e-Random-Relative-Target-Joint-Space-Random-Init-v0": {
        "env_cfg_entry_point": (
            "sweeping_policy.config.ur5e."
            "joint_pos_relative_target_random_init_env_cfg:"
            "UR5eSweepRelativeTargetRandomInitEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "sweeping_policy.config.ur5e.agents.rsl_rl_ppo_cfg_02:"
            "UR5eSweepPPORunnerCfg"
        ),
    },
}


class SweepingPolicyRegistrationTests(unittest.TestCase):
    def test_inspire_v1_mass_has_independent_config_and_runner(self):
        import sweeping_policy

        spec = gym.spec(sweeping_policy.RANDOM_SINGLE_INSPIRE_HAND_MASS_TASK_ID)
        self.assertEqual(spec.id, "Isaac-Sweep-Object-UR5e-Random-Single-Inspire-Open-v1-mass")
        self.assertEqual(spec.entry_point, "sweeping_policy.env_inspire_hand:ShelfSweepInspireHandEnv")
        self.assertEqual(spec.kwargs["env_cfg_entry_point"],
                         "sweeping_policy.config.ur5e.joint_pos_random_single_inspire_hand_mass_env_cfg:"
                         "UR5eSweepRandomSingleInspireHandMassEnvCfg")
        self.assertEqual(spec.kwargs["rsl_rl_cfg_entry_point"],
                         "sweeping_policy.config.ur5e.agents.rsl_rl_ppo_inspire_hand_mass_cfg:"
                         "UR5eSweepInspireHandMassPPORunnerCfg")

    def test_inspire_v2_registers_its_sensor_environment_and_runner(self):
        import sweeping_policy

        spec = gym.spec(sweeping_policy.RANDOM_SINGLE_INSPIRE_SENSORS_TASK_ID)
        self.assertEqual(spec.id, "Isaac-Sweep-Object-UR5e-Random-Single-Inspire-Open-v2")
        self.assertEqual(spec.entry_point,
                         "sweeping_policy.env_inspire_sensors:ShelfSweepInspireSensorsEnv")
        self.assertEqual(spec.kwargs, {
            "env_cfg_entry_point": (
                "sweeping_policy.config.ur5e.joint_pos_random_single_inspire_sensors_env_cfg:"
                "UR5eSweepRandomSingleInspireSensorsEnvCfg"
            ),
            "rsl_rl_cfg_entry_point": (
                "sweeping_policy.config.ur5e.agents.rsl_rl_ppo_inspire_sensors_cfg:"
                "UR5eSweepInspireSensorsPPORunnerCfg"
            ),
        })

    def test_inspire_v1_has_an_independent_entry_point_and_keeps_v0(self):
        import sweeping_policy

        spec = gym.spec(sweeping_policy.RANDOM_SINGLE_INSPIRE_HAND_TASK_ID)
        self.assertEqual(spec.id, "Isaac-Sweep-Object-UR5e-Random-Single-Inspire-Open-v1")
        self.assertEqual(spec.entry_point, "sweeping_policy.env_inspire_hand:ShelfSweepInspireHandEnv")
        self.assertEqual(spec.kwargs, {
            "env_cfg_entry_point": (
                "sweeping_policy.config.ur5e.joint_pos_random_single_inspire_hand_env_cfg:"
                "UR5eSweepRandomSingleInspireHandEnvCfg"
            ),
            "rsl_rl_cfg_entry_point": (
                "sweeping_policy.config.ur5e.agents.rsl_rl_ppo_inspire_hand_cfg:"
                "UR5eSweepInspireHandPPORunnerCfg"
            ),
        })
        self.assertEqual(gym.spec(sweeping_policy.RANDOM_SINGLE_INSPIRE_OPEN_TASK_ID).entry_point,
                         "sweeping_policy.env:ShelfSweepEnv")

    def test_root_import_registers_all_tasks_without_importing_isaac_lab(self):
        original_import = builtins.__import__
        package_modules = {
            name: module
            for name, module in tuple(sys.modules.items())
            if name == "sweeping_policy" or name.startswith("sweeping_policy.")
        }

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "isaaclab" or name.startswith("isaaclab."):
                raise AssertionError(f"eager Isaac Lab import attempted: {name}")
            return original_import(name, globals, locals, fromlist, level)

        for name in package_modules:
            sys.modules.pop(name, None)
        builtins.__import__ = guarded_import
        try:
            package = importlib.import_module("sweeping_policy")
        finally:
            builtins.__import__ = original_import

        self.assertEqual(package.TASK_ID, "Isaac-Sweep-Object-UR5e-v0")
        self.assertEqual(
            package.RANDOM_TASK_ID, "Isaac-Sweep-Object-UR5e-Random-v0"
        )
        self.assertEqual(
            package.RANDOM_EE_HOMING_TASK_ID,
            "Isaac-Sweep-Object-UR5e-Random-EE-Homing-v0",
        )
        self.assertEqual(
            package.RANDOM_NON_TRACKING_TASK_ID,
            "Isaac-Sweep-Object-UR5e-Random--Non-Tracking-v0",
        )
        self.assertEqual(
            package.RANDOM_NON_TRACKING_V1_TASK_ID,
            "Isaac-Sweep-Object-UR5e-Random--Non-Tracking-v1",
        )
        self.assertEqual(
            package.RANDOM_SINGLE_TASK_ID,
            "Isaac-Sweep-Object-UR5e-Random-Single-v0",
        )
        self.assertEqual(
            package.RANDOM_SINGLE_V1_TASK_ID,
            "Isaac-Sweep-Object-UR5e-Random-Single-v1",
        )
        self.assertEqual(
            package.RANDOM_SINGLE_OSC_OFFSET_TASK_ID,
            "Isaac-Sweep-Object-UR5e-Random-Single-OSC-Offset-Init-v0",
        )
        self.assertEqual(
            package.RANDOM_DIFF_IK_TASK_ID,
            "Isaac-Sweep-Object-UR5e-Random-v1",
        )
        self.assertEqual(
            package.RANDOM_OSC_TASK_ID,
            "Isaac-Sweep-Object-UR5e-Random-v2",
        )
        self.assertEqual(
            package.RANDOM_RELATIVE_TARGET_TASK_ID,
            "Isaac-Sweep-Object-UR5e-Random-Relative-Target-v0",
        )
        self.assertEqual(
            package.RANDOM_RELATIVE_TARGET_JOINT_SPACE_RANDOM_INIT_TASK_ID,
            "Isaac-Sweep-Object-UR5e-Random-Relative-Target-"
            "Joint-Space-Random-Init-v0",
        )

    def test_task_ids_resolve_to_expected_entry_points(self):
        import sweeping_policy  # noqa: F401

        for task_id, expected_kwargs in TASK_SPECS.items():
            with self.subTest(task_id=task_id):
                spec = gym.spec(task_id)
                self.assertEqual(
                    spec.entry_point, "sweeping_policy.env:ShelfSweepEnv"
                )
                self.assertTrue(spec.disable_env_checker)
                self.assertEqual(spec.kwargs, expected_kwargs)

    def test_registration_is_idempotent_and_legacy_id_exports_still_work(self):
        import sweeping_policy

        importlib.reload(sweeping_policy)

        from sweeping_policy.config import ur5e

        self.assertEqual(ur5e.TASK_ID, sweeping_policy.TASK_ID)
        self.assertEqual(ur5e.RANDOM_TASK_ID, sweeping_policy.RANDOM_TASK_ID)
        self.assertEqual(
            ur5e.RANDOM_EE_HOMING_TASK_ID,
            sweeping_policy.RANDOM_EE_HOMING_TASK_ID,
        )
        self.assertEqual(
            ur5e.RANDOM_NON_TRACKING_TASK_ID,
            sweeping_policy.RANDOM_NON_TRACKING_TASK_ID,
        )
        self.assertEqual(
            ur5e.RANDOM_NON_TRACKING_V1_TASK_ID,
            sweeping_policy.RANDOM_NON_TRACKING_V1_TASK_ID,
        )
        self.assertEqual(
            ur5e.RANDOM_SINGLE_TASK_ID,
            sweeping_policy.RANDOM_SINGLE_TASK_ID,
        )
        self.assertEqual(
            ur5e.RANDOM_SINGLE_V1_TASK_ID,
            sweeping_policy.RANDOM_SINGLE_V1_TASK_ID,
        )
        self.assertEqual(
            ur5e.RANDOM_SINGLE_OSC_OFFSET_TASK_ID,
            sweeping_policy.RANDOM_SINGLE_OSC_OFFSET_TASK_ID,
        )
        self.assertEqual(
            ur5e.RANDOM_DIFF_IK_TASK_ID,
            sweeping_policy.RANDOM_DIFF_IK_TASK_ID,
        )
        self.assertEqual(
            ur5e.RANDOM_OSC_TASK_ID,
            sweeping_policy.RANDOM_OSC_TASK_ID,
        )
        self.assertEqual(
            ur5e.RANDOM_RELATIVE_TARGET_TASK_ID,
            sweeping_policy.RANDOM_RELATIVE_TARGET_TASK_ID,
        )
        self.assertEqual(
            ur5e.RANDOM_RELATIVE_TARGET_JOINT_SPACE_RANDOM_INIT_TASK_ID,
            sweeping_policy.RANDOM_RELATIVE_TARGET_JOINT_SPACE_RANDOM_INIT_TASK_ID,
        )

    def test_public_api_matches_single_object_package_shape(self):
        import sweeping_policy

        expected = {
            "TASK_ID",
            "RANDOM_TASK_ID",
            "RANDOM_EE_HOMING_TASK_ID",
            "RANDOM_NON_TRACKING_TASK_ID",
            "RANDOM_NON_TRACKING_V1_TASK_ID",
            "RANDOM_SINGLE_TASK_ID",
            "RANDOM_SINGLE_V1_TASK_ID",
            "RANDOM_SINGLE_OSC_OFFSET_TASK_ID",
            "RANDOM_DIFF_IK_TASK_ID",
            "RANDOM_OSC_TASK_ID",
            "RANDOM_RELATIVE_TARGET_TASK_ID",
            "RANDOM_RELATIVE_TARGET_JOINT_SPACE_TASK_ID",
            "RANDOM_RELATIVE_TARGET_JOINT_SPACE_RANDOM_INIT_TASK_ID",
            "ShelfSweepEnv",
            "UR5eShelfEnvCfg",
            "UR5eSweepEnvCfg",
            "UR5eSweepEEHomingEnvCfg",
            "UR5eSweepRandomNonTrackingEnvCfg",
            "UR5eSweepRandomNonTrackingV1EnvCfg",
            "UR5eSweepRandomSingleEnvCfg",
            "UR5eSweepRandomSingleV1EnvCfg",
            "UR5eSweepRandomSingleOSCOffsetEnvCfg",
            "UR5eSweepDiffIKEnvCfg",
            "UR5eSweepOSCEnvCfg",
            "UR5eSweepDiffIKRelativeTargetEnvCfg",
            "UR5eSweepRelativeTargetRandomInitEnvCfg",
        }
        self.assertTrue(expected <= set(sweeping_policy.__all__))


if __name__ == "__main__":
    unittest.main()
