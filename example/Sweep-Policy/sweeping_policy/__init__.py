"""UR5e multi-object shelf-sweeping tasks for Isaac Lab.

Importing this package registers every environment in :mod:`sweeping_policy`
with Gymnasium.  Heavy Isaac Lab modules remain lazy so task IDs and their
entry-point metadata can be inspected before Isaac Sim is started.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym

# 기본 환경
TASK_ID = "Isaac-Sweep-Object-UR5e-v0"
RANDOM_TASK_ID = "Isaac-Sweep-Object-UR5e-Random-v0"

# 0812
RANDOM_OSC_TASK_ID = "Isaac-Sweep-Object-UR5e-Random-v2"
RANDOM_DIFF_IK_TASK_ID = "Isaac-Sweep-Object-UR5e-Random-v1"

# Diff-IK 운빨설 검증
RANDOM_DIFF_IK_RELATIVE_CONTROL_PARM_TASK_ID_1 = (
    "Isaac-Sweep-Object-UR5e-Random-Diff-IK-Relative-Control-Parm-AS-0.5"
)
RANDOM_DIFF_IK_RELATIVE_CONTROL_PARM_TASK_ID_2 = (
    "Isaac-Sweep-Object-UR5e-Random-Diff-IK-Relative-Control-Parm-AS-0.6"
)
RANDOM_DIFF_IK_RELATIVE_CONTROL_PARM_TASK_ID_3 = (
    "Isaac-Sweep-Object-UR5e-Random-Diff-IK-Relative-Control-Parm-AS-0.7"
)

# 0814
RANDOM_SINGLE_TASK_ID = "Isaac-Sweep-Object-UR5e-Random-Single-v0"  # 순수 단일 물체
RANDOM_SINGLE_INSPIRE_OPEN_TASK_ID = "Isaac-Sweep-Object-UR5e-Random-Single-Inspire-Open-v0"
RANDOM_SINGLE_INSPIRE_HAND_TASK_ID = "Isaac-Sweep-Object-UR5e-Random-Single-Inspire-Open-v1"
RANDOM_SINGLE_INSPIRE_HAND_MASS_TASK_ID = "Isaac-Sweep-Object-UR5e-Random-Single-Inspire-Open-v1-mass"
RANDOM_SINGLE_INSPIRE_HAND_MASS_LIMIT_TASK_ID = "Isaac-Sweep-Object-UR5e-Random-Single-Inspire-Open-v1-mass-limit"
RANDOM_SINGLE_INSPIRE_MASS_SENSORS_TASK_ID = "Isaac-Sweep-Object-UR5e-Random-Single-Inspire-Open-v2-mass"
RANDOM_SINGLE_INSPIRE_MASS_REACHING_TASK_ID = "Isaac-Sweep-Object-UR5e-Random-Single-Inspire-Open-v3-mass"
RANDOM_SINGLE_INSPIRE_SENSORS_TASK_ID = "Isaac-Sweep-Object-UR5e-Random-Single-Inspire-Open-v2"
RANDOM_SINGLE_INSPIRE_INITIAL_MASS_TASK_ID = "Isaac-Sweep-Object-UR5e-Random-Single-Inspire-Open-v3"
RANDOM_SINGLE_INSPIRE_PRIVILEGED_MASS_TASK_ID = "Isaac-Sweep-Object-UR5e-Random-Single-Inspire-Open-v4"
RANDOM_SINGLE_V1_TASK_ID = "Isaac-Sweep-Object-UR5e-Random-Single-v1"  # Base 랜덤화

# 0816
RANDOM_NON_TRACKING_TASK_ID = "Isaac-Sweep-Object-UR5e-Random--Non-Tracking-v0"
RANDOM_NON_TRACKING_V1_TASK_ID = "Isaac-Sweep-Object-UR5e-Random--Non-Tracking-v1"

# 0818
RANDOM_RELATIVE_TARGET_TASK_ID = "Isaac-Sweep-Object-UR5e-Random-Relative-Target-v0"  # Target Goal + Target Position w.r.t EEF Frame
RANDOM_RELATIVE_TARGET_V1_TASK_ID = (
    "Isaac-Sweep-Object-UR5e-Random-Relative-Target-v1"  # + except EEF Pose
)
RANDOM_OSC_RELATIVE_OBS_TASK_ID = "Isaac-Sweep-Object-UR5e-Random-Relative-Obs-v0"
RANDOM_OSC_RELATIVE_OBS_EE_HOMING_TASK_ID = (
    "Isaac-Sweep-Object-UR5e-Random-Relative-Obs-EE-Homing-v0"
)

# 0820
RANDOM_RELATIVE_TARGET_JOINT_SPACE_TASK_ID = (
    "Isaac-Sweep-Object-UR5e-Random-Relative-Target-Joint-Space-v0"  # + except EEF Pose
)

# 0828
RANDOM_RELATIVE_TARGET_JOINT_SPACE_RANDOM_INIT_TASK_ID = (
    "Isaac-Sweep-Object-UR5e-Random-Relative-Target-Joint-Space-Random-Init-v0"
)
RANDOM_EE_HOMING_TASK_ID = "Isaac-Sweep-Object-UR5e-Random-EE-Homing-v0"
HOMING_CARTESIAN_OSC_TASK_ID = "Isaac-Sweep-Object-UR5e-Random-Homing-Cartesian-OSC-v0"
HOMING_OSC_12D_ACTION_TASK_ID = (
    "Isaac-Sweep-Object-UR5e-Random-Homing-OSC-12D-Action-v0"
)

# Single-object, expanded random TCP initialization, fixed-impedance OSC
RANDOM_SINGLE_OSC_OFFSET_TASK_ID = (
    "Isaac-Sweep-Object-UR5e-Random-Single-OSC-Offset-Init-v0"
)

# 기본 환경
if TASK_ID not in gym.registry:
    gym.register(
        id=TASK_ID,
        entry_point="sweeping_policy.env:ShelfSweepEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                "sweeping_policy.config.ur5e.joint_pos_env_cfg:UR5eShelfEnvCfg"
            ),
            "rsl_rl_cfg_entry_point": (
                "sweeping_policy.config.ur5e.agents.rsl_rl_ppo_cfg:"
                "UR5eSweepPPORunnerCfg"
            ),
        },
    )

if RANDOM_TASK_ID not in gym.registry:
    gym.register(
        id=RANDOM_TASK_ID,
        entry_point="sweeping_policy.env:ShelfSweepEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                "sweeping_policy.config.ur5e.joint_pos_random_sweep_env_cfg:"
                "UR5eSweepEnvCfg"
            ),
            "rsl_rl_cfg_entry_point": (
                "sweeping_policy.config.ur5e.agents.rsl_rl_ppo_cfg_02:"
                "UR5eSweepPPORunnerCfg"
            ),
        },
    )

# 0812
if RANDOM_OSC_TASK_ID not in gym.registry:
    gym.register(
        id=RANDOM_OSC_TASK_ID,
        entry_point="sweeping_policy.env:ShelfSweepEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                "sweeping_policy.config.ur5e.osc_random_sweep_env_cfg:"
                "UR5eSweepOSCEnvCfg"
            ),
            "rsl_rl_cfg_entry_point": (
                "sweeping_policy.config.ur5e.agents.rsl_rl_ppo_cfg_02:"
                "UR5eSweepPPORunnerCfg"
            ),
        },
    )

if RANDOM_DIFF_IK_TASK_ID not in gym.registry:
    gym.register(
        id=RANDOM_DIFF_IK_TASK_ID,
        entry_point="sweeping_policy.env:ShelfSweepEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                "sweeping_policy.config.ur5e.diff_ik_random_sweep_env_cfg:"
                "UR5eSweepDiffIKEnvCfg"
            ),
            "rsl_rl_cfg_entry_point": (
                "sweeping_policy.config.ur5e.agents.rsl_rl_ppo_cfg_02:"
                "UR5eSweepPPORunnerCfg"
            ),
        },
    )

# Diff-IK 운빨설 검증
if RANDOM_DIFF_IK_RELATIVE_CONTROL_PARM_TASK_ID_1 not in gym.registry:
    gym.register(
        id=RANDOM_DIFF_IK_RELATIVE_CONTROL_PARM_TASK_ID_1,
        entry_point="sweeping_policy.env:ShelfSweepEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                "sweeping_policy.config.ur5e.diff_ik_relative_target_env_cfg:"
                "UR5eSweepDiffIK5RelativeTargetEnvCfg"
            ),
            "rsl_rl_cfg_entry_point": (
                "sweeping_policy.config.ur5e.agents.rsl_rl_ppo_cfg_02:"
                "UR5eSweepPPORunnerCfg"
            ),
        },
    )

if RANDOM_DIFF_IK_RELATIVE_CONTROL_PARM_TASK_ID_2 not in gym.registry:
    gym.register(
        id=RANDOM_DIFF_IK_RELATIVE_CONTROL_PARM_TASK_ID_2,
        entry_point="sweeping_policy.env:ShelfSweepEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                "sweeping_policy.config.ur5e.diff_ik_relative_target_env_cfg:"
                "UR5eSweepDiffIK6RelativeTargetEnvCfg"
            ),
            "rsl_rl_cfg_entry_point": (
                "sweeping_policy.config.ur5e.agents.rsl_rl_ppo_cfg_02:"
                "UR5eSweepPPORunnerCfg"
            ),
        },
    )

if RANDOM_DIFF_IK_RELATIVE_CONTROL_PARM_TASK_ID_3 not in gym.registry:
    gym.register(
        id=RANDOM_DIFF_IK_RELATIVE_CONTROL_PARM_TASK_ID_3,
        entry_point="sweeping_policy.env:ShelfSweepEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                "sweeping_policy.config.ur5e.diff_ik_relative_target_env_cfg:"
                "UR5eSweepDiffIK7RelativeTargetEnvCfg"
            ),
            "rsl_rl_cfg_entry_point": (
                "sweeping_policy.config.ur5e.agents.rsl_rl_ppo_cfg_02:"
                "UR5eSweepPPORunnerCfg"
            ),
        },
    )

# 0814
if RANDOM_SINGLE_TASK_ID not in gym.registry:
    gym.register(
        id=RANDOM_SINGLE_TASK_ID,
        entry_point="sweeping_policy.env:ShelfSweepEnv",
        disable_env_checker=True,
        kwargs={
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
    )

if RANDOM_SINGLE_INSPIRE_OPEN_TASK_ID not in gym.registry:
    gym.register(
        id=RANDOM_SINGLE_INSPIRE_OPEN_TASK_ID,
        entry_point="sweeping_policy.env:ShelfSweepEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                "sweeping_policy.config.ur5e.joint_pos_random_single_inspire_open_env_cfg:"
                "UR5eSweepRandomSingleInspireOpenEnvCfg"
            ),
            "rsl_rl_cfg_entry_point": (
                "sweeping_policy.config.ur5e.agents.rsl_rl_ppo_inspire_open_cfg:"
                "UR5eSweepInspireOpenPPORunnerCfg"
            ),
        },
    )

if RANDOM_SINGLE_INSPIRE_HAND_TASK_ID not in gym.registry:
    gym.register(
        id=RANDOM_SINGLE_INSPIRE_HAND_TASK_ID,
        entry_point="sweeping_policy.env_inspire_hand:ShelfSweepInspireHandEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                "sweeping_policy.config.ur5e.joint_pos_random_single_inspire_hand_env_cfg:"
                "UR5eSweepRandomSingleInspireHandEnvCfg"
            ),
            "rsl_rl_cfg_entry_point": (
                "sweeping_policy.config.ur5e.agents.rsl_rl_ppo_inspire_hand_cfg:"
                "UR5eSweepInspireHandPPORunnerCfg"
            ),
        },
    )

if RANDOM_SINGLE_INSPIRE_HAND_MASS_TASK_ID not in gym.registry:
    gym.register(
        id=RANDOM_SINGLE_INSPIRE_HAND_MASS_TASK_ID,
        entry_point="sweeping_policy.env_inspire_hand:ShelfSweepInspireHandEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                "sweeping_policy.config.ur5e.joint_pos_random_single_inspire_hand_mass_env_cfg:"
                "UR5eSweepRandomSingleInspireHandMassEnvCfg"
            ),
            "rsl_rl_cfg_entry_point": (
                "sweeping_policy.config.ur5e.agents.rsl_rl_ppo_inspire_hand_mass_cfg:"
                "UR5eSweepInspireHandMassPPORunnerCfg"
            ),
        },
    )

if RANDOM_SINGLE_INSPIRE_HAND_MASS_LIMIT_TASK_ID not in gym.registry:
    gym.register(
        id=RANDOM_SINGLE_INSPIRE_HAND_MASS_LIMIT_TASK_ID,
        entry_point="sweeping_policy.env_inspire_hand:ShelfSweepInspireHandEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                "sweeping_policy.config.ur5e.joint_pos_random_single_inspire_hand_mass_env_cfg:"
                "UR5eSweepRandomSingleInspireHandMassLimitEnvCfg"
            ),
            "rsl_rl_cfg_entry_point": (
                "sweeping_policy.config.ur5e.agents.rsl_rl_ppo_inspire_hand_mass_cfg:"
                "UR5eSweepInspireHandMassLimitPPORunnerCfg"
            ),
        },
    )

for task_id, cfg_module, cfg_class, env_class, runner_class in (
    (
        RANDOM_SINGLE_INSPIRE_MASS_SENSORS_TASK_ID,
        "joint_pos_random_single_inspire_mass_sensors_env_cfg",
        "UR5eSweepRandomSingleInspireMassSensorsEnvCfg",
        "sweeping_policy.env_inspire_sensors:ShelfSweepInspireSensorsEnv",
        "UR5eSweepInspireMassSensorsPPORunnerCfg",
    ),
    (
        RANDOM_SINGLE_INSPIRE_MASS_REACHING_TASK_ID,
        "joint_pos_random_single_inspire_mass_reaching_env_cfg",
        "UR5eSweepRandomSingleInspireMassReachingEnvCfg",
        "sweeping_policy.env_inspire_mass_reaching:ShelfSweepInspireMassReachingEnv",
        "UR5eSweepInspireMassReachingPPORunnerCfg",
    ),
):
    if task_id not in gym.registry:
        gym.register(
            id=task_id, entry_point=env_class, disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"sweeping_policy.config.ur5e.{cfg_module}:{cfg_class}",
                "rsl_rl_cfg_entry_point": (
                    "sweeping_policy.config.ur5e.agents.rsl_rl_ppo_inspire_mass_sensors_cfg:"
                    + runner_class
                ),
            },
        )

for task_id, cfg_class, runner_class in (
    (
        RANDOM_SINGLE_INSPIRE_INITIAL_MASS_TASK_ID,
        "UR5eSweepRandomSingleInspireInitialMassEnvCfg",
        "UR5eSweepInspireInitialMassPPORunnerCfg",
    ),
    (
        RANDOM_SINGLE_INSPIRE_PRIVILEGED_MASS_TASK_ID,
        "UR5eSweepRandomSingleInspirePrivilegedMassEnvCfg",
        "UR5eSweepInspirePrivilegedMassPPORunnerCfg",
    ),
):
    if task_id not in gym.registry:
        gym.register(
            id=task_id,
            entry_point="sweeping_policy.env_inspire_sensors:ShelfSweepInspireSensorsEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": (
                    "sweeping_policy.config.ur5e.joint_pos_random_single_inspire_initial_mass_env_cfg:"
                    + cfg_class
                ),
                "rsl_rl_cfg_entry_point": (
                    "sweeping_policy.config.ur5e.agents.rsl_rl_ppo_inspire_initial_mass_cfg:"
                    + runner_class
                ),
            },
        )

if RANDOM_SINGLE_INSPIRE_SENSORS_TASK_ID not in gym.registry:
    gym.register(
        id=RANDOM_SINGLE_INSPIRE_SENSORS_TASK_ID,
        entry_point="sweeping_policy.env_inspire_sensors:ShelfSweepInspireSensorsEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                "sweeping_policy.config.ur5e.joint_pos_random_single_inspire_sensors_env_cfg:"
                "UR5eSweepRandomSingleInspireSensorsEnvCfg"
            ),
            "rsl_rl_cfg_entry_point": (
                "sweeping_policy.config.ur5e.agents.rsl_rl_ppo_inspire_sensors_cfg:"
                "UR5eSweepInspireSensorsPPORunnerCfg"
            ),
        },
    )

if RANDOM_SINGLE_V1_TASK_ID not in gym.registry:
    gym.register(
        id=RANDOM_SINGLE_V1_TASK_ID,
        entry_point="sweeping_policy.env:ShelfSweepEnv",
        disable_env_checker=True,
        kwargs={
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
    )

# 0816
if RANDOM_NON_TRACKING_TASK_ID not in gym.registry:
    gym.register(
        id=RANDOM_NON_TRACKING_TASK_ID,
        entry_point="sweeping_policy.env:ShelfSweepEnv",
        disable_env_checker=True,
        kwargs={
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
    )

if RANDOM_NON_TRACKING_V1_TASK_ID not in gym.registry:
    gym.register(
        id=RANDOM_NON_TRACKING_V1_TASK_ID,
        entry_point="sweeping_policy.env:ShelfSweepEnv",
        disable_env_checker=True,
        kwargs={
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
    )

# 0818
if RANDOM_RELATIVE_TARGET_TASK_ID not in gym.registry:
    gym.register(
        id=RANDOM_RELATIVE_TARGET_TASK_ID,
        entry_point="sweeping_policy.env:ShelfSweepEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                "sweeping_policy.config.ur5e.diff_ik_relative_target_env_cfg:"
                "UR5eSweepDiffIKRelativeTargetEnvCfg"
            ),
            "rsl_rl_cfg_entry_point": (
                "sweeping_policy.config.ur5e.agents.rsl_rl_ppo_cfg_02:"
                "UR5eSweepPPORunnerCfg"
            ),
        },
    )

if RANDOM_RELATIVE_TARGET_V1_TASK_ID not in gym.registry:
    gym.register(
        id=RANDOM_RELATIVE_TARGET_V1_TASK_ID,
        entry_point="sweeping_policy.env:ShelfSweepEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                "sweeping_policy.config.ur5e.diff_ik_relative_target_env_cfg:"
                "UR5eSweepDiffIKRelativeTargetwoEEFEnvCfg"
            ),
            "rsl_rl_cfg_entry_point": (
                "sweeping_policy.config.ur5e.agents.rsl_rl_ppo_cfg_02:"
                "UR5eSweepPPORunnerCfg"
            ),
        },
    )

if RANDOM_OSC_RELATIVE_OBS_TASK_ID not in gym.registry:
    gym.register(
        id=RANDOM_OSC_RELATIVE_OBS_TASK_ID,
        entry_point="sweeping_policy.env:ShelfSweepEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                "sweeping_policy.config.ur5e.osc_random_sweep_env_cfg:"
                "UR5eSweepOSCRelativeObsEnvCfg"
            ),
            "rsl_rl_cfg_entry_point": (
                "sweeping_policy.config.ur5e.agents.rsl_rl_ppo_cfg_02:"
                "UR5eSweepPPORunnerCfg"
            ),
        },
    )

if RANDOM_OSC_RELATIVE_OBS_EE_HOMING_TASK_ID not in gym.registry:
    gym.register(
        id=RANDOM_OSC_RELATIVE_OBS_EE_HOMING_TASK_ID,
        entry_point="sweeping_policy.env:ShelfSweepEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                "sweeping_policy.config.ur5e.osc_random_sweep_env_cfg:"
                "UR5eSweepOSCRelativeObsEEHomingEnvCfg"
            ),
            "rsl_rl_cfg_entry_point": (
                "sweeping_policy.config.ur5e.agents.rsl_rl_ppo_cfg_02:"
                "UR5eSweepPPORunnerCfg"
            ),
        },
    )

# 0820
if RANDOM_RELATIVE_TARGET_JOINT_SPACE_TASK_ID not in gym.registry:
    gym.register(
        id=RANDOM_RELATIVE_TARGET_JOINT_SPACE_TASK_ID,
        entry_point="sweeping_policy.env:ShelfSweepEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                "sweeping_policy.config.ur5e.joint_pos_random_sweep_env_cfg:"
                "UR5eSweepRelativeTargetEnvCfg"
            ),
            "rsl_rl_cfg_entry_point": (
                "sweeping_policy.config.ur5e.agents.rsl_rl_ppo_cfg_02:"
                "UR5eSweepPPORunnerCfg"
            ),
        },
    )

# 0828
if RANDOM_RELATIVE_TARGET_JOINT_SPACE_RANDOM_INIT_TASK_ID not in gym.registry:
    gym.register(
        id=RANDOM_RELATIVE_TARGET_JOINT_SPACE_RANDOM_INIT_TASK_ID,
        entry_point="sweeping_policy.env:ShelfSweepEnv",
        disable_env_checker=True,
        kwargs={
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
    )

if RANDOM_EE_HOMING_TASK_ID not in gym.registry:
    gym.register(
        id=RANDOM_EE_HOMING_TASK_ID,
        entry_point="sweeping_policy.env:ShelfSweepEnv",
        disable_env_checker=True,
        kwargs={
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
    )

if HOMING_CARTESIAN_OSC_TASK_ID not in gym.registry:
    gym.register(
        id=HOMING_CARTESIAN_OSC_TASK_ID,
        entry_point="sweeping_policy.env:ShelfSweepEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                "sweeping_policy.config.ur5e.osc_random_sweep_env_cfg:"
                "UR5eSweepOSCHomingRewardCartesianEnvCfg"
            ),
            "rsl_rl_cfg_entry_point": (
                "sweeping_policy.config.ur5e.agents.rsl_rl_ppo_cfg_02:"
                "UR5eSweepPPORunnerCfg"
            ),
        },
    )

if HOMING_OSC_12D_ACTION_TASK_ID not in gym.registry:
    gym.register(
        id=HOMING_OSC_12D_ACTION_TASK_ID,
        entry_point="sweeping_policy.env:ShelfSweepEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                "sweeping_policy.config.ur5e.osc_random_sweep_env_cfg:"
                "UR5eSweepKpOSCRelativeObsEnvCfg"
            ),
            "rsl_rl_cfg_entry_point": (
                "sweeping_policy.config.ur5e.agents.rsl_rl_ppo_cfg_02:"
                "UR5eSweepPPORunnerCfg"
            ),
        },
    )

if RANDOM_SINGLE_OSC_OFFSET_TASK_ID not in gym.registry:
    gym.register(
        id=RANDOM_SINGLE_OSC_OFFSET_TASK_ID,
        entry_point="sweeping_policy.env:ShelfSweepEnv",
        disable_env_checker=True,
        kwargs={
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
    )


def __getattr__(name: str) -> Any:
    if name == "UR5eSweepRandomSingleInspireInitialMassEnvCfg":
        from .config.ur5e.joint_pos_random_single_inspire_initial_mass_env_cfg import (
            UR5eSweepRandomSingleInspireInitialMassEnvCfg,
        )
        return UR5eSweepRandomSingleInspireInitialMassEnvCfg
    if name == "UR5eSweepRandomSingleInspirePrivilegedMassEnvCfg":
        from .config.ur5e.joint_pos_random_single_inspire_initial_mass_env_cfg import (
            UR5eSweepRandomSingleInspirePrivilegedMassEnvCfg,
        )
        return UR5eSweepRandomSingleInspirePrivilegedMassEnvCfg
    if name == "UR5eSweepRandomSingleInspireMassSensorsEnvCfg":
        from .config.ur5e.joint_pos_random_single_inspire_mass_sensors_env_cfg import (
            UR5eSweepRandomSingleInspireMassSensorsEnvCfg,
        )
        return UR5eSweepRandomSingleInspireMassSensorsEnvCfg
    if name == "UR5eSweepRandomSingleInspireMassReachingEnvCfg":
        from .config.ur5e.joint_pos_random_single_inspire_mass_reaching_env_cfg import (
            UR5eSweepRandomSingleInspireMassReachingEnvCfg,
        )
        return UR5eSweepRandomSingleInspireMassReachingEnvCfg
    """Lazily expose environment and configuration classes."""

    if name == "ShelfSweepEnv":
        from .env import ShelfSweepEnv

        return ShelfSweepEnv
    if name == "UR5eShelfEnvCfg":
        from .config.ur5e.joint_pos_env_cfg import UR5eShelfEnvCfg

        return UR5eShelfEnvCfg
    if name == "UR5eSweepEnvCfg":
        from .config.ur5e.joint_pos_random_sweep_env_cfg import UR5eSweepEnvCfg

        return UR5eSweepEnvCfg
    if name == "UR5eSweepOSCEnvCfg":
        from .config.ur5e.osc_random_sweep_env_cfg import UR5eSweepOSCEnvCfg

        return UR5eSweepOSCEnvCfg
    if name == "UR5eSweepDiffIKEnvCfg":
        from .config.ur5e.diff_ik_random_sweep_env_cfg import UR5eSweepDiffIKEnvCfg

        return UR5eSweepDiffIKEnvCfg
    if name == "UR5eSweepRandomSingleEnvCfg":
        from .config.ur5e.joint_pos_random_single_sweep_env_cfg import (
            UR5eSweepRandomSingleEnvCfg,
        )

        return UR5eSweepRandomSingleEnvCfg
    if name == "UR5eSweepRandomSingleInspireHandEnvCfg":
        from .config.ur5e.joint_pos_random_single_inspire_hand_env_cfg import (
            UR5eSweepRandomSingleInspireHandEnvCfg,
        )

        return UR5eSweepRandomSingleInspireHandEnvCfg
    if name == "UR5eSweepRandomSingleInspireSensorsEnvCfg":
        from .config.ur5e.joint_pos_random_single_inspire_sensors_env_cfg import (
            UR5eSweepRandomSingleInspireSensorsEnvCfg,
        )

        return UR5eSweepRandomSingleInspireSensorsEnvCfg
    if name == "UR5eSweepRandomSingleInspireHandMassEnvCfg":
        from .config.ur5e.joint_pos_random_single_inspire_hand_mass_env_cfg import (
            UR5eSweepRandomSingleInspireHandMassEnvCfg,
        )

        return UR5eSweepRandomSingleInspireHandMassEnvCfg
    if name == "UR5eSweepRandomSingleInspireHandMassLimitEnvCfg":
        from .config.ur5e.joint_pos_random_single_inspire_hand_mass_env_cfg import (
            UR5eSweepRandomSingleInspireHandMassLimitEnvCfg,
        )
        return UR5eSweepRandomSingleInspireHandMassLimitEnvCfg
    if name == "UR5eSweepRandomSingleV1EnvCfg":
        from .config.ur5e.joint_pos_random_single_v1_sweep_env_cfg import (
            UR5eSweepRandomSingleV1EnvCfg,
        )

        return UR5eSweepRandomSingleV1EnvCfg
    if name == "UR5eSweepRandomSingleInspireOpenEnvCfg":
        from .config.ur5e.joint_pos_random_single_inspire_open_env_cfg import (
            UR5eSweepRandomSingleInspireOpenEnvCfg,
        )

        return UR5eSweepRandomSingleInspireOpenEnvCfg
    if name == "UR5eSweepRandomNonTrackingEnvCfg":
        from .config.ur5e.joint_pos_random_non_tracking_sweep_env_cfg import (
            UR5eSweepRandomNonTrackingEnvCfg,
        )

        return UR5eSweepRandomNonTrackingEnvCfg
    if name == "UR5eSweepRandomNonTrackingV1EnvCfg":
        from .config.ur5e.joint_pos_random_non_tracking_v1_sweep_env_cfg import (
            UR5eSweepRandomNonTrackingV1EnvCfg,
        )

        return UR5eSweepRandomNonTrackingV1EnvCfg
    if name == "UR5eSweepDiffIKRelativeTargetEnvCfg":
        from .config.ur5e.diff_ik_relative_target_env_cfg import (
            UR5eSweepDiffIKRelativeTargetEnvCfg,
        )

        return UR5eSweepDiffIKRelativeTargetEnvCfg
    if name == "UR5eSweepOSCRelativeObsEEHomingEnvCfg":
        from .config.ur5e.osc_random_sweep_env_cfg import (
            UR5eSweepOSCRelativeObsEEHomingEnvCfg,
        )

        return UR5eSweepOSCRelativeObsEEHomingEnvCfg
    if name == "UR5eSweepRelativeTargetRandomInitEnvCfg":
        from .config.ur5e.joint_pos_relative_target_random_init_env_cfg import (
            UR5eSweepRelativeTargetRandomInitEnvCfg,
        )

        return UR5eSweepRelativeTargetRandomInitEnvCfg
    if name == "UR5eSweepEEHomingEnvCfg":
        from .config.ur5e.joint_pos_random_ee_homing_sweep_env_cfg import (
            UR5eSweepEEHomingEnvCfg,
        )

        return UR5eSweepEEHomingEnvCfg
    if name == "UR5eSweepRandomSingleOSCOffsetEnvCfg":
        from .config.ur5e.osc_random_single_offset_env_cfg import (
            UR5eSweepRandomSingleOSCOffsetEnvCfg,
        )

        return UR5eSweepRandomSingleOSCOffsetEnvCfg
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__version__ = "0.1.0"

__all__ = [
    "TASK_ID",
    "RANDOM_TASK_ID",
    "RANDOM_OSC_TASK_ID",
    "RANDOM_DIFF_IK_TASK_ID",
    "RANDOM_SINGLE_TASK_ID",
    "RANDOM_SINGLE_INSPIRE_OPEN_TASK_ID",
    "RANDOM_SINGLE_INSPIRE_HAND_TASK_ID",
    "RANDOM_SINGLE_INSPIRE_HAND_MASS_TASK_ID",
    "RANDOM_SINGLE_INSPIRE_HAND_MASS_LIMIT_TASK_ID",
    "RANDOM_SINGLE_INSPIRE_MASS_SENSORS_TASK_ID",
    "RANDOM_SINGLE_INSPIRE_MASS_REACHING_TASK_ID",
    "RANDOM_SINGLE_INSPIRE_SENSORS_TASK_ID",
    "RANDOM_SINGLE_INSPIRE_INITIAL_MASS_TASK_ID",
    "RANDOM_SINGLE_INSPIRE_PRIVILEGED_MASS_TASK_ID",
    "RANDOM_SINGLE_V1_TASK_ID",
    "RANDOM_NON_TRACKING_TASK_ID",
    "RANDOM_NON_TRACKING_V1_TASK_ID",
    "RANDOM_RELATIVE_TARGET_TASK_ID",
    "RANDOM_OSC_RELATIVE_OBS_EE_HOMING_TASK_ID",
    "RANDOM_RELATIVE_TARGET_JOINT_SPACE_TASK_ID",
    "RANDOM_RELATIVE_TARGET_JOINT_SPACE_RANDOM_INIT_TASK_ID",
    "RANDOM_EE_HOMING_TASK_ID",
    "RANDOM_SINGLE_OSC_OFFSET_TASK_ID",
    "ShelfSweepEnv",
    "UR5eShelfEnvCfg",
    "UR5eSweepEnvCfg",
    "UR5eSweepOSCEnvCfg",
    "UR5eSweepDiffIKEnvCfg",
    "UR5eSweepRandomSingleEnvCfg",
    "UR5eSweepRandomSingleInspireOpenEnvCfg",
    "UR5eSweepRandomSingleInspireHandEnvCfg",
    "UR5eSweepRandomSingleInspireHandMassEnvCfg",
    "UR5eSweepRandomSingleInspireHandMassLimitEnvCfg",
    "UR5eSweepRandomSingleInspireMassSensorsEnvCfg",
    "UR5eSweepRandomSingleInspireMassReachingEnvCfg",
    "UR5eSweepRandomSingleInspireInitialMassEnvCfg",
    "UR5eSweepRandomSingleInspirePrivilegedMassEnvCfg",
    "UR5eSweepRandomSingleInspireSensorsEnvCfg",
    "UR5eSweepRandomSingleV1EnvCfg",
    "UR5eSweepRandomNonTrackingEnvCfg",
    "UR5eSweepRandomNonTrackingV1EnvCfg",
    "UR5eSweepDiffIKRelativeTargetEnvCfg",
    "UR5eSweepOSCRelativeObsEEHomingEnvCfg",
    "UR5eSweepRelativeTargetRandomInitEnvCfg",
    "UR5eSweepEEHomingEnvCfg",
    "UR5eSweepRandomSingleOSCOffsetEnvCfg",
]
