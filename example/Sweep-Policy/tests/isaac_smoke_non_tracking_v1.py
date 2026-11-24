"""Isaac Sim smoke check for Random--Non-Tracking-v1.

Run from the repository root with Isaac Lab's Python::

    ../../IsaacLab/isaaclab.bat -p tests/isaac_smoke_non_tracking_v1.py
    ../../IsaacLab/isaaclab.bat -p tests/isaac_smoke_non_tracking_v1.py --full-env
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from isaacsim import SimulationApp


simulation_app = SimulationApp({"headless": True})

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import sweeping_policy  # noqa: E402
from sweeping_policy.config.ur5e.joint_pos_random_non_tracking_v1_sweep_env_cfg import (  # noqa: E402
    UR5eSweepRandomNonTrackingV1EnvCfg,
)


TASK_ID = "Isaac-Sweep-Object-UR5e-Random--Non-Tracking-v1"


def _check_registration_and_cfg() -> UR5eSweepRandomNonTrackingV1EnvCfg:
    spec = gym.spec(TASK_ID)
    cfg = UR5eSweepRandomNonTrackingV1EnvCfg()

    assert sweeping_policy.RANDOM_NON_TRACKING_V1_TASK_ID == TASK_ID
    assert spec.entry_point == "sweeping_policy.env:ShelfSweepEnv"
    assert spec.kwargs["env_cfg_entry_point"].endswith(
        ":UR5eSweepRandomNonTrackingV1EnvCfg"
    )
    assert cfg.observations.policy.target_obs_state is None
    assert cfg.observations.policy.target_obj_width is None
    assert cfg.rewards.reaching.weight == 0.0
    assert cfg.events.initialize_reaching_pose.mode == "reset"
    assert cfg.events.initialize_reaching_pose.params["eef_offset"] == (
        0.13,
        0.0,
        0.0,
    )

    print(
        "V1|registered=True|policy_obs_dim=31|target_state=False"
        "|target_width=False|reaching_weight=0|reset_ik=True",
        flush=True,
    )
    return cfg


def _check_full_environment(
    cfg: UR5eSweepRandomNonTrackingV1EnvCfg,
    device: str,
) -> None:
    cfg.scene.num_envs = 1
    cfg.sim.device = device
    cfg.observations.policy.enable_corruption = False
    cfg.scene.ee_frame.debug_vis = False
    cfg.scene.finger_frame.debug_vis = False
    cfg.scene.wrist_frame.debug_vis = False
    cfg.commands.target_goal_pos.debug_vis = False
    cfg.scene.ground = None

    env = None
    try:
        env = gym.make(TASK_ID, cfg=cfg)
        observation, _ = env.reset()
        unwrapped = env.unwrapped

        assert unwrapped.action_manager.action_term_dim == [6, 1]
        assert observation["policy"].shape == (1, 31)
        assert unwrapped.observation_manager.active_terms["policy"] == [
            "joint_pos",
            "joint_vel",
            "actions",
            "ee_pose",
            "goal_pos",
        ]
        assert bool(unwrapped.reaching_ik_success.all())
        assert bool(torch.isfinite(unwrapped.target_init_pos_w).all())
        assert bool(torch.isfinite(unwrapped.desired_reaching_pose_w).all())

        zero_action = torch.zeros((1, 7), device=unwrapped.device)
        next_observation, reward, terminated, truncated, _ = env.step(
            zero_action
        )
        assert next_observation["policy"].shape == (1, 31)
        assert bool(torch.isfinite(reward).all())
        assert reward.shape == terminated.shape == truncated.shape == (1,)

        print(
            "ENV|num_envs=1|action_dims=6+1|policy_obs_dim=31"
            "|reset_ik_success=True|step=True",
            flush=True,
        )
    except BaseException as error:
        print(f"ENV_ERROR|{type(error).__name__}|{error}", flush=True)
        print(traceback.format_exc(), flush=True)
        raise
    finally:
        if env is not None:
            env.close()


def main(full_env: bool = False, device: str = "cuda:0") -> None:
    cfg = _check_registration_and_cfg()
    if full_env:
        _check_full_environment(cfg, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-env", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    arguments = parser.parse_args()
    try:
        main(full_env=arguments.full_env, device=arguments.device)
    finally:
        simulation_app.close()
