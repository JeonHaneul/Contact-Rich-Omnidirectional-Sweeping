"""Isaac Sim smoke check for the single-object absolute Diff-IK task."""

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

import single_object_sweep  # noqa: E402, F401
from single_object_sweep import DIFF_IK_TASK_ID  # noqa: E402
from single_object_sweep.diff_ik_env_cfg import (  # noqa: E402
    UR5eSingleObjectSweepDiffIKEnvCfg,
)


def _check_registration_and_cfg() -> UR5eSingleObjectSweepDiffIKEnvCfg:
    spec = gym.spec(DIFF_IK_TASK_ID)
    cfg = UR5eSingleObjectSweepDiffIKEnvCfg()
    action = cfg.actions.arm_action

    assert DIFF_IK_TASK_ID == "Isaac-Sweep-Single-Object-UR5e-Random-v1"
    assert spec.entry_point == "single_object_sweep.env:SingleObjectSweepEnv"
    assert (
        spec.kwargs["env_cfg_entry_point"]
        == "single_object_sweep.diff_ik_env_cfg:"
        "UR5eSingleObjectSweepDiffIKEnvCfg"
    )
    assert action.class_type.__name__ == "DefaultPoseDifferentialIKAction"
    assert action.body_name == "robotiq_base_link"
    assert tuple(action.body_offset.pos) == (0.13, 0.0, 0.0)
    assert action.controller.command_type == "pose"
    assert action.controller.use_relative_mode is False
    assert action.controller.ik_method == "dls"
    assert action.controller.ik_params == {"lambda_val": 0.01}
    assert tuple(action.scale) == (0.05, 0.05, 0.05, 0.25, 0.25, 0.25)

    print(
        "V1|registered=True|controller=dls|relative=False"
        "|reference=default_ee_pose|arm_dim=6|gripper_dim=1",
        flush=True,
    )
    return cfg


def _check_full_environment(
    cfg: UR5eSingleObjectSweepDiffIKEnvCfg,
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
        env = gym.make(DIFF_IK_TASK_ID, cfg=cfg)
        observation, _ = env.reset()
        unwrapped = env.unwrapped
        arm_action = unwrapped.action_manager.get_term("arm_action")

        assert unwrapped.action_manager.action_term_dim == [6, 1]
        assert observation["policy"].shape == (1, 35)

        zero_action = torch.zeros((1, 7), device=unwrapped.device)
        next_observation, reward, terminated, truncated, _ = env.step(zero_action)

        assert torch.allclose(
            arm_action._ik_controller.ee_pos_des,
            arm_action._default_ee_pos,
            atol=1.0e-6,
            rtol=1.0e-6,
        )
        assert torch.allclose(
            arm_action._ik_controller.ee_quat_des,
            arm_action._default_ee_quat,
            atol=1.0e-6,
            rtol=1.0e-6,
        )
        assert next_observation["policy"].shape == (1, 35)
        assert bool(torch.isfinite(reward).all())
        assert reward.shape == terminated.shape == truncated.shape == (1,)

        print(
            "ENV|num_envs=1|action_dims=6+1|policy_obs_dim=35"
            "|zero_action_targets_default_pose=True|step=True",
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
