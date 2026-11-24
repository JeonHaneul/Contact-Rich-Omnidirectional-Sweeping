"""Isaac Sim smoke check for the independent single-object sweep task.

Run registration/config/USD checks with Isaac Lab's Python executable::

    <isaac-lab>/isaaclab.bat -p tests/isaac_smoke_single_object.py

Add ``--full-env`` to construct one environment and perform reset/step checks.
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
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

import single_object_sweep  # noqa: E402, F401
from single_object_sweep import TASK_ID  # noqa: E402
from single_object_sweep.assets import CUP_1_USD_PATH  # noqa: E402
from single_object_sweep.env_cfg import (  # noqa: E402
    SPAWN_X_RANGE,
    SPAWN_Y_RANGE,
    SPAWN_Z,
    SWEEP_DISTANCE,
    TARGET_OBJECT_NAME,
    TARGET_OBJECT_WIDTH,
    UR5eSingleObjectSweepEnvCfg,
)


def _composition_counts(path: Path) -> tuple[int, int]:
    stage = Usd.Stage.Open(str(path), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"Could not open USD stage: {path}")
    mesh_count = sum(prim.IsA(UsdGeom.Mesh) for prim in stage.Traverse())
    rigid_body_count = sum(
        prim.HasAPI(UsdPhysics.RigidBodyAPI) for prim in stage.Traverse()
    )
    return mesh_count, rigid_body_count


def _check_registration_and_cfg() -> UR5eSingleObjectSweepEnvCfg:
    spec = gym.spec(TASK_ID)
    cfg = UR5eSingleObjectSweepEnvCfg()

    assert TASK_ID == "Isaac-Sweep-Single-Object-UR5e-Random-v0"
    assert spec.entry_point == "single_object_sweep.env:SingleObjectSweepEnv"
    assert (
        spec.kwargs["env_cfg_entry_point"]
        == "single_object_sweep.env_cfg:UR5eSingleObjectSweepEnvCfg"
    )
    assert (
        spec.kwargs["rsl_rl_cfg_entry_point"]
        == "single_object_sweep.agents.rsl_rl_ppo_cfg:"
        "UR5eSingleObjectSweepPPORunnerCfg"
    )

    target_cfg = cfg.scene.target_object
    assert TARGET_OBJECT_NAME == "cup_1"
    assert TARGET_OBJECT_WIDTH == 0.06
    assert tuple(SPAWN_X_RANGE) == (-0.75, -0.60)
    assert tuple(SPAWN_Y_RANGE) == (-0.20, 0.20)
    assert SPAWN_Z == 1.05
    assert SWEEP_DISTANCE == 0.18
    assert target_cfg.prim_path == "{ENV_REGEX_NS}/TargetObject"
    assert Path(target_cfg.spawn.usd_path).resolve() == Path(CUP_1_USD_PATH).resolve()

    arm_action = cfg.actions.arm_action
    assert len(arm_action.joint_names) == 6
    assert arm_action.scale == 0.5
    assert arm_action.use_default_offset is True
    assert len(cfg.actions.gripper_action.joint_names) == 8
    assert cfg.commands.target_goal_pos.asset_name == "target_object"
    assert cfg.commands.target_goal_pos.sweep_distance == 0.18
    assert cfg.commands.target_goal_pos.resampling_time_range == (1.0e6, 1.0e6)
    assert cfg.scene.num_envs == 4096
    assert cfg.scene.env_spacing == 2.5
    assert cfg.decimation == 2
    assert cfg.episode_length_s == 10.0
    assert cfg.sim.dt == 0.01
    assert cfg.curriculum is None

    spawn_params = cfg.events.target_spawn.params
    assert spawn_params["spawn_x_range"] == (-0.75, -0.60)
    assert spawn_params["spawn_y_range"] == (-0.20, 0.20)
    assert spawn_params["spawn_z"] == 1.05
    assert spawn_params["target_width_value"] == 0.06
    assert spawn_params["sweep_distance"] == 0.18

    print(
        "CFG|registered=True|target=cup_1|objects=1|arm_dim=6|gripper_dim=1"
        "|policy_obs_dim=35|curriculum=None",
        flush=True,
    )
    return cfg


def _check_target_usd() -> None:
    cup_path = Path(CUP_1_USD_PATH)
    assert cup_path.is_file(), cup_path
    mesh_count, rigid_body_count = _composition_counts(cup_path)
    assert mesh_count > 0
    assert rigid_body_count > 0
    print(
        f"USD|file={cup_path.name}|meshes={mesh_count}"
        f"|rigid_bodies={rigid_body_count}|resolved={cup_path.resolve()}",
        flush=True,
    )


def _check_full_environment(cfg: UR5eSingleObjectSweepEnvCfg) -> None:
    import omni.usd

    cfg.scene.num_envs = 1
    cfg.observations.policy.enable_corruption = False
    cfg.scene.ee_frame.debug_vis = False
    cfg.scene.finger_frame.debug_vis = False
    cfg.scene.wrist_frame.debug_vis = False
    cfg.commands.target_goal_pos.debug_vis = False
    # Pip Isaac Sim installations may not define a Nucleus content root;
    # GroundPlaneCfg otherwise attempts to resolve ``None/Isaac/...``.
    cfg.scene.ground = None

    env = None
    try:
        env = gym.make(TASK_ID, cfg=cfg)
        observation, _ = env.reset()
        unwrapped = env.unwrapped

        assert unwrapped.action_manager.action_term_dim == [6, 1]
        assert observation["policy"].shape == (1, 35)

        target = unwrapped.scene["target_object"]
        target_pos_w = target.data.root_pos_w.clone()
        target_pos_e = target_pos_w - unwrapped.scene.env_origins
        epsilon = 1.0e-5
        assert bool(
            (
                (target_pos_e[:, 0] >= SPAWN_X_RANGE[0] - epsilon)
                & (target_pos_e[:, 0] <= SPAWN_X_RANGE[1] + epsilon)
            ).all()
        ), target_pos_e
        assert bool(
            (
                (target_pos_e[:, 1] >= SPAWN_Y_RANGE[0] - epsilon)
                & (target_pos_e[:, 1] <= SPAWN_Y_RANGE[1] + epsilon)
            ).all()
        ), target_pos_e
        assert torch.allclose(
            target_pos_e[:, 2],
            torch.full_like(target_pos_e[:, 2], SPAWN_Z),
            atol=2.0e-2,
            rtol=0.0,
        ), target_pos_e

        command = unwrapped.command_manager.get_command("target_goal_pos")
        goal_delta = command[:, :3] - target_pos_w
        expected_delta = unwrapped.sweep_dir
        assert torch.allclose(goal_delta, expected_delta, atol=1.0e-5, rtol=0.0), (
            goal_delta,
            expected_delta,
        )
        assert torch.allclose(
            goal_delta[:, (0, 2)],
            torch.zeros_like(goal_delta[:, (0, 2)]),
            atol=1.0e-5,
            rtol=0.0,
        )
        assert torch.allclose(
            goal_delta[:, 1].abs(),
            torch.full_like(goal_delta[:, 1], SWEEP_DISTANCE),
            atol=1.0e-5,
            rtol=0.0,
        )

        stage = omni.usd.get_context().get_stage()
        target_prim_paths = [
            str(prim.GetPath())
            for prim in stage.Traverse()
            if prim.GetName() == "TargetObject"
        ]
        assert target_prim_paths == ["/World/envs/env_0/TargetObject"], target_prim_paths

        zero_action = torch.zeros((1, 7), device=unwrapped.device)
        next_observation, reward, terminated, truncated, _ = env.step(zero_action)
        assert next_observation["policy"].shape == (1, 35)
        assert reward.shape == (1,)
        assert terminated.shape == (1,)
        assert truncated.shape == (1,)
        assert bool(torch.isfinite(reward).all()), reward

        print(
            "ENV|num_envs=1|target_prims=1|spawn_bounds=True|goal_delta=+-0.18y"
            "|action_dims=6+1|policy_obs_dim=35|finite_reward=True|step=True",
            flush=True,
        )
    except BaseException as error:
        print(f"ENV_ERROR|{type(error).__name__}|{error}", flush=True)
        print(traceback.format_exc(), flush=True)
        raise
    finally:
        if env is not None:
            env.close()


def main(full_env: bool = False) -> None:
    cfg = _check_registration_and_cfg()
    _check_target_usd()
    if full_env:
        _check_full_environment(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full-env",
        action="store_true",
        help="Construct one environment and verify reset plus one zero-action step.",
    )
    arguments = parser.parse_args()
    try:
        main(full_env=arguments.full_env)
    finally:
        simulation_app.close()
