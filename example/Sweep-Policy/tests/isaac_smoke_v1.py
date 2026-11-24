"""Isaac Sim smoke check for v1 registration, IK config, and object USDs.

Run with Isaac Sim's Python executable after installing this package editable::

    <isaac-sim>/python.bat tests/isaac_smoke_v1.py
"""

from __future__ import annotations

import argparse
import traceback

from isaacsim import SimulationApp


simulation_app = SimulationApp({"headless": True})

import gymnasium as gym  # noqa: E402
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

import sweeping_policy  # noqa: E402, F401
from sweeping_policy.config.ur5e.diff_ik_random_sweep_env_cfg import (  # noqa: E402
    UR5eSweepDiffIKEnvCfg,
)
from sweeping_policy.src.assets import ASSET_ROOT, object_usd_path  # noqa: E402


TASK_ID = "Isaac-Sweep-Object-UR5e-Random-v1"
PROBLEM_ASSETS = (
    "Objects/Collected_Cup_1/Cup_1.usd",
    "Objects/Collected_Mug_3/Mug_3.usd",
)


def _composition_counts(path: str) -> tuple[int, int]:
    stage = Usd.Stage.Open(path, load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"Could not open USD stage: {path}")
    mesh_count = sum(prim.IsA(UsdGeom.Mesh) for prim in stage.Traverse())
    rigid_body_count = sum(prim.HasAPI(UsdPhysics.RigidBodyAPI) for prim in stage.Traverse())
    return mesh_count, rigid_body_count


def main(full_env: bool = False) -> None:
    spec = gym.spec(TASK_ID)
    cfg = UR5eSweepDiffIKEnvCfg()
    action = cfg.actions.arm_action

    assert spec.kwargs["env_cfg_entry_point"].endswith(":UR5eSweepDiffIKEnvCfg")
    assert action.controller.command_type == "pose"
    assert action.controller.use_relative_mode is True
    assert action.controller.ik_method == "dls"
    assert tuple(action.scale) == (0.05, 0.05, 0.05, 0.25, 0.25, 0.25)
    for rigid_object_cfg in cfg.scene.object_collection.rigid_objects.values():
        assert rigid_object_cfg.spawn.visual_material is None

    print(
        f"V1|registered=True|controller={action.controller.ik_method}"
        f"|relative={action.controller.use_relative_mode}|arm_dim=6",
        flush=True,
    )
    for relative_path in PROBLEM_ASSETS:
        wrapper = f"{ASSET_ROOT.rstrip('/')}/{relative_path.lstrip('/')}"
        resolved = object_usd_path(relative_path)
        wrapper_meshes, wrapper_rigid_bodies = _composition_counts(wrapper)
        resolved_meshes, resolved_rigid_bodies = _composition_counts(resolved)
        assert resolved_meshes > 0
        assert resolved_rigid_bodies > 0
        print(
            f"USD|{relative_path.rsplit('/', 1)[-1]}"
            f"|wrapper_meshes={wrapper_meshes}|wrapper_rigid={wrapper_rigid_bodies}"
            f"|resolved_meshes={resolved_meshes}|resolved_rigid={resolved_rigid_bodies}"
            f"|resolved={resolved}",
            flush=True,
        )

    if full_env:
        import torch

        cfg.scene.num_envs = 1
        cfg.observations.policy.enable_corruption = False
        # This host's pip Isaac Sim install has no Nucleus root configured;
        # debug frame/command markers otherwise resolve to ``None/Isaac/...``.
        cfg.scene.ee_frame.debug_vis = False
        cfg.scene.finger_frame.debug_vis = False
        cfg.scene.wrist_frame.debug_vis = False
        cfg.commands.target_goal_pos.debug_vis = False
        cfg.scene.ground = None
        env = None
        try:
            env = gym.make(TASK_ID, cfg=cfg)
            observation, _ = env.reset()
            action_dims = env.unwrapped.action_manager.action_term_dim
            assert action_dims == [6, 1]
            assert observation["policy"].shape == (1, 35)
            zero_action = torch.zeros((1, 7), device=env.unwrapped.device)
            env.step(zero_action)
            print("ENV|num_envs=1|action_dims=6+1|policy_obs_dim=35|step=True", flush=True)
        except BaseException as error:
            print(f"ENV_ERROR|{type(error).__name__}|{error}", flush=True)
            print(traceback.format_exc(), flush=True)
            raise
        finally:
            if env is not None:
                env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-env", action="store_true")
    args = parser.parse_args()
    try:
        main(full_env=args.full_env)
    finally:
        simulation_app.close()
