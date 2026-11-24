#!/usr/bin/env python3
"""Record deterministic policy execution before automatic resets hide motion."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-Sweep-Object-UR5e-Random-Single-Inspire-Open-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output", default="/tmp/inspire_policy_diagnostic.jsonl")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sweeping_policy  # noqa: F401,E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg


def main():
    device = args.device or "cuda:0"
    cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
    cfg.seed = args.seed
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(
        agent_cfg, importlib.metadata.version("rsl-rl-lib")
    )
    env = RslRlVecEnvWrapper(gym.make(args.task, cfg=cfg), clip_actions=agent_cfg.clip_actions)
    base = env.unwrapped
    robot = base.scene["robot"]
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(str(Path(args.checkpoint).expanduser().resolve()))
    policy = runner.get_inference_policy(device=device)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    def values(tensor):
        return tensor.detach().cpu().tolist()

    original_compute = base.termination_manager.compute
    step = 0
    last = {}
    try:
        with output.open("w") as stream:
            metadata = {
                "type": "metadata", "checkpoint": str(Path(args.checkpoint).resolve()),
                "joint_names": robot.joint_names, "seed": args.seed,
                "step_dt": base.step_dt,
                "initial_joint_pos": values(robot.data.joint_pos),
                "default_joint_pos": values(robot.data.default_joint_pos),
                "arm_action_joint_names": base.action_manager.get_term("arm_action")._joint_names,
                "note": "Snapshots are after physics and before automatic reset; training configuration is unchanged.",
            }
            stream.write(json.dumps(metadata) + "\n")

            def capture_before_reset():
                nonlocal last
                dones = original_compute()
                ids = base.target_id.squeeze(-1).long()
                rows = torch.arange(base.num_envs, device=base.device)
                obj = base.scene["object_collection"].data.object_pos_w[rows, ids]
                reach = obj.clone()
                reach[:, 0] -= 0.02
                reach[:, 1] -= base.target_width[:, 0] * torch.sign(base.sweep_dir[:, 1])
                reach[:, 2] += 0.09
                ee = base.scene["ee_frame"].data.target_pos_w[:, 0]
                wrist = base.scene["wrist_frame"].data.target_pos_w[:, 0]
                ee_distance = torch.linalg.vector_norm(reach - ee, dim=-1)
                wrist_y_error = (reach[:, 1] - wrist[:, 1]).abs()
                last = {
                    "type": "step", "step": step,
                    "episode_step": values(base.episode_length_buf),
                    "action": values(base.action_manager.action),
                    "joint_pos": values(robot.data.joint_pos),
                    "joint_vel": values(robot.data.joint_vel),
                    "joint_pos_target": values(robot.data.joint_pos_target),
                    "max_abs_joint_vel": values(robot.data.joint_vel.abs().amax(dim=1)),
                    "target_id": values(ids),
                    "target_pos": values(obj), "eef_pos": values(ee),
                    "eef_reach_distance": values(ee_distance),
                    "wrist_y_error": values(wrist_y_error),
                    "push_gate": values((ee_distance < 0.04) & (wrist_y_error < 0.04)),
                    "done": values(dones),
                    "terminations": {
                        name: values(base.termination_manager.get_term(name))
                        for name in base.termination_manager.active_terms
                    },
                }
                stream.write(json.dumps(last) + "\n")
                return dones

            # The normal compute and reset remain intact; only capture their state.
            base.termination_manager.compute = capture_before_reset
            obs = env.get_observations()
            with torch.inference_mode():
                for step in range(args.steps):
                    if not simulation_app.is_running():
                        break
                    action = policy(obs)
                    obs, _, dones, _ = env.step(action)
                    if hasattr(policy, "reset"):
                        policy.reset(dones)
                    if step < 5 or step % 100 == 0:
                        print("[DIAG] " + json.dumps({
                            "step": step, "episode_step": last["episode_step"][0],
                            "q": last["joint_pos"][0],
                            "target": last["joint_pos_target"][0],
                            "max_velocity": last["max_abs_joint_vel"][0],
                            "done": last["done"][0],
                            "terminations": [k for k, v in last["terminations"].items() if v[0]],
                        }), flush=True)
            stream.flush()
    finally:
        base.termination_manager.compute = original_compute
        env.close()
    print(f"[DIAG] Saved pre-reset states: {output}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
