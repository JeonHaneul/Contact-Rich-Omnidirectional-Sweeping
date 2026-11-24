"""Environment class with task-specific vectorized state buffers."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.envs import ViewerCfg


class ShelfSweepEnv(ManagerBasedRLEnv):
    """Manager-based environment for the shelf-sweeping configurations."""

    def __init__(
        self,
        cfg: ManagerBasedRLEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        # Command, reset, observation, reward, and termination terms access
        # these buffers while the base class is constructing its managers.

        shape = (cfg.scene.num_envs, 1)
        self.target_id = torch.zeros(shape, device=cfg.sim.device, dtype=torch.long)
        self.target_width = torch.zeros(shape, device=cfg.sim.device)
        self.sweep_dir = torch.zeros((cfg.scene.num_envs, 3), device=cfg.sim.device)

        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
