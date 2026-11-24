"""v2 inherits v1 hand reset and masks sensor samples invalidated by a reset."""

import torch

from .env_inspire_hand import ShelfSweepInspireHandEnv


class ShelfSweepInspireSensorsEnv(ShelfSweepInspireHandEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        # ObservationManager evaluates term shapes before the first physics step.
        self.sensor_data_fresh = torch.zeros(
            cfg.scene.num_envs, device=cfg.sim.device, dtype=torch.bool,
        )
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

    def _reset_idx(self, env_ids):
        self.sensor_data_fresh[env_ids] = False
        super()._reset_idx(env_ids)

    def step(self, action):
        # Auto-reset inside the parent step marks only reset rows stale again.
        self.sensor_data_fresh[:] = True
        return super().step(action)
