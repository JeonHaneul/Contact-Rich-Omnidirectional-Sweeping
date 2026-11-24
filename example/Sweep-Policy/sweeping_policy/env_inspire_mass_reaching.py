"""Sensor environment with reset history consistent with the solved arm pose."""

import torch

from .env_inspire_sensors import ShelfSweepInspireSensorsEnv


class ShelfSweepInspireMassReachingEnv(ShelfSweepInspireSensorsEnv):
    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        # Absolute JointPositionAction uses default positions as its offset.
        # Reset history must describe the IK pose, not a return to home.
        arm = self.action_manager.get_term("arm_action")
        robot = self.scene[arm.cfg.asset_name]
        offset = arm._offset[env_ids] if isinstance(arm._offset, torch.Tensor) else arm._offset
        scale = arm._scale[env_ids] if isinstance(arm._scale, torch.Tensor) else arm._scale
        positions = robot.data.joint_pos[env_ids][:, arm._joint_ids]
        raw = (positions - offset) / scale
        self.action_manager._action[env_ids, :arm.action_dim] = raw
        self.action_manager._prev_action[env_ids, :arm.action_dim] = raw
        arm._raw_actions[env_ids] = raw
        arm._processed_actions[env_ids] = positions
