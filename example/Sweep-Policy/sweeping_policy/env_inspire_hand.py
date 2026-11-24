"""v1 reset history starts with fully open hand commands, [1, 1]."""

from .env import ShelfSweepEnv


def initialize_hand_action_history(action_manager, env_ids):
    # v1 action order is [UR5e 6D, common openness, thumb opposition openness].
    action_manager._action[env_ids, -2:] = 1.0
    action_manager._prev_action[env_ids, -2:] = 1.0


class ShelfSweepInspireHandEnv(ShelfSweepEnv):
    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        initialize_hand_action_history(self.action_manager, env_ids)
