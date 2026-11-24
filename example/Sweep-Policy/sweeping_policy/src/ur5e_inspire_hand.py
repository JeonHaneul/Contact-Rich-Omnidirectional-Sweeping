"""Direct-mounted Inspire with two absolute openness commands in [0, 1]."""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from .ur5e_inspire_open import HAND_JOINT_NAMES, make_fixed_open_inspire_cfg


# Same zero-angle fully extended pose as v0. Thumb 1 controls opposition;
# thumb 2-4 and all four other fingers follow the common flexion command.
HAND_OPEN_POSITIONS = (0.0,) * 12
HAND_CLOSED_POSITIONS = (
    1.1641, 0.5864, 0.5864 * 0.8024, 0.5864 * 0.8024 * 0.9487,
    1.4381, 1.4381 * 1.0843,
    1.4381, 1.4381 * 1.0843,
    1.4381, 1.4381 * 1.0843,
    1.4381, 1.4381 * 1.0843,
)


def hand_joint_targets(openness, open_positions, closed_positions):
    """Expand [common_open, thumb_1_open] without filtering or rate limiting."""
    joint_openness = openness[..., :1].expand(*openness.shape[:-1], 12).clone()
    joint_openness[..., 0] = openness[..., 1]
    return closed_positions + joint_openness * (open_positions - closed_positions)


class InspireHandOpennessAction(ActionTerm):
    """Two absolute hand commands: 0 = closed, 1 = fully extended."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._joint_ids, names = self._asset.find_joints(list(HAND_JOINT_NAMES), preserve_order=True)
        if tuple(names) != HAND_JOINT_NAMES:
            raise ValueError(f"Unexpected Inspire hand joints: {names}")
        self._open_positions = torch.tensor(HAND_OPEN_POSITIONS, device=self.device)
        self._closed_positions = torch.tensor(HAND_CLOSED_POSITIONS, device=self.device)
        self._raw_actions = torch.ones(self.num_envs, 2, device=self.device)
        self._processed_actions = torch.ones_like(self._raw_actions)
        self._joint_targets = self._open_positions.expand(self.num_envs, -1).clone()

    @property
    def action_dim(self):
        return 2

    @property
    def raw_actions(self):
        return self._raw_actions

    @property
    def processed_actions(self):
        return self._processed_actions

    def process_actions(self, actions):
        self._raw_actions[:] = actions
        self._processed_actions[:] = actions.clamp(0.0, 1.0)
        self._joint_targets[:] = hand_joint_targets(
            self._processed_actions, self._open_positions, self._closed_positions
        )

    def apply_actions(self):
        self._asset.set_joint_position_target(self._joint_targets, joint_ids=self._joint_ids)

    def reset(self, env_ids=None):
        index = slice(None) if env_ids is None else env_ids
        self._raw_actions[index] = 1.0
        self._processed_actions[index] = 1.0
        self._joint_targets[index] = self._open_positions
        self._asset.set_joint_position_target(
            self._joint_targets[index], joint_ids=self._joint_ids, env_ids=env_ids
        )


@configclass
class InspireHandOpennessActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = InspireHandOpennessAction
    asset_name: str = "robot"


def make_movable_inspire_cfg():
    """Use v0 Arm settings and raw revolute fingers, without its fixed spawner."""
    cfg = make_fixed_open_inspire_cfg()
    cfg.spawn = sim_utils.UsdFileCfg(
        usd_path=cfg.spawn.usd_path,
        rigid_props=cfg.spawn.rigid_props.copy(),
        articulation_props=cfg.spawn.articulation_props.copy(),
    )
    cfg.init_state.joint_pos.update(dict(zip(HAND_JOINT_NAMES, HAND_OPEN_POSITIONS)))
    cfg.init_state.joint_vel.update({name: 0.0 for name in HAND_JOINT_NAMES})
    # Source Inspire position-hand gains; Arm gains remain those of Random-v0.
    cfg.actuators["hand"] = ImplicitActuatorCfg(
        joint_names_expr=list(HAND_JOINT_NAMES),
        effort_limit_sim=0.20, velocity_limit_sim=1.0,
        stiffness=1.2, damping=0.06, armature=0.0001,
    )
    return cfg
