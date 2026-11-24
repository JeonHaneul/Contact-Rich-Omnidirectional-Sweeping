"""Random-Single-v0 with planar UR5e base-position randomization."""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ... import mdp
from ...shelf_sweep_random_env_cfg import (
    EventCfg as RandomSweepEventCfg,
    ObservationsCfg as RandomSweepObservationsCfg,
)
from .joint_pos_random_single_sweep_env_cfg import UR5eSweepRandomSingleEnvCfg


def robot_base_xy_in_env_frame(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Return the robot-root XY position relative to each environment origin."""
    robot = env.scene[asset_cfg.name]
    return robot.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]


@configclass
class ObservationsCfg(RandomSweepObservationsCfg):
    """Random-Single-v1 observations with the robot-base XY position."""

    @configclass
    class PolicyCfg(RandomSweepObservationsCfg.PolicyCfg):
        base_pos = ObsTerm(func=robot_base_xy_in_env_frame)

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg(RandomSweepEventCfg):
    """Random-Single events with planar robot-base randomization."""

    robot_spawn = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.050, 0.050),
                "y": (-0.050, 0.050),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class UR5eSweepRandomSingleV1EnvCfg(UR5eSweepRandomSingleEnvCfg):
    """Random-Single-v0 with randomized and observed robot-base XY position."""

    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        super().__post_init__()


@configclass
class UR5eSweepRandomSingleV1EnvCfg_PLAY(UR5eSweepRandomSingleV1EnvCfg):
    """Small visualization configuration for Random-Single-v1."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
