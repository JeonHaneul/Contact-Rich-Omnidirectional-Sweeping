"""v3/v4: initial-only actor target, optionally privileged live critic state."""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import ConstantNoiseCfg, NoiseModelWithAdditiveBiasCfg, UniformNoiseCfg

from ... import mdp
from ...mdp import inspire_privileged_observations as privileged_obs
from .joint_pos_random_single_inspire_mass_sensors_env_cfg import (
    MassSensorObservationsCfg, UR5eSweepRandomSingleInspireMassSensorsEnvCfg,
)


@configclass
class UR5eSweepRandomSingleInspireInitialMassEnvCfg(UR5eSweepRandomSingleInspireMassSensorsEnvCfg):
    """Keep v2-mass physics/rewards; replace actor tracking with a reset latch."""

    def __post_init__(self):
        super().__post_init__()
        target = self.observations.policy.target_obs_state
        target.func = mdp.InitialTargetPositionInRRF
        target.params = {"command_name": "target_goal_pos"}
        target.noise = NoiseModelWithAdditiveBiasCfg(
            noise_cfg=ConstantNoiseCfg(bias=0.0),
            bias_noise_cfg=UniformNoiseCfg(n_min=-0.01, n_max=0.01, operation="abs"),
            sample_bias_per_component=True,
        )


@configclass
class InitialMassPrivilegedObservationsCfg(MassSensorObservationsCfg):
    @configclass
    class PrivilegedCfg(ObsGroup):
        target_state = ObsTerm(func=privileged_obs.target_state_in_robot_root)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    privileged: PrivilegedCfg = PrivilegedCfg()


@configclass
class UR5eSweepRandomSingleInspirePrivilegedMassEnvCfg(UR5eSweepRandomSingleInspireInitialMassEnvCfg):
    """Same actor/reward contract as v3, with live Scene GT for the critic."""

    observations: InitialMassPrivilegedObservationsCfg = InitialMassPrivilegedObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Reward inputs remain uncorrupted Scene GT. Do not read the frozen
        # actor target, or the critic's transformed/normalized input tensor.
        objects = SceneEntityCfg("object_collection")
        eef = SceneEntityCfg("ee_frame")
        self.rewards.reaching.params.update(object_collection_cfg=objects, ee_frame_cfg=eef)
        self.rewards.orientation.params.update(shelf_cfg=SceneEntityCfg("shelf"), ee_frame_cfg=eef)
        self.rewards.sweeping_object.params["object_collection_cfg"] = objects
        self.rewards.homing_after_sweep.params["object_collection_cfg"] = objects
        self.rewards.object_collision.params["object_collection_cfg"] = objects


@configclass
class UR5eSweepRandomSingleInspireInitialMassEnvCfg_PLAY(UR5eSweepRandomSingleInspireInitialMassEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = False


@configclass
class UR5eSweepRandomSingleInspirePrivilegedMassEnvCfg_PLAY(UR5eSweepRandomSingleInspirePrivilegedMassEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = False
