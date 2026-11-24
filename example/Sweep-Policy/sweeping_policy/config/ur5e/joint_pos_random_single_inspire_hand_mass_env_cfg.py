"""v1-mass: v1 with exactly one bottom-heavy cylindrical target."""

from isaaclab.assets import RigidObjectCfg
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

from ...src.mass_cylinder import MassCylinderCfg
from .joint_pos_random_single_inspire_hand_env_cfg import UR5eSweepRandomSingleInspireHandEnvCfg


@configclass
class UR5eSweepRandomSingleInspireHandMassEnvCfg(UR5eSweepRandomSingleInspireHandEnvCfg):
    def _load_object_catalog(self):
        # No external object USDs, parked objects, or unused collection members.
        return {
            "objects": {"mass_cylinder": None},
            "pose": {"mass_cylinder": [-0.75, 0.0, 1.05, 1.0, 0.0, 0.0, 0.0]},
            "id": {"mass_cylinder": 0},
            "width": {"mass_cylinder": 0.04},
        }

    def _make_object_cfg(self, name, usd_path, pose):
        del usd_path
        return RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{name}",
            init_state=RigidObjectCfg.InitialStateCfg(pos=pose[:3], rot=pose[3:]),
            spawn=MassCylinderCfg(
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True, contact_offset=0.001, rest_offset=0.0,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.4, 0.8)),
            ),
        )


@configclass
class UR5eSweepRandomSingleInspireHandMassEnvCfg_PLAY(UR5eSweepRandomSingleInspireHandMassEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = False


@configclass
class UR5eSweepRandomSingleInspireHandMassLimitEnvCfg(UR5eSweepRandomSingleInspireHandMassEnvCfg):
    """v1-mass with the original 1 rad/s Arm velocity termination."""

    def __post_init__(self):
        super().__post_init__()
        self.terminations.hand_velocity.params["threshold"] = 1.0


@configclass
class UR5eSweepRandomSingleInspireHandMassLimitEnvCfg_PLAY(UR5eSweepRandomSingleInspireHandMassLimitEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = False
