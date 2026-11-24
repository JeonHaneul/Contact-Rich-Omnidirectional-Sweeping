"""Exercise XY-only resets, rotated TCP IK, palm sensing and reset commands."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS

import gymnasium as gym
import pytest
import torch

from test_inspire_sensor_observations import _functions, math_utils, contact_magnitudes_and_bits

ROOT = Path(__file__).resolve().parents[1]


def test_new_tasks_are_independent_and_keep_the_original_mass_task():
    import sweeping_policy
    for version, cls in ((2, "Sensors"), (3, "Reaching")):
        spec = gym.spec(f"Isaac-Sweep-Object-UR5e-Random-Single-Inspire-Open-v{version}-mass")
        assert spec.kwargs["env_cfg_entry_point"].endswith(f"UR5eSweepRandomSingleInspireMass{cls}EnvCfg")
        assert spec.kwargs["rsl_rl_cfg_entry_point"].endswith(f"UR5eSweepInspireMass{cls}PPORunnerCfg")
    original = gym.spec(sweeping_policy.RANDOM_SINGLE_INSPIRE_HAND_MASS_TASK_ID)
    assert original.entry_point == "sweeping_policy.env_inspire_hand:ShelfSweepInspireHandEnv"


def test_palm_only_sensing_masks_stale_rows_and_excludes_fingers():
    ns = _functions("sweeping_policy/mdp/inspire_sensor_observations.py", {
        "torch": torch, "contact_magnitudes_and_bits": contact_magnitudes_and_bits,
    })
    sensor = NS(body_names=["inspire_palm_force_sensor"], data=NS(
        net_forces_w=torch.tensor([[[.05, 0, 0]], [[float("nan"), 0, 0]], [[0, 0, 0]]])))
    env = NS(num_envs=3, device="cpu", sensor_data_fresh=torch.tensor([True, False, True]),
             scene={"palm_tactile": sensor})
    assert ns["palm_contact_bit"](env).tolist() == [[1.], [0.], [0.]]
    sensor.body_names = ["inspire_thumb_force_sensor_1"]
    with pytest.raises(RuntimeError, match="only the palm"):
        ns["palm_contact_bit"](env)


class VirtualArm:
    """Six-DOF Cartesian/Euler chain with exact world-frame Jacobian."""
    is_fixed_base = True
    ids = [2, 5, 0, 7, 4, 8]

    def __init__(self, n):
        self.data = NS(default_joint_pos=torch.zeros(n, 18), joint_pos=torch.zeros(n, 18),
                       joint_vel=torch.zeros(n, 18),
                       soft_joint_pos_limits=torch.tensor([-10., 10.]).expand(n, 18, 2).clone(),
                       body_pos_w=torch.zeros(n, 3, 3),
                       body_quat_w=torch.tensor([1., 0, 0, 0]).expand(n, 3, 4).clone())
        self.root_physx_view = NS(get_jacobians=self.jacobian)

    def find_joints(self, names, preserve_order=True):
        return self.ids, names

    def find_bodies(self, name):
        return [2], [name]

    def set_joint_position_target(self, positions, env_ids):
        pass

    def set_joint_velocity_target(self, velocities, env_ids):
        pass

    def write_joint_state_to_sim(self, positions, velocities, env_ids):
        self.data.joint_pos[env_ids] = positions
        self.data.joint_vel[env_ids] = velocities
        q = positions[:, self.ids]
        self.data.body_pos_w[env_ids, 2] = q[:, :3]
        self.data.body_quat_w[env_ids, 2] = math_utils.quat_from_euler_xyz(q[:, 3], q[:, 4], q[:, 5])

    def jacobian(self):
        q = self.data.joint_pos[:, self.ids]
        n = len(q)
        zero = torch.zeros(n)
        rz = math_utils.matrix_from_quat(math_utils.quat_from_euler_xyz(zero, zero, q[:, 5]))
        rzy = math_utils.matrix_from_quat(math_utils.quat_from_euler_xyz(zero, q[:, 4], q[:, 5]))
        j = torch.zeros(n, 6, 6)
        j[:, :3, :3] = torch.eye(3)
        j[:, 3:, 3] = rzy[:, :, 0]
        j[:, 3:, 4] = rz[:, :, 1]
        j[:, 5, 5] = 1
        full = torch.zeros(n, 2, 6, 18)
        full[:, 1, :, self.ids] = j
        return full


def event_namespace():
    class Manager:
        def __init__(self, cfg, env):
            pass
    namespace = {
        "__name__": __name__, "ManagerTermBase": Manager,
        "EventTermCfg": object, "ManagerBasedRLEnv": object, "SceneEntityCfg": lambda *a, **k: NS(name=a[0]),
        "Articulation": object, "RigidObject": object, "RigidObjectCollection": object,
        "torch": torch, "math_utils": math_utils,
    }
    tree = ast.parse((ROOT / "sweeping_policy/mdp/events.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "initialize_robot_at_reaching_pose")
    exec(compile(ast.Module(body=[node], type_ignores=[]), "reaching_event", "exec"), namespace)
    return namespace


@pytest.mark.parametrize("frame_rotation", [(1., 0., 0., 0.), (.5, -.5, -.5, -.5)])
@pytest.mark.parametrize("use_column_axes", [False, True])
def test_actual_ik_event_has_xy_variation_fixed_z_and_rotated_frame(frame_rotation, use_column_axes):
    torch.manual_seed(19)
    robot = VirtualArm(4)
    positions = torch.tensor([[-.7, -.1, 1.05], [-.7, .1, 1.05], [-.6, 0, 1.05], [-.5, .2, 1.05]])
    states = torch.zeros(4, 1, 13)
    states[:, 0, :3] = positions
    env = NS(num_envs=4, device="cpu", target_id=torch.zeros(4, 1, dtype=torch.long),
             target_width=torch.full((4, 1), .04), sweep_dir=torch.tensor([[0., 1, 0], [0, -1, 0], [0, 1, 0], [0, -1, 0]]),
             scene={"robot": robot, "shelf": NS(data=NS(default_root_state=torch.tensor([0., 0, 0, 1, 0, 0, 0]).expand(4, 7))),
                    "object_collection": NS(data=NS(object_link_state_w=states))})
    params = dict(robot_cfg=NS(name="robot"), shelf_cfg=NS(name="shelf"),
                  object_collection_cfg=NS(name="object_collection"), arm_joint_names=tuple("abcdef"),
                  eef_body_name="axia80_link", eef_offset=(0., 0., .1427), eef_rotation=frame_rotation,
                  reaching_x_offset=-.02, reaching_z_offset=.04, position_noise=(.01, .01, 0.),
                  use_column_axes=use_column_axes,
                  max_iterations=80, damping=.05, step_size=.5, position_tolerance=.003,
                  orientation_tolerance=.01, joint_seed_offsets=((0.,) * 6,))
    event = event_namespace()["initialize_robot_at_reaching_pose"](NS(params=params), env)
    ids = torch.tensor([0, 1, 3])
    event(env, ids, **params)
    desired = env.desired_reaching_pose_w[ids]
    base = positions[ids].clone()
    base[:, 0] -= .02
    base[:, 1] -= .04 * env.sweep_dir[ids, 1]
    base[:, 2] += .04
    noise = desired[:, :3] - base
    assert torch.all(torch.abs(noise[:, :2]) <= .01)
    assert torch.count_nonzero(noise[:, :2]) == 6
    assert torch.equal(noise[:, 2], torch.zeros(3))
    assert env.reaching_ik_success[ids].all()
    assert torch.equal(robot.data.joint_pos[2], torch.zeros(18))
    assert torch.equal(robot.data.joint_pos[ids][:, 9:], torch.zeros(3, 9))
    body_q = robot.data.body_quat_w[ids, 2]
    tcp_pos = robot.data.body_pos_w[ids, 2] + math_utils.quat_apply(body_q, torch.tensor([0., 0., .1427]).expand(3, 3))
    assert torch.all(torch.linalg.norm(tcp_pos - desired[:, :3], dim=-1) < .003)
    tcp_q = math_utils.quat_mul(body_q, torch.tensor(frame_rotation).expand(3, 4))
    _, rot_error = math_utils.compute_pose_error(tcp_pos, tcp_q, tcp_pos, desired[:, 3:], rot_error_type="axis_angle")
    assert torch.all(torch.linalg.norm(rot_error, dim=-1) < .01)
    if use_column_axes:
        torch.testing.assert_close(math_utils.matrix_from_quat(desired[:, 3:])[:, :, 1],
                                   torch.tensor([0., 0., 1.]).expand(3, 3), atol=1e-6, rtol=1e-6)


def test_reset_history_represents_ik_position_with_scalar_action_scale():
    robot = NS(data=NS(joint_pos=torch.arange(24.).reshape(3, 8)))
    arm = NS(cfg=NS(asset_name="robot"), _joint_ids=list(range(6)), _offset=torch.ones(3, 6), _scale=.5,
             action_dim=6, _raw_actions=torch.zeros(3, 6), _processed_actions=torch.zeros(3, 6))
    class Parent:
        def _reset_idx(self, env_ids):
            self.action_manager._action[env_ids, -2:] = 1
            self.action_manager._prev_action[env_ids, -2:] = 1
    ns = _functions("sweeping_policy/env_inspire_mass_reaching.py", {"torch": torch, "ShelfSweepInspireSensorsEnv": Parent})
    env = ns["ShelfSweepInspireMassReachingEnv"]()
    env.scene = {"robot": robot}
    env.action_manager = NS(_action=torch.zeros(3, 8), _prev_action=torch.zeros(3, 8), get_term=lambda _: arm)
    env._reset_idx(torch.tensor([1]))
    expected = (robot.data.joint_pos[1, :6] - 1) / .5
    torch.testing.assert_close(env.action_manager._action[1, :6], expected)
    torch.testing.assert_close(arm._raw_actions[1], expected)
    torch.testing.assert_close(arm._processed_actions[1], robot.data.joint_pos[1, :6])
    assert env.action_manager._action[1, -2:].tolist() == [1., 1.]
    assert torch.count_nonzero(env.action_manager._action[[0, 2]]) == 0


def test_real_configclass_inheritance_keeps_prefix_and_orders_spawn_before_ik():
    from test_inspire_sensor_observations import REPO
    tree = ast.parse((REPO / "IsaacLab/source/isaaclab/isaaclab/utils/configclass.py").read_text())
    tree.body = [n for n in tree.body if not (isinstance(n, ast.ImportFrom) and n.level)]
    ns = {"__name__": __name__, "class_to_dict": lambda obj: vars(obj), "update_class_from_dict": lambda *a: None}
    exec(compile(tree, "configclass", "exec"), ns)
    configclass = ns["configclass"]

    @configclass
    class Term:
        func: object = None
        params: dict = {}
        mode: str = "reset"
        noise: object = None

    @configclass
    class Group:
        concatenate_terms: bool = True
        enable_corruption: bool = True

    ns.update(configclass=configclass, ObsTerm=Term, ObsGroup=Group, MISSING="unset",
              Unoise=lambda **kw: NS(**kw), mdp=NS(**{name: lambda env: None for name in (
                  "MA_joint_pos_rel", "MA_joint_vel_rel", "last_action", "MA_object_position_in_RRF",
                  "MA_object_width", "ee_pos_r", "MA_target_goal_command")}))
    tree = ast.parse((ROOT / "sweeping_policy/shelf_sweep_random_env_cfg.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ObservationsCfg")
    exec(compile(ast.Module(body=[node], type_ignores=[]), "observations", "exec"), ns)

    @configclass
    class Events:
        reset_all: Term = Term()
        object_spawn: Term = Term()

    @configclass
    class MassParent:
        observations = ns["ObservationsCfg"]()
        events: Events = Events()
        scene = NS()
        rewards = NS(reaching=NS(weight=3., func="legacy_reach", params={}),
                     sweeping_object=NS(weight=6., params={"command_name": "target_goal_pos"}),
                     orientation=NS(weight=2., func="legacy_align"))
        actions = NS(arm_action=NS(scale=.5), gripper_action=NS(action_dim=2))

        def __post_init__(self):
            self.events.object_spawn.func = "single_spawn"
            self.observations.policy.joint_pos.params = {"arm_only": True}
            self.observations.policy.goal_pos.params = {"command_name": "target_goal_pos"}

    def frame(targets, debug_vis=False):
        return NS(target_frames=[NS(prim_path="hand", offset=NS(pos=pos, rot=(.5, -.5, -.5, -.5))) for _, pos in targets])

    ns.update(UR5eSweepRandomSingleInspireHandMassEnvCfg=MassParent, ContactSensorCfg=NS,
              sensor_obs=NS(palm_contact_bit=lambda env: None, wrist_wrench_eef=lambda env: None),
              make_sensor_inspire_cfg=lambda: NS(replace=lambda **kw: NS(spawn=NS(sensor_height=.0254))),
              _tool_frame=frame, TARGET_Z_OFFSET=.04,
              mass_rewards=NS(reaching_at_object_height="new_reach", align_eef_y_with_shelf_z="new_align"))
    _functions("sweeping_policy/config/ur5e/joint_pos_random_single_inspire_mass_sensors_env_cfg.py", ns)
    ns.update(RandomSweepEventCfg=Events, EventTerm=Term, SceneEntityCfg=lambda name: NS(name=name),
              ARM_JOINT_NAMES=tuple("abcdef"), TOOL_ROTATION_H=(.5, -.5, -.5, -.5),
              mdp=NS(initialize_robot_at_reaching_pose="reaching_ik"))
    _functions("sweeping_policy/config/ur5e/joint_pos_random_single_inspire_mass_reaching_env_cfg.py", ns)
    parent = MassParent()
    v2 = ns["UR5eSweepRandomSingleInspireMassSensorsEnvCfg"]()
    v3 = ns["UR5eSweepRandomSingleInspireMassReachingEnvCfg"]()
    assert v2.actions == v3.actions == parent.actions
    prefix = [name for name, term in vars(parent.observations.policy).items() if isinstance(term, Term)]
    terms = [name for name, term in vars(v2.observations.policy).items() if isinstance(term, Term)]
    assert terms == prefix + ["tactile", "wrist_wrench"]
    assert v3.observations == v2.observations
    assert v2.observations.policy.joint_pos.params == {"arm_only": True}
    assert parent.rewards.reaching.func == "legacy_reach"
    assert parent.rewards.orientation.func == "legacy_align"
    assert "pushing_z_offset" not in parent.rewards.sweeping_object.params
    assert v2.rewards.reaching.func == "new_reach"
    assert v2.rewards.reaching.params == {"reaching_z_offset": .04}
    assert v2.rewards.orientation.func == "new_align"
    assert v2.rewards.orientation.weight == parent.rewards.orientation.weight
    assert v2.rewards.sweeping_object.params["pushing_z_offset"] == .04
    assert v3.rewards.reaching is None
    assert v3.rewards.sweeping_object == v2.rewards.sweeping_object
    assert [name for name, term in vars(v3.events).items() if isinstance(term, Term)] == ["reset_all", "object_spawn", "initialize_reaching_pose"]
    assert v3.events.object_spawn.func == "single_spawn"
    assert not hasattr(parent.events, "initialize_reaching_pose")
    assert not hasattr(parent.observations.policy, "tactile")
    params = v3.events.initialize_reaching_pose.params
    assert params["position_noise"] == (.01, .01, 0.)
    assert params["reaching_z_offset"] == .04
    assert params["use_column_axes"] is True
    assert params["eef_offset"] == v3.scene.ee_frame.target_frames[0].offset.pos
    assert params["eef_rotation"] == v3.scene.ee_frame.target_frames[0].offset.rot
    # Sensor center + TCP offset = old mount distance + full sensor thickness.
    height = v2.scene.robot.spawn.sensor_height
    assert v2.scene.ee_frame.target_frames[0].offset.pos == pytest.approx((0., 0., .13 + height / 2))
    assert height / 2 + v2.scene.ee_frame.target_frames[0].offset.pos[2] == pytest.approx(.13 + height)
    assert v2.scene.palm_tactile.prim_path.endswith("/inspire_palm_force_sensor")
