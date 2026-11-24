"""CPU checks for initial-only actor input and the privileged mass-task critic."""

import ast
from contextlib import contextmanager
from dataclasses import MISSING
from types import SimpleNamespace as NS

import gymnasium as gym
import pytest
import torch

from test_inspire_sensor_observations import ROOT, REPO, _functions, math_utils


CFG = "sweeping_policy/config/ur5e/joint_pos_random_single_inspire_initial_mass_env_cfg.py"
RUNNER = "sweeping_policy/config/ur5e/agents/rsl_rl_ppo_inspire_initial_mass_cfg.py"


def _nodes(path, names, namespace):
    """Run actual simulator-independent definitions without importing Kit."""
    tree = ast.parse((ROOT / path).read_text())
    nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    nodes += [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, path, "exec"), namespace)
    return namespace


def _latch_namespace():
    class Manager:
        def __init__(self, cfg, env):
            self._env = env

    return _nodes("sweeping_policy/mdp/observations.py", {
        "MA_initial_object_position_in_RRF", "InitialTargetPositionInRRF",
    }, {"torch": torch, "MISSING": MISSING, "ManagerTermBase": Manager,
        "SceneEntityCfg": lambda name: NS(name=name),
        "subtract_frame_transforms": math_utils.subtract_frame_transforms})


def _privileged_namespace():
    return _functions("sweeping_policy/mdp/inspire_privileged_observations.py", {
        "torch": torch, "SceneEntityCfg": lambda name: NS(name=name),
        "subtract_frame_transforms": math_utils.subtract_frame_transforms,
        "quat_apply_inverse": math_utils.quat_apply_inverse,
        "quat_unique": math_utils.quat_unique,
    })


def _state_env():
    # Different target indices reject accidental broadcasting/first-object use.
    root = torch.tensor([[3., -2., 1., 2**-.5, 0., 0., 2**-.5],
                         [-1., 2., .5, 1., 0., 0., 0.]])
    states = torch.full((2, 2, 13), 777.)
    states[0, 1] = torch.tensor([4., -2., 1.5, 2**-.5, 0., 2**-.5, 0., 2., 0., 0., 0., 3., 0.])
    states[1, 0] = torch.tensor([-.5, 2.25, 1., -1., 0., 0., 0., 1., 2., 3., -1., -2., -3.])
    snapshot = states[torch.arange(2), torch.tensor([1, 0])].clone()
    command = NS(target_init_state_w=snapshot)
    return NS(num_envs=2, device="cpu", target_id=torch.tensor([[1], [0]]),
              command_manager=NS(get_term=lambda _: command),
              scene={"robot": NS(data=NS(root_state_w=root)),
                     "object_collection": NS(data=NS(object_link_state_w=states,
                         # Deliberately distinct COM state: privileged input must
                         # retain the cylinder's bottom/link-origin convention.
                         object_state_w=states + 100.))})


def test_initial_actor_position_is_latched_after_command_reset_and_partial_resets():
    env = _state_env()
    latch = _latch_namespace()["InitialTargetPositionInRRF"](NS(), env)
    initial = latch(env, command_name="target_goal_pos")
    torch.testing.assert_close(initial, torch.tensor([[0., -1., .5], [.5, .25, .5]]), atol=1e-6, rtol=1e-6)

    # A consumer must not be able to modify the saved per-episode value.
    initial.add_(123.)
    captured = latch(env, command_name="target_goal_pos")
    live = env.scene["object_collection"].data.object_link_state_w
    live[..., :3] += .25
    env.scene["robot"].data.root_state_w[:, :3] += .1
    assert torch.equal(latch(env, command_name="target_goal_pos"), captured)

    # Manager reset comes before command reset. Capture is deferred until the
    # next call, when the new spawn snapshot is available; other rows stay fixed.
    latch.reset(torch.tensor([1]))
    command = env.command_manager.get_term("target_goal_pos")
    command.target_init_state_w[1, :3] = torch.tensor([-.25, 2.5, 1.25])
    result = latch(env, command_name="target_goal_pos")
    assert torch.equal(result[0], captured[0])
    torch.testing.assert_close(result[1], torch.tensor([.65, .4, .65]), atol=1e-6, rtol=1e-6)
    latch.reset()
    all_reset = latch(env, command_name="target_goal_pos")
    assert not torch.equal(all_reset[0], captured[0])


def test_privileged_state_uses_live_selected_link_state_and_transforms_all_components():
    env = _state_env()
    observe = _privileged_namespace()["target_state_in_robot_root"]
    actual = observe(env)
    assert actual.shape == (2, 13)
    torch.testing.assert_close(actual[:, :3], torch.tensor([[0., -1., .5], [.5, .25, .5]]), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual[:, 7:10], torch.tensor([[0., -2., 0.], [1., 2., 3.]]), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual[:, 10:13], torch.tensor([[3., 0., 0.], [-1., -2., -3.]]), atol=1e-6, rtol=1e-6)
    state = env.scene["object_collection"].data.object_link_state_w[torch.arange(2), env.target_id[:, 0]]
    expected_quat = math_utils.quat_mul(math_utils.quat_conjugate(env.scene["robot"].data.root_state_w[:, 3:7]), state[:, 3:7])
    torch.testing.assert_close(math_utils.matrix_from_quat(actual[:, 3:7]), math_utils.matrix_from_quat(expected_quat))
    assert torch.all(actual[:, 3] >= 0)

    latch = _latch_namespace()["InitialTargetPositionInRRF"](NS(), env)
    actor_initial = latch(env, command_name="target_goal_pos")
    env.scene["object_collection"].data.object_link_state_w[0, 1, 0] += .2
    env.scene["object_collection"].data.object_link_state_w[0, 1, 7] += .3
    moved = observe(env)
    assert torch.equal(latch(env, command_name="target_goal_pos"), actor_initial)
    assert not torch.equal(moved[0, :3], actual[0, :3])
    assert not torch.equal(moved[0, 7:10], actual[0, 7:10])
    assert torch.equal(moved[1], actual[1])


def test_scene_rewards_ignore_frozen_actor_values_and_explicit_sources_preserve_formulas():
    functions = _functions("sweeping_policy/mdp/inspire_mass_rewards.py", {
        "torch": torch, "matrix_from_quat": math_utils.matrix_from_quat,
    })
    functions = _nodes("sweeping_policy/mdp/reward_random_sweep.py", {"pushing_target", "homing_reward"},
                       {"torch": torch, "SceneEntityCfg": lambda name: NS(name=name), **functions})
    bottom = torch.tensor([[[-.75, 0., 1.05]]])
    eef = torch.tensor([[[-.77, -.04, 1.09]]])
    env = NS(num_envs=1, device="cpu", target_id=torch.zeros(1, 1, dtype=torch.long),
             target_width=torch.full((1, 1), .04), sweep_dir=torch.tensor([[0., 1., 0.]]),
             command_manager=NS(get_command=lambda _: torch.tensor([[-.75, .15, 1.05]])),
             observations={"policy": torch.full((1, 41), -999.), "privileged": torch.full((1, 13), -999.)},
             scene={"object_collection": NS(data=NS(object_pos_w=bottom, object_lin_vel_w=torch.tensor([[[0., .07, 0.]]]))),
                    "ee_frame": NS(data=NS(target_pos_w=eef, target_quat_w=torch.tensor([[[2**-.5, 2**-.5, 0., 0.]]]))),
                    "wrist_frame": NS(data=NS(target_pos_w=eef.clone())),
                    "shelf": NS(data=NS(default_root_state=torch.tensor([[0., 0., 0., 1., 0., 0., 0.]]))),
                    "robot": NS(data=NS(joint_pos=torch.zeros(1, 6), default_joint_pos=torch.zeros(1, 6)))})
    env.scene = type("Scene", (dict,), {"num_envs": 1})(env.scene)
    reach = functions["reaching_at_object_height"]
    align = functions["align_eef_y_with_shelf_z"]
    push = functions["pushing_target"]
    home = functions["homing_reward"]
    values = [reach(env), align(env), push(env, "target_goal_pos", pushing_z_offset=.04), home(env, "target_goal_pos")]
    torch.testing.assert_close(values[0], torch.ones(1))
    torch.testing.assert_close(values[1], torch.ones(1))
    torch.testing.assert_close(values[2], torch.tensor([2. / 3.]), atol=1e-6, rtol=1e-6)
    objects, frame, shelf = (NS(name=name) for name in ("object_collection", "ee_frame", "shelf"))
    explicit = [reach(env, object_collection_cfg=objects, ee_frame_cfg=frame),
                align(env, shelf_cfg=shelf, ee_frame_cfg=frame),
                push(env, "target_goal_pos", object_collection_cfg=objects, pushing_z_offset=.04),
                home(env, "target_goal_pos", object_collection_cfg=objects)]
    for before, after in zip(values, explicit):
        assert torch.equal(before, after)
    env.observations["policy"].fill_(999.)
    env.observations["privileged"].fill_(999.)
    assert torch.equal(reach(env), values[0])
    assert torch.equal(push(env, "target_goal_pos", pushing_z_offset=.04), values[2])
    bottom[..., 1] += .015
    assert not torch.equal(reach(env), values[0])
    assert not torch.equal(push(env, "target_goal_pos", pushing_z_offset=.04), values[2])

    # Explicit asset names are actually honored, rather than accepted and ignored.
    env.scene["other_objects"] = env.scene.pop("object_collection")
    env.scene["other_eef"] = env.scene.pop("ee_frame")
    env.scene["other_shelf"] = env.scene.pop("shelf")
    torch.testing.assert_close(reach(env, object_collection_cfg=NS(name="other_objects"), ee_frame_cfg=NS(name="other_eef")),
                               torch.exp(torch.tensor([-.15])))
    torch.testing.assert_close(align(env, shelf_cfg=NS(name="other_shelf"), ee_frame_cfg=NS(name="other_eef")), torch.ones(1))


def _configclass():
    tree = ast.parse((REPO / "IsaacLab/source/isaaclab/isaaclab/utils/configclass.py").read_text())
    tree.body = [node for node in tree.body if not (isinstance(node, ast.ImportFrom) and node.level)]
    ns = {"__name__": __name__, "class_to_dict": lambda obj: vars(obj), "update_class_from_dict": lambda *args: None}
    exec(compile(tree, "configclass", "exec"), ns)
    return ns["configclass"]


def _configs():
    configclass = _configclass()

    @configclass
    class Term:
        func: object = None
        params: dict = {}
        noise: object = None
        weight: float = 0.

    @configclass
    class Group:
        enable_corruption: bool = True
        concatenate_terms: bool = True

    original_mdp = NS(**{name: (lambda env: None) for name in (
        "MA_joint_pos_rel", "MA_joint_vel_rel", "last_action", "MA_object_position_in_RRF",
        "MA_object_width", "ee_pos_r", "MA_target_goal_command")})
    original_mdp.InitialTargetPositionInRRF = _latch_namespace()["InitialTargetPositionInRRF"]
    ns = {"__name__": __name__, "configclass": configclass, "ObsTerm": Term, "ObsGroup": Group,
          "mdp": original_mdp, "MISSING": MISSING, "Unoise": lambda **kwargs: NS(**kwargs)}
    _nodes("sweeping_policy/shelf_sweep_random_env_cfg.py", {"ObservationsCfg"}, ns)
    sensor_functions = NS(palm_contact_bit=lambda env: None, wrist_wrench_eef=lambda env: None)
    ns["sensor_obs"] = sensor_functions
    _nodes("sweeping_policy/config/ur5e/joint_pos_random_single_inspire_mass_sensors_env_cfg.py",
           {"MassSensorObservationsCfg"}, ns)

    @configclass
    class Parent:
        observations = ns["MassSensorObservationsCfg"]()
        scene = NS(num_envs=2048, robot="sensor_robot", object_collection="single_mass_target", ee_frame="z_01554_tcp")
        actions = NS(arm_action=NS(scale=.5), gripper_action=NS(action_dim=2))
        events = NS(reset_all="default", object_spawn="single_object")
        rewards = NS(reaching=Term(func="reach_gt", weight=3., params={"reaching_z_offset": .04}),
                     orientation=Term(func="align_gt", weight=2.),
                     sweeping_object=Term(func="push_gt", weight=6., params={"command_name": "target_goal_pos", "pushing_z_offset": .04}),
                     homing_after_sweep=Term(func="home_gt", weight=9., params={"command_name": "target_goal_pos"}),
                     object_collision=Term(func="collision_gt", weight=-.5),
                     action_rate=Term(func="rate", weight=-.03), joint_vel=Term(func="velocity", weight=-.03))

        def __post_init__(self):
            self.observations.policy.joint_pos.params = {"asset_cfg": NS(name="robot", joint_names=tuple("abcdef"))}
            self.observations.policy.goal_pos.params = {"command_name": "target_goal_pos"}

    ns.update(UR5eSweepRandomSingleInspireMassSensorsEnvCfg=Parent,
              ConstantNoiseCfg=lambda **kwargs: NS(**kwargs), UniformNoiseCfg=lambda **kwargs: NS(**kwargs),
              NoiseModelWithAdditiveBiasCfg=lambda **kwargs: NS(**kwargs),
              SceneEntityCfg=lambda name: NS(name=name), privileged_obs=NS(**_privileged_namespace()))
    _functions(CFG, ns)
    return Parent(), ns["UR5eSweepRandomSingleInspireInitialMassEnvCfg"](), ns["UR5eSweepRandomSingleInspirePrivilegedMassEnvCfg"](), ns, Term


def test_configs_keep_physics_reset_action_and_reward_contracts_and_fixed_initial_noise():
    parent, initial, privileged, ns, Term = _configs()
    for cfg in (initial, privileged):
        assert cfg.scene == parent.scene
        assert cfg.actions == parent.actions
        assert cfg.events == parent.events  # Reaching is not skipped in either new task.
        assert cfg.observations.policy.goal_pos == parent.observations.policy.goal_pos
        assert cfg.observations.policy.joint_pos == parent.observations.policy.joint_pos
        term = cfg.observations.policy.target_obs_state
        assert term.func is ns["mdp"].InitialTargetPositionInRRF
        assert term.params == {"command_name": "target_goal_pos"}
        assert term.noise.noise_cfg.bias == 0.
        assert term.noise.bias_noise_cfg.n_min == -.01
        assert term.noise.bias_noise_cfg.n_max == .01
        assert term.noise.bias_noise_cfg.operation == "abs"
        assert term.noise.sample_bias_per_component is True
        for name, before in vars(parent.rewards).items():
            after = getattr(cfg.rewards, name)
            assert after.func == before.func
            assert after.weight == before.weight
            for key, value in before.params.items():
                assert after.params[key] == value
    assert initial.rewards == parent.rewards
    assert not hasattr(parent.observations, "privileged")
    assert not hasattr(initial.observations, "privileged")
    assert parent.observations.policy.target_obs_state.func is ns["mdp"].MA_object_position_in_RRF
    assert privileged.observations.policy == initial.observations.policy
    terms = [name for name, value in vars(privileged.observations.privileged).items() if isinstance(value, Term)]
    assert terms == ["target_state"]
    assert not privileged.observations.privileged.enable_corruption
    assert privileged.observations.privileged.concatenate_terms
    assert privileged.observations.privileged.target_state.func is ns["privileged_obs"].target_state_in_robot_root
    for name in ("reaching", "sweeping_object", "homing_after_sweep", "object_collision"):
        assert getattr(privileged.rewards, name).params["object_collection_cfg"].name == "object_collection"
    assert privileged.rewards.reaching.params["ee_frame_cfg"].name == "ee_frame"
    assert privileged.rewards.orientation.params["ee_frame_cfg"].name == "ee_frame"
    assert privileged.rewards.orientation.params["shelf_cfg"].name == "shelf"
    for cls in ("UR5eSweepRandomSingleInspireInitialMassEnvCfg_PLAY", "UR5eSweepRandomSingleInspirePrivilegedMassEnvCfg_PLAY"):
        play = ns[cls]()
        assert play.scene.num_envs == 50
        assert not play.observations.policy.enable_corruption


def _runner_configs():
    configclass = _configclass()

    @configclass
    class Runner:
        obs_groups: dict = {}

    ns = {"__name__": __name__, "configclass": configclass, "RslRlOnPolicyRunnerCfg": Runner,
          "RslRlPpoActorCriticCfg": lambda **kwargs: NS(**kwargs), "RslRlPpoAlgorithmCfg": lambda **kwargs: NS(**kwargs)}
    for file in ("rsl_rl_ppo_cfg_02.py", "rsl_rl_ppo_inspire_open_cfg.py", "rsl_rl_ppo_inspire_hand_cfg.py",
                 "rsl_rl_ppo_inspire_hand_mass_cfg.py", "rsl_rl_ppo_inspire_mass_sensors_cfg.py"):
        _functions(f"sweeping_policy/config/ur5e/agents/{file}", ns)
    parent = ns["UR5eSweepInspireMassSensorsPPORunnerCfg"]()
    _functions(RUNNER, ns)
    initial = ns["UR5eSweepInspireInitialMassPPORunnerCfg"]()
    privileged = ns["UR5eSweepInspirePrivilegedMassPPORunnerCfg"]()
    return parent, initial, privileged


def test_ppo_routes_privileged_only_to_critic_and_preserves_all_parent_hyperparameters():
    parent, initial, privileged = _runner_configs()
    assert initial.obs_groups == {"actor": ["policy"], "critic": ["policy"]}
    assert privileged.obs_groups == {"actor": ["policy"], "critic": ["policy", "privileged"]}
    for cfg in (initial, privileged):
        for name, value in vars(parent).items():
            if name not in ("obs_groups", "experiment_name"):
                assert getattr(cfg, name) == value
    assert len({parent.experiment_name, initial.experiment_name, privileged.experiment_name}) == 3
    privileged.obs_groups["critic"].append("mutated")
    assert initial.obs_groups["critic"] == ["policy"]


@contextmanager
def _single_threaded():
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        yield
    finally:
        torch.set_num_threads(previous)


def test_actual_rsl_ppo_uses_41_actor_54_critic_inputs_and_updates_without_actor_leakage():
    PPO = pytest.importorskip("rsl_rl.algorithms").PPO
    TensorDict = pytest.importorskip("tensordict").TensorDict
    _, _, runner = _runner_configs()
    with torch.random.fork_rng(devices=[]), _single_threaded():
        torch.manual_seed(23)
        obs = TensorDict({"policy": torch.randn(4, 41), "privileged": torch.randn(4, 13)}, batch_size=[4])
        algorithm = dict(vars(runner.algorithm))
        algorithm.update(class_name="PPO")
        config = {
            "obs_groups": runner.obs_groups,
            "actor": {"class_name": "MLPModel", "hidden_dims": runner.policy.actor_hidden_dims,
                      "activation": runner.policy.activation, "obs_normalization": False,
                      "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": runner.policy.init_noise_std}},
            "critic": {"class_name": "MLPModel", "hidden_dims": runner.policy.critic_hidden_dims,
                       "activation": runner.policy.activation, "obs_normalization": False},
            "algorithm": algorithm, "num_steps_per_env": runner.num_steps_per_env, "multi_gpu": None,
        }
        ppo = PPO.construct_algorithm(obs, NS(num_envs=4, num_actions=8), config, "cpu")
        assert ppo.actor.obs_dim == 41
        assert ppo.critic.obs_dim == 54
        changed = obs.clone()
        changed["privileged"] += 1.
        with torch.no_grad():
            actor = ppo.actor(obs)
            assert torch.equal(ppo.actor(changed), actor)
            assert not torch.allclose(ppo.critic(changed), ppo.critic(obs))
            # Deployed actor inference needs no privileged input at all.
            assert torch.equal(ppo.get_policy()(TensorDict({"policy": obs["policy"]}, batch_size=[4])), actor)
        before = [parameter.detach().clone() for parameter in ppo.actor.parameters()]
        with torch.no_grad():
            for _ in range(runner.num_steps_per_env):
                assert ppo.act(obs).shape == (4, 8)
                next_obs = TensorDict({"policy": torch.randn(4, 41), "privileged": torch.randn(4, 13)}, batch_size=[4])
                ppo.process_env_step(next_obs, torch.tensor([-.1, .1, .2, .3]), torch.zeros(4, dtype=torch.bool), {})
                obs = next_obs
            ppo.compute_returns(obs)
        losses = ppo.update()
        assert losses
        assert all(torch.isfinite(torch.tensor(value)) for value in losses.values() if value is not None)
        assert any(not torch.equal(old, new) for old, new in zip(before, ppo.actor.parameters()))


def test_registrations_keep_previous_mass_reaching_task_and_use_distinct_new_runners():
    import sweeping_policy  # Registers metadata without launching a simulator.
    for version, label in ((3, "Initial"), (4, "Privileged")):
        task = gym.spec(f"Isaac-Sweep-Object-UR5e-Random-Single-Inspire-Open-v{version}")
        assert task.entry_point == "sweeping_policy.env_inspire_sensors:ShelfSweepInspireSensorsEnv"
        assert task.kwargs["env_cfg_entry_point"].endswith(f"UR5eSweepRandomSingleInspire{label}MassEnvCfg")
        assert task.kwargs["rsl_rl_cfg_entry_point"].endswith(f"UR5eSweepInspire{label}MassPPORunnerCfg")
    prior = gym.spec("Isaac-Sweep-Object-UR5e-Random-Single-Inspire-Open-v3-mass")
    assert prior.entry_point == "sweeping_policy.env_inspire_mass_reaching:ShelfSweepInspireMassReachingEnv"
    assert prior.kwargs["env_cfg_entry_point"].endswith("UR5eSweepRandomSingleInspireMassReachingEnvCfg")
