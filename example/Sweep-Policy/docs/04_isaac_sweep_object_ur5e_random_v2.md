# `Isaac-Sweep-Object-UR5e-Random-v2`

[환경 목록](README.md) · [등록과 상속](ENVIRONMENT_REGISTRY.md) · [설치·실행](../README.md)

현재 코드는 양방향 [Random-v0](02_isaac_sweep_object_ur5e_random_v0.md)를 상속해 arm action을 **relative pose fixed-impedance OSC**로 교체한다. OSC가 TCP 목표 pose 오차를 arm joint effort로 변환한다. 이 문서는 2026-10-14 코드 설정에 맞춰 이전 absolute pose 설명을 교정했다.

## 기준 환경과의 차이

| 항목 | 현재 설정 |
|---|---|
| Config / 직접 부모 | `UR5eSweepOSCEnvCfg` / `UR5eSweepEnvCfg` |
| Runtime class | `ShelfSweepEnv` |
| Arm action | relative TCP pose 6D `[dx, dy, dz, rx, ry, rz]` |
| Gripper action | Random-v0의 binary position command 1D |
| 전체 action / policy observation | 7D / 35D |
| 제어점 | `robotiq_base_link` local `(0.13, 0, 0) m` TCP |
| Arm implicit stiffness / damping | 모두 `0.0` |
| PPO runner | `rsl_rl_ppo_cfg_02:UR5eSweepPPORunnerCfg` |

```text
[dx, dy, dz, rx, ry, rz, g]
 └── relative TCP pose 6D ──┘  gripper 1D
```

위치 delta는 meter, 회전 delta는 axis-angle radian으로 해석한다. Position/orientation scale은 모두 `1.0`이며 이 config에 별도 action clip은 없다. 따라서 action 분포가 과도한 TCP delta를 요청할 수 있다. Policy가 직접 torque를 내보내지는 않는다.

## OSC 설정

[osc_random_sweep_env_cfg.py](../sweeping_policy/config/ur5e/osc_random_sweep_env_cfg.py)의 실제 `OperationalSpaceControllerCfg`가 기준이다.

| 항목 | 값 |
|---|---|
| `target_types` | `["pose_rel"]` |
| `impedance_mode` | `fixed` |
| Motion axes | translation·rotation 6축 모두 활성 |
| Wrench axes | 모두 비활성 |
| `motion_stiffness_task` | 6축 모두 `100.0` |
| `motion_damping_ratio_task` | 6축 모두 `1.0` |
| `inertial_dynamics_decoupling` | `True` |
| `partial_inertial_dynamics_decoupling` | `False` |
| `gravity_compensation` | `False` |
| `nullspace_control` / target | `none` / `none` |

Arm implicit position drive gain을 0으로 만들어 OSC effort가 drive torque와 동시에 더해지는 것을 피한다. 그리퍼 actuator는 기준 환경을 유지한다. Stiffness와 damping은 고정된 controller 설정이며 action에 포함되지 않는다.

## 유지하는 MDP

Scene, command, reset, observation term, reward, termination, curriculum과 gripper는 [Random-v0 상세 설명](02_isaac_sweep_object_ur5e_random_v0.md)을 따른다. `last_action`이 기준과 같은 7D여서 observation도 `8 + 6 + 7 + 3 + 1 + 7 + 3 = 35D`다. `ee_pose`의 quaternion은 observation이며 6D relative arm action과 구분한다.

Reward의 `action_rate`는 relative pose 6D와 gripper 1D의 raw action 변화량에 적용된다. Joint-velocity penalty와 종료 조건은 OSC가 만든 실제 joint state를 읽는다. 큰 delta로 인한 속도·접촉 변화는 `hand_velocity`, `push_fast`, `shelf_collision` 종료로 이어질 수 있다. 목표 도달 성공 함수는 active termination에 등록되어 있지 않다.

## 실행

Workspace 루트에서:

```bash
./IsaacLab/isaaclab.sh -p example/Sweep-Policy/train.py \
  --task Isaac-Sweep-Object-UR5e-Random-v2 \
  --num_envs 4096 --run_name random_relative_osc --headless
```

재생 방법과 자산 설정은 [README](../README.md)를 본다.

## 남아 있는 감사 항목

[test_v2_osc_contract.py](../tests/test_v2_osc_contract.py)는 `pose_abs`, gravity compensation 활성, gain `80/40`을 기대한다. 실제 설정은 위 표처럼 다르다. 이 정리는 학습 동작을 변경하지 않고 문서와 class docstring을 현재 구현에 맞췄다. OSC의 의도된 계약과 테스트 기대값을 맞추는 작업은 별도로 수행해야 한다. 이전 8D absolute pose 정책과 현재 7D relative pose 정책의 체크포인트 호환성을 가정하지 않는다.
