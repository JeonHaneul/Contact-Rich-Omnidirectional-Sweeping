# `Isaac-Sweep-Object-UR5e-Random--Non-Tracking-v1`

[환경 목록](README.md) · [등록과 상속](ENVIRONMENT_REGISTRY.md) · [설치·실행](../README.md)

이 환경은 `Isaac-Sweep-Object-UR5e-Random-v0`를 직접 상속하며,
target 물체의 state와 width를 policy에 주지 않는다. 대신 reset 이벤트와
reward는 필요한 실시간 상태를 privileged information으로 사용한다.
task ID의 `Random` 뒤 이중 하이픈(`--`)은 실제 등록 문자열의 일부다.

## 차이점

### Policy observation

Random-v0의 observation에서 다음 두 term을 완전히 제거한다.

- `target_obs_state` 3D
- `target_obj_width` 1D

정책이 받는 term은 `joint_pos`, `joint_vel`, `actions`, `ee_pose`,
`goal_pos`만 남는다. joint-position action이 7D이므로 총 차원은 다음과 같다.

```text
8 + 6 + 7 + 7 + 3 = 31D
```

`goal_pos`는 제거하지 않는다. 기존 command가 reset 시점 target 위치와
sweep 방향으로 생성한 고정 목표를 그대로 제공한다.

### Privileged reset IK

reset 이벤트는 다음 순서로 실행된다.

1. `reset_all`: scene과 robot을 default 상태로 복구
2. `object_spawn`: target 선택, 물체 배치, width와 sweep 방향 저장
3. `initialize_reaching_pose`: 선택된 target 옆으로 arm을 이동

세 번째 이벤트는 policy observation이나 command snapshot을 읽지 않는다.
`object_collection` 내부 tensor의 초기 target 위치, `env.target_width`,
`env.sweep_dir`만 privileged하게 사용한다. command manager는 reset event 전체가
끝난 후 reset되므로, 이 순서에서 이전 episode의 command를 참조하지 않는다.

목표 TCP 위치는 기존 reaching reward의 pre-contact 위치와 같다.

```text
x = target_init_x - 0.02
y = target_init_y - target_width * sign(sweep_dir_y)
z = target_init_z + 0.09
```

각 xyz 성분에 reset당 uniform noise `[-0.01, 0.01] m`를 더한다.
TCP는 `robotiq_base_link`에서 local `(0.13, 0, 0) m` 지점으로 정의하며,
선반 z-axis에 EEF y-axis를 맞추되 nominal EEF x-axis의 twist를 가능한
유지한다.

DLS IK 설정은 다음과 같다.

- 최대 iteration: `40`
- damping: `0.05`
- step size: `0.5`, joint update clip `±0.2 rad`
- 위치 tolerance: `0.003 m`
- 회전 tolerance: `0.001 rad`
- joint seed offset: zero, `(+0.15, -0.10, +0.10, 0, 0, 0)`,
  `(-0.15, +0.10, -0.10, 0, 0, 0)`

모든 seed가 실패하면 해당 reset에서 `RuntimeError`를 발생시켜 잘못된
초기 pose로 episode가 시작되지 않게 한다. 성공한 joint state는 첫 policy
step의 position target으로 유지하고 velocity target은 `0`으로 초기화한다.

### Reward의 privileged information

`RewardsCfg`는 Random-v0의 term을 유지한다. pushing, homing, orientation,
shelf/object collision은 policy observation이 아니라 simulator의 현재 object pose·velocity,
robot/frame state와 내부 width/sweep 정보를 직접 사용한다.

reset 직후 TCP가 이미 target 옆의 pre-contact 위치에 있으므로
`reaching` reward weight만 `3.0`에서 `0.0`으로 변경한다.

그 밖의 scene, action, command, termination, curriculum과 PPO runner 설정은
Random-v0와 같다.

## 실행

```bash
/path/to/IsaacLab/isaaclab.sh -p train.py \
  --task Isaac-Sweep-Object-UR5e-Random--Non-Tracking-v1 \
  --num_envs 4096 \
  --headless
```

## 구현 파일

- 환경 override: [`joint_pos_random_non_tracking_v1_sweep_env_cfg.py`](../sweeping_policy/config/ur5e/joint_pos_random_non_tracking_v1_sweep_env_cfg.py)
- privileged reset IK: [`events.py`](../sweeping_policy/mdp/events.py)
- 실시간 reward: [`reward_random_sweep.py`](../sweeping_policy/mdp/reward_random_sweep.py)
- 환경 등록: [`__init__.py`](../sweeping_policy/__init__.py)
- Isaac Sim smoke check: [`isaac_smoke_non_tracking_v1.py`](../tests/isaac_smoke_non_tracking_v1.py)
