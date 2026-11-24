# `Isaac-Sweep-Object-UR5e-Random-v1`

[환경 목록](README.md) · [등록과 상속](ENVIRONMENT_REGISTRY.md) · [설치·실행](../README.md)

이 환경은 `Isaac-Sweep-Object-UR5e-Random-v0`의 양방향 sweep 과제를 그대로 두고, arm policy action만 joint position에서 **relative Cartesian differential IK**로 바꾼 환경이다. Scene, command, observation 항목, reset, reward, termination, curriculum과 binary gripper는 Random-v0를 상속한다.

## 환경 요약

| 항목 | 설정 |
|---|---|
| 환경 config | `UR5eSweepDiffIKEnvCfg` |
| Arm 제어 | 6D relative Cartesian pose → DLS IK → joint-position target |
| Gripper 제어 | 1D binary joint-position target |
| 전체 policy action | 7D |
| Policy observation | 35D |
| Sweep 방향 | reset 때 `+Y` 또는 `-Y`, 거리 `0.18 m` |
| 물체 질량 | 물체당 `1.0 kg` |
| Physics / policy 주기 | `100 Hz` / `50 Hz` |
| Episode 길이 | 최대 `10 s`, 약 500 policy step |

물체에는 별도의 preview material이나 색상을 덮어쓰지 않으며, 가져온 USD의 원래 외형을 그대로 사용한다.

## 1. `CommandsCfg`

유일한 command는 `target_goal_pos`다.

- reset된 target world 위치에서 world/shelf Y축으로 `±0.18 m` 이동한 위치를 goal로 만든다.
- 방향은 `sweep_dir_options=True`에 의해 reset마다 `+Y` 또는 `-Y`로 정해진다.
- command 내부 값은 goal world position 3D와 reset 당시 target quaternion 4D를 합친 7D pose다.
- policy의 `goal_pos`에는 위치 3D만 robot-root frame으로 변환되어 들어간다.
- reward도 goal position을 사용하며 command quaternion을 추종하지 않는다.
- 시간 기반 resampling 범위가 `1e6 s`이므로 episode 안에서는 goal이 다시 뽑히지 않는다.

Arm action도 robot-root frame에서 해석되므로 `target_obs_state`, `ee_pose`, `goal_pos`와 action의 위치 좌표계가 일치한다.

## 2. `ActionsCfg`

### 전체 action 배치

```text
[dx, dy, dz, rx, ry, rz, g]
 └──── relative TCP pose 6D ────┘  └ gripper 1D
```

전체 action은 7D다. Random-v0와 숫자상 차원은 같지만 앞의 6개 값이 뜻하는 물리량은 완전히 다르다.

### 2.1 Arm action: relative Cartesian pose

| Action 성분 | 단위 | 의미 |
|---|---|---|
| `dx, dy, dz` | m | 현재 TCP 위치에 적용할 robot-root frame 병진 변화량 |
| `rx, ry, rz` | rad | 현재 TCP orientation에 적용할 axis-angle 회전 벡터 |

이 action은 다음처럼 처리된다.

1. `robotiq_base_link`에서 local `(0.13, 0, 0) m` 떨어진 TCP의 현재 pose를 robot root 기준으로 읽는다.
2. 현재 pose에 policy의 relative translation과 axis-angle rotation을 적용해 이번 step의 desired TCP pose를 만든다.
3. TCP Jacobian과 DLS(damped least squares) IK를 사용해 필요한 joint 변화량을 계산한다.
4. 계산 결과를 현재 6개 arm joint 위치에 더해 absolute joint-position target으로 actuator에 전달한다.

따라서 policy 관점에서는 Cartesian delta action이지만, simulator에 최종 적용되는 arm 명령은 six-joint position target이다. DLS damping parameter는 `lambda=0.01`이다.

제어점은 observation/reward가 사용하는 `ee_frame`과 같은 TCP offset이므로, policy가 관측하는 EE pose와 IK가 움직이는 점이 일치한다.

#### 현재 scale에 대한 중요 사항

현재 실행 config는 다음과 같다.

```text
scale = 0.5
clip = 없음
```

따라서 실제 처리값은 raw action의 절반이다. 예를 들어 `dx=0.1`은 현재 설정상 `0.05 m` 상대 이동이고 `rz=0.2`는 `0.1 rad` axis-angle 회전이다.

이전 설명의 `scale=1.0`과 과거 제한 주석의 `0.05 m / 0.25 rad` tuple은 현재 설정과 다르다. [test_v1_contract.py](../tests/test_v1_contract.py)는 아직 해당 tuple을 기대하므로 현재 소스와 불일치한다. 이 문서는 실행되는 scalar `scale=0.5`를 기준으로 설명한다. 별도 clip이 없어 action 크기에 대한 최대값을 이 scalar만으로 보장하지 않는다.

### 2.2 Gripper action

마지막 scalar `g`는 Random-v0와 같은 binary gripper position action이다.

- `g >= 0`: open, 8개 gripper joint target을 모두 `0.0`으로 설정
- `g < 0`: close, finger/knuckle별 `+0.5`, `-0.5` 또는 `0.0` target 사용

Gripper는 IK에 포함되지 않으며 arm action과 독립된 별도 action term이다.

## 3. `ObservationsCfg`

Policy observation은 35D다.

| 순서 | Term | 차원 | 의미 | Noise |
|---:|---|---:|---|---|
| 1 | `joint_pos` | 8 | articulation의 첫 8개 joint 현재 위치 - default 위치 | 없음 |
| 2 | `joint_vel` | 6 | 첫 6개 arm joint 현재 속도 - default 속도 | 없음 |
| 3 | `actions` | 7 | 직전 relative pose 6D와 gripper 1D raw action | 없음 |
| 4 | `target_obs_state` | 3 | robot-root frame target 위치 | 각 성분 `[-0.01, 0.01]` |
| 5 | `target_obj_width` | 1 | target 물체 폭 | `[-0.01, 0.01]` |
| 6 | `ee_pose` | 7 | robot-root frame TCP 위치 3D + unique quaternion `(w,x,y,z)` | 없음 |
| 7 | `goal_pos` | 3 | robot-root frame sweep goal 위치 | 없음 |

```text
8 + 6 + 7 + 3 + 1 + 7 + 3 = 35
```

`actions` 항목의 7개 값은 actuator에 적용된 joint target이 아니라 **policy가 출력한 raw Cartesian delta 6D와 gripper scalar**다. 즉 policy는 자신의 직전 task-space 명령을 다시 관측한다.

`joint_pos`가 8D인 이유는 코드가 articulation tensor의 첫 8개 joint를 직접 slice하기 때문이다. 정확한 마지막 두 joint 의미는 USD articulation 순서에 의존한다. `joint_vel`은 6개 arm joint만 포함한다.

## 4. `EventCfg`

v1은 Random-v0 event를 변경하지 않는다.

### `reset_all`

Robot, shelf와 모든 물체를 default state로 복구한다.

### `object_spawn`

1. 여섯 물체 중 target 하나를 선택한다.
2. 물체 이름을 shuffle하여 2×3 slot에 새로 배치한다.
3. 각 물체 XY에 `[-0.02, 0.02] m` jitter를 더한다.
4. yaw를 무작위화하고 초기 속도를 0으로 만든다.
5. sweep direction을 `(0, +0.18, 0)` 또는 `(0, -0.18, 0) m`로 선택한다.
6. 방향에 따라 앞/측면/대각선의 방해 물체를 `z=1.8 m`로 옮긴다.

Target 종류, 물체 순열, XY jitter와 방향은 reset 함수 호출 단위로 공유될 수 있다. 특히 Python `choice`로 선택하는 sweep 방향은 한 호출의 `env_ids`마다 독립적이지 않고 reset batch 전체에 같은 값이 적용된다. Yaw는 환경별로 생성된다.

## 5. `RewardsCfg`

Reward는 Random-v0와 동일하다. 각 raw term에 아래 weight와 `step_dt=0.02 s`가 곱해진다.

| Term | Weight | 역할 |
|---|---:|---|
| `action_rate` | -0.03 | relative Cartesian action 6D와 gripper 1D의 step 간 변화 억제 |
| `joint_vel` | -0.03 | 6개 arm joint 속도 제곱합 억제 |
| `shelf_collision` | -0.5 | shelf 이동과 finger/wrist/TCP의 shelf 근접 패널티 |
| `object_collision` | -0.5 | target 이외 물체의 움직임 패널티 |
| `reaching` | +3.0 | sweep 방향에 맞는 target 측면 접근 유도 |
| `orientation` | +2.0 | TCP Y축과 shelf Z축 정렬 유도 |
| `sweeping_object` | +6.0 | target의 goal 방향 이동 및 적정 속도 보상 |
| `homing_after_sweep` | +9.0 | goal 도달 후 default arm pose 복귀 유도 |

### `action_rate`

현재와 이전 raw action의 차이 제곱합이다. v1에서는 meter 단위 translation, radian 단위 rotation vector와 gripper scalar가 한 벡터에 함께 포함된다. 좌표마다 별도의 단위 정규화 weight는 없다.

### `joint_vel`

6개 arm joint의 실제 속도 제곱합이다. Policy action은 Cartesian이지만 이 penalty는 IK 결과로 생긴 joint-space 움직임을 평가한다.

### `shelf_collision`

- Shelf 초기 위치 변화 norm과 6D root velocity norm의 합이 `0.005` 초과이면 raw `1`.
- TCP가 shelf 기준점 `z+1.06 m`에서 `0.2 m` 이내일 때, finger 두 개의 `0.02 m` 수직 여유와 wrist의 `0.08 m` 여유가 작을수록 각각 최대 `1`의 penalty.

Contact force가 아니라 pose/velocity 기반 heuristic이다.

### `object_collision`

Target 이외 물체들의 반올림된 XYZ 선속도 절댓값 합에 `tanh`를 적용한다. 실제 collision event 자체가 아니라 주변 물체가 움직인 정도를 측정한다.

### `reaching`

접근점은 target에서 X `-0.02 m`, Z `+0.09 m`, Y `-target_width×sign(sweep_dir_y)`만큼 이동한 지점이다. TCP가 이 지점에 가까워질수록 `exp(-10×distance)` 값이 커진다. 계산된 wrist 거리는 최종 reaching reward에 사용되지 않는다.

### `orientation`

TCP Y축과 shelf Z축 내적의 부호 유지 제곱값이다. 같은 방향이면 양수, 반대 방향이면 음수, 직교하면 0이다.

### `sweeping_object`

- Target-goal 3D 거리 `<0.03 m`: `2×exp(-5×distance)` 성공 구간 보상.
- 그 밖에서는 TCP-접근점 거리 `<0.04 m`이고 wrist Y 오차 `<0.04 m`일 때만 진행 보상을 준다.
- `0.05 < |target_vy| < 0.1 m/s`이면 `+0.5`, `|target_vy| >= 0.1 m/s`이면 `-0.5` 속도 보정이 적용된다.
- 속도는 절댓값만 보므로 goal과 반대 방향인지까지는 검사하지 않는다.

### `homing_after_sweep`

Target과 goal의 YZ 거리 `0.03 m` 부근에서 smooth `tanh` gate가 켜진다. 첫 5개 arm joint가 default 위치에 가까울수록 보상이 증가하며 `wrist_3_joint`는 home 오차에서 빠진다.

## 6. `TerminationsCfg`

v1도 Random-v0와 같은 termination을 사용한다.

| Term | 종료 조건 |
|---|---|
| `time_out` | `10 s`, 약 500 policy step |
| `object_drop` | 여섯 물체 중 하나라도 `z < 1.04 m` 또는 `|roll|/|pitch| > 0.9 rad` |
| `push_fast` | target의 3D 선속도 norm `> 0.3 m/s` |
| `shelf_collision` | shelf 6D root velocity norm `> 0.1`, finger Z `< shelf Z+1.06 m`, 또는 wrist Z `< shelf Z+1.12 m` |
| `hand_velocity` | 6개 arm joint 중 하나라도 `|joint_vel| > 1.0 rad/s` |

Differential IK 명령이 커서 arm joint 속도가 급격히 증가하면 `hand_velocity` 또는 `push_fast`가 빠르게 episode를 끝낼 수 있다. 특히 현재 action scale이 `0.5`이고 clip이 없다는 점과 함께 해석해야 한다.

성공 판정 함수는 active termination으로 등록되어 있지 않으므로 target이 goal에 도달해도 즉시 끝나지 않는다. 이후 homing reward를 받으며 움직일 수 있다.

## 7. `CurriculumCfg`

`obj_collision` curriculum은 global common step이 `250,000`을 넘으면 `object_collision` weight를 `-0.5`로 설정한다. 초기 weight도 이미 `-0.5`이므로 실질적인 변화는 없다. 주석 처리된 homing curriculum은 활성화되지 않는다.

## 학습 runner 참고

Random-v0와 같은 `rsl_rl_ppo_cfg_02.py`를 사용한다. 주요 값은 최대 `90,000` iteration, `gamma=0.98`, `entropy_coef=0.005`다. Actor output은 7D지만 그중 arm 6D의 의미가 joint target이 아니라 relative Cartesian pose라는 점이 Random-v0와의 핵심 차이다.

## 주요 구현 파일

- v1 override: [`diff_ik_random_sweep_env_cfg.py`](../sweeping_policy/config/ur5e/diff_ik_random_sweep_env_cfg.py)
- 상속하는 Random-v0 config: [`joint_pos_random_sweep_env_cfg.py`](../sweeping_policy/config/ur5e/joint_pos_random_sweep_env_cfg.py)
- Random 공통 MDP: [`shelf_sweep_random_env_cfg.py`](../sweeping_policy/shelf_sweep_random_env_cfg.py)
- reward: [`reward_random_sweep.py`](../sweeping_policy/mdp/reward_random_sweep.py)
- termination: [`terminations.py`](../sweeping_policy/mdp/terminations.py)

[통합 비교 문서로 돌아가기](README.md)
