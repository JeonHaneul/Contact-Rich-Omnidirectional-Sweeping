# `Isaac-Sweep-Object-UR5e-Random-v0`

[환경 목록](README.md) · [등록과 상속](ENVIRONMENT_REGISTRY.md) · [설치·실행](../README.md)

이 환경은 UR5e가 선반 위 target 물체를 `+Y` 또는 `-Y` 방향으로 쓸어내도록 학습하는 양방향 joint-position 환경이다. Random-v1과 Random-v2가 공통으로 상속하는 기준 scene, command, observation, event, reward와 termination을 정의한다.

## 환경 요약

| 항목 | 설정 |
|---|---|
| 환경 config | `UR5eSweepEnvCfg` |
| Arm 제어 | 6D joint-position target |
| Gripper 제어 | 1D binary joint-position target |
| 전체 policy action | 7D |
| Policy observation | 35D |
| Sweep 방향 | reset 때 `+Y` 또는 `-Y`, 거리 `0.18 m` |
| 물체 질량 | 물체당 `1.0 kg` |
| Physics / policy 주기 | `100 Hz` / `50 Hz` |
| Episode 길이 | 최대 `10 s`, 약 500 policy step |

## 1. `CommandsCfg`

등록된 command는 `target_goal_pos` 하나다.

- reset 때 선택한 target 물체의 초기 world 위치에서 sweep 방향으로 `0.18 m` 떨어진 목표를 만든다.
- `sweep_dir_options=True`이므로 Y 방향 부호는 reset 때 `+1` 또는 `-1`로 선택된다.
- 실제 goal 위치는 `target의 reset 직후 world 위치 + (0, ±0.18, 0) m`다.
- command 출력은 목표 위치 3D와 target의 초기 quaternion 4D를 합친 7D pose다.
- policy의 `goal_pos` observation은 그중 위치 3D만 robot-root frame으로 변환해 사용한다.
- reward도 주로 command 위치를 사용하며 target orientation을 goal로 맞추지는 않는다.
- command의 시간 기반 resampling interval은 `1e6 s`이므로 episode 중에는 사실상 고정되고 reset 때 갱신된다.

## 2. `ActionsCfg`

### 전체 action 배치

```text
[a0, a1, a2, a3, a4, a5, g]
 └──────── arm 6D ────────┘  └ gripper 1D
```

전체 action은 7D이며 `Isaac-Sweep-Object-UR5e-v0`와 동일한 표현을 사용한다.

### 2.1 Arm action: default 자세 기준 absolute joint-position target

아래 표는 config가 의도하는 표준 UR5e joint 순서다. `preserve_order=True`를 지정하지 않았으므로 최종 action component 순서는 runtime에서 USD articulation joint 순서로 resolve된다는 점은 유의해야 한다.

| Action index | Joint | Default position |
|---:|---|---:|
| 0 | `shoulder_pan_joint` | `0.0 rad` |
| 1 | `shoulder_lift_joint` | `-2.2 rad` |
| 2 | `elbow_joint` | `2.2 rad` |
| 3 | `wrist_1_joint` | `0.0 rad` |
| 4 | `wrist_2_joint` | `1.57 rad` |
| 5 | `wrist_3_joint` | `0.785 rad` |

각 arm action은 다음 joint-position target으로 변환된다.

```text
desired joint position = default joint position + 0.5 × raw arm action
```

중요한 물리적 의미는 다음과 같다.

- 현재 joint 위치에 action을 더하는 relative/incremental 명령이 아니다.
- action `0`은 직전 자세를 유지하라는 뜻이 아니라 default joint target으로 돌아가라는 뜻이다.
- 처리된 값은 absolute joint-position target이며 implicit actuator의 PD 동작이 이를 추종한다.
- 별도 action clip은 설정되어 있지 않다. `[-1, 1]`을 가정하면 default에서 약 `±0.5 rad`지만 환경이 이 범위를 강제하지 않는다.
- policy가 직접 torque를 출력하지 않는다.

### 2.2 Gripper action

마지막 1D 값은 binary position 명령이다.

- `g >= 0`: open
- `g < 0`: close

Open일 때 8개 gripper target은 모두 `0.0`이다. Close일 때 주요 finger/knuckle target은 부호에 따라 `±0.5 rad`이고 `left_outer_finger_joint`, `right_outer_finger_joint`는 `0.0`을 유지한다.

| Gripper joint | Close target |
|---|---:|
| `finger_joint` | 0.5 |
| `right_outer_knuckle_joint` | 0.5 |
| `left_outer_finger_joint` | 0.0 |
| `left_inner_finger_knuckle_joint` | -0.5 |
| `left_inner_finger_joint` | -0.5 |
| `right_outer_finger_joint` | 0.0 |
| `right_inner_finger_joint` | 0.5 |
| `right_inner_finger_knuckle_joint` | -0.5 |

## 3. `ObservationsCfg`

Policy observation은 35D다.

| 순서 | Term | 차원 | 의미 | Noise |
|---:|---|---:|---|---|
| 1 | `joint_pos` | 8 | articulation joint index `0:8`의 현재 위치 - default 위치 | 없음 |
| 2 | `joint_vel` | 6 | 첫 6개 arm joint의 현재 속도 - default 속도 | 없음 |
| 3 | `actions` | 7 | 현재 observation 직전에 입력된 전체 raw action | 없음 |
| 4 | `target_obs_state` | 3 | target 물체의 robot-root frame 위치 | 각 성분 `[-0.01, 0.01]` |
| 5 | `target_obj_width` | 1 | 선택된 target의 폭 | `[-0.01, 0.01]` |
| 6 | `ee_pose` | 7 | robot-root frame TCP 위치 + unique quaternion `(w, x, y, z)` | 없음 |
| 7 | `goal_pos` | 3 | robot-root frame sweep goal 위치 | 없음 |

```text
8 + 6 + 7 + 3 + 1 + 7 + 3 = 35
```

`joint_pos`는 이름 기반 선택이 아니라 articulation tensor의 첫 8개 joint를 직접 사용한다. 이 asset에서 의도한 구성은 6개 arm DoF와 앞쪽 gripper 관련 DoF를 함께 보는 것이다. `joint_vel`은 arm 6개만 포함한다. 좌우 sweep 방향은 별도의 scalar observation으로 제공되지 않지만, target 위치와 반대쪽에 놓인 `goal_pos`를 함께 보면 방향을 알 수 있다.

## 4. `EventCfg`

### `reset_all`

Robot, shelf와 물체를 기본 상태로 복원한다.

### `object_spawn`

reset 때 다음 순서로 scene을 구성한다.

1. bottle/cup/mug/can을 포함한 여섯 물체 중 target 하나를 무작위로 선택한다.
2. target의 collection ID와 YAML 폭을 `env.target_id`, `env.target_width`에 저장한다.
3. 물체 이름 순서를 shuffle하여 2×3 shelf slot에 새로 배치한다.
4. 각 물체 XY 위치에 `[-0.02, 0.02] m` 오차를 추가한다.
5. 물체 yaw를 무작위화하고 초기 선속도·각속도를 0으로 만든다.
6. sweep Y 방향을 `+0.18 m` 또는 `-0.18 m`로 선택한다.
7. 양의 방향이면 target의 앞/오른쪽/앞-오른쪽, 음의 방향이면 앞/왼쪽/앞-왼쪽에 있는 방해 물체를 `z=1.8 m`로 옮긴다.

구현상 target 선택, 물체 순서 shuffle과 sweep 방향 선택은 Python 난수를 event 호출당 한 번 사용한다. 따라서 한 reset 호출에 함께 들어온 `env_ids`는 같은 target 종류, slot 순열과 방향을 공유한다. “각 병렬 환경이 항상 독립적으로 좌우를 샘플링한다”는 의미는 아니다. yaw는 환경별로 생성된다.

## 5. `RewardsCfg`

모든 raw term에는 설정 weight와 `step_dt=0.02 s`가 곱해진다.

| Term | Weight | 역할과 실제 판정 |
|---|---:|---|
| `action_rate` | -0.03 | 7D action의 step 간 변화량 제곱합을 패널티로 준다. |
| `joint_vel` | -0.03 | 6개 arm joint 속도의 제곱합을 패널티로 준다. |
| `shelf_collision` | -0.5 | shelf 이동 및 TCP/finger/wrist의 shelf 근접도를 패널티로 준다. |
| `object_collision` | -0.5 | target 이외 물체의 움직임을 패널티로 준다. |
| `reaching` | +3.0 | 현재 sweep 방향의 반대편 접근점으로 TCP를 유도한다. |
| `orientation` | +2.0 | TCP Y축과 shelf Z축의 올바른 정렬을 유도한다. |
| `sweeping_object` | +6.0 | 양방향 target 진행, 접근 위치와 적정 속도를 보상한다. |
| `homing_after_sweep` | +9.0 | target이 goal에 가까워지면 arm의 default 자세 복귀를 보상한다. |

### `action_rate`

현재와 이전 전체 action 차이의 제곱합이다. Arm Cartesian/joint 의미와 무관하게 action 좌표계의 raw 변화량을 측정하며 gripper 부호 변화도 포함한다.

### `joint_vel`

첫 6개 arm joint의 속도 제곱합이다. gripper joint는 제외한다.

### `shelf_collision`

다음 두 값을 합친다.

- 초기 shelf 위치 변화 norm과 shelf `root_vel_w` norm의 합이 `0.005`를 넘으면 `1`.
- TCP가 `shelf root + z 1.06 m` 기준점의 `0.2 m` 안에 있을 때 finger 두 개와 wrist의 낮은 높이에 비례한 `0~3` 근접 penalty.

Finger는 기준점 위 `0.02 m`, wrist는 `0.08 m` 범위 안에서 가까워질수록 penalty가 커진다. 실제 contact force가 아니라 pose와 shelf 상태에 기반한 proxy다.

### `object_collision`

모든 물체의 선속도를 `0.01` 단위로 반올림하고 target 속도는 제외한다. 나머지 물체의 모든 선속도 성분 절댓값을 합한 뒤 `tanh`를 적용한다. 주변 물체 하나라도 움직이면 값이 증가하고 큰 움직임에서는 1에 가까워진다.

### `reaching`

접근점은 target 중심에서 X `-0.02 m`, Z `+0.09 m` 이동하고, Y는 sweep 부호에 따라 다음처럼 정한다.

```text
approach_y = target_y - target_width × sign(sweep_direction_y)
```

즉 `+Y`로 밀 때는 target의 `-Y` 측면, `-Y`로 밀 때는 `+Y` 측면에 접근하게 한다. TCP와 접근점의 3D 거리에 `exp(-10×distance)` 감쇠를 적용한다. wrist 거리도 계산되지만 최종 reaching 값에는 사용하지 않는다.

### `orientation`

TCP local Y축과 shelf Z축 내적의 부호를 유지하면서 제곱한다. 올바른 같은 방향 정렬은 양의 보상, 반대 방향 정렬은 음의 값, 직교는 0이다.

### `sweeping_object`

Target과 goal의 3D 거리를 중심으로 계산한다.

- goal 거리 `< 0.03 m`: raw reward는 `2×exp(-5×distance)`로, goal 중심에 가까울수록 2에 접근한다.
- 그 밖의 구간: TCP가 접근점 `0.04 m` 이내이고 wrist Y 오차가 `0.04 m` 미만일 때 진행 reward가 활성화된다.
- target Y 속도 절댓값이 `0.05~0.1 m/s`이면 `+0.5`, `0.1 m/s` 이상이면 `-0.5`다.
- 속도에 절댓값을 사용하므로 좌우 어느 방향도 같은 기준으로 평가한다.
- 진행량은 goal 거리와 기본 sweep 거리 `0.18 m`를 비교해 계산한다.

### `homing_after_sweep`

Target-goal 거리는 YZ 평면에서 측정한다. 첫 5개 arm joint의 default 자세 오차가 작을수록 home-pose 값이 커진다. v0의 hard on/off와 달리 goal 거리 `0.03 m` 주변에서 steep한 `tanh` gate를 사용하므로 경계가 연속적으로 변한다. 여섯 번째 wrist joint는 home 오차에 포함하지 않는다.

## 6. `TerminationsCfg`

| Term | 종료 조건 | Time-out 여부 |
|---|---|---|
| `time_out` | `10 s` 경과 | 예 |
| `object_drop` | 여섯 물체 중 하나라도 `z < 1.04 m`, 또는 `|roll| > 0.9 rad`, 또는 `|pitch| > 0.9 rad` | 아니오 |
| `push_fast` | target의 3D 선속도 크기 `> 0.3 m/s` | 아니오 |
| `shelf_collision` | shelf `root_vel_w` norm `> 0.1`, finger Z `< shelf Z+1.06 m`, 또는 wrist Z `< shelf Z+1.12 m` | 아니오 |
| `hand_velocity` | 6개 arm joint 중 하나라도 속도 절댓값 `> 1.0 rad/s` | 아니오 |

추가로 알아둘 점:

- `object_drop`은 target이 아닌 물체도 포함해 모든 여섯 물체를 검사한다.
- `shelf_collision`은 contact sensor 기반이 아니라 shelf 속도와 frame 높이 기반이다.
- shelf `root_vel_w`의 선속도와 각속도 성분이 하나의 norm에 함께 들어간다.
- `success_sweeping`은 termination으로 등록되지 않았다. Goal 도달 후에도 homing을 수행하거나 다른 failure/time-out이 발생할 때까지 episode가 계속된다.

## 7. `CurriculumCfg`

`obj_collision` curriculum은 global common step이 `250,000`을 넘으면 `object_collision` weight를 `-0.5`로 설정한다.

그러나 이 환경의 초기 weight가 이미 `-0.5`이므로 실제 reward 강도 변화는 없다. Random-v1과 Random-v2도 이 설정을 그대로 상속하므로 동일하게 실질적인 no-op curriculum이다. 주석 처리된 homing curriculum은 실행되지 않는다.

## 학습 runner 참고

이 task는 `rsl_rl_ppo_cfg_02.py`를 사용한다. 주요 값은 최대 `90,000` iteration, `gamma=0.98`, `entropy_coef=0.005`이며 experiment 이름은 `UR5e_shelf_sweep_random`이다.

## 주요 구현 파일

- 환경별 config: [`joint_pos_random_sweep_env_cfg.py`](../sweeping_policy/config/ur5e/joint_pos_random_sweep_env_cfg.py)
- Random 공통 MDP config: [`shelf_sweep_random_env_cfg.py`](../sweeping_policy/shelf_sweep_random_env_cfg.py)
- reward: [`reward_random_sweep.py`](../sweeping_policy/mdp/reward_random_sweep.py)
- event: [`events.py`](../sweeping_policy/mdp/events.py)
- termination: [`terminations.py`](../sweeping_policy/mdp/terminations.py)

[통합 비교 문서로 돌아가기](README.md)
