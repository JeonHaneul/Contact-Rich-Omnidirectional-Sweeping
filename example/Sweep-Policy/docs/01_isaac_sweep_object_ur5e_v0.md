# `Isaac-Sweep-Object-UR5e-v0`

[환경 목록](README.md) · [등록과 상속](ENVIRONMENT_REGISTRY.md) · [설치·실행](../README.md)

이 환경은 UR5e가 선반 위의 여러 물체 중 하나를 골라 항상 `+Y` 방향으로 쓸어내는 joint-position 제어 환경이다. 이름에 `Random`이 없지만 target 물체, 물체 배치, 작은 위치 오차와 yaw는 reset마다 바뀐다. Random 계열과 구별되는 핵심은 sweep 방향이 고정되어 있고 기존 reward 설계를 사용한다는 점이다.

## 환경 요약

| 항목 | 설정 |
|---|---|
| 환경 config | `UR5eShelfEnvCfg` |
| Arm 제어 | 6D joint-position target |
| Gripper 제어 | 1D binary joint-position target |
| 전체 policy action | 7D |
| Policy observation | 35D |
| Sweep 방향 | 항상 world/shelf Y축의 `+0.18 m` 방향 |
| 물체 질량 | 물체당 `0.5 kg` |
| Physics / policy 주기 | `100 Hz` / `50 Hz` |
| Episode 길이 | 최대 `10 s`, 약 500 policy step |

## 1. `CommandsCfg`

등록된 command는 `target_goal_pos` 하나다.

- reset 때 선택된 target 물체의 world 위치를 기준으로 목표를 만든다.
- `sweep_dir_options=False`이므로 sweep vector는 항상 `(0, +0.18, 0) m`다.
- 목표 위치는 `target의 reset 직후 world 위치 + sweep vector`다.
- command 자체는 목표 위치 3D와 target의 초기 quaternion 4D를 합친 7D pose를 반환한다.
- policy observation과 reward는 이 command 중 위치 3D를 주로 사용한다. 목표 orientation을 추종하는 reward는 없다.
- `resampling_time_range=(1e6, 1e6)`이므로 episode 중 시간에 따른 재샘플링은 사실상 없고, reset 때 갱신된다.

즉, policy가 받는 `goal_pos`는 “현재 target을 오른쪽으로 18 cm 이동시켜야 하는 위치”를 robot-root frame으로 변환한 값이다.

## 2. `ActionsCfg`

### 전체 action 배치

```text
[a0, a1, a2, a3, a4, a5, g]
 └──────── arm 6D ────────┘  └ gripper 1D
```

전체 action은 7D다.

### 2.1 Arm action: default 자세 기준 absolute joint-position target

앞의 6개 값은 다음 UR5e arm joint에 적용된다. 아래 표는 config가 의도하는 표준 UR5e joint 순서를 나타낸다.

| Action index | Joint | Default position |
|---:|---|---:|
| 0 | `shoulder_pan_joint` | `0.0 rad` |
| 1 | `shoulder_lift_joint` | `-2.2 rad` |
| 2 | `elbow_joint` | `2.2 rad` |
| 3 | `wrist_1_joint` | `0.0 rad` |
| 4 | `wrist_2_joint` | `1.57 rad` |
| 5 | `wrist_3_joint` | `0.785 rad` |

`preserve_order=True`를 지정하지 않았으므로 최종 action component 순서는 runtime에서 USD articulation joint 순서로 resolve된다.

각 값은 현재 joint 위치에 더하는 incremental action이 아니다. 처리된 target은 다음 의미를 갖는다.

```text
desired joint position = default joint position + 0.5 × raw arm action
```

따라서:

- arm action `0`은 해당 joint를 default 위치로 보내는 명령이다.
- 예를 들어 첫 action이 `0.4`라면 shoulder pan target은 `0 + 0.5×0.4 = 0.2 rad`다.
- 매 step의 target은 default 자세를 기준으로 다시 해석된다. 현재 joint 위치 기준 delta가 아니다.
- action config에는 별도 clip이 없다. 일반적으로 policy 출력을 `[-1, 1]` 부근으로 운용하면 default에서 약 `±0.5 rad` 범위를 뜻하지만, 환경 자체가 그 범위를 강제로 자르지는 않는다.
- target은 UR5e implicit actuator의 stiffness/damping을 통해 실제 joint torque로 변환된다. policy가 torque를 직접 출력하지 않는다.

### 2.2 Gripper action: binary open/close

마지막 값 `g` 하나가 8개 gripper joint의 position target 묶음으로 변환된다.

- `g >= 0`: open configuration
- `g < 0`: close configuration

| Gripper joint | Open target | Close target |
|---|---:|---:|
| `finger_joint` | 0.0 | 0.5 |
| `right_outer_knuckle_joint` | 0.0 | 0.5 |
| `left_outer_finger_joint` | 0.0 | 0.0 |
| `left_inner_finger_knuckle_joint` | 0.0 | -0.5 |
| `left_inner_finger_joint` | 0.0 | -0.5 |
| `right_outer_finger_joint` | 0.0 | 0.0 |
| `right_inner_finger_joint` | 0.0 | 0.5 |
| `right_inner_finger_knuckle_joint` | 0.0 | -0.5 |

## 3. `ObservationsCfg`

Policy observation은 아래 순서로 concatenate된 35D 벡터다.

| 순서 | Term | 차원 | 의미 | Noise |
|---:|---|---:|---|---|
| 1 | `joint_pos` | 8 | articulation joint index `0:8`의 현재 위치에서 default 위치를 뺀 값 | 없음 |
| 2 | `joint_vel` | 6 | arm joint index `0:6`의 현재 속도에서 default 속도를 뺀 값 | 없음 |
| 3 | `actions` | 7 | 직전에 환경에 입력되어 현재 적용된 전체 raw action | 없음 |
| 4 | `target_obs_state` | 3 | 선택된 target 물체의 robot-root frame 위치 `(x, y, z)` | 각 성분 `[-0.01, 0.01]` |
| 5 | `target_obj_width` | 1 | YAML에 저장된 target 물체 폭 | `[-0.01, 0.01]` |
| 6 | `ee_pose` | 7 | robot-root frame TCP 위치 3D + quaternion `(w, x, y, z)` 4D | 없음 |
| 7 | `goal_pos` | 3 | robot-root frame의 target sweep 목표 위치 | 없음 |

```text
8 + 6 + 7 + 3 + 1 + 7 + 3 = 35
```

주의할 점은 `joint_pos`가 이름으로 joint를 선택하지 않고 articulation tensor의 첫 8개 값을 직접 자른다는 것이다. 이 asset에서는 6개 arm DoF와 앞쪽 gripper 관련 DoF를 함께 관측하려는 구성이다. 반면 `joint_vel`은 첫 6개 arm joint만 관측한다. `ee_pose` quaternion은 같은 회전을 두 부호로 표현하는 문제를 줄이기 위해 실수부가 양수가 되도록 unique 처리된다.

## 4. `EventCfg`

두 event 모두 reset mode에서 실행된다.

### `reset_all`

`reset_scene_to_default`를 호출하여 robot, shelf와 물체 상태를 기본 상태로 복구한다.

### `object_spawn`

`randomize_scene`이 다음 작업을 수행한다.

1. 여섯 물체 중 target 하나를 무작위로 선택하고 `target_id`, `target_width`를 기록한다.
2. 여섯 물체 이름을 shuffle하여 YAML의 2×3 slot에 다시 대응시킨다.
3. 각 물체의 XY 위치에 `[-0.02, 0.02] m` 오차를 더한다.
4. 각 환경의 물체 yaw를 무작위로 정하고 초기 선속도와 각속도를 0으로 만든다.
5. sweep vector를 `(0, +0.18, 0) m`로 설정한다.
6. target의 앞쪽, 오른쪽과 앞-오른쪽 대각선에 있어 sweep 경로를 막을 수 있는 물체를 `z=1.8 m`로 이동한다.

Python의 `choice`와 `shuffle`을 reset event 호출당 한 번 사용하므로, 한 번의 호출에 포함된 `env_ids`는 같은 target 종류, 물체-slot 순열과 sweep 방향을 공유한다. 물체 yaw는 환경별로 샘플링된다.

## 5. `RewardsCfg`

Isaac Lab reward manager는 아래 raw term에 설정 weight와 공통 `step_dt=0.02 s`를 곱해 합산한다. 표의 weight는 config에 선언된 값이다.

| Term | Weight | 역할과 실제 판정 |
|---|---:|---|
| `action_rate` | -0.01 | 전체 7D action이 이전 step보다 급격히 변하는 것을 억제한다. raw 값은 각 action 변화량 제곱의 합이다. |
| `joint_vel` | -0.01 | 6개 arm joint의 속도 제곱합을 패널티로 준다. gripper 속도는 포함하지 않는다. |
| `shelf_collision` | -0.1 | shelf가 움직였거나 TCP/finger/wrist가 shelf 기준 높이에 너무 가까운 정도를 패널티로 준다. |
| `object_collision` | -0.1, curriculum 후 -0.5 | target을 제외한 나머지 물체가 움직이면 패널티를 준다. |
| `reaching` | +2.0 | TCP가 target을 밀기 전 접근 위치에 가까워질수록 보상한다. |
| `orientation` | +2.0 | TCP의 로컬 Y축이 shelf의 Z축과 같은 방향으로 정렬되도록 보상한다. |
| `sweeping_object` | +6.0 | target을 goal 쪽으로 밀고, 적절한 접근 위치와 속도를 유지하도록 보상한다. |
| `homing_after_sweep` | +12.0 | target이 goal에 도달한 뒤 arm이 default 자세로 돌아오도록 보상한다. |

### `action_rate`

현재 전체 raw action과 이전 전체 raw action의 차이를 제곱해 합한다. arm뿐 아니라 gripper action의 부호 전환도 패널티에 포함된다.

### `joint_vel`

첫 6개 arm joint의 `joint_vel²`을 합한다. 빠른 움직임을 지속적으로 억제하며, 별도의 `hand_velocity` termination보다 부드러운 사전 억제 역할을 한다.

### `shelf_collision`

두 부분을 더한 raw penalty다.

- Shelf 이동 판정: 초기 위치로부터 shelf 위치 변화량의 norm과 shelf `root_vel_w` norm의 합이 `0.005`를 넘으면 `1`.
- 근접 판정: TCP가 `shelf root + z 1.06 m` 지점에서 `0.2 m` 이내일 때 활성화된다. 각 finger가 기준 높이보다 `0.02 m` 이내, wrist가 `0.08 m` 이내로 접근할수록 `0~1` 패널티가 각각 증가한다.

따라서 raw 값은 shelf 이동 0/1과 두 finger 및 wrist 근접도 최대 3을 합친 값이다. 실제 contact sensor가 아니라 shelf 상태와 frame 위치로 계산한다.

### `object_collision`

모든 물체의 3D 선속도를 소수 둘째 자리로 반올림한 뒤 target 물체 속도만 0으로 지운다. 나머지 물체 속도 성분 절댓값의 합에 `tanh`를 적용한다. 즉, target을 움직이는 것은 이 term이 벌하지 않지만 주변 물체를 건드려 움직이는 것은 패널티가 된다.

### `reaching`

목표 접근점은 target 중심에서 다음과 같이 이동한 위치다.

- X: `-0.02 m`
- Y: `-target_width`
- Z: `+0.09 m`

TCP와 이 접근점 사이의 3D 거리에 `exp(-10×distance)` 형태의 감쇠를 적용한다. 코드에서 wrist 거리도 계산하지만 최종 reward에는 TCP 거리만 사용된다.

### `orientation`

TCP rotation matrix의 Y축과 shelf rotation matrix의 Z축 내적을 사용한다. 같은 방향으로 정확히 정렬되면 raw reward가 `+1`에 가깝고, 직교하면 `0`, 반대 방향이면 음수가 된다. 내적의 부호를 유지한 채 제곱하므로 잘못 뒤집힌 정렬도 패널티 방향으로 강하게 반영된다.

### `sweeping_object`

먼저 target과 goal의 3D 거리를 계산한다.

- 거리가 `0.03 m` 미만이면 raw reward는 약 `2`에 가까운 성공 구간 보상이다.
- 그 밖에서는 TCP가 위의 접근점에서 `0.04 m` 이내이고 wrist의 Y 오차가 `0.04 m` 미만일 때만 진행 보상이 활성화된다.
- target의 `+Y` 속도가 `0.05~0.1 m/s`이면 `+0.5`, `0.1 m/s` 이상이면 `-0.5` 속도 보정이 들어간다.
- goal까지 남은 거리 `0.18 m`를 기준으로 가까워질수록 진행 보상이 커진다.

이 구현은 `+Y` sweep을 전제로 하므로 음의 Y 속도는 진행 속도로 인정하지 않는다.

### `homing_after_sweep`

target과 goal 사이의 YZ 평면 거리가 `0.03 m`보다 작을 때만 활성화된다. 그 후 arm의 첫 5개 joint가 default 자세에 가까울수록 보상이 커진다. 여섯 번째 wrist joint는 home-pose 오차 계산에 포함하지 않는다.

## 6. `TerminationsCfg`

| Term | 종료 조건 | Time-out 여부 |
|---|---|---|
| `time_out` | episode 시간이 `10 s`에 도달 | 예 |
| `object_drop` | 여섯 물체 중 하나라도 `z < 1.04 m`, 또는 roll/pitch 절댓값 중 하나가 `0.9 rad` 초과 | 아니오 |
| `push_fast` | 선택된 target 물체의 3D 선속도 norm이 `0.3 m/s` 초과 | 아니오 |
| `shelf_collision` | shelf `root_vel_w` norm이 `0.1` 초과, 또는 finger/wrist가 지정된 shelf 높이 아래로 진입 | 아니오 |
| `hand_velocity` | 6개 arm joint 중 하나라도 `|joint_vel| > 1.0 rad/s` | 아니오 |

`shelf_collision`의 frame 높이 조건은 구체적으로 다음과 같다.

- left 또는 right finger Z가 `shelf root Z + 1.06 m`보다 낮음
- wrist Z가 `shelf root Z + 1.12 m`보다 낮음

`object_drop`은 target만 검사하지 않고 **모든 여섯 물체**를 검사한다. 또한 shelf velocity 조건은 6D `root_vel_w` 전체 norm을 사용하므로 선속도와 각속도 성분이 하나의 threshold 판정에 함께 들어간다.

`success_sweeping` 함수는 구현되어 있지만 이 환경의 `TerminationsCfg`에는 등록되지 않았다. 따라서 target이 goal에 도달해도 성공만으로 즉시 종료하지 않고, 이후 homing reward를 받을 수 있다.

## 7. `CurriculumCfg`

활성 curriculum은 `obj_collision` 하나다.

- 초기 `object_collision` weight: `-0.1`
- global common step counter가 `250,000`을 초과한 뒤: `-0.5`
- 변화 방식: 선형 증가가 아니라 임계 step 이후 한 번에 weight를 교체하는 step schedule

즉, 학습 초반에는 주변 물체 이동을 비교적 약하게 벌하고, 이후에는 주변 물체를 건드리지 않는 행동을 더 강하게 요구한다. 주석 처리된 homing curriculum은 실행되지 않는다.

## 학습 runner 참고

이 task는 기본적으로 `rsl_rl_ppo_cfg.py`를 사용한다. 주요 값은 최대 `30,000` iteration, `gamma=0.95`, `entropy_coef=0.01`이며 experiment 이름은 `UR5e_shelf_sweep`이다.

## 주요 구현 파일

- 환경별 config: [`joint_pos_env_cfg.py`](../sweeping_policy/config/ur5e/joint_pos_env_cfg.py)
- 공통 MDP config: [`shelf_sweep_env_cfg.py`](../sweeping_policy/shelf_sweep_env_cfg.py)
- reward: [`reward_sweep.py`](../sweeping_policy/mdp/reward_sweep.py)
- reset event: [`events.py`](../sweeping_policy/mdp/events.py)
- termination: [`terminations.py`](../sweeping_policy/mdp/terminations.py)

[통합 비교 문서로 돌아가기](README.md)
