# Inspire Hand 환경 변형

[환경 문서 목록](README.md) · [전체 등록과 상속](ENVIRONMENT_REGISTRY.md) · [실행 방법](../README.md)

이 문서는 Sweep-Policy 안에서 추가한 Inspire Hand 환경의 설정 차이만 다룬다.
전체 Contact-Rich-Manipulation 흐름은 [문서 서브모듈](../../Contact-Rich-Manipulation-Context/README.md)을 본다.
아래 명령은 workspace 루트 기준이다. 다른 실행 위치와 설치 절차는 [Sweep-Policy README](../README.md)에 있다.
설정·차원은 코드 감사 결과이며, 학습 성공이나 실물 재현을 보장하는 실험 결과가 아니다.

## 환경 비교

아래 ID는 `Isaac-Sweep-Object-UR5e-Random-Single-Inspire-Open-` 뒤에 붙는 접미사다.

| 접미사 | 주요 변경 | Action | Policy observation |
|---|---|---:|---:|
| `v0` | 완전히 편 손을 fixed joint로 고정 | 6D | 32D |
| `v1` | 두 openness action으로 손 관절 제어 | 8D | 34D |
| `v1-mass` | 하단 중심 질량을 가진 단일 실린더 | 8D | 34D |
| `v1-mass-limit` | arm 속도 종료 1 rad/s | 8D | 34D |
| `v2` | 물체 카탈로그 유지, pad 17D + F/T 6D | 8D | 57D |
| `v2-mass` | 단일 실린더, palm 1D + F/T 6D | 8D | 41D |
| `v3-mass` | v2-mass + reaching IK reset | 8D | 41D |
| `v3` | v2-mass + initial-only target 위치 | 8D | 41D |
| `v4` | v3 + live privileged critic state | 8D | actor 41D / critic 54D |

## 고정 손 v0

`Random-Single-Inspire-Open-v0`는 `Random-Single-v0`를 상속한다.
UR5e에는 Axia80 없이 Inspire Hand를 직접 장착하고, 원본 USD의 손을 완전히 편
zero-angle 자세에서 손 관절 12개를 fixed joint로 변환한다. 손 actuator와
gripper action은 없으며 접촉 중에도 손가락을 움직일 수 없다. Action은 기존
Arm joint position 6개만 남아 **7D → 6D**다. Arm의 초기 자세, PD gains,
torque/velocity 한도와 action scale `0.5`는 Random-v0 그대로다.
손의 인접 링크 메시가 겹치는 부분을 고려해 robot self-collision은 끈다.
손과 물체·선반 사이의 충돌은 유지한다.

보상 가중치·수식(Reach 높이 +9 cm, Y−width, XYZ/wrist 4 cm 게이트, homing 포함),
curriculum, +Y 18 cm 밀기와 PPO 설정도 유지한다. Arm 속도 종료는 기존 1 rad/s에서
3 rad/s로 완화하고, 나머지 종료 조건은 유지한다. OSC나 target-based IK
reset을 추가하지 않는다. EEF는 원본 example의 전방 13 cm에 맞춘 Inspire 로컬
`(0, 0, 0.13)`이며, 프레임 회전으로
원본 tool +Y를 Inspire +X에 대응시켜 원본 자세 보상식을 유지한다. 원본의 가상
finger 프레임은 로컬 `(±0.07, 0, 0.13)`, wrist 프레임은 `(0, 0, -0.17)`이다.
finger 프레임의 높이는 선반 접근 패널티와 선반 충돌 종료에 사용하므로 유지한다.
모든 프레임 marker scale은 원본과 같은 `(0.05, 0.05, 0.05)`이며 EEF만 표시한다.
관측은 움직일 수 있는 Arm joint position/velocity와 원본 나머지 항을 사용해 32D이며,
손 관측·tactile·F/T 항을 추가하지 않는다.

단일 물체 처리는 `mdp/events.py`의 `randomize_single_object_scene`을 그대로 사용한다.
카탈로그에서 선택한 물체 하나만 선반에 두고, 나머지는 작업 영역 밖으로 이동하여
gravity를 끈다. 모든 물체를 scene에서 제거하는 방식은 아니다.

로봇 USD는 이 workspace의
`src/trash/hand_manipulation_rl/hand_manipulation_rl/assets/data/ur5e_inspire_usd/ur5e_inspire.usd`를
직접 참조하며 Axia80를 추가하는 spawner는 사용하지 않는다. 다른 설치 위치에서는
`SWEEP_POLICY_INSPIRE_ROBOT_USD_PATH`로 직접 장착형 USD 경로를 지정할 수 있다.

workspace 루트에서 학습:

```bash
./IsaacLab/isaaclab.sh -p example/Sweep-Policy/train.py \
  --task Isaac-Sweep-Object-UR5e-Random-Single-Inspire-Open-v0 \
  --num_envs 2048 --run_name inspire_open_joint_position --headless
```

재생:

```bash
./IsaacLab/isaaclab.sh -p example/Sweep-Policy/play.py \
  --task Isaac-Sweep-Object-UR5e-Random-Single-Inspire-Open-v0 \
  --num_envs 1 --checkpoint "<CHECKPOINT_PATH>" --real-time
```

로그 experiment는 `UR5e_shelf_sweep_random_single_inspire_open`이다.
Action/관측 크기가 바뀌므로 원본 Robotiq 정책 체크포인트를 그대로 재사용할 수 없다.

## 두 openness action을 사용하는 v1

`Random-Single-Inspire-Open-v1`은 위 v0를 상속하고 손 관절 12개를 가동 관절로
사용한다. v0의 고정 손 spawner·환경 ID는 유지한다. Axia80 없이 직접 장착하며,
Arm 제어·초기 Arm 자세·EEF·보상·싱글 오브젝트 Event는 v0와 같다.

- Action은 **Arm 6D + Hand 2D = 8D**다.
- 첫 Hand Action은 엄지 2–4번과 나머지 네 손가락을 함께 굽힌다.
- 두 번째 Hand Action은 엄지를 손바닥 쪽으로 움직이는 `inspire_left_thumb_1_joint`를 제어한다.
- 두 값 모두 **0 = 최대 굽힘, 1 = 최대 펼침**이다. v0와 같은 zero-angle 자세로 시작한다.
- Reset 시 Hand target과 직전 Hand Action을 `[1, 1]`로 초기화한다. 이후에는
  `[0, 1]` 범위 제한과 관절각 선형 변환만 적용하며, 속도에 따른 target 완충·최소 펼침 제한은 없다.
- 손 actuator는 기존 Inspire 코드의 position-hand 설정(K=1.2, D=0.06,
  effort=0.20 Nm, velocity=1 rad/s, armature=0.0001)을 사용한다.
- 관측은 v0의 Arm/물체/EEF 항과 8D 직전 Action으로 **34D**다.
  Hand 상태·Tactile·F/T 관측은 추가하지 않는다.

v1 학습:

```bash
./IsaacLab/isaaclab.sh -p example/Sweep-Policy/train.py \
  --task Isaac-Sweep-Object-UR5e-Random-Single-Inspire-Open-v1 \
  --num_envs 2048 --run_name inspire_hand_2d --headless
```

v1 재생:

```bash
./IsaacLab/isaaclab.sh -p example/Sweep-Policy/play.py \
  --task Isaac-Sweep-Object-UR5e-Random-Single-Inspire-Open-v1 \
  --num_envs 1 --checkpoint "<CHECKPOINT_PATH>" --real-time
```

v1 experiment는 `UR5e_shelf_sweep_random_single_inspire_hand_v1`이며,
Action/관측 차원이 달라 v0 체크포인트는 그대로 로드할 수 없다.

## 단일 실린더 v1-mass

`Random-Single-Inspire-Open-v1-mass`는 v1을 상속해 전도에 덜 민감한 물체로
Sweeping을 테스트한다. 지름 8 cm, 높이 10 cm는 원본 `cup_1`의 대략적인 크기다.
질량은 1.0 kg이고, 하단 2 cm에 질량 80%, 나머지 상단에 20%를 배치한 모델의
COM·관성을 명시한다. COM은 바닥에서 2 cm, 관성은 COM 기준
`(0.00093333, 0.00093333, 0.0008) kg·m²`다. 실제 내부 구조를 따로 생성하지는 않는다.

실린더 하나만 scene에 생성하며 다른 물체를 주차하거나 외부 물체 USD를 읽지 않는다.
물체 body 원점은 바닥 중앙으로 유지하여 기존 Reach Z+9 cm의 기준이 바뀌지 않는다.
`target_width`는 반경 4 cm로 설정하며, 원본 컵 카탈로그의 6 cm와는 다르다.
초기 위치는 기존 컵 위치 `(-0.75, 0, 1.05)`에서 원본 reset의 XY ±2 cm 및
Yaw randomization을 적용한다. 밀기 방향 ±Y, 거리 18 cm, Arm/Hand 제어·보상·PPO·
속도 종료 3 rad/s는 v1을 상속한다. Action 8D, 관측 34D이며 센서는 추가하지 않는다.
원본 환경도 물체에 mass=1.0 kg를 지정하므로 주요 차이는 형상과 낮은 COM이다.

```bash
./IsaacLab/isaaclab.sh -p example/Sweep-Policy/train.py \
  --task Isaac-Sweep-Object-UR5e-Random-Single-Inspire-Open-v1-mass \
  --num_envs 2048 --run_name inspire_hand_mass_cylinder --headless
```

experiment는 `UR5e_shelf_sweep_random_single_inspire_hand_v1_mass`다.

## Arm 속도 제한 v1-mass-limit

`Isaac-Sweep-Object-UR5e-Random-Single-Inspire-Open-v1-mass-limit`는 `v1-mass`를
상속하고 `hand_velocity` 종료 임계값만 3.0에서 1.0 rad/s로 강화한다.
이 항목은 이름으로 선택한 Arm 6축 중 하나라도 절대 관절 속도가 1.0 rad/s를
초과하면 episode를 종료한다. 손가락 속도나 EEF 선속도 제한은 아니다.
원본 `v1-mass`의 임계값 3.0과 다른 설정은 유지하며, PPO 설정도 상속한다.
experiment는 `UR5e_shelf_sweep_random_single_inspire_hand_v1_mass_limit`다.

```bash
./IsaacLab/isaaclab.sh -p example/Sweep-Policy/train.py \
  --task Isaac-Sweep-Object-UR5e-Random-Single-Inspire-Open-v1-mass-limit \
  --num_envs 2048 --run_name inspire_hand_mass_limit_1rad --headless
```

## Mass 환경의 센서 및 Reach 초기화 버전

`v2-mass`는 `v1-mass`를 상속한다. Hand는 이미 전체 굽힘과 엄지 opposition을
분리한 2D openness `[0, 1]` 명령이므로 이를 그대로 사용한다. Arm 6D를 합해
Action은 8D다. 손을 완전히 편 상태 `[1, 1]`에서 시작한다.

- 원통형 F/T 하우징과 손바닥 패드 하나를 사용한다. 손가락 패드는 관측하지 않는다.
- 손바닥 접촉 여부 1D와 EEF 기준 F/T 6D를 추가하여 policy 관측은 41D다.
  기존 34D 관측을 유지하며, 센서는 보상·게이트·제어에 연결하지 않는다.
  정책 입력으로 제공되므로 PPO는 센서값을 학습에 활용할 수 있다.
- F/T 하우징은 반경 4.1 cm, 길이 2.54 cm, 가상 질량 0.001 kg이다.
  손 전체가 local Z로 2.54 cm 이동한다. EEF는 하우징 중심 기준
  `(0, 0, 0.1427)`이며, 기존 장착부 기준 `(0, 0, 0.1554)`다.
  손 기준의 기존 13 cm는 유지해 센서 길이를 중복 적용하지 않는다.
- 원래 단일 실린더, controller, Arm/Hand 초기 관절값, 보상 가중치, PPO는 유지한다.
- Reach와 밀기 게이트의 목표 높이는 물체 바닥 원점 +4 cm다. 정렬 보상은
  별도 `inspire_mass_rewards.align_eef_y_with_shelf_z`를 사용해 회전행렬의
  열에서 실제 EEF Y와 선반 Z축을 읽는다. 기존 example의 보상 함수는 유지한다.

`v3-mass`는 `v2-mass`를 상속하고 물체 스폰 이후 Reach 목표 주변에서 시작한다.
`initialize_robot_at_reaching_pose`에 XYZ별 노이즈와 EEF 회전 offset 지원을
추가해 사용한다. 다른 초기화 이벤트의 넓은 선반 분포 대신, 물체 기준 Reach
목표 주변으로 분포를 제한한다.

- 기준 목표: `(object X - 0.02, object Y - width * sweep_sign, object Z + 0.04)`.
- X와 Y는 각각 독립적인 uniform ±1 cm. Z와 회전에는 노이즈를 넣지 않는다.
  ±1 cm는 이번 실험용 가정이며 실측 접근 오차가 아니다.
- IK는 실제 EEF offset·회전을 반영하며 Arm만 이동한다. 실패 시 진단을 출력하고
  reset을 중단한다. 수렴 허용오차는 위치 3 mm, 회전 0.01 rad다.
- Reach 보상은 제거한다. 나머지 보상·게이트 및 policy 관측은 유지한다.
  reset 이후 Arm Action 기록은 IK 자세를 나타내고 Hand는 `[1, 1]`이다.
- 관측의 기존 실시간 물체 GT는 유지한다.

각 환경은 별도 experiment로 기록하며 기존 `v1-mass`, `v2` 등록을 보존한다.
센서가 추가된 버전은 관측 차원이 달라 체크포인트 호환성을 확인해야 한다.

```bash
./IsaacLab/isaaclab.sh -p example/Sweep-Policy/train.py \
  --task Isaac-Sweep-Object-UR5e-Random-Single-Inspire-Open-v2-mass \
  --num_envs 2048 --run_name inspire_mass_palm_ft --headless

./IsaacLab/isaaclab.sh -p example/Sweep-Policy/train.py \
  --task Isaac-Sweep-Object-UR5e-Random-Single-Inspire-Open-v3-mass \
  --num_envs 2048 --run_name inspire_mass_reaching_xy --headless
```

experiment는 각각 `UR5e_shelf_sweep_random_single_inspire_hand_v2_mass`,
`UR5e_shelf_sweep_random_single_inspire_hand_v3_mass`다.

## Initial Object Position 및 privileged critic — v3 / v4

이 두 환경은 `v2-mass`를 기반으로 하며, 기존 Reach 스킵 실험 `v3-mass`와
다른 ID다. `v2-mass`의 단일 실린더, 센서, 8D Action, controller, reset,
Z+4 cm 목표, 보상 수식·가중치와 PPO 하이퍼파라미터를 유지한다.

- **v3:** 물체 위치 관측을 episode-initial 위치로 교체한다. reset 때 한 번
  취득한 robot root 좌표계의 3D 위치를 episode 내내 같은 값으로 제공한다.
  기존 ±1 cm 관측 오차도 reset마다 한 번 샘플링한 고정 bias로 바꾼다.
  물체가 움직여도 해당 관측을 갱신하지 않는다. goal도 초기 snapshot 기반이다.
  정책 관측은 41D이며 actor와 critic이 같은 policy 그룹을 사용한다.
- **v4:** v3의 policy 그룹을 유지하고 privileged 그룹에 현재 물체의 위치 3D,
  quaternion 4D, 선속도 3D, 각속도 3D를 추가한다. 모두 robot root 좌표계로
  표현하며, 위치 원점은 실린더 바닥인 object link다. 노이즈 없이 Scene GT를 읽는다.
  `obs_groups = {"actor": ["policy"], "critic": ["policy", "privileged"]}`로
  actor 41D와 critic 54D를 분리한다. 재생 정책에는 privileged 입력이 필요 없다.
- 두 환경의 보상·종료 판정은 현재 Scene GT를 계속 사용한다. v4는 보상 입력의
  SceneEntityCfg를 명시한다. 보상이 초기 actor 위치나 정규화된 critic tensor를
  읽지 않으며, 수식·계수는 바꾸지 않는다.
- 초기 위치 latch는 부분 reset에서도 해당 환경만 갱신한다. 센서 관측과
  물체 폭·EEF pose·goal 관측은 유지한다. Hand는 `[1, 1]`에서 시작한다.

```bash
./IsaacLab/isaaclab.sh -p example/Sweep-Policy/train.py \
  --task Isaac-Sweep-Object-UR5e-Random-Single-Inspire-Open-v3 \
  --num_envs 2048 --run_name inspire_mass_initial_object --headless

./IsaacLab/isaaclab.sh -p example/Sweep-Policy/train.py \
  --task Isaac-Sweep-Object-UR5e-Random-Single-Inspire-Open-v4 \
  --num_envs 2048 --run_name inspire_mass_privileged_critic --headless
```

experiment는 각각 `UR5e_shelf_sweep_random_single_inspire_initial_v3_mass`,
`UR5e_shelf_sweep_random_single_inspire_privileged_v4_mass`다. runner는 RSL-RL의
`obs_groups` 설정을 사용한다. 실제 설치 버전은 실행 로그에 기록한다. v4는 critic 입력 크기가 달라 새 학습으로 시작한다.

## 물체 카탈로그를 유지하는 센서 v2

`Random-Single-Inspire-Open-v2`는 v1을 상속하고 Axia80을 모티브로 한 원통형
F/T 하우징을 wrist와 hand 사이에 삽입한다. 센서 구현은 이 workspace의 보관 패키지
`src/trash/hand_manipulation_rl`을 재사용하므로 해당 소스와 USD가 함께 있어야 한다.

- 하우징은 반경 4.1 cm, 두께 2.54 cm다. 기존 센서 구현의 0.001 kg 가상 질량을
  사용하며, 실제 Axia80의 질량·정확도를 재현한 모델은 아니다.
- 물리적인 삽입으로 hand가 장착축 방향으로 2.54 cm 앞으로 이동한다.
  EEF의 hand-local 위치 `(0, 0, 0.13)`과 프레임 회전은 v1과 같다.
- Tactile은 실제 손바닥·손가락 pad 17개의 **접촉 여부**다. 0.05 N 이상을 1로
  읽으며, 특정 물체 ID로 필터링하지 않는다.
- F/T는 hand를 child로 하는 고정 관절의 실제 reaction **6D**다.
  EEF 원점과 EEF 축으로 변환한 `[Fx, Fy, Fz, Mx, My, Mz]`(N, Nm)를 읽는다.
  손 자중·관성 성분을 포함하며 중력 보상이나 별도 필터는 적용하지 않는다.
- Reset 직후 센서는 0으로 표시하고 다음 physics step부터 측정값을 제공한다.
- Action은 v1과 같은 **8D**, 관측은 **34 + 17 + 6 = 57D**다.
  보상·게이트·termination·controller는 센서 관측을 참조하지 않는다.
  정책 입력에 포함되므로 PPO 정책은 학습 과정에서 센서를 활용할 수 있다.
- 초기 Arm/Hand 관절값, 단일 물체 spawn, 보상 가중치, PPO 설정은 v1을 상속한다.
  센서의 물리적인 추가로 접촉과 동역학은 달라질 수 있다.

v2 학습:

```bash
./IsaacLab/isaaclab.sh -p example/Sweep-Policy/train.py \
  --task Isaac-Sweep-Object-UR5e-Random-Single-Inspire-Open-v2 \
  --num_envs 2048 --run_name inspire_hand_tactile_ft --headless
```

v2 재생:

```bash
./IsaacLab/isaaclab.sh -p example/Sweep-Policy/play.py \
  --task Isaac-Sweep-Object-UR5e-Random-Single-Inspire-Open-v2 \
  --num_envs 1 --checkpoint "<CHECKPOINT_PATH>" --real-time
```

v2 experiment는 `UR5e_shelf_sweep_random_single_inspire_sensors_v2`다.
관측 차원이 바뀌므로 v0/v1 체크포인트를 그대로 로드할 수 없다.
