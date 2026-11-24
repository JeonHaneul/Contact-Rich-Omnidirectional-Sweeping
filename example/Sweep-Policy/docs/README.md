# Sweep-Policy 환경 문서

이 폴더는 `sweeping_policy`가 구현하는 환경의 설정과 변형을 설명한다. 전체 프로젝트 목표·개발 원칙·활성 패키지는 [Contact-Rich-Manipulation 문서](../../Contact-Rich-Manipulation-Context/README.md), 일반 지식은 [workspace docs](../../../docs/README.md)를 본다.

[설치·실행](../README.md) · [모든 등록 ID와 상속](ENVIRONMENT_REGISTRY.md) · [개발 로그](DEVELOPMENT_LOG.md)

## 환경 선택

| 문서 | 확인할 내용 |
|---|---|
| [01 고정 방향 joint position](01_isaac_sweep_object_ur5e_v0.md) | 고정 +Y sweep와 기준 MDP |
| [02 Random-v0](02_isaac_sweep_object_ur5e_random_v0.md) | 양방향 sweep, joint action, 공통 observation·reward·termination |
| [03 Random-v1](03_isaac_sweep_object_ur5e_random_v1.md) | relative differential IK action과 실제 scale |
| [04 Random-v2](04_isaac_sweep_object_ur5e_random_v2.md) | 현재 relative pose fixed-impedance OSC, 예전 설명과의 차이 |
| [05 Relative-Target-v0](05_isaac_sweep_object_ur5e_random_relative_target_v0.md) | EEF frame의 target·goal observation |
| [06 Random-Single-v0](06_isaac_sweep_object_ur5e_random_single_v0.md) | target 하나만 작업 영역에 남기는 reset |
| [07 Random-Single-v1](07_isaac_sweep_object_ur5e_random_single_v1.md) | base XY randomization과 2D observation 추가 |
| [08 Non-Tracking-v0](08_isaac_sweep_object_ur5e_random_non_tracking_v0.md) | reset 때 고정한 target 위치와 bias |
| [09 Non-Tracking-v1](09_isaac_sweep_object_ur5e_random_non_tracking_v1.md) | target/width observation 제거, privileged reaching IK reset |
| [10 Joint-Space Random-Init](10_isaac_sweep_object_ur5e_relative_target_joint_space_random_init_v0.md) | 연속 single-object 분포와 arm reset IK |
| [11 Inspire Hand 변형](11_inspire_hand_variants.md) | 고정 손·2D hand action·mass·센서·reaching reset·initial target·privileged critic |
| [12 제어·관측·homing 변형](12_control_observation_variants.md) | Diff-IK scale 0.5/0.6/0.7, EEF pose 제거, relative OSC observation·homing·variable Kp·single OSC offset reset |

## 기준 환경을 읽는 방법

고정 방향 `UR5e-v0`와 양방향 `Random-v0`는 각각 별도의 공통 config에서 로봇별 설정을 완성한다. 많은 변형이 `UR5eSweepEnvCfg`를 상속하므로 [Random-v0](02_isaac_sweep_object_ur5e_random_v0.md)를 먼저 읽고 override를 확인한다. config 상속과 실행 환경 class 상속은 [등록 문서](ENVIRONMENT_REGISTRY.md)에서 따로 확인한다.

기본 Robotiq 계열은 6개 arm joint action과 1D binary gripper를 사용하지만 OSC/variable Kp 및 Inspire 변형은 action·관측 구성이 다르다. 특히 단일 mass 실린더 환경은 여섯 물체를 생성한 뒤 비타겟을 주차하는 reset과 다르다. 기준 환경의 수치가 모든 변형에 공통이라고 가정하지 않는다.

환경 설명은 코드의 계약을 기록한다. 실험 실행 결과는 task ID·명령·로그 경로와 함께 개발 로그에 남긴다. 문서 확인일은 2026-10-14이며, 이 정리 작업에서 학습 성공을 새로 측정한 것은 아니다.
