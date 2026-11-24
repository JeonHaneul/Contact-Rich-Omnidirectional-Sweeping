# 제어·관측·Homing 변형

[환경 목록](README.md) · [정확한 등록 ID와 상속](ENVIRONMENT_REGISTRY.md) · [설치·실행](../README.md)

기존 개별 문서에서 빠졌던 Robotiq 계열 변형을 현재 코드 기준으로 정리한다. 아래 짧은 이름에는 `Isaac-Sweep-Object-UR5e-` 접두사를 붙인다. Config의 직접 부모와 PPO 연결은 [등록 표](ENVIRONMENT_REGISTRY.md)가 기준이다.

## Differential IK scale와 EEF pose 제거

[diff_ik_relative_target_env_cfg.py](../sweeping_policy/config/ur5e/diff_ik_relative_target_env_cfg.py)에 정의되어 있다.

| ID의 접미사 | 차이 | Action / observation |
|---|---|---|
| `Random-Diff-IK-Relative-Control-Parm-AS-0.5` | `UR5eSweepDiffIKEnvCfg` 상속, arm scale `0.5`, target·goal을 EEF frame으로 변경 | 7D / 35D |
| `Random-Diff-IK-Relative-Control-Parm-AS-0.6` | 위와 같은 변경, scale `0.6` | 7D / 35D |
| `Random-Diff-IK-Relative-Control-Parm-AS-0.7` | 위와 같은 변경, scale `0.7` | 7D / 35D |
| `Random-Relative-Target-v1` | Diff-IK 기준에서 target·goal EEF frame 변경, policy `ee_pose` 7D 제거 | 7D / 28D |

Scale은 위치와 회전 delta에 함께 적용되는 scalar다. Config에 별도 meter/radian clip을 추가하지 않는다. `v1`의 live target과 goal은 EEF 기준으로 남아 있으므로 EEF pose를 제거한 사실만으로 target 추적이 사라지는 것은 아니다. 이 파일의 reaching IK `EventCfg`는 해당 등록 config에서 활성화하지 않은 코드이므로 이 네 환경을 reaching-reset 환경으로 설명하지 않는다.

## Joint-space relative-target와 Cartesian homing

| ID의 접미사 | 구현과 차이 |
|---|---|
| `Random-Relative-Target-Joint-Space-v0` | [UR5eSweepRelativeTargetEnvCfg](../sweeping_policy/config/ur5e/joint_pos_random_sweep_env_cfg.py)는 `UR5eSweepEnvCfg`에서 joint action을 유지하고 target·goal 관측만 EEF frame으로 변경한다. Action 7D, observation 35D다. |
| `Random-EE-Homing-v0` | [UR5eSweepEEHomingEnvCfg](../sweeping_policy/config/ur5e/joint_pos_random_ee_homing_sweep_env_cfg.py)는 Random-v0의 `homing_after_sweep.func`만 `EndEffectorHomingReward`로 바꾼다. Action 7D, observation 35D다. |

Joint-space random-init reset은 [별도 설명](10_isaac_sweep_object_ur5e_relative_target_joint_space_random_init_v0.md)을 본다. Homing 함수들은 [reward_random_sweep.py](../sweeping_policy/mdp/reward_random_sweep.py)에서 확인한다. 이름이 유사하더라도 `EndEffectorHomingReward`와 `EndEffectorHomingReward2`는 별도 함수다.

## OSC relative observation와 homing

[osc_random_sweep_env_cfg.py](../sweeping_policy/config/ur5e/osc_random_sweep_env_cfg.py)의 현재 기준 OSC는 [Random-v2](04_isaac_sweep_object_ur5e_random_v2.md)와 같이 `pose_rel`, fixed gain 100, gravity compensation 비활성이다.

| ID의 접미사 | 변경 | Action / observation |
|---|---|---|
| `Random-Relative-Obs-v0` | 기준 OSC에서 target·goal을 EEF frame으로 변경 | 7D / 35D |
| `Random-Relative-Obs-EE-Homing-v0` | 위 relative observation + homing 함수 `EndEffectorHomingReward2` | 7D / 35D |
| `Random-Homing-Cartesian-OSC-v0` | relative observation + homing 함수 `EndEffectorHomingReward` | 7D / 35D |
| `Random-Homing-OSC-12D-Action-v0` | 기준 OSC에서 `variable_kp`, target·goal EEF frame 변경 | **전체 13D** / 41D |

마지막 ID의 `12D`는 **arm action**만 가리킨다. Relative pose 6D와 stiffness 6D에 상속된 binary gripper 1D를 더해 전체 action은 13D다. `last_action`도 13D여서 observation은 `8 + 6 + 13 + 3 + 1 + 7 + 3 = 41D`다. Stiffness scale은 100, controller의 stiffness 제한은 `[50, 200]`, damping ratio는 6축 모두 1이다. 이 class의 homing 함수 교체 코드는 주석 처리되어 있으므로 기본 joint homing을 유지한다.

Action 차원은 workspace의 Isaac Lab [OperationalSpaceController](../../../IsaacLab/source/isaaclab/isaaclab/controllers/operational_space.py) 구현의 `pose_rel=6`, `variable_kp=target_dim+6`에서 확인했다. 실제 설치 Isaac Lab이 바뀌면 다시 확인한다.

## Single OSC offset 초기화

`Random-Single-OSC-Offset-Init-v0`의 [UR5eSweepRandomSingleOSCOffsetEnvCfg](../sweeping_policy/config/ur5e/osc_random_single_offset_env_cfg.py)는 `UR5eSweepEnvCfg`를 직접 상속한다.

- target 하나만 작업 영역에 남기는 `randomize_single_object_scene`을 사용한다.
- `homing_after_sweep` reward term을 제거한다.
- arm implicit stiffness/damping을 0으로 바꾸고 fixed-impedance relative pose OSC를 구성한다. Arm 6D + binary gripper 1D, observation 35D다.
- 기본 TCP 후보는 base에서 target 방향의 안전 영역 X, Y `[-0.25, 0.25] m`, Z `[1.15, 1.32] m`다.
- 확률 `1/3`로 target 중심에서 sweep 반대 방향 `0.14–0.22 m` 떨어진 측면 후보를 시도한다. 측면 clearance는 `0.07 m`, X 허용 범위는 `0.10 m`다. 경계에서 공간이 부족하면 기본 영역으로 돌아간다.
- IK에서 `wrist_3_joint`를 고정하고 heading 후보를 18° 이내에서 샘플링한다. 최종 heading 제한은 20°다.
- 온라인 재시도 및 pose bank 검증을 모두 통과하지 못하면 reset 오류를 낸다. 후보 분포·fallback은 [events.py](../sweeping_policy/mdp/events.py)의 `initialize_robot_in_random_shelf_pose_with_target_offset`가 구현한다.

설정상 `max_iterations=120`, `max_pose_attempts=8`, `pose_bank_fallback_attempts=8`이다. 이 수치는 구현 파라미터이며 실물 안전성이나 실측 접근 오차를 뜻하지 않는다.

## 실행 예시

Workspace 루트에서 task ID를 선택해 공통 launcher를 쓴다.

```bash
./IsaacLab/isaaclab.sh -p example/Sweep-Policy/train.py \
  --task Isaac-Sweep-Object-UR5e-Random-Single-OSC-Offset-Init-v0 \
  --num_envs 4096 --run_name single_osc_offset --headless

./IsaacLab/isaaclab.sh -p example/Sweep-Policy/train.py \
  --task Isaac-Sweep-Object-UR5e-Random-Homing-OSC-12D-Action-v0 \
  --num_envs 4096 --run_name relative_osc_variable_kp --headless
```

각 branch의 action·observation 차원과 homing 함수는 체크포인트·실험 해석에 영향을 준다. 등록 ID, 실행 명령과 결과를 [개발 로그](DEVELOPMENT_LOG.md)에 함께 남긴다.
