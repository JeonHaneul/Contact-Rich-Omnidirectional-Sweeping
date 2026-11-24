# `Isaac-Sweep-Object-UR5e-Random-Single-v1`

[환경 목록](README.md) · [등록과 상속](ENVIRONMENT_REGISTRY.md) · [설치·실행](../README.md)

이 환경은 `Isaac-Sweep-Object-UR5e-Random-Single-v0`를 직접 상속하고,
UR5e articulation root(base_link)의 reset 위치를 평면에서 randomize하는 event 하나만
추가한다. 또한 환경 원점 대비 base XY 위치를 2차원 policy observation으로 추가한다.
그 외 scene, object event, action, 기존 observation term, command, reward,
termination, curriculum과 RSL-RL PPO 설정은 모두 v0와 같다.

## Base 위치 randomization

reset 때 기본 robot root 상태에 다음 world-axis offset을 더한다.

```text
dx ~ Uniform(-0.05, 0.05) m
dy ~ Uniform(-0.05, 0.05) m
dz = 0
droll = dpitch = dyaw = 0
```

XY randomization 영역은 `10 cm × 10 cm` 정사각형이다. Z는 기존 `0.79505 m`,
quaternion은 기존 값, root 선속도와 각속도는 기존 기본값을 유지한다. 각 병렬
환경과 partial reset 대상은 독립적인 XY sample을 받는다.

## Base 위치 observation

기존 35차원 policy observation의 마지막에 다음 2차원 값이 추가된다.

```text
base_pos = robot_root_pos_w[:, :2] - env_origins[:, :2]
```

world 좌표의 각 vectorized environment 원점을 제거하므로, base가 해당 원점에서
`x=-0.03 m`, `y=0.01 m`만큼 이동했다면 관측값은 정확히
`[-0.03, 0.01]`이다. Z와 회전은 이 관측에 포함하지 않으며 전체 policy
observation은 37차원이다.

## 학습 실행

```bash
../../IsaacLab/isaaclab.sh -p train.py \
  --task Isaac-Sweep-Object-UR5e-Random-Single-v1 \
  --num_envs 4096 \
  --headless
```

## 주요 구현 파일

- 환경, event 및 observation override: [`joint_pos_random_single_v1_sweep_env_cfg.py`](../sweeping_policy/config/ur5e/joint_pos_random_single_v1_sweep_env_cfg.py)
- 상속하는 v0: [`joint_pos_random_single_sweep_env_cfg.py`](../sweeping_policy/config/ur5e/joint_pos_random_single_sweep_env_cfg.py)

[통합 비교 문서로 돌아가기](README.md)
