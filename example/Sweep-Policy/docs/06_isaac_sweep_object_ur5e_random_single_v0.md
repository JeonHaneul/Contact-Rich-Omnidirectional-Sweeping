# `Isaac-Sweep-Object-UR5e-Random-Single-v0`

[환경 목록](README.md) · [등록과 상속](ENVIRONMENT_REGISTRY.md) · [설치·실행](../README.md)

이 환경은 `Isaac-Sweep-Object-UR5e-Random-v0`를 직접 상속하고 reset의
`object_spawn` event 함수만 교체한다. 따라서 robot, shelf, action, observation,
command, reward, termination, curriculum과 RSL-RL PPO 설정은 Random-v0와 같다.

## Target-only reset

reset 때 다음 동작을 수행한다.

1. 여섯 물체 종류 중 target 하나를 무작위로 선택한다.
2. 기존 2×3 slot 순서를 shuffle하고 target을 해당 slot에 배치한다.
3. target의 XY 오차, yaw와 `+Y`/`-Y` sweep 방향을 기존과 같이 randomize한다.
4. 나머지 다섯 물체는 `z=1.8 m`의 작업 영역 밖으로 이동시키고 속도를 0으로 만든다.
5. 비타겟의 gravity를 비활성화하고 target의 gravity를 활성화하여 비타겟이 episode 중 선반으로 돌아오지 않게 한다.

랜덤 target 종류를 매 reset 유지하고 기존 MDP term의 collection index 계약을
보존하기 위해 여섯 physics actor 자체는 scene에 유지한다. 이 환경에서
target-only는 **활성 작업 선반에 target 하나만 존재한다**는 의미다.

## 학습 실행

```bash
../../IsaacLab/isaaclab.sh -p train.py \
  --task Isaac-Sweep-Object-UR5e-Random-Single-v0 \
  --num_envs 4096 \
  --headless
```

## 주요 구현 파일

- 환경 override: [`joint_pos_random_single_sweep_env_cfg.py`](../sweeping_policy/config/ur5e/joint_pos_random_single_sweep_env_cfg.py)
- reset event: [`events.py`](../sweeping_policy/mdp/events.py)
- 상속하는 Random-v0: [`joint_pos_random_sweep_env_cfg.py`](../sweeping_policy/config/ur5e/joint_pos_random_sweep_env_cfg.py)

[통합 비교 문서로 돌아가기](README.md)
