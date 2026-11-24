# `Isaac-Sweep-Object-UR5e-Random--Non-Tracking-v0`

[환경 목록](README.md) · [등록과 상속](ENVIRONMENT_REGISTRY.md) · [설치·실행](../README.md)

이 환경은 `Isaac-Sweep-Object-UR5e-Random-v0`를 직접 상속하며, policy가
target 물체의 이동을 추적하지 못하도록 `target_obs_state`만 변경한다. task ID의
`Random` 뒤 이중 하이픈(`--`)은 실제 등록 문자열의 일부다.

## 차이점

### `target_obs_state`

reset 때 `DynamicObjectGoalPosCommand`가 선택된 target의 world pose를
`target_init_state_w`에 저장한다. 새 관측 함수는 live object tensor 대신 이
snapshot의 position을 robot-root frame으로 변환한다. 따라서 target이 움직여도
policy에 전달되는 3D 위치는 다음 reset까지 바뀌지 않는다.

기존 Random-v0의 `[-0.01, 0.01] m` observation error 범위는 유지한다. 다만
매 step 새로운 noise를 더하면 최종 관측값이 고정되지 않으므로, 이 환경에서는
reset당 한 번 샘플링하는 additive bias noise model을 사용한다. 각 병렬 환경이
reset될 때 해당 환경의 3개 성분만 새 bias를 받는다. PLAY config에서 corruption을
끄면 이 bias도 적용되지 않는다.

### `goal_pos`

변경하지 않는다. 기존 command가 reset 시점의 target 위치에 `env.sweep_dir`을
더해 goal을 만들고 episode 중 다시 resample하지 않으므로, 초기 target 관측과
이미 일관된 고정 목표다.

### Reward의 privileged information

`RewardsCfg`를 그대로 상속한다. reaching, pushing, homing reward 등은 policy의
고정 관측을 참조하지 않고 simulator의 현재 target 위치와 속도를 직접 읽는다.
따라서 policy는 초기 target 위치만 받지만 reward 계산은 실시간 상태를 privileged
information으로 계속 활용한다.

그 밖의 scene, action, observation 순서와 차원(35D), command, reset event,
termination, curriculum 및 PPO runner 설정은 Random-v0와 같다.

## 실행

```bash
/path/to/IsaacLab/isaaclab.sh -p train.py \
  --task Isaac-Sweep-Object-UR5e-Random--Non-Tracking-v0 \
  --num_envs 4096 \
  --headless
```

## 구현 파일

- 환경 override: [`joint_pos_random_non_tracking_sweep_env_cfg.py`](../sweeping_policy/config/ur5e/joint_pos_random_non_tracking_sweep_env_cfg.py)
- 초기 target 관측: [`observations.py`](../sweeping_policy/mdp/observations.py)
- reset snapshot과 goal command: [`position_command.py`](../sweeping_policy/mdp/commands/position_command.py)
- 실시간 reward: [`reward_random_sweep.py`](../sweeping_policy/mdp/reward_random_sweep.py)
