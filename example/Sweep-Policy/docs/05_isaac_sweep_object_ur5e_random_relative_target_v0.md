# `Isaac-Sweep-Object-UR5e-Random-Relative-Target-v0`

[환경 목록](README.md) · [등록과 상속](ENVIRONMENT_REGISTRY.md) · [설치·실행](../README.md)

이 환경은 `Isaac-Sweep-Object-UR5e-Random-v1`을 직접 상속하며,
`ObservationsCfg.PolicyCfg.target_obs_state`와 `goal_pos`의 관측 함수만 변경한다. 따라서
scene, action, command, 다른 observation term, reward, event, termination,
curriculum과 PPO runner는 Random-v1과 동일하다.

## 변경된 관측

Random-v1은 target object와 goal의 world position을 robot root frame으로 변환한다.
이 환경은 두 위치를 모두 end-effector frame으로 변환한다.

```text
p_target_eef = R_world_eef^T (p_target_world - p_eef_world)
p_goal_eef   = R_world_eef^T (p_goal_world   - p_eef_world)
```

여기서 EEF는 `robotiq_base_link` 원점이 아니라 해당 link의 local
`(0.13, 0, 0) m` 지점이다. 자세는 `robotiq_base_link`와 같으며, 이는 상속된
`ee_frame`의 `end_effector` target frame과 동일하다.

구현은 `ee_frame.data.target_pos_w`와 `target_quat_w`를 기준 pose로 사용해
`subtract_frame_transforms`를 적용한다. 따라서 단순 world-frame 위치 차이가
아니라 EEF의 회전축까지 반영된 상대 위치 3D가 각 term에서 반환된다.

## 유지되는 계약

- `target_obs_state` 출력: 3D
- `goal_pos` 출력: 3D
- 관측 noise: 성분별 uniform `[-0.01, 0.01]`
- 전체 policy observation: 35D
- arm action: Random-v1의 6D relative Cartesian differential IK
- gripper action: 1D binary joint-position command

## 주요 구현 파일

- 환경 override: [`diff_ik_relative_target_env_cfg.py`](../sweeping_policy/config/ur5e/diff_ik_relative_target_env_cfg.py)
- EEF 관측 함수: [`observations.py`](../sweeping_policy/mdp/observations.py)
- 상속하는 Random-v1: [`diff_ik_random_sweep_env_cfg.py`](../sweeping_policy/config/ur5e/diff_ik_random_sweep_env_cfg.py)

[통합 비교 문서로 돌아가기](README.md)
