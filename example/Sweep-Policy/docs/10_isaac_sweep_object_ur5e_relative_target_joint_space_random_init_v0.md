# Relative-Target Joint-Space Random-Init v0

[환경 목록](README.md) · [등록과 상속](ENVIRONMENT_REGISTRY.md) · [설치·실행](../README.md)

Task ID:
`Isaac-Sweep-Object-UR5e-Random-Relative-Target-Joint-Space-Random-Init-v0`

이 환경은
`Isaac-Sweep-Object-UR5e-Random-Relative-Target-Joint-Space-v0`의 config인
`UR5eSweepRelativeTargetEnvCfg`를 직접 상속한다. 따라서 observation, action,
command, reward, termination, curriculum, simulation 주기와 RSL-RL PPO config는
기준 환경과 동일하다. 달라지는 것은 reset event뿐이다.

## 물체 초기화

- 여섯 물체 중 target 하나를 환경별로 독립적으로 균등 선택한다.
- 기존 2×3 grid 중심 범위와 기존 XY uncertainty `±0.02 m`를 합친 실제 spawn
  envelope 안에서 target 중심 XY를 연속 균등 샘플링한다.
  - environment-local X: `[-0.77, -0.58] m`
  - environment-local Y: `[-0.22, 0.22] m`
  - Z: 기존과 같은 `1.05 m`
- 나머지 다섯 물체는 서로 분리된 기존 grid XY에 두되 `z=1.8 m`로 올리고,
  해당 환경에서 gravity를 비활성화한다.
- target yaw와 sweep 방향 `(0, ±0.18, 0) m`도 환경별로 독립 샘플링한다.

## 로봇 초기화

`reset_all`이 기본 상태를 복구하고 물체 event가 target을 정한 다음,
`initialize_random_shelf_pose`가 arm joint만 변경한다. Robot root pose에는 쓰지 않으므로
base의 위치 `pos=(0, 0, 0.79505)`와 orientation은 기준 환경 그대로 고정된다.

TCP 위치는 다음 안전 범위에서 샘플링한다.

- X: base에서 target 방향으로 이동하되 base에서 `0.38 m`, target에서 `0.15 m`
  이상 떨어지는 내부 구간
- Y: `[-0.25, 0.25] m`
- Z: `[1.15, 1.32] m`

Y/Z 범위는 현재 환경에서 사용하는 shelf 내부 Y 경계 `[-0.50, 0.50] m`와 현재 층의 바닥/천장
`z=1.05/1.50 m`에서 gripper 여유를 둔 값이다. X 여유는 base 부근의 접힌-arm
singularity와 target 선접촉을 피하기 위한 값이다. 가장 먼 조합도 base root 기준
직선거리가 약 `0.85 m` 이내가 되도록 Y/Z 범위를 보수적으로 줄여 UR5e가 거의 완전히
펴지는 자세를 피한다.

기준 시작 자세의 tool local `+X`를 heading으로 사용한다. Reset마다 기준 heading에서
최대 `18°`인 cone 내부 방향을 샘플링한다. 최종 상태는 기준 heading과의 각도를 다시
검사하며 반드시 `20°` 미만이어야 한다.

IK는 TCP position 3축과 heading 2자유도를 첫 다섯 arm joint로 푼다.
`wrist_3_joint`는 IK 변수에서 제외하며 모든 seed, iteration 및 최종 simulator/drive
target write에서 기준 환경의 default 값 `0.785 rad`로 복구한다. 따라서 heading을
randomize한 뒤에도 wrist-3 초기값은 기존 환경과 정확히 같다.

목표 위치는 위 전체 범위에서 샘플링하며 경계 주변을 별도로 제거하지 않는다. IK의
수렴 판정에는 목표 오차뿐 아니라 X clearance, Y/Z 범위, 기준 heading `20°` 및
`wrist_3_joint` 고정 조건을 함께 적용한다. 한 pose에서 모든 joint seed가 실패하면
물체 위치는 유지한 채 실패한 environment의 로봇 목표 pose만 전체 분포에서 다시
샘플링하며, 이를 최대 8회 수행한다.

온라인 재샘플링이 모두 실패한 경우에도 하나의 고정 joint pose를 사용하지 않는다.
성공한 random pose를 X/Y/Z/heading의 16개 cell로 나누어 cell당 최대 32개씩 보관하고,
cell과 pose를 무작위로 선택한 뒤 현재 물체에 대한 hard constraint를 다시 검증한다.
따라서 pose-bank 사용은 특정 고정 자세를 반복하거나 특정 workspace strip을 영구히
제외하지 않는다. `random_init_pose_attempt_count`와
`random_init_used_pose_bank`가 reset 결과에 대한 내부 진단값으로 유지된다. 모든
재샘플과 pose-bank 검증까지 실패할 때만 constraint 이름을 포함한 reset 오류를
반환한다.

## 학습 실행

```bash
../../IsaacLab/isaaclab.sh -p train.py \
  --task Isaac-Sweep-Object-UR5e-Random-Relative-Target-Joint-Space-Random-Init-v0 \
  --num_envs 4096 \
  --headless
```
