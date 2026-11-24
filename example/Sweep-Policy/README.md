# Sweep-Policy

UR5e 선반 sweeping 강화학습 환경을 보관하는 참고 프로젝트다. 기존 Robotiq 환경과 이 저장소에서 추가한 Inspire Hand·OSC·관측·초기화 변형을 비교할 때 사용한다. 현재 workspace의 활성 패키지와 전체 Contact-Rich-Manipulation 흐름은 [문서 서브모듈](../Contact-Rich-Manipulation-Context/README.md)에서 확인한다.

이곳의 [docs](docs/README.md)는 **Sweep-Policy 환경의 동작과 설정 차이**를 설명한다. 프로젝트 전체 개발 원칙·진행 상황은 문서 서브모듈에, ROS Python 등 일반 지식은 [workspace docs](../../docs/README.md)에 둔다. 기존 ROS Python 문서는 [일반 지식 문서](../../docs/ROS2_PYTHON_ENVIRONMENTS.md)로 이동했다.

## 읽는 순서

1. [환경 문서 목록](docs/README.md): 비교할 환경 선택
2. [등록 ID와 상속 구조](docs/ENVIRONMENT_REGISTRY.md): 정확한 task ID, config·runtime class, PPO runner 연결
3. 선택한 환경의 설명: 기준 환경에서 바뀐 observation/action/reset/reward 확인
4. [개발 로그](docs/DEVELOPMENT_LOG.md): 변경 근거·검증 범위·실행 기록

## 코드 구성

| 경로 | 역할 |
|---|---|
| [sweeping_policy/__init__.py](sweeping_policy/__init__.py) | Gymnasium 환경 ID와 config·runner entry point 등록 |
| [shelf_sweep_env_cfg.py](sweeping_policy/shelf_sweep_env_cfg.py), [shelf_sweep_random_env_cfg.py](sweeping_policy/shelf_sweep_random_env_cfg.py) | scene, command, action, observation, event, reward, termination, curriculum의 Manager 기반 정의 |
| [config/ur5e](sweeping_policy/config/ur5e) | 로봇 설정과 환경별 상속·override |
| [config/ur5e/agents](sweeping_policy/config/ur5e/agents) | RSL-RL PPO runner 설정과 experiment 이름 |
| [mdp](sweeping_policy/mdp) | manager가 사용하는 관측·보상·종료·초기화 함수 |
| [src](sweeping_policy/src) | YAML, asset 경로, 로봇·물체 spawner, hand action 구현 |
| [env.py](sweeping_policy/env.py), [env_inspire_hand.py](sweeping_policy/env_inspire_hand.py), [env_inspire_sensors.py](sweeping_policy/env_inspire_sensors.py), [env_inspire_mass_reaching.py](sweeping_policy/env_inspire_mass_reaching.py) | ManagerBasedRLEnv 확장과 reset/history/sensor 처리 |
| [train.py](train.py), [play.py](play.py) | 환경을 등록한 뒤 Isaac Lab의 공식 RSL-RL script 실행 |
| [tests](tests) | 정적 설정 계약과 Isaac Sim smoke script |

## 설치와 자산 의존성

아래 명령은 **workspace 루트**에서 실행한다. Isaac Lab과 필요한 Python 패키지가 설치되어 있어야 한다.

```bash
./IsaacLab/isaaclab.sh -p -m pip install -e example/Sweep-Policy
```

`train.py`와 `play.py`는 기본적으로 workspace의 `IsaacLab/scripts/reinforcement_learning/rsl_rl`을 사용한다. Isaac Lab이 다른 곳에 설치되어 있으면 해당 checkout의 launcher와 다음 변수를 함께 사용한다.

```bash
export ISAACLAB_PATH=/path/to/IsaacLab
```

선반과 기본 물체 USD는 [assets.py](sweeping_policy/src/assets.py)의 `SWEEP_POLICY_ASSET_ROOT`에서 읽는다. 기본값은 `assets/shelf`다. 다른 Nucleus 서버나 로컬 asset 디렉터리를 사용하려면 다음을 설정한다.

```bash
export SWEEP_POLICY_ASSET_ROOT=/path/to/shelf-assets
```

Inspire 환경은 보관 패키지 `src/trash/hand_manipulation_rl`의 직접 장착형 USD를 명시적으로 재사용한다. 기본 파일은 `hand_manipulation_rl/assets/data/ur5e_inspire_usd/ur5e_inspire.usd`이며 다른 USD는 다음 변수로 지정한다.

```bash
export SWEEP_POLICY_INSPIRE_ROBOT_USD_PATH=/path/to/ur5e_inspire.usd
```

센서가 있는 Inspire 환경은 같은 보관 패키지의 `sensors.py`와 `assets/robot.py`도 사용한다. 위 변수는 USD만 교체하므로 센서 소스 의존성을 대체하지 않는다. 보관 패키지를 ROS 빌드에서 제외해도 이 환경의 Python import는 명시적으로 해당 소스를 읽는다. 이 submodule만 단독 checkout하면 그 보관 자산·소스는 포함되지 않는다.

## 실행 명령

[등록 목록](docs/ENVIRONMENT_REGISTRY.md)의 정확한 ID를 `SWEEP_TASK`에 넣는다. 예시는 양방향 joint-position 기준 환경이다.

```bash
export SWEEP_TASK=Isaac-Sweep-Object-UR5e-Random-v0
./IsaacLab/isaaclab.sh -p example/Sweep-Policy/train.py \
  --task "$SWEEP_TASK" --num_envs 4096 --run_name reference_random_joint --headless

./IsaacLab/isaaclab.sh -p example/Sweep-Policy/play.py \
  --task "$SWEEP_TASK" --num_envs 1 --checkpoint "<CHECKPOINT_PATH>" --real-time
```

`example/Sweep-Policy` 안에서 실행할 때는 launcher를 `../../IsaacLab/isaaclab.sh`, script를 `train.py` 또는 `play.py`로 바꾼다. `--num_envs`는 사용 가능한 GPU 메모리에 맞춰 조절한다.

Gymnasium 등록 메타데이터만 확인할 때는 Isaac Sim을 시작하지 않는다.

```bash
./IsaacLab/isaaclab.sh -p -c 'import gymnasium as gym; import sweeping_policy; print("\n".join(sorted(task for task in gym.registry if task.startswith("Isaac-Sweep-Object-UR5e-"))))'
```

Isaac Lab trainer의 기본 로그 위치는 실행 디렉터리의 `logs/rsl_rl/<experiment_name>/...`다. 비교 실험은 task ID, 상속 config, 실행한 디렉터리·명령, seed, checkpoint, 의존성 버전, 결과와 실패를 함께 기록한다. 관측·action·critic 차원이 바뀐 환경 간 체크포인트 재사용은 별도로 확인한다.

[diagnose_policy.py](diagnose_policy.py)는 자동 reset 직전 state를 JSONL로 기록한다. 이 script의 reach 높이와 gate 진단값은 초기 Inspire 기준의 `+0.09 m`/`0.04 m`로 고정되어 있으므로 `+0.04 m`를 사용하는 mass sensor 환경을 평가할 때는 환경 설정과 대조해야 한다.

## 확인 범위

문서는 현재 등록과 코드 설정을 기준으로 정리했다. `Random-v2`의 현재 OSC는 `pose_rel` 6D를 사용한다. 이전 absolute pose 7D 설명과 오래된 테스트 기대값은 [v2 문서](docs/04_isaac_sweep_object_ur5e_random_v2.md)의 감사 기록에서 구분한다. 정적 확인만으로 시뮬레이터 실행·학습 수렴·실물 센서 타당성이 검증되지는 않는다.
