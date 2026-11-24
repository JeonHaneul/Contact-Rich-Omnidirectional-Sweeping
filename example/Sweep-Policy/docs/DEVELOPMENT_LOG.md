# Sweep-Policy 개발 로그

[환경 목록](README.md) · [등록과 상속](ENVIRONMENT_REGISTRY.md) · [설치·실행](../README.md)

이 로그는 Sweep-Policy 내부 환경의 변경과 검증을 기록한다. Contact-Rich-Manipulation 전체 흐름·workspace 운영 기록은 [문서 서브모듈](../../Contact-Rich-Manipulation-Context/README.md)에 둔다. 파일이 남아 있다는 사실만으로 실험 성공이나 배포 완료를 추정하지 않는다.

## 2026-10-14 — Workspace 정리에 따른 환경 문서 감사

**변경:** README를 설치·실행·의존성·문서 탐색의 진입점으로 정리했다. 기존 Inspire 상세 내용은 [11번 문서](11_inspire_hand_variants.md)에 모으고, 누락된 Diff-IK scale / observation / homing / variable Kp / single OSC offset 변형은 [12번 문서](12_control_observation_variants.md)에 추가했다. [등록 표](ENVIRONMENT_REGISTRY.md)에 30개 task의 config 직접 부모, runtime class와 PPO runner를 연결했다. 기존 ROS Python 일반 지식은 [workspace docs](../../../docs/ROS2_PYTHON_ENVIRONMENTS.md)로 이동했다.

**현재 코드와 과거 설명의 차이:**

| 환경 | 현재 코드 | 교정한 설명 |
|---|---|---|
| `Random-v1` | Diff-IK scalar scale `0.5`, 전체 action 7D | 기존 문서의 scale `1.0` 및 소스의 `5 cm / 0.25 rad` 최대값 주석 |
| `Random-v2` | `pose_rel`, fixed gain 100, gravity compensation `False`, action 7D / observation 35D | absolute pose 7D arm / 전체 action 8D / observation 36D 설명과 class docstring |
| `Random-Homing-OSC-12D-Action-v0` | arm 12D + gripper 1D = 전체 13D | ID의 12D와 전체 action 차원의 구분을 추가 |
| Inspire `v3-mass` / `v3` / `v4` | reaching IK / initial target / privileged critic의 별도 branch | 부모·관측·runtime 차이를 명시 |

**패키지 이동 대응:** Inspire 기본 USD와 센서 source resolver를 `src/trash/hand_manipulation_rl`로 바꿨다. 같은 자산을 읽는 기존 테스트 3개의 경로도 함께 갱신했다. `shelf_utils.py`의 단독 실행 예제가 참조하던 옛 `src/shelf_policy` YAML을 패키지 내부 `ENVIRONMENT_YAML_PATH`로 교체했다. OSC 주석의 이전 비교 구현 경로를 `src/trash/test/rtde_osc_velocity.py`로 갱신했다. 학습 action, reward, reset·controller 설정값은 변경하지 않았다.

**검증:** Isaac Lab용 Python 환경에서 다음 명령을 실행했다. ROS `launch_testing` pytest plugin의 `lark` 의존성이 이 환경에 없으므로 이번 독립 테스트에는 자동 plugin 로딩을 껐다. 실행 위치는 `example/Sweep-Policy`다.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest -q \
  tests/test_inspire_open_asset.py tests/test_inspire_sensor_asset.py \
  tests/test_inspire_sensor_observations.py tests/test_sweeping_policy_registration.py
```

결과는 **14 passed, 2 skipped, 14 subtests passed**다. `pxr`가 해당 Python 환경에 없어 USD topology 테스트 두 모듈을 건너뛰었다. Gym 등록은 Isaac Lab을 eager import하지 않는 기존 계약 테스트로 확인했다. 등록 코드의 모든 30개 entry point와 config class·직접 부모를 정적으로 대조했고, 로컬 문서 링크와 `git diff --check`를 확인했다.

**확인한 기존 테스트 불일치:** `test_v1_contract.py` 실행은 4 passed / 1 failed, `test_v2_osc_contract.py` 실행은 2 passed / 1 failed였다. 실패 원인은 각각 과거 tuple action scale과 absolute OSC 계약을 기대하는 것이다. 두 config의 실행문은 이번 변경 전 HEAD와 동일함을 AST로 대조했다. 테스트 기대값이나 controller 동작을 이 정리 작업에서 바꾸지는 않았다.

**확인 범위:** Isaac Sim smoke script, GPU 학습·재생, Nucleus 접근 및 실제 장비 실행은 수행하지 않았다. 문서상의 차원과 파라미터는 코드에서 확인한 값이다. `[+0.09 m]`로 고정된 `diagnose_policy.py`의 reach/gate 진단은 mass sensor branch의 `+0.04 m` 기준과 구분해야 한다.

## 이후 기록 형식

새 환경 추가나 실행 기록에는 다음 내용을 함께 남긴다.

- 날짜, 정확한 task ID, 변경 목적
- config class와 직접 부모, runtime class, 기준 환경에서 바뀐 manager term
- action·policy·critic 차원, asset·sensor·checkpoint 의존성
- 실행한 디렉터리와 전체 명령, seed, 의존성 버전, 로그·checkpoint 경로
- 검증 결과와 실패, 코드로 확인한 내용과 실제 실행으로 확인한 내용

예시:

```text
날짜: YYYY-MM-DD
Task: Isaac-Sweep-Object-UR5e-...
Config / 부모 / Runtime: ...
변경한 manager term과 이유: ...
Action / actor / critic 차원: ...
실행 디렉터리와 명령: ...
Seed / 의존성 버전 / 로그 / checkpoint: ...
결과와 근거 / 실패 및 다음 확인: ...
```
