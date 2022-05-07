# recommendation-systems

## 개발 환경
- 환경 및 패키지 의존성 관리를 위해 `Anaconda` 사용
- `environment.yaml`을 만들 때는 `prefix`가 포함되지 않도록 다음 명령어 사용
    - 윈도우: `conda env export | findstr -v "^prefix: " > environment.yaml`
    - 맥, 리눅스: `conda env export | grep -v "^prefix: " > environment.yaml`
- 파이토치 버전은 1.10.2, CUDA 버전은 11.3을 사용하며, 사용하는 GPU는 `NVIDIA GeForce GRX 1070 Ti`
- 코드 포맷터로는 `black`, Import 포맷터로는 `isort`, Code style checker로는 `flak8`을 이용하며, PR이 없는 개인 관리 저장소의 특징을 생각하여 `git hook`를 이용하여 commit 이전에 코드 퀄리티 확인 (`pre-commit install` 명령어를 통해 활성화해야 함)

