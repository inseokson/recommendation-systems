# Recommendation Systems Implementation in PyTorch
PyTorch를 이용한 개인화 추천 시스템 구현

# Workflow

## Developer's Environment
- 본 저장소의 코드는 Window OS, 1.10.2 버전의 PyTorch(CUDA 11.3, NVIDIA GeForce GRX 1070 Ti)를 기준으로 작성되었습니다.

## Development Workflow
1. 환경 및 패키지 의존성 관리를 위해 `Anaconda Environment` 생성 (with `python 3.8.13`): `conda env create --name [name] --file=environment.yaml`
2. `black`, `isort` 그리고 `flake8`을 통해 코드 퀄리티를 유지할 수 있도록 Git hook 이용: `pre-commit install`
3. `recommenders` 패키지 설치: `pip install .`(`python setup.py install`) 혹은 `pip install -e .`(`python setup.py develop`)

## Release Workflow
1. 패키지 업데이트에 따른 `environment.yaml` 수정 시 `prefix`가 포함되지 않도록 다음 명령어 사용: 
    - 윈도우: `conda env export | findstr -v "^prefix: " > environment.yaml`
    - 맥, 리눅스: `conda env export | grep -v "^prefix: " > environment.yaml`

# Data
`./data` 폴더에 데이터별로 지정된 폴더명으로 저장
1. `/ml-100k`: GroupLens의 MovieLens 100K Dataset (Released 4/1998, [Link](https://grouplens.org/datasets/movielens/100k/))

