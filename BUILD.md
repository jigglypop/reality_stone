# Reality Stone 빌드 가이드

## Docker 빌드


### 1. 개발 환경 실행

```bash
docker compose down -v
docker compose up -d dev
docker compose logs -f dev
docker compose exec -it reality_stone-dev-1 python tests/test.py
```

### 2. 수동 빌드 (필요시)

컨테이너 내부에서 직접 빌드가 필요한 경우:

```bash
docker compose exec -it dev bash
maturin develop --release
python -c "import reality_stone; print(reality_stone.__file__)"
```

### 3. 완전 초기화 및 캐시 삭제

컴퓨터가 느려지거나 디스크 공간 문제가 발생할 때, 아래 명령어로 모든 Docker 관련 데이터를 깨끗하게 삭제할 수 있습니다.

```bash
docker compose down -v
docker volume prune -f
docker builder prune -a -f
docker image prune -a -f
rm -rf target/
# 1. 깨끗한 상태에서 컨테이너 빌드 및 실행
DOCKER_BUILDKIT=1 docker compose up -d --build dev
docker compose exec -it dev bash
source /opt/venv/bin/activate
uv pip install patchelf 
maturin develop --release
python tests/test.py
```

### 6. 빌드 시간 예상
- 첫 빌드: 5-10분 (네트워크에 따라 다름)
- 캐시된 빌드: 1-2분
- 소스만 변경: 30초 이내

```bash

docker compose down -v
docker compose up -d --build dev
# 2. 컨테이너 접속
uv pip install -e .
uv pip install -r requirements.txt
docker compose exec -it dev bash
source /opt/venv/bin/activate
maturin develop --release

# 4. 테스트
python tests/test.py
```


```bash
docker compose exec -it dev bash
source ./.venv/bin/activate

cargo watch -w src -s "maturin develop"
```