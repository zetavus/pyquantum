# PyQuantum Makefile
# 개발, 테스트, 배포를 위한 편의 명령어들

.PHONY: help install install-dev test test-fast lint format clean build upload docker docs

# 기본 명령어 (help 출력)
help:
	@echo "🌟 PyQuantum 개발 도구"
	@echo "===================="
	@echo ""
	@echo "📦 설치 관련:"
	@echo "  install      - 기본 설치"
	@echo "  install-dev  - 개발 환경 설치"
	@echo "  uninstall    - 패키지 제거"
	@echo ""
	@echo "🧪 테스트 관련:"
	@echo "  test         - 전체 테스트 실행"
	@echo "  test-fast    - 빠른 테스트만 실행"
	@echo "  test-gpu     - GPU 테스트 실행"
	@echo "  test-cover   - 커버리지와 함께 테스트"
	@echo ""
	@echo "🔧 코드 품질:"
	@echo "  lint         - 코드 검사"
	@echo "  format       - 코드 포맷팅"
	@echo "  type-check   - 타입 검사"
	@echo ""
	@echo "📊 예제 실행:"
	@echo "  demo         - 기본 데모 실행"
	@echo "  demo-bell    - 벨 상태 데모"
	@echo "  demo-xor     - XOR 양자신경망 데모"
	@echo "  demo-all     - 모든 예제 실행"
	@echo ""
	@echo "🐳 Docker 관련:"
	@echo "  docker-build - Docker 이미지 빌드"
	@echo "  docker-run   - Docker 컨테이너 실행"
	@echo "  docker-jupyter - Jupyter Lab with Docker"
	@echo "  docker-clean - Docker 정리"
	@echo ""
	@echo "📚 문서화:"
	@echo "  docs         - 문서 생성"
	@echo "  docs-serve   - 문서 서버 실행"
	@echo ""
	@echo "🚀 배포:"
	@echo "  build        - 패키지 빌드"
	@echo "  upload       - PyPI 업로드"
	@echo "  release      - 릴리즈 준비"
	@echo ""
	@echo "🧹 정리:"
	@echo "  clean        - 빌드 파일 정리"
	@echo "  clean-all    - 모든 임시 파일 정리"

# 설치 관련
install:
	@echo "📦 PyQuantum 기본 설치 중..."
	pip install -e .
	@echo "✅ 설치 완료!"

install-dev:
	@echo "📦 PyQuantum 개발 환경 설치 중..."
	pip install -e ".[dev]"
	pip install pytest pytest-cov black flake8 mypy jupyterlab
	@echo "✅ 개발 환경 설치 완료!"

uninstall:
	@echo "🗑️ PyQuantum 제거 중..."
	pip uninstall pyquantum -y
	@echo "✅ 제거 완료!"

# 테스트 관련
test:
	@echo "🧪 전체 테스트 실행 중..."
	python -m pytest tests/ -v --tb=short
	@echo "✅ 테스트 완료!"

test-fast:
	@echo "⚡ 빠른 테스트 실행 중..."
	python -m pytest tests/test_basic.py -v
	@echo "✅ 빠른 테스트 완료!"

test-gpu:
	@echo "🚀 GPU 테스트 실행 중..."
	python -c "import torch; print(f'CUDA 사용 가능: {torch.cuda.is_available()}')"
	python -c "from pyquantum import test_installation; test_installation()"
	@echo "✅ GPU 테스트 완료!"

test-cover:
	@echo "📊 커버리지 테스트 실행 중..."
	python -m pytest tests/ --cov=pyquantum --cov-report=html --cov-report=term
	@echo "✅ 커버리지 테스트 완료! htmlcov/index.html 확인"

# 코드 품질
lint:
	@echo "🔍 코드 검사 중..."
	flake8 pyquantum/ --max-line-length=88 --extend-ignore=E203,W503
	@echo "✅ 코드 검사 완료!"

format:
	@echo "🎨 코드 포맷팅 중..."
	black pyquantum/ examples/ tests/ --line-length=88
	@echo "✅ 포맷팅 완료!"

type-check:
	@echo "🔎 타입 검사 중..."
	mypy pyquantum/ --ignore-missing-imports
	@echo "✅ 타입 검사 완료!"

# 예제 실행
demo:
	@echo "🎯 기본 데모 실행 중..."
	python -c "from pyquantum import test_installation; test_installation()"

demo-bell:
	@echo "🔗 벨 상태 데모 실행 중..."
	python examples/bell_state.py

demo-xor:
	@echo "🧠 XOR 양자신경망 데모 실행 중..."
	python examples/xor_qnn.py

demo-all:
	@echo "🎪 모든 예제 실행 중..."
	@make demo-bell
	@make demo-xor
	python examples/basic_gates.py
	@echo "✅ 모든 데모 완료!"

# Docker 관련
docker-build:
	@echo "🐳 Docker 이미지 빌드 중..."
	docker build -t pyquantum:latest .
	@echo "✅ Docker 이미지 빌드 완료!"

docker-run:
	@echo "🚀 Docker 컨테이너 실행 중..."
	docker-compose up -d pyquantum
	docker-compose exec pyquantum bash
	@echo "✅ Docker 컨테이너 실행 완료!"

docker-jupyter:
	@echo "📊 Jupyter Lab with Docker 실행 중..."
	docker-compose --profile jupyter up -d jupyter
	@echo "✅ Jupyter Lab 실행됨! http://localhost:8889 접속"

docker-clean:
	@echo "🧹 Docker 정리 중..."
	docker-compose down
	docker system prune -f
	@echo "✅ Docker 정리 완료!"

# 문서화
docs:
	@echo "📚 문서 생성 중..."
	cd docs && make html
	@echo "✅ 문서 생성 완료!"

docs-serve:
	@echo "🌐 문서 서버 실행 중..."
	cd docs/_build/html && python -m http.server 8000
	@echo "✅ 문서 서버 실행! http://localhost:8000 접속"

# 배포 관련
build:
	@echo "🔨 패키지 빌드 중..."
	python -m build
	@echo "✅ 빌드 완료!"

upload:
	@echo "📤 PyPI 업로드 중..."
	python -m twine upload dist/*
	@echo "✅ 업로드 완료!"

release: clean build upload
	@echo "🎉 릴리즈 완료!"

# 정리
clean:
	@echo "🧹 빌드 파일 정리 중..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	@echo "✅ 정리 완료!"

clean-all: clean
	@echo "🧹 모든 임시 파일 정리 중..."
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf results/
	rm -rf experiments/
	@echo "✅ 전체 정리 완료!"

# 개발 워크플로우
dev-setup: install-dev
	@echo "🛠️ 개발 환경 설정 완료!"
	@echo "다음 명령어들을 사용할 수 있습니다:"
	@echo "  make test      - 테스트"
	@echo "  make format    - 코드 포맷팅"
	@echo "  make demo      - 데모 실행"

quick-check: format lint test-fast
	@echo "✅ 빠른 품질 검사 완료!"

full-check: format lint type-check test
	@echo "✅ 전체 품질 검사 완료!"