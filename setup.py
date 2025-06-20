"""
PyQuantum 패키지 설정
pip install -e . 로 개발 모드 설치 가능
"""

from setuptools import setup, find_packages
import os

# 현재 디렉토리
here = os.path.abspath(os.path.dirname(__file__))

# README 파일 읽기
def read_readme():
    try:
        with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "PyQuantum - PyTorch-native quantum computing library"

# requirements.txt 파일 읽기
def read_requirements():
    try:
        with open(os.path.join(here, 'requirements.txt'), encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        return ['torch>=1.8.0', 'numpy>=1.19.0']

# 버전 정보 읽기
def get_version():
    try:
        # __init__.py에서 버전 읽기
        init_file = os.path.join(here, 'pyquantum', '__init__.py')
        with open(init_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('__version__'):
                    return line.split('=')[1].strip().strip('"\'')
    except (FileNotFoundError, IndexError):
        return "0.1.0"

setup(
    # 기본 정보
    name="pyquantum",
    version=get_version(),
    description="PyTorch-native quantum computing library for intuitive quantum machine learning",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    
    # 작성자 정보
    author="PyQuantum Team",
    author_email="pyquantum@example.com",
    maintainer="PyQuantum Team",
    maintainer_email="pyquantum@example.com",
    
    # 프로젝트 URL들
    url="https://github.com/zetavus/pyquantum",
    project_urls={
        "Documentation": "https://pyquantum.readthedocs.io",
        "Source": "https://github.com/zetavus/pyquantum",
        "Tracker": "https://github.com/zetavus/pyquantum/issues",
        "Discussions": "https://github.com/zetavus/pyquantum/discussions",
    },
    
    # 패키지 정보
    packages=find_packages(exclude=['tests*', 'examples*', 'docs*']),
    include_package_data=True,
    
    # Python 버전 요구사항
    python_requires=">=3.7",
    
    # 의존성
    install_requires=read_requirements(),
    
    # 선택적 의존성
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'black>=21.0',
            'flake8>=3.8',
            'mypy>=0.900',
            'pre-commit>=2.15',
        ],
        'docs': [
            'sphinx>=4.0',
            'sphinx-rtd-theme>=1.0',
            'myst-parser>=0.15',
            'nbsphinx>=0.8',
        ],
        'viz': [
            'matplotlib>=3.3.0',
            'plotly>=5.0.0',
            'networkx>=2.5',
        ],
        'jupyter': [
            'jupyter>=1.0.0',
            'ipywidgets>=7.6.0',
            'voila>=0.3.0',
        ],
        'all': [
            'pytest>=6.0', 'pytest-cov>=2.0', 'black>=21.0', 'flake8>=3.8',
            'sphinx>=4.0', 'sphinx-rtd-theme>=1.0',
            'matplotlib>=3.3.0', 'plotly>=5.0.0',
            'jupyter>=1.0.0', 'ipywidgets>=7.6.0',
        ]
    },
    
    # 분류 정보
    classifiers=[
        # 개발 상태
        "Development Status :: 3 - Alpha",
        
        # 대상 사용자
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        
        # 주제 분야
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics", 
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Education",
        
        # 라이선스
        "License :: OSI Approved :: MIT License",
        
        # 운영체제
        "Operating System :: OS Independent",
        
        # Python 버전들
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8", 
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        
        # 기타
        "Natural Language :: English",
        "Natural Language :: Korean",
    ],
    
    # 키워드
    keywords=[
        "quantum computing", "quantum machine learning", "pytorch", 
        "quantum circuits", "quantum simulation", "quantum algorithms",
        "교육", "양자컴퓨팅", "양자머신러닝", "파이토치"
    ],
    
    # 콘솔 스크립트 (명령행 도구)
    entry_points={
        'console_scripts': [
            'pyquantum-test=pyquantum.cli:test_command',
            'pyquantum-demo=pyquantum.cli:demo_command',
        ],
    },
    
    # 패키지 데이터
    package_data={
        'pyquantum': [
            'data/*.json',
            'examples/*.py',
            'tutorials/*.ipynb',
        ],
    },
    
    # 라이선스
    license="MIT",
    
    # zip_safe 설정
    zip_safe=False,
    
    # 추가 메타데이터
    platforms=["any"],
)