"""
pyquantum/__init__.py - 전체 API 구조 파악

PyQuantum - PyTorch 스타일의 직관적인 양자 시뮬레이터
"딥러닝 하듯이 양자 얽힘을 써보자"

PyTorch nn.Module 완전 통합
"""

__version__ = "0.2.0"  
__author__ = "zetavus PyQuantum Team"
__description__ = "PyTorch-native quantum computing library for intuitive quantum machine learning"

# 버전 체크 및 기본 설정
import sys
import warnings

# Python 버전 체크
if sys.version_info < (3, 7):
    raise RuntimeError("PyQuantum은 Python 3.7 이상이 필요합니다")

# PyTorch 체크
try:
    import torch
    if not hasattr(torch, '__version__'):
        raise ImportError("PyTorch가 올바르게 설치되지 않았습니다")
    
    torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
    if torch_version < (1, 8):
        warnings.warn(f"PyTorch {torch.__version__}가 감지되었습니다. PyTorch 1.8+ 권장합니다.", UserWarning)
        
except ImportError as e:
    print("X PyTorch가 설치되지 않았습니다.")
    print("   pip install torch 명령어로 설치해주세요.")
    raise e

# 핵심 모듈들 임포트 (오류 처리 포함)
try:
    # Phase 1: 기본 양자 컴퓨팅 모듈들
    from .qubit import (
        QubitState,
        zero_state,
        one_state,
        plus_state,
        bell_state
    )

    from .gates import (
        # 단일 큐비트 게이트 클래스
        PauliX, PauliY, PauliZ,
        Hadamard, Phase, TGate, Identity,
        RX, RY, RZ, PhaseShift,
        
        # 2큐비트 게이트 클래스
        CNOT, CZ, SWAP, CRZ,
        
        # 편의 함수
        X, Y, Z, H, S, T, I,
        Rx, Ry, Rz
    )

    from .circuit import (
        QuantumCircuit,
        create_bell_circuit,
        create_ghz_circuit,
        create_superposition_circuit
    )
    
    # Phase 1 임포트 성공
    _phase1_available = True
    
except ImportError as e:
    print(f"X PyQuantum Phase 1 모듈 임포트 실패: {e}")
    _phase1_available = False

# Phase 2: PyTorch 통합 모듈들
try:
    from .torch_layer import (
        # 핵심 클래스들
        QuantumLayer,
        HybridNet,
        QuantumFunction,
        
        # 편의 함수들
        create_quantum_classifier,
        create_quantum_regressor
    )
    
    # Phase 2 임포트 성공
    _phase2_available = True
    
    print("PyQuantum Phase 2: PyTorch 통합 모듈 로드됨!")
    
except ImportError as e:
    print(f"! PyQuantum Phase 2 (PyTorch 통합) 모듈 임포트 실패: {e}")
    print("   torch_layer.py 파일을 확인해주세요.")
    _phase2_available = False

# 임포트 성공 여부에 따른 처리
if _phase1_available and _phase2_available:
    _import_success = True
    print("PyQuantum 완전체 로드 성공!")
elif _phase1_available:
    _import_success = True
    print("V PyQuantum Phase 1 로드 성공 (Phase 2는 선택사항)")
else:
    print("X PyQuantum 핵심 모듈 로드 실패")
    _import_success = False

if _import_success:
    # GPU 지원 확인
    _cuda_available = torch.cuda.is_available()
    
    # 환영 메시지 (환경변수로 비활성화 가능)
    import os
    if os.environ.get('PYQUANTUM_QUIET') != '1':
        if _cuda_available:
            print("PyQuantum: CUDA GPU 가속이 활성화되었습니다!")
        else:
            print("PyQuantum: CPU 모드로 실행됩니다.")
        
        print(f"PyQuantum v{__version__}이 로드되었습니다!")
        
        if _phase2_available:
            print("PyTorch nn.Module 통합 완료 - 양자 신경망 사용 가능!")

# 주요 상수들
PI = 3.141592653589793
SQRT2 = 1.4142135623730951

# 전역 설정 클래스
class Config:
    """PyQuantum 전역 설정"""
    
    # 기본 데이터 타입
    DEFAULT_DTYPE = torch.complex64
    
    # 수치 정밀도
    NUMERICAL_TOLERANCE = 1e-10
    
    # GPU 사용 여부 (자동 감지)
    USE_GPU = _cuda_available if _import_success else False
    
    # Phase 2 기능 사용 여부
    ENABLE_PYTORCH_INTEGRATION = _phase2_available
    
    # 시각화 설정
    PLOT_STYLE = "default"
    FIGURE_SIZE = (8, 6)
    
    # 시뮬레이션 설정  
    MAX_QUBITS_WARNING = 20  # 이 개수 이상이면 경고
    DEFAULT_SHOTS = 1000
    
    # QML 설정 (Phase 2)
    DEFAULT_QUANTUM_LAYERS = 2
    DEFAULT_ANSATZ = 'RY'
    PARAMETER_SHIFT_DELTA = 1.5708  # π/2
    
    @classmethod
    def set_device(cls, device: str):
        """계산 디바이스 설정"""
        device = device.lower()
        if device in ['cpu']:
            cls.USE_GPU = False
            print("디바이스를 CPU로 설정했습니다.")
        elif device in ['cuda', 'gpu'] and torch.cuda.is_available():
            cls.USE_GPU = True
            print("디바이스를 GPU로 설정했습니다.")
        elif device in ['cuda', 'gpu']:
            print("! CUDA가 사용 불가능합니다. CPU 모드를 유지합니다.")
        else:
            raise ValueError(f"지원되지 않는 디바이스: {device}")
    
    @classmethod  
    def info(cls):
        """현재 설정 정보"""
        print("PyQuantum 설정:")
        print(f"   버전: {__version__}")
        print(f"   PyTorch: {torch.__version__}")
        print(f"   데이터 타입: {cls.DEFAULT_DTYPE}")
        print(f"   GPU 사용: {cls.USE_GPU}")
        print(f"   CUDA 사용 가능: {torch.cuda.is_available()}")
        print(f"   Phase 1 (기본): {'V' if _phase1_available else 'X'}")
        print(f"   Phase 2 (PyTorch): {'V' if _phase2_available else 'X'}")
        print(f"   수치 허용 오차: {cls.NUMERICAL_TOLERANCE}")
        print(f"   기본 측정 횟수: {cls.DEFAULT_SHOTS}")
        
        if _phase2_available:
            print(f"   기본 양자 레이어 수: {cls.DEFAULT_QUANTUM_LAYERS}")
            print(f"   기본 앤사츠: {cls.DEFAULT_ANSATZ}")

# 전역 설정 인스턴스
config = Config()

# 도움말 함수들
def quick_start():
    """PyQuantum 빠른 시작 예제"""
    if not _import_success:
        print("X PyQuantum 모듈이 올바르게 로드되지 않았습니다.")
        return
    
    print("PyQuantum 빠른 시작 예제")
    print("=" * 40)
    
    try:
        # Phase 1: 기본 양자 컴퓨팅
        print("Phase 1: 기본 양자 컴퓨팅")
        print("-" * 30)
        
        # 1. 단일 큐비트 중첩 상태
        qc = QuantumCircuit(1)
        qc.H(0)  # Hadamard 게이트 적용
        print(f"1. |+⟩ 상태: {qc.get_state()}")
        
        # 2. 벨 상태 (양자 얽힘)
        bell_circuit = create_bell_circuit()
        print(f"2. 벨 상태: {bell_circuit.get_state()}")
        
        # 3. 측정
        counts = bell_circuit.sample(shots=100)
        print(f"3. 측정 결과 (100회): {counts}")
        
        # Phase 2: PyTorch 통합 (가능한 경우)
        if _phase2_available:
            print(f"\nPhase 2: PyTorch 신경망 통합")
            print("-" * 35)
            
            # 4. 양자 신경망 레이어
            qlayer = QuantumLayer(n_qubits=3, n_layers=2)
            x = torch.randn(2, 3)
            output = qlayer(x)
            print(f"4. 양자층: {x.shape} → {output.shape}")
            
            # 5. 하이브리드 모델
            hybrid_model = create_quantum_classifier(
                input_size=4, n_qubits=3, n_classes=2
            )
            x = torch.randn(2, 4)
            output = hybrid_model(x)
            print(f"5. 하이브리드 모델: {x.shape} → {output.shape}")
            
            # 6. 자동 미분
            loss = torch.sum(output)
            loss.backward()
            grad_norm = sum(p.grad.norm().item() for p in hybrid_model.parameters())
            print(f"6. 자동 미분: 그래디언트 크기 {grad_norm:.4f}")
        
        print("\nV PyQuantum 사용법을 익혔습니다!")
        print("더 많은 예제는 examples/ 폴더를 참고하세요.")
        
        if _phase2_available:
            print("이제 실제 QML 문제를 해결할 수 있습니다!")
        
    except Exception as e:
        print(f"X 예제 실행 중 오류: {e}")
        print("설치 상태를 확인해주세요.")

def help():
    """PyQuantum 도움말"""
    help_text = """
PyQuantum - PyTorch 스타일 양자 컴퓨팅 라이브러리

Phase 1: 기본 양자 컴퓨팅
    from pyquantum import QuantumCircuit
    
    # 양자 회로 생성
    qc = QuantumCircuit(2)
    
    # 게이트 적용 (체이닝 방식)
    qc.H(0).CNOT(0, 1)  # Hadamard + CNOT
    
    # 상태 확인 및 측정
    print(qc.get_state())
    results = qc.sample(shots=1000)
"""
    
    if _phase2_available:
        help_text += """
Phase 2: PyTorch 신경망 통합
    from pyquantum import QuantumLayer, HybridNet
    import torch.nn as nn
    
    # 양자 신경망 레이어
    qlayer = QuantumLayer(n_qubits=4, n_layers=2)
    
    # 하이브리드 모델 구성
    model = nn.Sequential(
        nn.Linear(8, 4),
        qlayer,
        nn.Linear(4, 2)
    )
    
    # 일반 PyTorch 모델처럼 사용
    optimizer = torch.optim.Adam(model.parameters())
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()
"""
    
    help_text += f"""
주요 기능:
    V PyTorch 네이티브 (GPU 가속 지원)
    V 직관적인 체이닝 API
    V 교육 친화적 한글 지원
    {'V 양자 신경망 레이어 (Phase 2)' if _phase2_available else '! Phase 2 (PyTorch 통합) 비활성화'}
    {'V 자동 미분 지원 (Phase 2)' if _phase2_available else ''}
    {'V 하이브리드 모델 구축 (Phase 2)' if _phase2_available else ''}

더 많은 정보:
    - quick_start() : 빠른 시작 예제
    - config.info() : 설정 정보
    - GitHub: https://github.com/zetavus/pyquantum
    """
    
    print(help_text)

def version_info():
    """버전 정보 출력"""
    print(f"PyQuantum v{__version__}")
    print(f"PyTorch v{torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
    print(f"Python {sys.version}")
    print(f"Phase 1 (기본): {'V' if _phase1_available else 'X'}")
    print(f"Phase 2 (PyTorch): {'V' if _phase2_available else 'X'}")

def test_installation():
    """설치 상태 테스트"""
    print("PyQuantum 설치 테스트")
    print("=" * 30)
    
    tests = [
        ("PyTorch 임포트", lambda: __import__('torch')),
        ("Phase 1 모듈", lambda: _phase1_available),
        ("QubitState 생성", lambda: QubitState(n_qubits=1) if _phase1_available else None),
        ("QuantumCircuit 생성", lambda: QuantumCircuit(1) if _phase1_available else None),
        ("벨 상태 생성", lambda: create_bell_circuit() if _phase1_available else None),
    ]
    
    # Phase 2 테스트 추가
    if _phase2_available:
        tests.extend([
            ("QuantumLayer 생성", lambda: QuantumLayer(n_qubits=2, n_layers=1)),
            ("HybridNet 생성", lambda: HybridNet(2, 8, 2, 1, 2)),
            ("양자 분류기 생성", lambda: create_quantum_classifier(4, 3, 2)),
            ("자동 미분 테스트", lambda: _test_autograd()),
        ])
    
    success_count = 0
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result is not None and result is not False:
                print(f"V {test_name}")
                success_count += 1
            else:
                print(f"X {test_name}")
        except Exception as e:
            print(f"X {test_name}: {e}")
    
    print(f"\n테스트 결과: {success_count}/{len(tests)} 통과")
    
    if success_count == len(tests):
        print("PyQuantum이 완벽하게 설치되었습니다!")
        if _phase2_available:
            print("PyTorch 통합 기능까지 모두 사용 가능합니다!")
        return True
    else:
        print("! 일부 기능에 문제가 있습니다.")
        if not _phase2_available and _phase1_available:
            print("Phase 1은 정상이지만 Phase 2 (PyTorch 통합)를 사용하려면 torch_layer.py를 확인해주세요.")
        return False

def _test_autograd():
    """자동 미분 테스트 (내부 함수)"""
    if not _phase2_available:
        return False
    
    try:
        model = QuantumLayer(n_qubits=2, n_layers=1)
        x = torch.randn(1, 2, requires_grad=True)
        output = model(x)
        loss = torch.sum(output)
        loss.backward()
        
        # 그래디언트가 계산되었는지 확인
        has_grad = model.quantum_weights.grad is not None and x.grad is not None
        return has_grad
    except:
        return False

# __all__ 정의 (from pyquantum import *에서 가져올 것들)
_base_all = [
    # 유틸리티
    'quick_start', 'help', 'version_info', 'test_installation', 'config',
    'PI', 'SQRT2'
]

if _phase1_available:
    _base_all.extend([
        # 핵심 클래스들
        'QubitState', 'QuantumCircuit',
        
        # 상태 생성 함수들
        'zero_state', 'one_state', 'plus_state', 'bell_state',
        
        # 게이트 클래스들  
        'PauliX', 'PauliY', 'PauliZ', 'Hadamard', 'Phase', 'TGate',
        'RX', 'RY', 'RZ', 'PhaseShift',
        'CNOT', 'CZ', 'SWAP', 'CRZ',
        
        # 게이트 편의 함수들
        'X', 'Y', 'Z', 'H', 'S', 'T', 'I',
        'Rx', 'Ry', 'Rz',
        
        # 회로 생성 함수들
        'create_bell_circuit', 'create_ghz_circuit', 'create_superposition_circuit',
    ])

if _phase2_available:
    _base_all.extend([
        # Phase 2: PyTorch 통합
        'QuantumLayer', 'HybridNet', 'QuantumFunction',
        'create_quantum_classifier', 'create_quantum_regressor',
    ])

__all__ = _base_all

# Phase별 기능 확인 함수들
def check_phase1():
    """Phase 1 기능 사용 가능 여부"""
    return _phase1_available

def check_phase2():
    """Phase 2 기능 사용 가능 여부"""
    return _phase2_available

def get_available_features():
    """사용 가능한 기능 목록"""
    features = []
    
    if _phase1_available:
        features.extend([
            "기본 양자 회로 시뮬레이션",
            "양자 게이트 라이브러리",
            "양자 상태 측정",
            "벨 상태 및 얽힘 실험",
            "GPU 가속 지원"
        ])
    
    if _phase2_available:
        features.extend([
            "PyTorch nn.Module 통합",
            "양자 신경망 레이어",
            "자동 미분 (Autograd)",
            "하이브리드 모델 구축",
            "Parameter Shift Rule",
            "양자 머신러닝 예제"
        ])
    
    return features

def what_can_i_do():
    """사용 가능한 기능 안내"""
    print("PyQuantum으로 할 수 있는 것들:")
    print("=" * 40)
    
    features = get_available_features()
    for i, feature in enumerate(features, 1):
        print(f"{i:2d}. {feature}")
    
    if not _phase2_available and _phase1_available:
        print(f"\n추가 기능을 원한다면:")
        print("   torch_layer.py를 추가하여 Phase 2 기능을 활성화하세요!")
    
    print(f"\n시작하려면: quick_start() 실행")
    print(f"설정 확인: config.info() 실행")

# 경고 및 추천사항
def _show_recommendations():
    """추천사항 출력"""
    import os
    
    if os.environ.get('PYQUANTUM_QUIET') == '1':
        return
    
    recommendations = []
    
    # GPU 관련 추천
    if not torch.cuda.is_available():
        recommendations.append("GPU 가속을 위해 CUDA 버전 PyTorch 설치를 고려해보세요")
    
    # Phase 2 관련 추천
    if _phase1_available and not _phase2_available:
        recommendations.append("PyTorch 신경망 통합을 위해 torch_layer.py를 추가해보세요")
    
    # 버전 관련 추천
    if torch_version < (2, 0):
        recommendations.append("더 나은 성능을 위해 PyTorch 2.0+ 업그레이드를 고려해보세요")
    
    if recommendations:
        print("\n추천사항:")
        for rec in recommendations:
            print(f"   {rec}")

# 초기화 시 추천사항 표시
if _import_success:
    _show_recommendations()

# 버전별 호환성 정보
COMPATIBILITY_INFO = {
    "python": "3.7+",
    "torch": "1.8+",
    "numpy": "1.19+",
    "cuda": "10.2+ (선택사항)",
    "phase1_features": [
        "QuantumCircuit", "QubitState", "양자 게이트", "측정", "시뮬레이션"
    ],
    "phase2_features": [
        "QuantumLayer", "HybridNet", "자동 미분", "양자 신경망", "Parameter Shift Rule"
    ]
}

def compatibility_info():
    """호환성 정보 출력"""
    print("PyQuantum 호환성 정보:")
    print("=" * 30)
    
    for key, value in COMPATIBILITY_INFO.items():
        if key in ["phase1_features", "phase2_features"]:
            phase_num = "1" if "phase1" in key else "2"
            available = _phase1_available if "phase1" in key else _phase2_available
            status = "V" if available else "X"
            print(f"Phase {phase_num} {status}: {', '.join(value)}")
        else:
            print(f"{key}: {value}")

# 개발자용 디버그 정보
def debug_info():
    """개발자용 디버그 정보"""
    print("PyQuantum 디버그 정보:")
    print("=" * 30)
    print(f"Phase 1 available: {_phase1_available}")
    print(f"Phase 2 available: {_phase2_available}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Python version: {sys.version}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
    
    print(f"Available modules: {[name for name in __all__ if name[0].isupper()]}")

# 사용자 경험 개선을 위한 함수들
def get_started_guide():
    """초보자를 위한 가이드"""
    print("PyQuantum 시작 가이드")
    print("=" * 25)
    
    steps = [
        ("설치 확인", "test_installation()"),
        ("빠른 예제", "quick_start()"),
        ("기본 사용법", "help()"),
        ("설정 확인", "config.info()"),
    ]
    
    if _phase2_available:
        steps.extend([
            ("XOR 예제 실행", "python examples/xor_qnn.py"),
            ("하이브리드 모델", "from pyquantum import HybridNet"),
        ])
    
    for i, (step_name, command) in enumerate(steps, 1):
        print(f"{i}. {step_name}: {command}")
    
    print(f"\n튜토리얼: docs/ 폴더 참고")
    print(f"질문이 있으시면 GitHub Issues에 남겨주세요!")

# 최종 상태 요약
def status():
    """PyQuantum 현재 상태 요약"""
    print("PyQuantum 상태 요약")
    print("=" * 25)
    
    status_items = [
        ("버전", __version__),
        ("Phase 1 (기본)", "V 활성화" if _phase1_available else "X 비활성화"),
        ("Phase 2 (PyTorch)", "V 활성화" if _phase2_available else "X 비활성화"),
        ("GPU 가속", "V 사용 가능" if torch.cuda.is_available() else "X 사용 불가"),
        ("준비 상태", "완전 준비" if _phase1_available and _phase2_available else 
                     "V 기본 준비" if _phase1_available else "X 설정 필요"),
    ]
    
    for item, value in status_items:
        print(f"{item:15}: {value}")
    
    print(f"\n다음 단계: {'what_can_i_do()' if _import_success else 'test_installation()'}")

# 모듈 로딩 완료 후 상태 출력 (간단히)
if _import_success and os.environ.get('PYQUANTUM_QUIET') != '1':
    phase_status = "완전체" if _phase2_available else "Phase 1"
    print(f"PyQuantum {phase_status} 준비 완료! 시작하려면 quick_start() 실행")