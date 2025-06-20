# PyQuantum 기여 가이드

PyQuantum 프로젝트에 기여해주셔서 감사합니다! 이 문서는 효과적인 기여를 위한 가이드라인을 제공합니다.

## 기여 방법

### 1. 개발 환경 설정

```bash
# 저장소 포크 및 클론
git clone https://github.com/zetavus/pyquantum.git
cd pyquantum

# 개발 환경 설치
make install-dev

# 또는 수동 설치
pip install -e ".[dev]"
```

### 2. 개발 워크플로우

```bash
# 새 브랜치 생성
git checkout -b feature/your-feature-name

# 개발 진행
# ... 코드 작성 ...

# 코드 품질 검사
make format        # 코드 포맷팅
make lint         # 코드 검사
make test         # 테스트 실행

# 또는 한 번에
make full-check

# 커밋 및 푸시
git add .
git commit -m "Add: your feature description"
git push origin feature/your-feature-name
```

### 3. Pull Request 생성

1. GitHub에서 Pull Request 생성
2. 명확한 제목과 설명 작성
3. 관련 이슈 번호 링크
4. 테스트 결과 첨부

## 코딩 스타일

### Python 코드 스타일

```python
# Good
def quantum_gate(qubit_index: int, angle: float) -> QuantumCircuit:
    """
    양자 게이트를 적용합니다.
    
    Args:
        qubit_index: 큐비트 인덱스
        angle: 회전 각도
        
    Returns:
        수정된 양자 회로
    """
    pass

# Bad
def qgate(i,a):
    pass
```

### 문서화 스타일

```python
class QuantumCircuit:
    """
    양자 회로 클래스
    
    PyTorch 스타일의 직관적인 양자 회로 구현체입니다.
    체이닝 방식으로 게이트를 연속적으로 적용할 수 있습니다.
    
    Examples:
        >>> qc = QuantumCircuit(2)
        >>> qc.H(0).CNOT(0, 1)
        >>> print(qc.get_state())
    """
```

## 테스트 작성

### 단위 테스트

```python
import pytest
import torch
from pyquantum import QuantumCircuit

def test_hadamard_gate():
    """하다마드 게이트 테스트"""
    qc = QuantumCircuit(1)
    qc.H(0)
    
    state = qc.get_state()
    expected = torch.tensor([1/torch.sqrt(torch.tensor(2.0)), 
                           1/torch.sqrt(torch.tensor(2.0))])
    
    torch.testing.assert_close(state.state, expected, atol=1e-6)

def test_bell_state():
    """벨 상태 생성 테스트"""
    qc = QuantumCircuit(2)
    qc.H(0).CNOT(0, 1)
    
    # 벨 상태 확인
    probs = qc.get_probabilities()
    assert torch.allclose(probs[0], torch.tensor(0.5), atol=1e-6)
    assert torch.allclose(probs[3], torch.tensor(0.5), atol=1e-6)
    assert torch.allclose(probs[1], torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(probs[2], torch.tensor(0.0), atol=1e-6)
```

### 성능 테스트

```python
import pytest
from pyquantum import QuantumCircuit

@pytest.mark.benchmark
def test_large_circuit_performance(benchmark):
    """대형 회로 성능 테스트"""
    def create_large_circuit():
        qc = QuantumCircuit(10)
        for i in range(10):
            qc.H(i)
        for i in range(9):
            qc.CNOT(i, i+1)
        return qc.get_state()
    
    result = benchmark(create_large_circuit)
    assert result is not None
```

## 새로운 기능 추가

### 1. 새로운 양자 게이트 추가

```python
# pyquantum/gates.py에 추가
def toffoli_gate():
    """토폴리 게이트 (3큐비트 제어 NOT)"""
    gate = torch.zeros(8, 8, dtype=torch.complex64)
    # 구현...
    return gate

# pyquantum/circuit.py에 메서드 추가
class QuantumCircuit:
    def Toffoli(self, control1: int, control2: int, target: int):
        """토폴리 게이트 적용"""
        # 구현...
        return self
```

### 2. 새로운 양자 알고리즘 추가

```python
# examples/grover_algorithm.py
from pyquantum import QuantumCircuit

def grover_search(target_items, n_qubits):
    """
    그로버 검색 알고리즘
    
    Args:
        target_items: 찾을 항목들
        n_qubits: 큐비트 수
        
    Returns:
        측정 결과
    """
    qc = QuantumCircuit(n_qubits)
    
    # 초기화
    for i in range(n_qubits):
        qc.H(i)
    
    # 그로버 반복
    iterations = int(torch.pi/4 * torch.sqrt(2**n_qubits))
    for _ in range(iterations):
        # Oracle
        # Diffusion operator
        pass
    
    return qc.sample(shots=1000)
```

## 문서화

### API 문서

모든 공개 함수와 클래스에는 docstring이 필요합니다:

```python
def create_bell_circuit(bell_type: str = "phi_plus") -> QuantumCircuit:
    """
    벨 상태를 생성하는 양자 회로를 만듭니다.
    
    Args:
        bell_type: 벨 상태 타입 ("phi_plus", "phi_minus", "psi_plus", "psi_minus")
        
    Returns:
        벨 상태를 생성하는 양자 회로
        
    Raises:
        ValueError: 지원하지 않는 벨 상태 타입인 경우
        
    Examples:
        >>> bell_circuit = create_bell_circuit("phi_plus")
        >>> print(bell_circuit.get_state())
        (|00⟩ + |11⟩)/√2
        
        >>> counts = bell_circuit.sample(shots=1000)
        >>> print(counts)
        {'00': ~500, '11': ~500}
    """
```

### 튜토리얼 작성

```markdown
# 새로운 기능 튜토리얼

## 개요
이 튜토리얼에서는 새로 추가된 기능을 사용하는 방법을 배웁니다.

## 예제 코드
\```python
from pyquantum import QuantumCircuit

# 기본 사용법
qc = QuantumCircuit(3)
qc.new_feature()
\```

## 주의사항
- 성능 고려사항
- 제한사항
- 모범 사례
```

## 이슈 제보

### 버그 리포트

버그를 발견하시면 다음 정보를 포함해주세요:

1. **환경 정보**
   - Python 버전
   - PyTorch 버전
   - PyQuantum 버전
   - 운영체제

2. **재현 방법**
   ```python
   # 최소한의 재현 가능한 예제
   from pyquantum import QuantumCircuit
   qc = QuantumCircuit(1)
   # 버그가 발생하는 코드
   ```

3. **예상 결과 vs 실제 결과**

4. **오류 메시지** (있다면)

### 기능 요청

새로운 기능을 제안하실 때는:

1. **사용 사례** 설명
2. **현재의 제한사항**
3. **제안하는 API** 설계
4. **참고 자료** (논문, 다른 구현체 등)

## 코드 리뷰 가이드라인

### 리뷰어를 위한 가이드

- 코드 스타일 준수 확인
- 테스트 커버리지 확인
- 성능 영향 검토
- 문서화 품질 확인
- API 설계 일관성 검토

### 기여자를 위한 가이드

- 작은 단위로 PR 분할
- 명확한 커밋 메시지
- 충분한 테스트 작성
- 문서 업데이트

## 릴리즈 프로세스

### 버전 관리

- Semantic Versioning (MAJOR.MINOR.PATCH) 사용
- 주요 변경사항은 CHANGELOG.md에 기록

### 릴리즈 체크리스트

1. [ ] 모든 테스트 통과
2. [ ] 문서 업데이트
3. [ ] 버전 번호 업데이트
4. [ ] CHANGELOG.md 업데이트
5. [ ] 태그 생성 및 푸시

## 커뮤니티

### 소통 채널

- GitHub Issues: 버그 리포트, 기능 요청
- GitHub Discussions: 일반적인 질문, 아이디어 논의
- Discord: 실시간 채팅 (개발 예정)

### 행동 강령

모든 기여자는 서로를 존중하고 배려하는 환경을 만들어야 합니다:

- 건설적인 피드백 제공
- 다양한 관점 존중
- 초보자 친화적인 환경 조성
- 포용적인 언어 사용

## 감사의 말

PyQuantum에 기여해주시는 모든 분들께 진심으로 감사드립니다. 여러분의 기여가 양자 컴퓨팅을 더 많은 사람들이 접근할 수 있게 만듭니다.

---

더 자세한 정보가 필요하시면 [GitHub Discussions](https://github.com/zetavus/pyquantum/discussions)에서 질문해주세요!