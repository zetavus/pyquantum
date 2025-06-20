"""
pyquantum/circuit.py - 메인 QuantumCircuit 클래스

게이트를 큐비트에 적용하는 회로 구성 및 실행
"""

import torch
from typing import Union, List, Tuple, Optional, Dict, Any
import copy

# 상대 임포트 사용
try:
    # 패키지 내에서 실행될 때
    from .qubit import QubitState, zero_state
    from .gates import (
        QuantumGate, SingleQubitGate, TwoQubitGate, ParameterizedGate,
        Hadamard, PauliX, PauliY, PauliZ, Phase, TGate,
        RX, RY, RZ, CNOT, CZ, SWAP
    )
except ImportError:
    # 직접 실행되거나 개발 중일 때
    try:
        from qubit import QubitState, zero_state
        from gates import (
            QuantumGate, SingleQubitGate, TwoQubitGate, ParameterizedGate,
            Hadamard, PauliX, PauliY, PauliZ, Phase, TGate,
            RX, RY, RZ, CNOT, CZ, SWAP
        )
    except ImportError:
        print("모듈 임포트 오류: qubit.py와 gates.py 파일이 같은 디렉토리에 있는지 확인해주세요.")
        raise


class QuantumCircuit:
    """양자 회로 클래스 - PyTorch 스타일의 체이닝 API"""
    
    def __init__(self, n_qubits: int, initial_state: Optional[QubitState] = None):
        """
        Args:
            n_qubits: 큐비트 개수
            initial_state: 초기 상태 (None이면 |0...0⟩)
        """
        if n_qubits <= 0:
            raise ValueError("큐비트 개수는 1 이상이어야 합니다")
        
        self.n_qubits = n_qubits
        self.initial_state = initial_state or zero_state(n_qubits)
        self.current_state = copy.deepcopy(self.initial_state)
        
        # 회로 기록 (디버깅/시각화용)
        self.operations = []  # (gate_name, qubits, parameters) 튜플들
        self.measurement_results = {}  # 측정 결과 저장
        
        # 성능 최적화용
        self._compiled = False
        self._gate_sequence = []
    
    def reset(self) -> 'QuantumCircuit':
        """회로를 초기 상태로 리셋"""
        self.current_state = copy.deepcopy(self.initial_state)
        self.operations.clear()
        self.measurement_results.clear()
        self._compiled = False
        self._gate_sequence.clear()
        return self
    
    def copy(self) -> 'QuantumCircuit':
        """회로 복사"""
        new_circuit = QuantumCircuit(self.n_qubits, copy.deepcopy(self.initial_state))
        new_circuit.current_state = copy.deepcopy(self.current_state)
        new_circuit.operations = copy.deepcopy(self.operations)
        new_circuit.measurement_results = copy.deepcopy(self.measurement_results)
        return new_circuit
    
    def _validate_qubit(self, qubit: int):
        """큐비트 인덱스 유효성 검사"""
        if qubit < 0 or qubit >= self.n_qubits:
            raise ValueError(f"큐비트 인덱스 {qubit}이 범위를 벗어났습니다 (0-{self.n_qubits-1})")
    
    # ====== 단일 큐비트 게이트 메소드 (체이닝 API) ======
    
    def H(self, qubit: int) -> 'QuantumCircuit':
        """Hadamard 게이트 적용"""
        self._validate_qubit(qubit)
        gate = Hadamard()
        self.current_state = gate.apply_to_state(self.current_state, qubit)
        self.operations.append(("H", [qubit], {}))
        return self
    
    def X(self, qubit: int) -> 'QuantumCircuit':
        """Pauli-X 게이트 적용"""
        self._validate_qubit(qubit)
        gate = PauliX()
        self.current_state = gate.apply_to_state(self.current_state, qubit)
        self.operations.append(("X", [qubit], {}))
        return self
    
    def Y(self, qubit: int) -> 'QuantumCircuit':
        """Pauli-Y 게이트 적용"""
        self._validate_qubit(qubit)
        gate = PauliY()
        self.current_state = gate.apply_to_state(self.current_state, qubit)
        self.operations.append(("Y", [qubit], {}))
        return self
    
    def Z(self, qubit: int) -> 'QuantumCircuit':
        """Pauli-Z 게이트 적용"""
        self._validate_qubit(qubit)
        gate = PauliZ()
        self.current_state = gate.apply_to_state(self.current_state, qubit)
        self.operations.append(("Z", [qubit], {}))
        return self
    
    def S(self, qubit: int) -> 'QuantumCircuit':
        """Phase 게이트 적용"""
        self._validate_qubit(qubit)
        gate = Phase()
        self.current_state = gate.apply_to_state(self.current_state, qubit)
        self.operations.append(("S", [qubit], {}))
        return self
    
    def T(self, qubit: int) -> 'QuantumCircuit':
        """T 게이트 적용"""
        self._validate_qubit(qubit)
        gate = TGate()
        self.current_state = gate.apply_to_state(self.current_state, qubit)
        self.operations.append(("T", [qubit], {}))
        return self
    
    def Rx(self, qubit: int, theta: float) -> 'QuantumCircuit':
        """X축 회전 게이트 적용"""
        self._validate_qubit(qubit)
        gate = RX(theta)
        self.current_state = gate.apply_to_state(self.current_state, qubit)
        self.operations.append(("Rx", [qubit], {"theta": theta}))
        return self
    
    def Ry(self, qubit: int, theta: float) -> 'QuantumCircuit':
        """Y축 회전 게이트 적용"""
        self._validate_qubit(qubit)
        gate = RY(theta)
        self.current_state = gate.apply_to_state(self.current_state, qubit)
        self.operations.append(("Ry", [qubit], {"theta": theta}))
        return self
    
    def Rz(self, qubit: int, theta: float) -> 'QuantumCircuit':
        """Z축 회전 게이트 적용"""
        self._validate_qubit(qubit)
        gate = RZ(theta)
        self.current_state = gate.apply_to_state(self.current_state, qubit)
        self.operations.append(("Rz", [qubit], {"theta": theta}))
        return self
    
    # ====== 2큐비트 게이트 메소드 ======
    
    def CNOT(self, control: int, target: int) -> 'QuantumCircuit':
        """CNOT 게이트 적용"""
        self._validate_qubit(control)
        self._validate_qubit(target)
        if control == target:
            raise ValueError("제어 큐비트와 타겟 큐비트가 같을 수 없습니다")
        
        gate = CNOT()
        self.current_state = gate.apply_to_state(self.current_state, control, target)
        self.operations.append(("CNOT", [control, target], {}))
        return self
    
    def CX(self, control: int, target: int) -> 'QuantumCircuit':
        """CNOT의 별칭"""
        return self.CNOT(control, target)
    
    def CZ(self, control: int, target: int) -> 'QuantumCircuit':
        """제어 Z 게이트 적용"""
        self._validate_qubit(control)
        self._validate_qubit(target)
        if control == target:
            raise ValueError("제어 큐비트와 타겟 큐비트가 같을 수 없습니다")
        
        gate = CZ()
        self.current_state = gate.apply_to_state(self.current_state, control, target)
        self.operations.append(("CZ", [control, target], {}))
        return self
    
    def SWAP(self, qubit1: int, qubit2: int) -> 'QuantumCircuit':
        """SWAP 게이트 적용"""
        self._validate_qubit(qubit1)
        self._validate_qubit(qubit2)
        if qubit1 == qubit2:
            raise ValueError("같은 큐비트를 SWAP할 수 없습니다")
        
        gate = SWAP()
        self.current_state = gate.apply_to_state(self.current_state, qubit1, qubit2)
        self.operations.append(("SWAP", [qubit1, qubit2], {}))
        return self
    
    # ====== 범용 게이트 적용 메소드 ======
    
    def apply(self, gate: QuantumGate, *qubits) -> 'QuantumCircuit':
        """범용 게이트 적용 메소드"""
        if isinstance(gate, SingleQubitGate):
            if len(qubits) != 1:
                raise ValueError("단일 큐비트 게이트는 정확히 1개의 큐비트가 필요합니다")
            self._validate_qubit(qubits[0])
            self.current_state = gate.apply_to_state(self.current_state, qubits[0])
        elif isinstance(gate, TwoQubitGate):
            if len(qubits) != 2:
                raise ValueError("2큐비트 게이트는 정확히 2개의 큐비트가 필요합니다")
            self._validate_qubit(qubits[0])
            self._validate_qubit(qubits[1])
            self.current_state = gate.apply_to_state(self.current_state, qubits[0], qubits[1])
        else:
            raise NotImplementedError(f"게이트 타입 {type(gate)}은 아직 지원되지 않습니다")
        
        # 파라미터 추출
        params = {}
        if isinstance(gate, ParameterizedGate):
            params['parameter'] = gate.parameter
        
        self.operations.append((gate.name, list(qubits), params))
        return self
    
    # ====== 측정 메소드 ======
    
    def measure(self, qubit: int, classical_bit: str = None) -> 'QuantumCircuit':
        """단일 큐비트 측정"""
        self._validate_qubit(qubit)
        result, new_state = self.current_state.measure(qubit)
        self.current_state = new_state
        
        bit_name = classical_bit or f"c{qubit}"
        self.measurement_results[bit_name] = result
        self.operations.append(("measure", [qubit], {"classical_bit": bit_name}))
        return self
    
    def measure_all(self, classical_register: str = "c") -> 'QuantumCircuit':
        """모든 큐비트 측정"""
        results, new_state = self.current_state.measure()
        self.current_state = new_state
        
        # 결과를 개별 클래식 비트에 저장
        if isinstance(results, list):
            for i, result in enumerate(results):
                self.measurement_results[f"{classical_register}{i}"] = result
        else:
            self.measurement_results[f"{classical_register}0"] = results
        
        self.operations.append(("measure_all", list(range(self.n_qubits)), {"register": classical_register}))
        return self
    
    def sample(self, shots: int = 1000) -> Dict[str, int]:
        """여러 번 측정하여 통계 수집"""
        if shots <= 0:
            raise ValueError("측정 횟수는 1 이상이어야 합니다")
        
        counts = {}
        original_state = copy.deepcopy(self.current_state)
        
        for _ in range(shots):
            # 상태 복원
            self.current_state = copy.deepcopy(original_state)
            
            # 측정 수행
            results, _ = self.current_state.measure()
            
            # 비트 문자열로 변환
            if isinstance(results, list):
                bit_string = ''.join(map(str, results))
            else:
                bit_string = str(results)
            
            counts[bit_string] = counts.get(bit_string, 0) + 1
        
        # 원래 상태로 복원
        self.current_state = original_state
        return counts
    
    # ====== 상태 조회 메소드 ======
    
    def get_state(self) -> QubitState:
        """현재 양자 상태 반환"""
        return copy.deepcopy(self.current_state)
    
    def get_probabilities(self) -> torch.Tensor:
        """측정 확률 분포 반환"""
        return self.current_state.probability()
    
    def get_amplitudes(self) -> torch.Tensor:
        """상태 진폭 반환"""
        return self.current_state.state.clone()
    
    def expectation_value(self, observable: Union[QuantumGate, torch.Tensor]) -> float:
        """관측값의 기댓값 계산 (간단한 구현)"""
        if isinstance(observable, QuantumGate):
            if observable.n_qubits != 1:
                raise NotImplementedError("다중 큐비트 관측값은 아직 지원되지 않습니다")
            
            # 첫 번째 큐비트에 대해서만 계산 (간단한 구현)
            temp_state = copy.deepcopy(self.current_state)
            measured_state = observable.apply_to_state(temp_state, 0)
            
            # ⟨ψ|O|ψ⟩ 계산
            expectation = torch.real(torch.vdot(self.current_state.state, measured_state.state))
            return expectation.item()
        
        elif isinstance(observable, torch.Tensor):
            # 전체 상태에 대한 기댓값
            result_state = torch.matmul(observable, self.current_state.state)
            expectation = torch.real(torch.vdot(self.current_state.state, result_state))
            return expectation.item()
        
        raise ValueError("지원되지 않는 관측값 타입입니다")
    
    # ====== 유틸리티 메소드 ======
    
    def depth(self) -> int:
        """회로 깊이 계산 (순차 실행 기준)"""
        return len([op for op in self.operations if op[0] not in ["measure", "measure_all"]])
    
    def count_gates(self) -> Dict[str, int]:
        """게이트 종류별 개수 세기"""
        counts = {}
        for op in self.operations:
            gate_name = op[0]
            if gate_name not in ["measure", "measure_all"]:
                counts[gate_name] = counts.get(gate_name, 0) + 1
        return counts
    
    def to_qasm(self) -> str:
        """OpenQASM 형식으로 변환 (기본 구현)"""
        qasm = f"OPENQASM 2.0;\ninclude \"qelib1.inc\";\n"
        qasm += f"qreg q[{self.n_qubits}];\n"
        qasm += f"creg c[{self.n_qubits}];\n\n"
        
        for op in self.operations:
            gate_name, qubits, params = op
            
            if gate_name == "H":
                qasm += f"h q[{qubits[0]}];\n"
            elif gate_name == "X":
                qasm += f"x q[{qubits[0]}];\n"
            elif gate_name == "Y":
                qasm += f"y q[{qubits[0]}];\n"
            elif gate_name == "Z":
                qasm += f"z q[{qubits[0]}];\n"
            elif gate_name == "CNOT":
                qasm += f"cx q[{qubits[0]}],q[{qubits[1]}];\n"
            elif gate_name == "Rx":
                qasm += f"rx({params['theta']}) q[{qubits[0]}];\n"
            elif gate_name == "Ry":
                qasm += f"ry({params['theta']}) q[{qubits[0]}];\n"
            elif gate_name == "Rz":
                qasm += f"rz({params['theta']}) q[{qubits[0]}];\n"
            elif gate_name == "measure":
                qasm += f"measure q[{qubits[0]}] -> c[{qubits[0]}];\n"
            elif gate_name == "measure_all":
                for i in range(self.n_qubits):
                    qasm += f"measure q[{i}] -> c[{i}];\n"
        
        return qasm
    
    def __str__(self) -> str:
        """회로 정보 문자열"""
        info = f"QuantumCircuit({self.n_qubits} qubits, {len(self.operations)} operations)\n"
        info += f"Current state: {self.current_state}\n"
        
        if self.measurement_results:
            info += f"Measurements: {self.measurement_results}\n"
        
        return info
    
    def __repr__(self) -> str:
        return f"QuantumCircuit(n_qubits={self.n_qubits}, operations={len(self.operations)})"


# ====== 편의 함수들 ======

def create_bell_circuit(n_qubits: int = 2) -> QuantumCircuit:
    """벨 상태 생성 회로"""
    if n_qubits < 2:
        raise ValueError("벨 상태는 최소 2개의 큐비트가 필요합니다")
    
    qc = QuantumCircuit(n_qubits)
    qc.H(0).CNOT(0, 1)
    return qc

def create_ghz_circuit(n_qubits: int = 3) -> QuantumCircuit:
    """GHZ 상태 생성 회로"""
    if n_qubits < 2:
        raise ValueError("GHZ 상태는 최소 2개의 큐비트가 필요합니다")
    
    qc = QuantumCircuit(n_qubits)
    qc.H(0)
    for i in range(1, n_qubits):
        qc.CNOT(0, i)
    return qc

def create_superposition_circuit(n_qubits: int = 1) -> QuantumCircuit:
    """모든 큐비트에 중첩 상태 생성"""
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.H(i)
    return qc


# 테스트 코드 (파일 직접 실행 시)
if __name__ == "__main__":
    print("PyQuantum Circuit 테스트")
    
    try:
        # 기본 테스트
        qc = QuantumCircuit(2)
        print(f"2큐비트 회로 생성: {qc}")
        
        # 벨 상태 테스트  
        bell = create_bell_circuit()
        bell_state = bell.get_state()
        print(f"벨 상태: {bell_state}")
        
        # 측정 테스트
        counts = bell.sample(shots=100)
        print(f"측정 결과: {counts}")
        
        print("모든 테스트 통과!")
        
    except Exception as e:
        print(f"테스트 실패: {e}")
        import traceback
        traceback.print_exc()