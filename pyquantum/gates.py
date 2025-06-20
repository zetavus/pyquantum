"""
pyquantum/gates.py - 양자 게이트 구현들

Hadamard, Pauli-X/Y/Z, CNOT 등 게이트 정의
"""

import torch
import numpy as np
from math import sqrt, pi, cos, sin, exp
from typing import Union, List, Optional
from abc import ABC, abstractmethod


class QuantumGate(ABC):
    """양자 게이트의 추상 기본 클래스"""
    
    def __init__(self, name: str, n_qubits: int, matrix: torch.Tensor):
        self.name = name
        self.n_qubits = n_qubits  # 게이트가 작용하는 큐비트 수
        self.matrix = matrix.to(dtype=torch.complex64)
        
        # GPU 사용 가능하면 자동으로 이동
        if torch.cuda.is_available():
            self.matrix = self.matrix.cuda()
    
    def __str__(self):
        return f"{self.name} Gate"
    
    def __repr__(self):
        return f"{self.name}Gate({self.n_qubits} qubits)"
    
    @abstractmethod
    def apply_to_state(self, state: 'QubitState', target_qubits: Union[int, List[int]]) -> 'QubitState':
        """상태에 게이트 적용"""
        pass


class SingleQubitGate(QuantumGate):
    """단일 큐비트 게이트"""
    
    def __init__(self, name: str, matrix: torch.Tensor):
        super().__init__(name, 1, matrix)
    
    def apply_to_state(self, state: 'QubitState', target_qubit: int) -> 'QubitState':
        """단일 큐비트에 게이트 적용"""
        if target_qubit >= state.n_qubits:
            raise ValueError(f"타겟 큐비트 {target_qubit}이 상태의 큐비트 수 {state.n_qubits}를 초과합니다")
        
        # 새로운 상태 벡터 생성
        new_state = torch.zeros_like(state.state)
        
        # 각 기저 상태에 대해 게이트 적용
        for i in range(state.dim):
            # i번째 기저 상태에서 target_qubit의 값
            qubit_val = (i >> (state.n_qubits - 1 - target_qubit)) & 1
            
            # 게이트 적용 후 새로운 진폭 계산
            for new_qubit_val in range(2):
                # 새로운 기저 상태 인덱스 계산
                new_i = i
                if qubit_val != new_qubit_val:
                    new_i ^= (1 << (state.n_qubits - 1 - target_qubit))
                
                # 게이트 행렬 요소
                gate_element = self.matrix[new_qubit_val, qubit_val]
                new_state[new_i] += gate_element * state.state[i]
        
        return state.__class__(new_state, state.n_qubits)


class TwoQubitGate(QuantumGate):
    """2큐비트 게이트"""
    
    def __init__(self, name: str, matrix: torch.Tensor):
        super().__init__(name, 2, matrix)
    
    def apply_to_state(self, state: 'QubitState', control_qubit: int, target_qubit: int) -> 'QubitState':
        """2큐비트에 게이트 적용"""
        if max(control_qubit, target_qubit) >= state.n_qubits:
            raise ValueError("큐비트 인덱스가 상태의 큐비트 수를 초과합니다")
        
        if control_qubit == target_qubit:
            raise ValueError("제어와 타겟 큐비트가 같을 수 없습니다")
        
        new_state = torch.zeros_like(state.state)
        
        for i in range(state.dim):
            # 현재 기저 상태에서 제어/타겟 큐비트 값
            control_val = (i >> (state.n_qubits - 1 - control_qubit)) & 1
            target_val = (i >> (state.n_qubits - 1 - target_qubit)) & 1
            
            # 2큐비트 상태 인덱스 (control이 상위 비트)
            two_qubit_state = (control_val << 1) | target_val
            
            # 게이트 적용 후 모든 가능한 출력 상태
            for new_two_qubit_state in range(4):
                new_control_val = (new_two_qubit_state >> 1) & 1
                new_target_val = new_two_qubit_state & 1
                
                # 새로운 기저 상태 인덱스 계산
                new_i = i
                if control_val != new_control_val:
                    new_i ^= (1 << (state.n_qubits - 1 - control_qubit))
                if target_val != new_target_val:
                    new_i ^= (1 << (state.n_qubits - 1 - target_qubit))
                
                # 게이트 행렬 요소
                gate_element = self.matrix[new_two_qubit_state, two_qubit_state]
                new_state[new_i] += gate_element * state.state[i]
        
        return state.__class__(new_state, state.n_qubits)


class ParameterizedGate(SingleQubitGate):
    """매개변수가 있는 게이트"""
    
    def __init__(self, name: str, parameter: float):
        self.parameter = parameter
        matrix = self._compute_matrix(parameter)
        super().__init__(name, matrix)
    
    @abstractmethod
    def _compute_matrix(self, parameter: float) -> torch.Tensor:
        """매개변수에 따른 행렬 계산"""
        pass
    
    def update_parameter(self, new_parameter: float):
        """매개변수 업데이트"""
        self.parameter = new_parameter
        self.matrix = self._compute_matrix(new_parameter)
        if torch.cuda.is_available():
            self.matrix = self.matrix.cuda()


# ====== 기본 단일 큐비트 게이트 ======

class PauliX(SingleQubitGate):
    """Pauli-X (NOT) 게이트"""
    def __init__(self):
        matrix = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
        super().__init__("X", matrix)

class PauliY(SingleQubitGate):
    """Pauli-Y 게이트"""
    def __init__(self):
        matrix = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)
        super().__init__("Y", matrix)

class PauliZ(SingleQubitGate):
    """Pauli-Z 게이트"""
    def __init__(self):
        matrix = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)
        super().__init__("Z", matrix)

class Hadamard(SingleQubitGate):
    """Hadamard 게이트"""
    def __init__(self):
        matrix = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64) / sqrt(2)
        super().__init__("H", matrix)

class Phase(SingleQubitGate):
    """Phase (S) 게이트"""
    def __init__(self):
        matrix = torch.tensor([[1, 0], [0, 1j]], dtype=torch.complex64)
        super().__init__("S", matrix)

class TGate(SingleQubitGate):
    """T 게이트"""
    def __init__(self):
        matrix = torch.tensor([[1, 0], [0, exp(1j * pi / 4)]], dtype=torch.complex64)
        super().__init__("T", matrix)

class Identity(SingleQubitGate):
    """항등 게이트"""
    def __init__(self):
        matrix = torch.tensor([[1, 0], [0, 1]], dtype=torch.complex64)
        super().__init__("I", matrix)


# ====== 매개변수 게이트 ======

class RX(ParameterizedGate):
    """X축 회전 게이트"""
    def __init__(self, theta: float):
        super().__init__("RX", theta)
    
    def _compute_matrix(self, theta: float) -> torch.Tensor:
        c = cos(theta / 2)
        s = sin(theta / 2)
        return torch.tensor([[c, -1j*s], [-1j*s, c]], dtype=torch.complex64)

class RY(ParameterizedGate):
    """Y축 회전 게이트"""
    def __init__(self, theta: float):
        super().__init__("RY", theta)
    
    def _compute_matrix(self, theta: float) -> torch.Tensor:
        c = cos(theta / 2)
        s = sin(theta / 2)
        return torch.tensor([[c, -s], [s, c]], dtype=torch.complex64)

class RZ(ParameterizedGate):
    """Z축 회전 게이트"""
    def __init__(self, theta: float):
        super().__init__("RZ", theta)
    
    def _compute_matrix(self, theta: float) -> torch.Tensor:
        return torch.tensor([[exp(-1j*theta/2), 0], [0, exp(1j*theta/2)]], dtype=torch.complex64)

class PhaseShift(ParameterizedGate):
    """일반적인 위상 시프트 게이트"""
    def __init__(self, phi: float):
        super().__init__("P", phi)
    
    def _compute_matrix(self, phi: float) -> torch.Tensor:
        return torch.tensor([[1, 0], [0, exp(1j*phi)]], dtype=torch.complex64)


# ====== 2큐비트 게이트 ======

class CNOT(TwoQubitGate):
    """제어 NOT 게이트"""
    def __init__(self):
        matrix = torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=torch.complex64)
        super().__init__("CNOT", matrix)

class CZ(TwoQubitGate):
    """제어 Z 게이트"""
    def __init__(self):
        matrix = torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, -1]
        ], dtype=torch.complex64)
        super().__init__("CZ", matrix)

class SWAP(TwoQubitGate):
    """SWAP 게이트"""
    def __init__(self):
        matrix = torch.tensor([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=torch.complex64)
        super().__init__("SWAP", matrix)

class CRZ(TwoQubitGate):
    """제어 RZ 게이트"""
    def __init__(self, theta: float):
        self.theta = theta
        matrix = torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, exp(-1j*theta/2), 0],
            [0, 0, 0, exp(1j*theta/2)]
        ], dtype=torch.complex64)
        super().__init__("CRZ", matrix)


# ====== 편의 함수 ======

def X() -> PauliX:
    """Pauli-X 게이트 생성"""
    return PauliX()

def Y() -> PauliY:
    """Pauli-Y 게이트 생성"""
    return PauliY()

def Z() -> PauliZ:
    """Pauli-Z 게이트 생성"""
    return PauliZ()

def H() -> Hadamard:
    """Hadamard 게이트 생성"""
    return Hadamard()

def S() -> Phase:
    """Phase 게이트 생성"""
    return Phase()

def T() -> TGate:
    """T 게이트 생성"""
    return TGate()

def I() -> Identity:
    """항등 게이트 생성"""
    return Identity()

def Rx(theta: float) -> RX:
    """X축 회전 게이트 생성"""
    return RX(theta)

def Ry(theta: float) -> RY:
    """Y축 회전 게이트 생성"""
    return RY(theta)

def Rz(theta: float) -> RZ:
    """Z축 회전 게이트 생성"""
    return RZ(theta)


# ====== 특수 게이트 조합 ======

def hadamard_test_gate(unitary_gate: SingleQubitGate) -> List[QuantumGate]:
    """하다마드 테스트를 위한 게이트 시퀀스"""
    return [H(), unitary_gate, H()]

def create_bell_circuit() -> List[tuple]:
    """벨 상태 생성 회로 (게이트, 큐비트) 튜플 리스트"""
    return [
        (H(), 0),
        (CNOT(), 0, 1)
    ]