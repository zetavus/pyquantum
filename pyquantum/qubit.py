"""
pyquantum/qubit.py - 양자 상태 관리

상태벡터 표현, 중첩 생성, 단일 큐비트 연산
"""

import torch
import numpy as np
from typing import Union, List, Tuple, Optional
from math import sqrt, pi, cos, sin


class QubitState:
    """양자 상태를 나타내는 클래스"""
    
    def __init__(self, state_vector: Optional[torch.Tensor] = None, n_qubits: int = 1):
        """
        Args:
            state_vector: 상태 벡터 (None이면 |0⟩ 상태로 초기화)
            n_qubits: 큐비트 개수
        """
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits
        
        if state_vector is None:
            # |0⟩ 상태로 초기화
            self.state = torch.zeros(self.dim, dtype=torch.complex64)
            self.state[0] = 1.0
        else:
            self.state = state_vector.clone()
            
        # GPU 사용 가능하면 자동으로 이동
        if torch.cuda.is_available():
            self.state = self.state.cuda()
    
    @classmethod
    def from_classical(cls, bits: Union[int, str, List[int]]) -> 'QubitState':
        """고전 비트 문자열에서 양자 상태 생성
        
        Args:
            bits: '01', 3, [0,1,1] 등의 형태
        """
        if isinstance(bits, int):
            # 10진수를 비트 문자열로 변환
            n_qubits = max(1, bits.bit_length())
            bit_str = format(bits, f'0{n_qubits}b')
        elif isinstance(bits, str):
            bit_str = bits
            n_qubits = len(bits)
        elif isinstance(bits, list):
            bit_str = ''.join(map(str, bits))
            n_qubits = len(bits)
        else:
            raise ValueError("bits는 int, str, 또는 List[int] 타입이어야 합니다")
        
        # |bit_str⟩ 상태 생성
        state_index = int(bit_str, 2)
        dim = 2 ** n_qubits
        state_vector = torch.zeros(dim, dtype=torch.complex64)
        state_vector[state_index] = 1.0
        
        return cls(state_vector, n_qubits)
    
    @classmethod
    def superposition(cls, amplitudes: List[complex], n_qubits: int = None) -> 'QubitState':
        """중첩 상태 직접 생성
        
        Args:
            amplitudes: 각 기저 상태의 진폭
            n_qubits: 큐비트 개수 (None이면 자동 계산)
        """
        if n_qubits is None:
            n_qubits = int(np.log2(len(amplitudes)))
        
        if len(amplitudes) != 2 ** n_qubits:
            raise ValueError(f"진폭 개수가 2^{n_qubits}와 맞지 않습니다")
        
        state_vector = torch.tensor(amplitudes, dtype=torch.complex64)
        # 정규화
        state_vector = state_vector / torch.norm(state_vector)
        
        return cls(state_vector, n_qubits)
    
    def normalize(self) -> 'QubitState':
        """상태 정규화"""
        norm = torch.norm(self.state)
        if norm > 1e-10:
            self.state = self.state / norm
        return self
    
    def probability(self, qubit_idx: int = None) -> torch.Tensor:
        """측정 확률 계산
        
        Args:
            qubit_idx: 특정 큐비트만 측정 (None이면 전체)
        """
        if qubit_idx is None:
            # 전체 상태의 확률 분포
            return torch.abs(self.state) ** 2
        else:
            # 특정 큐비트의 확률
            prob_0 = 0.0
            prob_1 = 0.0
            
            for i in range(self.dim):
                bit_val = (i >> (self.n_qubits - 1 - qubit_idx)) & 1
                prob = torch.abs(self.state[i]) ** 2
                if bit_val == 0:
                    prob_0 += prob
                else:
                    prob_1 += prob
            
            return torch.tensor([prob_0, prob_1])
    
    def measure(self, qubit_indices: Union[int, List[int]] = None) -> Tuple[Union[int, List[int]], 'QubitState']:
        """양자 측정 수행
        
        Args:
            qubit_indices: 측정할 큐비트 인덱스 (None이면 전체)
            
        Returns:
            (측정 결과, 측정 후 상태)
        """
        if qubit_indices is None:
            # 전체 측정
            probs = self.probability()
            outcome = torch.multinomial(probs, 1).item()
            
            # 측정 후 상태는 해당 기저 상태
            new_state = torch.zeros_like(self.state)
            new_state[outcome] = 1.0
            
            # 비트 문자열로 변환
            bit_string = format(outcome, f'0{self.n_qubits}b')
            result = [int(b) for b in bit_string]
            
            return result if len(result) > 1 else result[0], QubitState(new_state, self.n_qubits)
        
        elif isinstance(qubit_indices, int):
            # 단일 큐비트 측정
            probs = self.probability(qubit_indices)
            outcome = torch.multinomial(probs, 1).item()
            
            # 측정 후 상태 계산 (부분 측정)
            new_state = torch.zeros_like(self.state)
            norm = 0.0
            
            for i in range(self.dim):
                bit_val = (i >> (self.n_qubits - 1 - qubit_indices)) & 1
                if bit_val == outcome:
                    new_state[i] = self.state[i]
                    norm += torch.abs(self.state[i]) ** 2
            
            if norm > 1e-10:
                new_state = new_state / torch.sqrt(norm)
            
            return outcome, QubitState(new_state, self.n_qubits)
        
        else:
            # 다중 큐비트 측정 (재귀적으로 처리)
            current_state = self
            results = []
            
            for idx in sorted(qubit_indices, reverse=True):  # 역순으로 측정
                result, current_state = current_state.measure(idx)
                results.append(result)
            
            return list(reversed(results)), current_state
    
    def bloch_vector(self, qubit_idx: int = 0) -> torch.Tensor:
        """블로흐 구면 좌표 계산 (단일 큐비트만)"""
        if self.n_qubits != 1 and qubit_idx >= self.n_qubits:
            raise ValueError("블로흐 벡터는 단일 큐비트 상태에만 적용 가능합니다")
        
        if self.n_qubits == 1:
            # 단일 큐비트
            alpha = self.state[0]  # |0⟩ 진폭
            beta = self.state[1]   # |1⟩ 진폭
        else:
            # 다중 큐비트에서 특정 큐비트 추출 (부분 추적)
            # 간단한 구현: 다른 큐비트들을 무시하고 해당 큐비트만 고려
            raise NotImplementedError("다중 큐비트에서 부분 추적은 아직 구현되지 않았습니다")
        
        # 블로흐 벡터 계산
        x = 2 * torch.real(alpha * torch.conj(beta))
        y = 2 * torch.imag(alpha * torch.conj(beta))
        z = torch.abs(alpha) ** 2 - torch.abs(beta) ** 2
        
        return torch.tensor([x, y, z], dtype=torch.float32)
    
    def fidelity(self, other: 'QubitState') -> float:
        """다른 상태와의 충실도 계산"""
        if self.n_qubits != other.n_qubits:
            raise ValueError("다른 큐비트 개수의 상태와는 충실도를 계산할 수 없습니다")
        
        overlap = torch.vdot(self.state, other.state)
        return torch.abs(overlap) ** 2
    
    def entropy(self) -> float:
        """폰 노이만 엔트로피 계산 (순수 상태의 경우 항상 0)"""
        # 순수 상태의 엔트로피는 0
        # 혼합 상태를 위해서는 밀도 행렬이 필요
        return 0.0
    
    def __str__(self) -> str:
        """상태 표현 문자열"""
        terms = []
        for i, amp in enumerate(self.state):
            if torch.abs(amp) > 1e-10:
                bit_str = format(i, f'0{self.n_qubits}b')
                amp_val = amp.item()
                if isinstance(amp_val, complex):
                    if abs(amp_val.imag) < 1e-10:
                        amp_str = f"{amp_val.real:.3f}"
                    else:
                        amp_str = f"({amp_val.real:.3f}{amp_val.imag:+.3f}i)"
                else:
                    amp_str = f"{amp_val:.3f}"
                
                terms.append(f"{amp_str}|{bit_str}⟩")
        
        return " + ".join(terms) if terms else "0"
    
    def __repr__(self) -> str:
        return f"QubitState({self.n_qubits} qubits): {str(self)}"


# 편의 함수들
def zero_state(n_qubits: int = 1) -> QubitState:
    """모든 큐비트가 |0⟩인 상태"""
    return QubitState(n_qubits=n_qubits)

def one_state(n_qubits: int = 1) -> QubitState:
    """모든 큐비트가 |1⟩인 상태"""
    state_vector = torch.zeros(2 ** n_qubits, dtype=torch.complex64)
    state_vector[-1] = 1.0  # |111...1⟩
    return QubitState(state_vector, n_qubits)

def plus_state(n_qubits: int = 1) -> QubitState:
    """모든 큐비트가 |+⟩ = (|0⟩ + |1⟩)/√2 인 상태"""
    dim = 2 ** n_qubits
    state_vector = torch.ones(dim, dtype=torch.complex64) / sqrt(dim)
    return QubitState(state_vector, n_qubits)

def bell_state(bell_type: str = "phi_plus") -> QubitState:
    """벨 상태 생성
    
    Args:
        bell_type: "phi_plus", "phi_minus", "psi_plus", "psi_minus"
    """
    if bell_type == "phi_plus":
        # |Φ+⟩ = (|00⟩ + |11⟩)/√2
        amplitudes = [1/sqrt(2), 0, 0, 1/sqrt(2)]
    elif bell_type == "phi_minus":
        # |Φ-⟩ = (|00⟩ - |11⟩)/√2
        amplitudes = [1/sqrt(2), 0, 0, -1/sqrt(2)]
    elif bell_type == "psi_plus":
        # |Ψ+⟩ = (|01⟩ + |10⟩)/√2
        amplitudes = [0, 1/sqrt(2), 1/sqrt(2), 0]
    elif bell_type == "psi_minus":
        # |Ψ-⟩ = (|01⟩ - |10⟩)/√2
        amplitudes = [0, 1/sqrt(2), -1/sqrt(2), 0]
    else:
        raise ValueError("bell_type은 'phi_plus', 'phi_minus', 'psi_plus', 'psi_minus' 중 하나여야 합니다")
    
    return QubitState.superposition(amplitudes, 2)