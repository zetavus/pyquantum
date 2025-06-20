"""
pyquantum/torch_layer.py

PyTorch nn.Module 기반 양자 신경망 레이어
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
from typing import Optional, List, Tuple, Union, Callable
import math

# 상대 임포트
try:
    from .circuit import QuantumCircuit
    from .gates import RX, RY, RZ, CNOT
    from .qubit import QubitState
except ImportError:
    try:
        from circuit import QuantumCircuit
        from gates import RX, RY, RZ, CNOT
        from qubit import QubitState
    except ImportError:
        print("PyQuantum 모듈을 찾을 수 없습니다.")
        raise


class QuantumFunction(Function):
    """
    PyTorch autograd를 위한 양자 함수
    순전파와 역전파를 정의
    """
    
    @staticmethod
    def forward(ctx, inputs, weights, n_qubits, n_layers, ansatz_type='RY'):
        """
        순전파: 고전 입력 → 양자 처리 → 고전 출력
        
        Args:
            ctx: PyTorch context for backward
            inputs: 입력 데이터 (batch_size, input_dim)
            weights: 양자 회로 파라미터 (n_params,)
            n_qubits: 큐비트 개수
            n_layers: 레이어 개수
            ansatz_type: 앤사츠 타입
        
        Returns:
            outputs: 출력 데이터 (batch_size, n_qubits)
        """
        batch_size = inputs.shape[0]
        device = inputs.device
        
        # 컨텍스트에 저장 (역전파용)
        ctx.save_for_backward(inputs, weights)
        ctx.n_qubits = n_qubits
        ctx.n_layers = n_layers
        ctx.ansatz_type = ansatz_type
        ctx.device = device  # 디바이스 정보 저장
        
        # 배치 처리
        outputs = torch.zeros(batch_size, n_qubits, device=device, dtype=torch.float32)
        
        for i in range(batch_size):
            # 개별 샘플 처리
            single_input = inputs[i]
            single_output = QuantumFunction._process_single_sample(
                single_input, weights, n_qubits, n_layers, ansatz_type
            )
            # 디바이스 일관성 확보
            outputs[i] = single_output.to(device)
        
        return outputs
    
    @staticmethod
    def _process_single_sample(input_data, weights, n_qubits, n_layers, ansatz_type):
        """단일 샘플 양자 처리"""
        # 양자 회로 생성
        qc = QuantumCircuit(n_qubits)
        
        # 1. 데이터 인코딩 (입력을 양자 상태로)
        QuantumFunction._encode_classical_data(qc, input_data, n_qubits)
        
        # 2. 파라미터화 양자 회로 (앤사츠) 적용
        QuantumFunction._apply_ansatz(qc, weights, n_qubits, n_layers, ansatz_type)
        
        # 3. 측정 (양자 → 고전)
        probabilities = qc.get_probabilities()
        
        # 4. 각 큐비트의 기댓값 계산
        expectations = QuantumFunction._compute_expectations(probabilities, n_qubits)
        
        # 5. 디바이스 통일 (입력과 같은 디바이스로)
        device = input_data.device if hasattr(input_data, 'device') else 'cpu'
        expectations = expectations.to(device)
        
        return expectations
    
    @staticmethod
    def _encode_classical_data(qc, data, n_qubits):
        """고전 데이터를 양자 상태로 인코딩"""
        # 간단한 각도 인코딩: 데이터 값을 회전 각도로 사용
        for i in range(min(len(data), n_qubits)):
            # 데이터를 [0, 2π] 범위로 정규화
            angle = float(data[i]) * math.pi  # [-π, π] → [-π², π²] → 적절히 조정
            qc.Ry(i, angle)
    
    @staticmethod
    def _apply_ansatz(qc, weights, n_qubits, n_layers, ansatz_type):
        """파라미터화 양자 회로 (앤사츠) 적용"""
        weight_idx = 0
        
        for layer in range(n_layers):
            # 단일 큐비트 회전
            for qubit in range(n_qubits):
                if ansatz_type == 'RY':
                    if weight_idx < len(weights):
                        qc.Ry(qubit, float(weights[weight_idx]))
                        weight_idx += 1
                elif ansatz_type == 'RX':
                    if weight_idx < len(weights):
                        qc.Rx(qubit, float(weights[weight_idx]))
                        weight_idx += 1
                elif ansatz_type == 'full':
                    # RX, RY만 사용 (RZ는 복소수 문제로 제외)
                    for gate_type in ['RX', 'RY']:
                        if weight_idx < len(weights):
                            angle = float(weights[weight_idx])
                            if gate_type == 'RX':
                                qc.Rx(qubit, angle)
                            elif gate_type == 'RY':
                                qc.Ry(qubit, angle)
                            weight_idx += 1
            
            # 큐비트 간 얽힘 (인접한 큐비트들끼리 CNOT)
            for qubit in range(n_qubits - 1):
                qc.CNOT(qubit, (qubit + 1) % n_qubits)
    
    @staticmethod
    def _compute_expectations(probabilities, n_qubits):
        """확률 분포에서 각 큐비트의 기댓값 계산"""
        # 디바이스 정보 보존
        device = probabilities.device
        
        # 각 큐비트에 대해 <Z> 기댓값 계산
        expectations = torch.zeros(n_qubits, device=device)
        
        for qubit in range(n_qubits):
            prob_0 = 0.0  # |0⟩ 확률
            prob_1 = 0.0  # |1⟩ 확률
            
            # 모든 기저 상태에 대해 해당 큐비트 값 확인
            for state_idx in range(len(probabilities)):
                # state_idx를 이진수로 변환해서 qubit 위치의 비트 확인
                bit_val = (state_idx >> (n_qubits - 1 - qubit)) & 1
                
                if bit_val == 0:
                    prob_0 += probabilities[state_idx]
                else:
                    prob_1 += probabilities[state_idx]
            
            # <Z> = P(0) - P(1)
            expectations[qubit] = prob_0 - prob_1
        
        return expectations
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        역전파: 파라미터 shift rule 사용
        
        Args:
            ctx: forward에서 저장된 컨텍스트
            grad_output: 출력에 대한 그래디언트
        
        Returns:
            input_grad, weight_grad: 입력과 가중치에 대한 그래디언트
        """
        inputs, weights = ctx.saved_tensors
        n_qubits = ctx.n_qubits
        n_layers = ctx.n_layers
        ansatz_type = ctx.ansatz_type
        
        # 디바이스 정보 안전하게 가져오기
        if hasattr(ctx, 'device'):
            device = ctx.device
        else:
            device = inputs.device  # fallback
        
        batch_size = inputs.shape[0]
        n_params = len(weights)
        
        # 그래디언트 초기화
        input_grad = torch.zeros_like(inputs)
        weight_grad = torch.zeros_like(weights)
        
        # Parameter Shift Rule을 사용한 그래디언트 계산
        shift = math.pi / 2  # π/2 shift
        
        for param_idx in range(n_params):
            # 각 배치에 대해 그래디언트 계산
            param_grad_sum = 0.0
            
            for batch_idx in range(batch_size):
                # weights[param_idx] + π/2
                weights_plus = weights.clone()
                weights_plus[param_idx] += shift
                
                # weights[param_idx] - π/2  
                weights_minus = weights.clone()
                weights_minus[param_idx] -= shift
                
                # 순전파 (shift된 파라미터들로)
                output_plus = QuantumFunction._process_single_sample(
                    inputs[batch_idx], weights_plus, n_qubits, n_layers, ansatz_type
                )
                output_minus = QuantumFunction._process_single_sample(
                    inputs[batch_idx], weights_minus, n_qubits, n_layers, ansatz_type
                )
                
                # 디바이스 통일
                output_plus = output_plus.to(device)
                output_minus = output_minus.to(device)
                grad_output_batch = grad_output[batch_idx].to(device)
                
                # Parameter shift rule: ∂f/∂θ = (f(θ+π/2) - f(θ-π/2)) / 2
                param_grad = (output_plus - output_minus) / 2.0
                
                # Chain rule: grad_output과 곱해서 최종 그래디언트
                final_grad = torch.sum(grad_output_batch * param_grad)
                param_grad_sum += final_grad
            
            weight_grad[param_idx] = param_grad_sum / batch_size
        
        # 입력에 대한 그래디언트는 현재 구현하지 않음 (필요시 추가)
        return input_grad, weight_grad, None, None, None


class QuantumLayer(nn.Module):
    """
    PyTorch nn.Module 기반 양자 신경망 레이어
    
    사용법:
        layer = QuantumLayer(n_qubits=4, n_layers=2)
        output = layer(input_tensor)
    """
    
    def __init__(
        self, 
        n_qubits: int,
        n_layers: int = 1,
        ansatz_type: str = 'RY',
        input_scaling: float = 1.0,
        initialization: str = 'random'
    ):
        """
        Args:
            n_qubits: 양자 비트 개수
            n_layers: 양자 회로 레이어 개수  
            ansatz_type: 앤사츠 종류 ('RY', 'RX', 'full')
            input_scaling: 입력 스케일링 팩터
            initialization: 파라미터 초기화 방법 ('random', 'zeros', 'normal')
        """
        super(QuantumLayer, self).__init__()
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.ansatz_type = ansatz_type
        self.input_scaling = input_scaling
        
        # 파라미터 개수 계산
        if ansatz_type == 'RY' or ansatz_type == 'RX':
            # 레이어당 각 큐비트에 하나씩
            self.n_params = n_qubits * n_layers
        elif ansatz_type == 'full':
            # 레이어당 각 큐비트에 RX, RY (RZ 제외)
            self.n_params = n_qubits * n_layers * 2
        else:
            raise ValueError(f"지원되지 않는 앤사츠 타입: {ansatz_type}")
        
        # 양자 회로 파라미터 (학습 가능한 파라미터)
        self.quantum_weights = nn.Parameter(torch.zeros(self.n_params))
        
        # 파라미터 초기화
        self._initialize_parameters(initialization)
    
    def _initialize_parameters(self, method: str):
        """파라미터 초기화"""
        with torch.no_grad():
            if method == 'random':
                # [0, 2π] 범위에서 랜덤 초기화
                self.quantum_weights.uniform_(0, 2 * math.pi)
            elif method == 'normal':
                # 정규분포로 초기화
                self.quantum_weights.normal_(0, math.pi / 4)
            elif method == 'zeros':
                # 0으로 초기화
                self.quantum_weights.zero_()
            else:
                raise ValueError(f"지원되지 않는 초기화 방법: {method}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파
        
        Args:
            x: 입력 텐서 (batch_size, input_features)
            
        Returns:
            output: 출력 텐서 (batch_size, n_qubits)
        """
        # 입력 크기 확인
        if x.dim() != 2:
            raise ValueError(f"입력은 2D 텐서여야 합니다. 받은 크기: {x.shape}")
        
        batch_size, input_features = x.shape
        
        # 입력 특성이 큐비트 수보다 많으면 잘라내기, 적으면 패딩
        if input_features > self.n_qubits:
            x = x[:, :self.n_qubits]
        elif input_features < self.n_qubits:
            # 0으로 패딩
            padding = torch.zeros(batch_size, self.n_qubits - input_features, device=x.device)
            x = torch.cat([x, padding], dim=1)
        
        # 입력 스케일링
        x = x * self.input_scaling
        
        # 양자 함수 적용
        output = QuantumFunction.apply(
            x, self.quantum_weights, self.n_qubits, self.n_layers, self.ansatz_type
        )
        
        return output
    
    def extra_repr(self) -> str:
        """레이어 정보 문자열"""
        return f'n_qubits={self.n_qubits}, n_layers={self.n_layers}, ansatz_type={self.ansatz_type}, n_params={self.n_params}'


class HybridNet(nn.Module):
    """
    고전-양자 하이브리드 신경망 예제
    고전층 → 양자층 → 고전층 구조
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_qubits: int,
        n_layers: int,
        output_size: int,
        quantum_ansatz: str = 'RY'
    ):
        super(HybridNet, self).__init__()
        
        # 전처리 고전 층
        self.classical_pre = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_qubits),
            nn.Tanh()  # [-1, 1] 범위로 정규화
        )
        
        # 양자 층
        self.quantum_layer = QuantumLayer(
            n_qubits=n_qubits,
            n_layers=n_layers,
            ansatz_type=quantum_ansatz,
            input_scaling=math.pi  # [-π, π] 범위로 스케일링
        )
        
        # 후처리 고전 층
        self.classical_post = nn.Sequential(
            nn.Linear(n_qubits, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        # 고전 전처리
        x = self.classical_pre(x)
        
        # 양자 처리
        x = self.quantum_layer(x)
        
        # 고전 후처리
        x = self.classical_post(x)
        
        return x


# 편의 함수들
def create_quantum_classifier(input_size: int, n_qubits: int, n_classes: int, **kwargs) -> HybridNet:
    """양자 분류기 생성 편의 함수"""
    return HybridNet(
        input_size=input_size,
        hidden_size=kwargs.get('hidden_size', 16),
        n_qubits=n_qubits,
        n_layers=kwargs.get('n_layers', 2),
        output_size=n_classes,
        quantum_ansatz=kwargs.get('ansatz', 'RY')
    )

def create_quantum_regressor(input_size: int, n_qubits: int, **kwargs) -> HybridNet:
    """양자 회귀기 생성 편의 함수"""
    return HybridNet(
        input_size=input_size,
        hidden_size=kwargs.get('hidden_size', 16),
        n_qubits=n_qubits,
        n_layers=kwargs.get('n_layers', 2),
        output_size=1,
        quantum_ansatz=kwargs.get('ansatz', 'RY')
    )


# 테스트 코드
if __name__ == "__main__":
    print("QuantumLayer 테스트")
    
    # 기본 테스트
    torch.manual_seed(42)
    
    # 양자 레이어 생성
    qlayer = QuantumLayer(n_qubits=4, n_layers=2, ansatz_type='RY')
    print(f"QuantumLayer 생성: {qlayer}")
    
    # 테스트 입력
    batch_size = 3
    input_features = 4
    x = torch.randn(batch_size, input_features)
    print(f"입력 크기: {x.shape}")
    
    # 순전파 테스트
    try:
        output = qlayer(x)
        print(f"순전파 성공: {x.shape} → {output.shape}")
        print(f"출력 예시: {output[0]}")
    except Exception as e:
        print(f"순전파 실패: {e}")
    
    # 하이브리드 네트워크 테스트
    try:
        hybrid_net = create_quantum_classifier(input_size=4, n_qubits=3, n_classes=2)
        output = hybrid_net(x)
        print(f"하이브리드 네트워크 성공: {x.shape} → {output.shape}")
    except Exception as e:
        print(f"하이브리드 네트워크 실패: {e}")
    
    # 역전파 테스트
    try:
        output = qlayer(x)
        loss = torch.sum(output)
        loss.backward()
        print(f"역전파 성공: 그래디언트 크기 {qlayer.quantum_weights.grad.norm().item():.4f}")
    except Exception as e:
        print(f"역전파 실패: {e}")
    
    print("QuantumLayer 기본 테스트 완료!")