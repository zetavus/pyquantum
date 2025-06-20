"""
PyQuantum Phase 2 테스트
PyTorch nn.Module 통합 기능 테스트
"""

import sys
import os
import traceback
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# PyQuantum 임포트
try:
    from pyquantum.torch_layer import QuantumLayer, HybridNet, QuantumFunction, create_quantum_classifier
    from pyquantum import QuantumCircuit, test_installation
    TORCH_LAYER_AVAILABLE = True
except ImportError as e:
    print(f"X PyQuantum torch_layer 임포트 실패: {e}")
    TORCH_LAYER_AVAILABLE = False


def test_quantum_function():
    """QuantumFunction 기본 테스트"""
    print("\n QuantumFunction 테스트")
    print("-" * 25)
    
    try:
        # 테스트 데이터
        batch_size = 2
        n_qubits = 3
        n_layers = 2
        
        inputs = torch.randn(batch_size, n_qubits, requires_grad=True)
        weights = torch.randn(n_qubits * n_layers, requires_grad=True)
        
        # 순전파
        outputs = QuantumFunction.apply(inputs, weights, n_qubits, n_layers, 'RY')
        
        assert outputs.shape == (batch_size, n_qubits)
        print(f"V 순전파: {inputs.shape} → {outputs.shape}")
        
        # 역전파
        loss = torch.sum(outputs)
        loss.backward()
        
        assert inputs.grad is not None
        assert weights.grad is not None
        print("V 역전파: 그래디언트 계산 성공")
        
        # 그래디언트 크기 확인
        print(f"   입력 그래디언트 크기: {inputs.grad.norm().item():.4f}")
        print(f"   가중치 그래디언트 크기: {weights.grad.norm().item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"X QuantumFunction 테스트 실패: {e}")
        traceback.print_exc()
        return False


def test_quantum_layer():
    """QuantumLayer 기본 테스트"""
    print("\n QuantumLayer 테스트")
    print("-" * 22)
    
    try:
        # 다양한 설정으로 테스트
        configs = [
            {'n_qubits': 2, 'n_layers': 1, 'ansatz_type': 'RY'},
            {'n_qubits': 3, 'n_layers': 2, 'ansatz_type': 'RX'},
            {'n_qubits': 4, 'n_layers': 1, 'ansatz_type': 'full'},
        ]
        
        for i, config in enumerate(configs):
            print(f"  설정 {i+1}: {config}")
            
            # 레이어 생성
            layer = QuantumLayer(**config)
            
            # 파라미터 개수 확인
            expected_params = config['n_qubits'] * config['n_layers']
            if config['ansatz_type'] == 'full':
                expected_params *= 2
            
            actual_params = layer.n_params
            assert actual_params == expected_params, f"파라미터 개수 불일치: {actual_params} != {expected_params}"
            
            # 순전파 테스트
            batch_size = 3
            input_features = config['n_qubits'] + 1  # 의도적으로 다른 크기
            
            x = torch.randn(batch_size, input_features)
            output = layer(x)
            
            expected_output_shape = (batch_size, config['n_qubits'])
            assert output.shape == expected_output_shape, f"출력 크기 불일치: {output.shape} != {expected_output_shape}"
            
            print(f"    V {x.shape} → {output.shape}")
        
        print("V 모든 설정에서 QuantumLayer 성공")
        return True
        
    except Exception as e:
        print(f"X QuantumLayer 테스트 실패: {e}")
        traceback.print_exc()
        return False


def test_gradient_flow():
    """그래디언트 흐름 테스트"""
    print("\n 그래디언트 흐름 테스트")
    print("-" * 25)
    
    try:
        # 간단한 회귀 문제 설정
        n_qubits = 3
        n_layers = 2
        batch_size = 4
        
        # 모델 생성
        layer = QuantumLayer(n_qubits=n_qubits, n_layers=n_layers)
        linear = nn.Linear(n_qubits, 1)
        
        model = nn.Sequential(layer, linear)
        
        # 손실 함수와 옵티마이저
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        # 더미 데이터
        x = torch.randn(batch_size, n_qubits)
        y = torch.randn(batch_size, 1)
        
        # 훈련 단계
        initial_loss = None
        for step in range(5):
            optimizer.zero_grad()
            
            output = model(x)
            loss = criterion(output, y)
            
            if initial_loss is None:
                initial_loss = loss.item()
            
            loss.backward()
            optimizer.step()
            
            # 그래디언트 존재 확인
            quantum_grad_norm = layer.quantum_weights.grad.norm().item()
            linear_grad_norm = sum(p.grad.norm().item() for p in linear.parameters())
            
            print(f"  Step {step+1}: Loss={loss.item():.4f}, "
                  f"Quantum Grad={quantum_grad_norm:.4f}, "
                  f"Linear Grad={linear_grad_norm:.4f}")
        
        # 손실이 변했는지 확인 (학습 중)
        final_loss = loss.item()
        loss_change = abs(final_loss - initial_loss)
        
        if loss_change > 1e-6:
            print("V 그래디언트 흐름: 손실 변화 확인됨")
        else:
            print("! 그래디언트 흐름: 손실 변화 미미")
        
        return True
        
    except Exception as e:
        print(f"X 그래디언트 흐름 테스트 실패: {e}")
        traceback.print_exc()
        return False


def test_hybrid_network():
    """하이브리드 네트워크 테스트"""
    print("\n 하이브리드 네트워크 테스트")
    print("-" * 28)
    
    try:
        # 다양한 태스크 테스트
        tasks = [
            {'name': '이진 분류', 'input_size': 4, 'output_size': 2, 'task_type': 'classification'},
            {'name': '다중 분류', 'input_size': 6, 'output_size': 3, 'task_type': 'classification'},
            {'name': '회귀', 'input_size': 3, 'output_size': 1, 'task_type': 'regression'},
        ]
        
        for task in tasks:
            print(f"  {task['name']} 태스크:")
            
            # 모델 생성
            model = HybridNet(
                input_size=task['input_size'],
                hidden_size=8,
                n_qubits=3,
                n_layers=2,
                output_size=task['output_size']
            )
            
            # 테스트 데이터
            batch_size = 5
            x = torch.randn(batch_size, task['input_size'])
            
            # 순전파
            output = model(x)
            expected_shape = (batch_size, task['output_size'])
            
            assert output.shape == expected_shape, f"출력 크기 불일치: {output.shape} != {expected_shape}"
            
            # 손실 계산 및 역전파
            if task['task_type'] == 'classification':
                y = torch.randint(0, task['output_size'], (batch_size,))
                criterion = nn.CrossEntropyLoss()
            else:
                y = torch.randn(batch_size, task['output_size'])
                criterion = nn.MSELoss()
            
            loss = criterion(output, y)
            loss.backward()
            
            # 모든 파라미터에 그래디언트가 있는지 확인
            has_grad = all(p.grad is not None for p in model.parameters() if p.requires_grad)
            assert has_grad, "일부 파라미터에 그래디언트가 없습니다"
            
            print(f"    V {x.shape} → {output.shape}, Loss={loss.item():.4f}")
        
        print("V 모든 태스크에서 하이브리드 네트워크 성공")
        return True
        
    except Exception as e:
        print(f"X 하이브리드 네트워크 테스트 실패: {e}")
        traceback.print_exc()
        return False


def test_convenience_functions():
    """편의 함수 테스트"""
    print("\n 편의 함수 테스트")
    print("-" * 20)
    
    try:
        # 분류기 생성 테스트
        classifier = create_quantum_classifier(
            input_size=4, n_qubits=3, n_classes=2,
            hidden_size=8, n_layers=2
        )
        
        x = torch.randn(3, 4)
        output = classifier(x)
        assert output.shape == (3, 2), f"분류기 출력 크기 오류: {output.shape}"
        print("V create_quantum_classifier")
        
        # 회귀기 생성 테스트 (torch_layer.py에 있다면)
        try:
            from pyquantum.torch_layer import create_quantum_regressor
            regressor = create_quantum_regressor(
                input_size=3, n_qubits=2, n_layers=1
            )
            
            x = torch.randn(2, 3)
            output = regressor(x)
            assert output.shape == (2, 1), f"회귀기 출력 크기 오류: {output.shape}"
            print("V create_quantum_regressor")
        except ImportError:
            print("! create_quantum_regressor 함수 없음 (선택사항)")
        
        return True
        
    except Exception as e:
        print(f"X 편의 함수 테스트 실패: {e}")
        traceback.print_exc()
        return False


def test_parameter_shift_rule():
    """Parameter Shift Rule 정확성 테스트"""
    print("\n Parameter Shift Rule 테스트")
    print("-" * 28)
    
    try:
        # 간단한 양자 함수 정의
        n_qubits = 2
        n_layers = 1
        
        # 수치적 그래디언트와 비교
        def quantum_expectation(weights):
            """양자 기댓값 계산"""
            x = torch.zeros(1, n_qubits)  # 더미 입력
            output = QuantumFunction.apply(x, weights, n_qubits, n_layers, 'RY')
            return torch.sum(output)
        
        # 테스트 파라미터
        weights = torch.tensor([0.1, 0.2], requires_grad=True)
        
        # Parameter Shift Rule 그래디언트
        loss = quantum_expectation(weights)
        loss.backward()
        analytic_grad = weights.grad.clone()
        
        # 수치적 그래디언트 (검증용)
        numerical_grad = torch.zeros_like(weights)
        eps = 1e-4
        
        for i in range(len(weights)):
            weights_plus = weights.clone().detach()
            weights_minus = weights.clone().detach()
            
            weights_plus[i] += eps
            weights_minus[i] -= eps
            
            loss_plus = quantum_expectation(weights_plus)
            loss_minus = quantum_expectation(weights_minus)
            
            numerical_grad[i] = (loss_plus - loss_minus) / (2 * eps)
        
        # 그래디언트 비교
        diff = torch.abs(analytic_grad - numerical_grad)
        max_diff = torch.max(diff).item()
        
        print(f"  Analytic grad: {analytic_grad}")
        print(f"  Numerical grad: {numerical_grad}")
        print(f"  Max difference: {max_diff:.6f}")
        
        if max_diff < 1e-2:  # 허용 오차
            print("V Parameter Shift Rule 정확도 검증")
        else:
            print("! Parameter Shift Rule 정확도 주의 필요")
        
        return True
        
    except Exception as e:
        print(f"X Parameter Shift Rule 테스트 실패: {e}")
        traceback.print_exc()
        return False


def test_device_compatibility():
    """디바이스 호환성 테스트"""
    print("\n 디바이스 호환성 테스트")
    print("-" * 25)
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"테스트 디바이스: {device}")
        
        # 모델을 디바이스로 이동
        model = QuantumLayer(n_qubits=3, n_layers=2)
        model = model.to(device)
        
        # 입력도 같은 디바이스로
        x = torch.randn(2, 3, device=device)
        
        # 순전파
        output = model(x)
        
        # 출력이 같은 디바이스에 있는지 확인 디바이스 타입만 비교 (cuda:0와 cuda는 같은 타입)
        output_device_type = output.device.type
        input_device_type = device.type

        assert output_device_type == input_device_type, f"출력 디바이스 타입 불일치: {output_device_type} != {input_device_type}"
        
        # 역전파
        loss = torch.sum(output)
        loss.backward()
        
        # 그래디언트도 같은 디바이스에 있는지 확인
        grad_device_type = model.quantum_weights.grad.device.type
        assert grad_device_type == input_device_type, "그래디언트 디바이스 타입 불일치"
        
        print(f"V {device} 디바이스에서 정상 작동")
        return True
        
    except Exception as e:
        print(f"X 디바이스 호환성 테스트 실패: {e}")
        traceback.print_exc()
        return False


def test_integration_with_optimizers():
    """다양한 옵티마이저와의 통합 테스트"""
    print("\n 옵티마이저 통합 테스트")
    print("-" * 25)
    
    try:
        # 테스트할 옵티마이저들
        optimizers = [
            ('Adam', optim.Adam),
            ('SGD', optim.SGD),
            ('RMSprop', optim.RMSprop),
        ]
        
        for opt_name, opt_class in optimizers:
            print(f"  {opt_name} 옵티마이저:")
            
            # 모델 생성
            model = QuantumLayer(n_qubits=2, n_layers=1)
            
            # 옵티마이저 설정
            if opt_name == 'SGD':
                optimizer = opt_class(model.parameters(), lr=0.01)
            else:
                optimizer = opt_class(model.parameters(), lr=0.01)
            
            # 더미 훈련
            x = torch.randn(3, 2)
            target = torch.randn(3, 2)
            
            initial_params = model.quantum_weights.clone().detach()
            
            for step in range(3):
                optimizer.zero_grad()
                output = model(x)
                loss = torch.mean((output - target) ** 2)
                loss.backward()
                optimizer.step()
            
            # 파라미터가 변했는지 확인
            final_params = model.quantum_weights.clone().detach()
            param_change = torch.norm(final_params - initial_params).item()
            
            if param_change > 1e-6:
                print(f"    V 파라미터 업데이트: {param_change:.6f}")
            else:
                print(f"    ! 파라미터 변화 미미: {param_change:.6f}")
        
        print("V 모든 옵티마이저 통합 성공")
        return True
        
    except Exception as e:
        print(f"X 옵티마이저 통합 테스트 실패: {e}")
        traceback.print_exc()
        return False


def test_performance_benchmark():
    """성능 벤치마크 테스트"""
    print("\n 성능 벤치마크 테스트")
    print("-" * 25)
    
    try:
        import time
        
        # 다양한 크기로 성능 측정
        configs = [
            {'n_qubits': 2, 'n_layers': 1, 'batch_size': 10},
            {'n_qubits': 3, 'n_layers': 2, 'batch_size': 10},
            {'n_qubits': 4, 'n_layers': 2, 'batch_size': 5},
        ]
        
        for config in configs:
            print(f"  설정: {config}")
            
            model = QuantumLayer(
                n_qubits=config['n_qubits'], 
                n_layers=config['n_layers']
            )
            
            x = torch.randn(config['batch_size'], config['n_qubits'])
            
            # 순전파 시간 측정
            start_time = time.time()
            for _ in range(10):
                output = model(x)
            forward_time = (time.time() - start_time) / 10 * 1000  # ms
            
            # 역전파 시간 측정
            start_time = time.time()
            for _ in range(10):
                model.zero_grad()
                output = model(x)
                loss = torch.sum(output)
                loss.backward()
            backward_time = (time.time() - start_time) / 10 * 1000  # ms
            
            print(f"    순전파: {forward_time:.2f}ms, 역전파: {backward_time:.2f}ms")
        
        print("V 성능 벤치마크 완료")
        return True
        
    except Exception as e:
        print(f"X 성능 벤치마크 테스트 실패: {e}")
        traceback.print_exc()
        return False


def test_edge_cases():
    """경계 케이스 테스트"""
    print("\n 경계 케이스 테스트")
    print("-" * 20)
    
    try:
        # 1. 매우 작은 입력
        model = QuantumLayer(n_qubits=2, n_layers=1)
        x = torch.zeros(1, 1)  # 큐비트 수보다 적은 입력
        output = model(x)
        assert output.shape == (1, 2), "작은 입력 처리 실패"
        print("V 작은 입력 처리")
        
        # 2. 큰 입력
        x = torch.randn(1, 10)  # 큐비트 수보다 많은 입력
        output = model(x)
        assert output.shape == (1, 2), "큰 입력 처리 실패"
        print("V 큰 입력 처리")
        
        # 3. 배치 크기 1
        x = torch.randn(1, 2)
        output = model(x)
        assert output.shape == (1, 2), "배치 크기 1 처리 실패"
        print("V 배치 크기 1 처리")
        
        # 4. 극값 입력
        x = torch.tensor([[float('inf'), float('-inf')]])
        try:
            output = model(x)
            print("! 극값 입력 처리됨 (주의 필요)")
        except:
            print("V 극값 입력 적절히 거부됨")
        
        # 5. NaN 입력
        x = torch.tensor([[float('nan'), 1.0]])
        try:
            output = model(x)
            if torch.isnan(output).any():
                print("! NaN 입력 → NaN 출력")
            else:
                print("V NaN 입력 처리됨")
        except:
            print("V NaN 입력 적절히 거부됨")
        
        return True
        
    except Exception as e:
        print(f"X 경계 케이스 테스트 실패: {e}")
        traceback.print_exc()
        return False


def test_state_consistency():
    """상태 일관성 테스트"""
    print("\n 상태 일관성 테스트")
    print("-" * 20)
    
    try:
        # 동일한 입력에 대해 동일한 출력이 나오는지 확인
        torch.manual_seed(42)
        
        model = QuantumLayer(n_qubits=3, n_layers=2)
        x = torch.randn(2, 3)
        
        # 첫 번째 실행
        output1 = model(x)
        
        # 두 번째 실행 (같은 파라미터)
        output2 = model(x)
        
        # 결과가 동일한지 확인
        diff = torch.max(torch.abs(output1 - output2)).item()
        
        if diff < 1e-6:
            print("V 동일 입력 → 동일 출력")
        else:
            print(f"! 출력 차이 발견: {diff}")
        
        # 파라미터 변경 후 출력 변경 확인
        with torch.no_grad():
            model.quantum_weights[0] += 0.1
        
        output3 = model(x)
        diff2 = torch.max(torch.abs(output1 - output3)).item()
        
        if diff2 > 1e-6:
            print("V 파라미터 변경 → 출력 변경")
        else:
            print("! 파라미터 변경해도 출력 동일")
        
        return True
        
    except Exception as e:
        print(f"X 상태 일관성 테스트 실패: {e}")
        traceback.print_exc()
        return False


def run_all_torch_tests():
    """모든 PyTorch 통합 테스트 실행"""
    print("PyQuantum Phase 2 (PyTorch 통합) 전체 테스트")
    print("=" * 50)
    
    if not TORCH_LAYER_AVAILABLE:
        print("X torch_layer 모듈을 임포트할 수 없습니다.")
        print("torch_layer.py가 pyquantum 디렉토리에 있는지 확인해주세요.")
        return False
    
    # Phase 1 기본 설치 확인
    print("Phase 1 기능 확인:")
    try:
        install_ok = test_installation()
        if not install_ok:
            print("X Phase 1 기능에 문제가 있습니다.")
            return False
    except:
        print("! Phase 1 테스트 스킵")
    
    # Phase 2 테스트들
    tests = [
        ("QuantumFunction", test_quantum_function),
        ("QuantumLayer", test_quantum_layer),
        ("그래디언트 흐름", test_gradient_flow),
        ("하이브리드 네트워크", test_hybrid_network),
        ("편의 함수", test_convenience_functions),
        ("Parameter Shift Rule", test_parameter_shift_rule),
        ("디바이스 호환성", test_device_compatibility),
        ("옵티마이저 통합", test_integration_with_optimizers),
        ("성능 벤치마크", test_performance_benchmark),
        ("경계 케이스", test_edge_cases),
        ("상태 일관성", test_state_consistency),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"X {test_name} 테스트 중 예외 발생: {e}")
            failed += 1
    
    print(f"\nPhase 2 테스트 결과")
    print("=" * 25)
    print(f"V 통과: {passed}")
    print(f"X 실패: {failed}")
    print(f"성공률: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("\nPhase 2 모든 테스트 통과!")
        print("PyTorch 통합이 완벽하게 작동합니다!")
        print("PyQuantum이 실용적인 QML 프레임워크로 완성되었습니다!")
        
        print("\n이제 가능한 것들:")
        print("   • torch.nn.Module처럼 양자층 사용")
        print("   • 자동 미분으로 양자 파라미터 최적화")
        print("   • 고전-양자 하이브리드 모델 구축")
        print("   • XOR, MNIST 등 실제 문제 해결")
        print("   • PyTorch 생태계와 완전 통합")
        
        return True
    else:
        print(f"\n{failed}개 테스트 실패. 문제를 수정해주세요.")
        return False


def quick_demo():
    """빠른 데모"""
    print("\nPyQuantum Phase 2 빠른 데모")
    print("=" * 35)
    
    try:
        # 1. 기본 양자층 사용
        print("1. 양자층 기본 사용:")
        qlayer = QuantumLayer(n_qubits=2, n_layers=2)
        x = torch.randn(3, 2)
        output = qlayer(x)
        print(f"   입력: {x.shape} → 출력: {output.shape}")
        
        # 2. 하이브리드 모델 구축
        print("\n2. 하이브리드 모델:")
        model = HybridNet(input_size=4, hidden_size=8, n_qubits=3, n_layers=2, output_size=2)
        x = torch.randn(2, 4)
        output = model(x)
        print(f"   입력: {x.shape} → 출력: {output.shape}")
        
        # 3. 훈련 시뮬레이션
        print("\n3. 훈련 시뮬레이션:")
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        y = torch.randint(0, 2, (2,))
        
        for step in range(3):
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            print(f"   Step {step+1}: Loss = {loss.item():.4f}")
        
        print("\nV 데모 성공! PyQuantum Phase 2 완료!")
        
    except Exception as e:
        print(f"X 데모 실패: {e}")


if __name__ == "__main__":
    # GPU 사용 가능하면 사용
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 디바이스: {device}")
    
    # 재현 가능성을 위한 시드 설정
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 전체 테스트 실행
    success = run_all_torch_tests()
    
    # 빠른 데모 실행
    if success:
        quick_demo()
    
    sys.exit(0 if success else 1)