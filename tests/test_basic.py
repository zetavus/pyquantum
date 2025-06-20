"""
PyQuantum 기본 테스트 (최종 수정 버전)
GPU/CPU 디바이스 불일치 문제 해결
"""

import sys
import os
import traceback
import torch
import numpy as np

# PyQuantum 패키지 임포트 시도
try:
    from pyquantum import (
        QuantumCircuit, QubitState, 
        create_bell_circuit, create_ghz_circuit,
        zero_state, one_state, plus_state, bell_state,
        test_installation
    )
    PYQUANTUM_AVAILABLE = True
except ImportError as e:
    print(f"PyQuantum 임포트 실패: {e}")
    print("패키지 구조를 확인해주세요.")
    PYQUANTUM_AVAILABLE = False

def get_device():
    """현재 PyQuantum이 사용하는 디바이스 반환"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_tensor_on_device(data, dtype=torch.complex64):
    """현재 디바이스에 맞는 텐서 생성"""
    device = get_device()
    return torch.tensor(data, dtype=dtype, device=device)

def test_qubit_states():
    """큐비트 상태 테스트"""
    print("\n큐비트 상태 테스트")
    print("-" * 25)
    
    try:
        # |0⟩ 상태 테스트 - 같은 디바이스에서 비교
        zero = zero_state(1)
        expected_zero = create_tensor_on_device([1.0, 0.0])
        assert torch.allclose(zero.state, expected_zero)
        print("OK |0⟩ 상태 생성")
        
        # |1⟩ 상태 테스트  
        one = one_state(1)
        expected_one = create_tensor_on_device([0.0, 1.0])
        assert torch.allclose(one.state, expected_one)
        print("OK |1⟩ 상태 생성")
        
        # |+⟩ 상태 테스트
        plus = plus_state(1)
        expected_plus = create_tensor_on_device([1.0, 1.0]) / np.sqrt(2)
        assert torch.allclose(plus.state, expected_plus, atol=1e-6)
        print("OK |+⟩ 상태 생성")
        
        # 벨 상태 테스트
        bell = bell_state("phi_plus")
        expected_bell = create_tensor_on_device([1.0, 0.0, 0.0, 1.0]) / np.sqrt(2)
        assert torch.allclose(bell.state, expected_bell, atol=1e-6)
        print("OK 벨 상태 생성")
        
        # 확률 계산 테스트 - 확률은 항상 CPU로 이동
        probs = bell.probability()
        expected_probs = torch.tensor([0.5, 0.0, 0.0, 0.5], dtype=torch.float32)
        # GPU 텐서를 CPU로 이동해서 비교
        probs_cpu = probs.cpu() if probs.is_cuda else probs
        assert torch.allclose(probs_cpu, expected_probs, atol=1e-6)
        print("OK 확률 계산")
        
        return True
        
    except Exception as e:
        print(f"FAIL 큐비트 상태 테스트 실패: {e}")
        traceback.print_exc()
        return False

def test_single_qubit_gates():
    """단일 큐비트 게이트 테스트"""
    print("\n단일 큐비트 게이트 테스트")
    print("-" * 30)
    
    try:
        # Hadamard 게이트 테스트
        qc = QuantumCircuit(1)
        qc.H(0)
        state = qc.get_state()
        expected = create_tensor_on_device([1.0, 1.0]) / np.sqrt(2)
        assert torch.allclose(state.state, expected, atol=1e-6)
        print("OK Hadamard 게이트")
        
        # Pauli-X 게이트 테스트
        qc = QuantumCircuit(1)
        qc.X(0)
        state = qc.get_state()
        expected = create_tensor_on_device([0.0, 1.0])
        assert torch.allclose(state.state, expected)
        print("OK Pauli-X 게이트")
        
        # Pauli-Y 게이트 테스트
        qc = QuantumCircuit(1)
        qc.Y(0)
        state = qc.get_state()
        expected = create_tensor_on_device([0.0, 1j])
        assert torch.allclose(state.state, expected)
        print("OK Pauli-Y 게이트")
        
        # Pauli-Z 게이트 테스트
        qc = QuantumCircuit(1)
        qc.H(0).Z(0)  # |+⟩에 Z 적용 → |-⟩
        state = qc.get_state()
        expected = create_tensor_on_device([1.0, -1.0]) / np.sqrt(2)
        assert torch.allclose(state.state, expected, atol=1e-6)
        print("OK Pauli-Z 게이트")
        
        # 회전 게이트 테스트
        qc = QuantumCircuit(1)
        qc.Rx(0, np.pi)  # π 회전 = X 게이트
        state = qc.get_state()
        expected = create_tensor_on_device([0.0, -1j])  # RX(π)|0⟩ = -i|1⟩
        assert torch.allclose(state.state, expected, atol=1e-6)
        print("OK 회전 게이트 (Rx)")
        
        return True
        
    except Exception as e:
        print(f"FAIL 단일 큐비트 게이트 테스트 실패: {e}")
        traceback.print_exc()
        return False

def test_two_qubit_gates():
    """2큐비트 게이트 테스트"""
    print("\n2큐비트 게이트 테스트")
    print("-" * 25)
    
    try:
        # CNOT 게이트 테스트
        qc = QuantumCircuit(2)
        qc.X(0).CNOT(0, 1)  # |10⟩ → |11⟩
        state = qc.get_state()
        expected = create_tensor_on_device([0.0, 0.0, 0.0, 1.0])  # |11⟩
        assert torch.allclose(state.state, expected)
        print("OK CNOT 게이트")
        
        # 벨 상태 생성 테스트
        bell_qc = create_bell_circuit()
        bell_state = bell_qc.get_state()
        expected = create_tensor_on_device([1.0, 0.0, 0.0, 1.0]) / np.sqrt(2)
        assert torch.allclose(bell_state.state, expected, atol=1e-6)
        print("OK 벨 상태 생성 회로")
        
        # CZ 게이트 테스트
        qc = QuantumCircuit(2)
        qc.H(0).H(1).CZ(0, 1)  # |++⟩에 CZ 적용
        state = qc.get_state()
        # CZ|++⟩ = (|00⟩ + |01⟩ + |10⟩ - |11⟩)/2
        expected = create_tensor_on_device([1.0, 1.0, 1.0, -1.0]) / 2.0
        assert torch.allclose(state.state, expected, atol=1e-6)
        print("OK CZ 게이트")
        
        return True
        
    except Exception as e:
        print(f"FAIL 2큐비트 게이트 테스트 실패: {e}")
        traceback.print_exc()
        return False

def test_measurement():
    """측정 테스트"""
    print("\n측정 테스트")
    print("-" * 15)
    
    try:
        # 확정적 측정 테스트
        qc = QuantumCircuit(1)
        qc.X(0)  # |1⟩ 상태
        result, final_state = qc.get_state().measure(0)
        assert result == 1
        expected_final = create_tensor_on_device([0.0, 1.0])
        assert torch.allclose(final_state.state, expected_final)
        print("OK 확정적 측정")
        
        # 확률적 측정 테스트 (여러 번)
        bell_qc = create_bell_circuit()
        counts = bell_qc.sample(shots=1000)
        
        # 벨 상태는 00과 11만 나와야 함
        assert set(counts.keys()).issubset({'00', '11'})
        
        # 대략 50:50 비율 (통계적 오차 고려)
        total_shots = sum(counts.values())
        assert total_shots == 1000
        
        # 각각 최소 150회 이상은 나와야 함 (통계적 변동 고려)
        for outcome in counts:
            assert counts[outcome] >= 150, f"{outcome}: {counts[outcome]} (너무 적음)"
        
        print("OK 확률적 측정 (벨 상태)")
        print(f"   측정 결과: {counts}")
        
        return True
        
    except Exception as e:
        print(f"FAIL 측정 테스트 실패: {e}")
        traceback.print_exc()
        return False

def test_circuit_operations():
    """회로 연산 테스트"""
    print("\n회로 연산 테스트")
    print("-" * 20)
    
    try:
        # 체이닝 API 테스트
        qc = QuantumCircuit(3)
        qc.H(0).CNOT(0, 1).CNOT(1, 2)  # GHZ 상태
        
        # 연산 기록 확인
        assert len(qc.operations) == 3
        assert qc.operations[0][0] == "H"
        assert qc.operations[1][0] == "CNOT"
        assert qc.operations[2][0] == "CNOT"
        print("OK 체이닝 API")
        
        # 회로 정보 확인
        depth = qc.depth()
        assert depth == 3
        print(f"OK 회로 깊이: {depth}")
        
        gate_counts = qc.count_gates()
        assert gate_counts["H"] == 1
        assert gate_counts["CNOT"] == 2
        print(f"OK 게이트 개수: {gate_counts}")
        
        # 회로 복사 테스트
        qc_copy = qc.copy()
        assert len(qc_copy.operations) == len(qc.operations)
        assert torch.allclose(qc_copy.get_state().state, qc.get_state().state)
        print("OK 회로 복사")
        
        # 회로 리셋 테스트
        qc.reset()
        assert len(qc.operations) == 0
        expected_reset = create_tensor_on_device([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        assert torch.allclose(qc.get_state().state, expected_reset)
        print("OK 회로 리셋")
        
        return True
        
    except Exception as e:
        print(f"FAIL 회로 연산 테스트 실패: {e}")
        traceback.print_exc()
        return False

def test_error_handling():
    """오류 처리 테스트"""
    print("\n오류 처리 테스트")
    print("-" * 20)
    
    try:
        # 잘못된 큐비트 인덱스
        qc = QuantumCircuit(2)
        try:
            qc.H(5)  # 존재하지 않는 큐비트
            assert False, "오류가 발생해야 함"
        except ValueError:
            print("OK 큐비트 인덱스 검증")
        
        # 같은 큐비트에 CNOT
        try:
            qc.CNOT(0, 0)  # 같은 큐비트
            assert False, "오류가 발생해야 함"
        except ValueError:
            print("OK CNOT 큐비트 검증")
        
        # 0개 큐비트 회로
        try:
            QuantumCircuit(0)
            assert False, "오류가 발생해야 함"
        except ValueError:
            print("OK 큐비트 개수 검증")
        
        # 음수 측정 횟수
        try:
            qc.sample(shots=-1)
            assert False, "오류가 발생해야 함"
        except ValueError:
            print("OK 측정 횟수 검증")
        
        return True
        
    except Exception as e:
        print(f"FAIL 오류 처리 테스트 실패: {e}")
        traceback.print_exc()
        return False

def test_convenience_functions():
    """편의 함수 테스트"""
    print("\n편의 함수 테스트")
    print("-" * 20)
    
    try:
        # 벨 상태 생성 함수
        bell = create_bell_circuit()
        assert bell.n_qubits == 2
        assert len(bell.operations) == 2  # H + CNOT
        print("OK create_bell_circuit()")
        
        # GHZ 상태 생성 함수
        ghz = create_ghz_circuit(3)
        assert ghz.n_qubits == 3
        assert len(ghz.operations) == 3  # H + CNOT + CNOT
        print("OK create_ghz_circuit()")
        
        # 중첩 상태 생성 함수
        try:
            from pyquantum.circuit import create_superposition_circuit
            sup = create_superposition_circuit(2)
            assert sup.n_qubits == 2
            assert len(sup.operations) == 2  # H + H
            print("OK create_superposition_circuit()")
        except ImportError:
            print("WARNING create_superposition_circuit() 함수를 찾을 수 없음 (선택사항)")
        
        return True
        
    except Exception as e:
        print(f"FAIL 편의 함수 테스트 실패: {e}")
        traceback.print_exc()
        return False

def test_performance():
    """성능 테스트"""
    print("\n성능 테스트")
    print("-" * 15)
    
    try:
        import time
        
        # 큰 회로 생성 시간 측정
        start_time = time.time()
        for _ in range(100):
            qc = QuantumCircuit(4)
            qc.H(0).CNOT(0, 1).CNOT(1, 2).CNOT(2, 3)
            _ = qc.get_probabilities()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100 * 1000  # ms
        print(f"OK 4큐비트 회로 평균 처리 시간: {avg_time:.2f}ms")
        
        # 큰 측정 시간 측정
        bell = create_bell_circuit()
        start_time = time.time()
        counts = bell.sample(shots=3000)  # 측정 횟수 더 줄여서 시간 단축
        end_time = time.time()
        
        measurement_time = (end_time - start_time) * 1000  # ms
        print(f"OK 3000회 측정 시간: {measurement_time:.2f}ms")
        
        # GPU 테스트 (가능한 경우)
        if torch.cuda.is_available():
            print("GPU CUDA GPU 감지됨")
            # GPU 메모리 사용량 체크
            if hasattr(torch.cuda, 'memory_allocated'):
                memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
                print(f"MEMORY GPU 메모리 사용량: {memory_mb:.1f}MB")
        else:
            print("CPU CPU 모드")
        
        return True
        
    except Exception as e:
        print(f"FAIL 성능 테스트 실패: {e}")
        traceback.print_exc()
        return False

def test_data_types():
    """데이터 타입 테스트"""
    print("\n데이터 타입 테스트")
    print("-" * 20)
    
    try:
        # 상태 벡터는 복소수여야 함
        qc = QuantumCircuit(1)
        state = qc.get_state()
        assert state.state.dtype == torch.complex64
        print("OK 상태 벡터 타입: complex64")
        
        # 확률은 실수여야 함
        probs = qc.get_probabilities()
        assert probs.dtype in [torch.float32, torch.float64]
        print("OK 확률 타입: float")
        
        # 디바이스 일관성 확인
        device = get_device()
        assert state.state.device.type == device.type
        print(f"OK 디바이스 일관성: {device}")
        
        # GPU 텐서 확인 (CUDA 사용 가능한 경우)
        if torch.cuda.is_available():
            assert state.state.is_cuda
            print("OK GPU 텐서 확인")
        else:
            assert not state.state.is_cuda
            print("OK CPU 텐서 확인")
        
        return True
        
    except Exception as e:
        print(f"FAIL 데이터 타입 테스트 실패: {e}")
        traceback.print_exc()
        return False

def test_device_compatibility():
    """디바이스 호환성 테스트 (추가)"""
    print("\n디바이스 호환성 테스트")
    print("-" * 25)
    
    try:
        device = get_device()
        print(f"현재 디바이스: {device}")
        
        # 여러 큐비트 수에서 디바이스 일관성 확인
        for n_qubits in [1, 2, 3, 4]:
            qc = QuantumCircuit(n_qubits)
            qc.H(0)
            state = qc.get_state()
            assert state.state.device.type == device.type
            
        print("OK 다양한 큐비트 수에서 디바이스 일관성")
        
        # 벨 상태에서 디바이스 일관성
        bell = create_bell_circuit()
        bell_state = bell.get_state()
        assert bell_state.state.device.type == device.type
        print("OK 벨 상태 디바이스 일관성")
        
        # 측정 후에도 디바이스 유지
        result, measured_state = bell_state.measure(0)
        assert measured_state.state.device.type == device.type
        print("OK 측정 후 디바이스 일관성")
        
        return True
        
    except Exception as e:
        print(f"FAIL 디바이스 호환성 테스트 실패: {e}")
        traceback.print_exc()
        return False

def run_all_tests():
    """모든 테스트 실행"""
    print("PyQuantum Phase 1 전체 테스트")
    print("=" * 40)
    
    if not PYQUANTUM_AVAILABLE:
        print("FAIL PyQuantum을 임포트할 수 없습니다.")
        print("INFO 파일 구조와 설치 상태를 확인해주세요.")
        return False
    
    # 설치 테스트 먼저 실행
    print("INFO 설치 상태 확인:")
    install_ok = test_installation()
    
    if not install_ok:
        print("FAIL 설치 상태에 문제가 있습니다.")
        return False
    
    # 개별 테스트들
    tests = [
        ("데이터 타입", test_data_types),
        ("디바이스 호환성", test_device_compatibility),  # 새로 추가
        ("큐비트 상태", test_qubit_states),
        ("단일 큐비트 게이트", test_single_qubit_gates),
        ("2큐비트 게이트", test_two_qubit_gates),
        ("측정", test_measurement),
        ("회로 연산", test_circuit_operations),
        ("오류 처리", test_error_handling),
        ("편의 함수", test_convenience_functions),
        ("성능", test_performance),
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
            print(f"FAIL {test_name} 테스트 중 예외 발생: {e}")
            failed += 1
    
    print(f"\n테스트 결과")
    print("=" * 20)
    print(f"PASS 통과: {passed}")
    print(f"FAIL 실패: {failed}")
    print(f"RATE 성공률: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("\nSUCCESS 모든 테스트 통과! Phase 1 완료!")
        print("GPU PyQuantum이 GPU에서 완벽하게 작동합니다!")
        print("NEXT 다음 단계: Phase 2 (PyTorch nn.Module 통합) 또는 커뮤니티 공유")
        return True
    else:
        print(f"\nERROR {failed}개 테스트 실패. 문제를 수정해주세요.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)