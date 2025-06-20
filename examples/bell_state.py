"""
PyQuantum 예제: 벨 상태 생성 및 측정
양자 얽힘의 가장 기본적인 예제
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyquantum import QuantumCircuit, create_bell_circuit, QubitState
from pyquantum import H, X, CNOT
import torch

def main():
    print("PyQuantum 벨 상태 생성 예제")
    print("=" * 50)
    
    # 방법 1: 직접 회로 구성
    print("1. 직접 회로 구성으로 벨 상태 만들기")
    print("-" * 30)
    
    # 2큐비트 회로 생성
    qc = QuantumCircuit(2)
    print(f"초기 상태: {qc.get_state()}")
    
    # Hadamard + CNOT으로 벨 상태 생성
    qc.H(0).CNOT(0, 1)
    
    print(f"벨 상태: {qc.get_state()}")
    print(f"확률 분포: {qc.get_probabilities()}")
    
    # 방법 2: 편의 함수 사용
    print("\n2. 편의 함수로 벨 상태 만들기")
    print("-" * 30)
    
    bell_circuit = create_bell_circuit()
    print(f"벨 상태: {bell_circuit.get_state()}")
    
    # 방법 3: 다양한 벨 상태들
    print("\n3. 4가지 벨 상태 비교")
    print("-" * 30)
    
    bell_states = {
        "Φ+ (phi_plus)": create_bell_circuit(),
        "Φ- (phi_minus)": QuantumCircuit(2).H(0).Z(1).CNOT(0, 1),
        "Ψ+ (psi_plus)": QuantumCircuit(2).H(0).CNOT(0, 1).X(1),
        "Ψ- (psi_minus)": QuantumCircuit(2).H(0).X(1).CNOT(0, 1).X(1)
    }
    
    for name, circuit in bell_states.items():
        print(f"{name:15}: {circuit.get_state()}")
    
    # 4. 양자 얽힘 확인 - 측정 통계
    print("\n4. 벨 상태 측정 통계 (양자 얽힘 확인)")
    print("-" * 30)
    
    bell_circuit = create_bell_circuit()
    
    # 1000번 측정
    print("1000번 측정 결과:")
    counts = bell_circuit.sample(shots=1000)
    
    for outcome, count in sorted(counts.items()):
        percentage = count / 1000 * 100
        bar = "█" * int(percentage // 2)
        print(f"|{outcome}⟩: {count:4d}회 ({percentage:5.1f}%) {bar}")
    
    # 5. 개별 큐비트 측정으로 얽힘 확인
    print("\n5. 개별 큐비트 측정으로 얽힘 확인")
    print("-" * 30)
    
    # 첫 번째 큐비트만 측정
    bell_circuit_copy = create_bell_circuit()
    result_0, state_after = bell_circuit_copy.get_state().measure(0)
    
    print(f"첫 번째 큐비트 측정 결과: {result_0}")
    print(f"측정 후 상태: {state_after}")
    
    # 두 번째 큐비트의 확률 분포
    prob_1 = state_after.probability(1)
    print(f"두 번째 큐비트 확률: |0⟩={prob_1[0]:.3f}, |1⟩={prob_1[1]:.3f}")
    
    if result_0 == 0:
        print("→ 첫 번째가 |0⟩이면 두 번째도 확실히 |0⟩! (완벽한 상관관계)")
    else:
        print("→ 첫 번째가 |1⟩이면 두 번째도 확실히 |1⟩! (완벽한 상관관계)")
    
    # 6. 벨 부등식 위반 시뮬레이션 (간단 버전)
    print("\n6. 벨 부등식 위반 시뮬레이션")
    print("-" * 30)
    
    def measure_correlation(angle1, angle2, shots=1000):
        """두 각도에서 측정했을 때 상관관계 계산"""
        correlations = []
        
        for _ in range(shots):
            # 새로운 벨 상태 생성
            qc = create_bell_circuit()
            
            # 각도에 따른 측정 기저 변경 (간단한 구현)
            qc.Ry(0, angle1).Ry(1, angle2)
            
            # 측정
            results, _ = qc.get_state().measure()
            
            # 상관관계: 같으면 +1, 다르면 -1
            correlation = 1 if results[0] == results[1] else -1
            correlations.append(correlation)
        
        return sum(correlations) / len(correlations)
    
    # CHSH 부등식 테스트
    angles = [0, torch.pi/4, torch.pi/2, 3*torch.pi/4]
    
    print("CHSH 부등식 테스트:")
    E_00 = measure_correlation(angles[0], angles[0])  # 0, 0
    E_01 = measure_correlation(angles[0], angles[1])  # 0, π/4  
    E_10 = measure_correlation(angles[1], angles[0])  # π/4, 0
    E_11 = measure_correlation(angles[1], angles[1])  # π/4, π/4
    
    S = abs(E_00 + E_01 + E_10 - E_11)
    
    print(f"E(0°,0°) = {E_00:.3f}")
    print(f"E(0°,45°) = {E_01:.3f}")  
    print(f"E(45°,0°) = {E_10:.3f}")
    print(f"E(45°,45°) = {E_11:.3f}")
    print(f"S = |E₀₀ + E₀₁ + E₁₀ - E₁₁| = {S:.3f}")
    
    if S > 2.0:
        print("벨 부등식 위반! (S > 2) → 양자역학적 얽힘 확인!")
    else:
        print("벨 부등식 만족 (S ≤ 2) → 고전적 상관관계")
    
    print(f"이론적 최대값: S = 2√2 ≈ {2 * (2**0.5):.3f}")
    
    # 7. 충실도 및 기타 특성 분석
    print("\n7. 벨 상태 특성 분석")
    print("-" * 30)
    
    bell = create_bell_circuit()
    state = bell.get_state()
    
    # 이론적 벨 상태와 비교
    from pyquantum import bell_state
    theoretical_bell = bell_state("phi_plus")
    
    fidelity = state.fidelity(theoretical_bell)
    print(f"이론적 벨 상태와의 충실도: {fidelity:.6f}")
    
    # 엔트로피 (순수 상태이므로 0)
    entropy = state.entropy()
    print(f"폰 노이만 엔트로피: {entropy:.6f}")
    
    # 확률 분포 분석
    probs = state.probability()
    print(f"확률 분포: {probs}")
    print(f"최대 확률: {torch.max(probs):.6f}")
    print(f"확률 엔트로피: {-torch.sum(probs * torch.log2(probs + 1e-12)):.6f} bits")
    
    # 8. 회로 정보 및 성능 측정
    print("\n8. 회로 정보 및 성능")
    print("-" * 30)
    
    bell_circuit = create_bell_circuit()
    print(f"회로 깊이: {bell_circuit.depth()}")
    print(f"게이트 개수: {bell_circuit.count_gates()}")
    
    # 성능 측정
    import time
    
    start_time = time.time()
    for _ in range(1000):
        qc = QuantumCircuit(2)
        qc.H(0).CNOT(0, 1)
        _ = qc.get_probabilities()
    end_time = time.time()
    
    print(f"1000회 벨 상태 생성+확률계산 시간: {(end_time - start_time)*1000:.2f}ms")
    print(f"평균 시간: {(end_time - start_time):.6f}ms per circuit")
    
    # GPU vs CPU 비교 (가능한 경우)
    if torch.cuda.is_available():
        print("\nGPU 가속 성능:")
        
        # CPU 버전
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(1000):
            qc = QuantumCircuit(2)
            qc.H(0).CNOT(0, 1)
            _ = qc.get_probabilities().cpu()
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time
        
        print(f"GPU 가속 1000회 실행: {gpu_time*1000:.2f}ms")
    
    print("\n벨 상태 예제 완료!")
    print("핵심 포인트:")
    print("   • 벨 상태는 H + CNOT으로 생성")
    print("   • 측정 시 완벽한 상관관계 (00 또는 11만 나옴)")  
    print("   • 벨 부등식 위반으로 양자 얽힘 확인")
    print("   • PyQuantum으로 쉽게 양자 얽힘 실험!")


def interactive_demo():
    """인터랙티브 벨 상태 실험"""
    print("\n인터랙티브 벨 상태 실험")
    print("=" * 30)
    
    while True:
        print("\n실험을 선택하세요:")
        print("1. 벨 상태 생성 및 측정")
        print("2. 다른 각도에서 측정")
        print("3. 다른 벨 상태 비교")
        print("4. 사용자 정의 회로")
        print("5. 종료")
        
        try:
            choice = input("\n선택 (1-5): ").strip()
            
            if choice == '1':
                shots = int(input("측정 횟수 (기본 1000): ") or "1000")
                bell = create_bell_circuit()
                counts = bell.sample(shots=shots)
                
                print(f"\n{shots}번 측정 결과:")
                for outcome, count in sorted(counts.items()):
                    print(f"|{outcome}⟩: {count}회 ({count/shots*100:.1f}%)")
            
            elif choice == '2':
                angle1 = float(input("첫 번째 큐비트 회전 각도 (라디안, 기본 0): ") or "0")
                angle2 = float(input("두 번째 큐비트 회전 각도 (라디안, 기본 0): ") or "0")
                
                qc = create_bell_circuit()
                qc.Ry(0, angle1).Ry(1, angle2)
                
                counts = qc.sample(shots=1000)
                print(f"\n각도 ({angle1:.2f}, {angle2:.2f})에서 측정:")
                for outcome, count in sorted(counts.items()):
                    print(f"|{outcome}⟩: {count}회 ({count/1000*100:.1f}%)")
            
            elif choice == '3':
                print("\n4가지 벨 상태 중 선택:")
                print("1. Φ+ = (|00⟩ + |11⟩)/√2")
                print("2. Φ- = (|00⟩ - |11⟩)/√2") 
                print("3. Ψ+ = (|01⟩ + |10⟩)/√2")
                print("4. Ψ- = (|01⟩ - |10⟩)/√2")
                
                bell_choice = input("벨 상태 선택 (1-4): ").strip()
                
                if bell_choice == '1':
                    qc = create_bell_circuit()
                elif bell_choice == '2':
                    qc = QuantumCircuit(2).H(0).Z(1).CNOT(0, 1)
                elif bell_choice == '3':
                    qc = QuantumCircuit(2).H(0).CNOT(0, 1).X(1)
                elif bell_choice == '4':
                    qc = QuantumCircuit(2).H(0).X(1).CNOT(0, 1).X(1)
                else:
                    print("잘못된 선택입니다.")
                    continue
                
                print(f"상태: {qc.get_state()}")
                counts = qc.sample(shots=1000)
                for outcome, count in sorted(counts.items()):
                    print(f"|{outcome}⟩: {count}회")
            
            elif choice == '4':
                print("\n사용자 정의 2큐비트 회로 (예: H 0, CNOT 0 1, X 1)")
                commands = input("명령어들 (스페이스로 구분): ").strip().split()
                
                qc = QuantumCircuit(2)
                i = 0
                while i < len(commands):
                    cmd = commands[i].upper()
                    
                    if cmd == 'H' and i + 1 < len(commands):
                        qc.H(int(commands[i + 1]))
                        i += 2
                    elif cmd == 'X' and i + 1 < len(commands):
                        qc.X(int(commands[i + 1]))
                        i += 2
                    elif cmd == 'Y' and i + 1 < len(commands):
                        qc.Y(int(commands[i + 1]))
                        i += 2
                    elif cmd == 'Z' and i + 1 < len(commands):
                        qc.Z(int(commands[i + 1]))
                        i += 2
                    elif cmd == 'CNOT' and i + 2 < len(commands):
                        qc.CNOT(int(commands[i + 1]), int(commands[i + 2]))
                        i += 3
                    else:
                        print(f"알 수 없는 명령어: {cmd}")
                        i += 1
                
                print(f"최종 상태: {qc.get_state()}")
                counts = qc.sample(shots=1000)
                for outcome, count in sorted(counts.items()):
                    print(f"|{outcome}⟩: {count}회")
            
            elif choice == '5':
                print("실험을 종료합니다.")
                break
                
            else:
                print("잘못된 선택입니다.")
                
        except KeyboardInterrupt:
            print("\n\n실험을 종료합니다.")
            break
        except Exception as e:
            print(f"오류가 발생했습니다: {e}")


if __name__ == "__main__":
    main()
    
    # 인터랙티브 모드 실행 여부 확인  
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_demo()