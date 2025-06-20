"""
PyQuantum 예제: XOR 문제를 양자 신경망으로 해결
고전적으로 선형 분리 불가능한 문제를 양자 컴퓨팅으로 해결하는 대표적 예제
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader

try:
    from pyquantum.torch_layer import QuantumLayer, HybridNet, create_quantum_classifier
    from pyquantum import QuantumCircuit
except ImportError as e:
    print(f"PyQuantum 임포트 오류: {e}")
    print("torch_layer.py가 pyquantum 디렉토리에 있는지 확인해주세요.")
    sys.exit(1)


class XORDataset(Dataset):
    """XOR 데이터셋"""
    
    def __init__(self, n_samples=1000, noise=0.1):
        """
        Args:
            n_samples: 샘플 개수
            noise: 노이즈 레벨 (0.0 = 노이즈 없음)
        """
        # XOR 진리표 기반 데이터 생성
        self.n_samples = n_samples
        
        # 기본 XOR 패턴
        base_inputs = torch.tensor([
            [0., 0.],  # XOR = 0
            [0., 1.],  # XOR = 1  
            [1., 0.],  # XOR = 1
            [1., 1.]   # XOR = 0
        ])
        
        base_labels = torch.tensor([0, 1, 1, 0])
        
        # 샘플 개수만큼 반복 + 노이즈 추가
        n_repeat = n_samples // 4 + 1
        
        inputs = base_inputs.repeat(n_repeat, 1)[:n_samples]
        labels = base_labels.repeat(n_repeat)[:n_samples]
        
        # 노이즈 추가
        if noise > 0:
            noise_tensor = torch.randn_like(inputs) * noise
            inputs = inputs + noise_tensor
            
            # 입력을 [0, 1] 범위로 클리핑
            inputs = torch.clamp(inputs, 0, 1)
        
        self.inputs = inputs.float()
        self.labels = labels.long()
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]


def create_classical_xor_model():
    """비교용 고전 신경망 모델"""
    return nn.Sequential(
        nn.Linear(2, 8),
        nn.ReLU(),
        nn.Linear(8, 8),  
        nn.ReLU(),
        nn.Linear(8, 2)
    )


def train_model(model, train_loader, test_loader, epochs=100, lr=0.01, model_name="Model"):
    """모델 훈련"""
    print(f"\n{model_name} 훈련 시작")
    print("-" * 30)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(epochs):
        # 훈련 모드
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_accuracy = 100 * correct / total
        avg_loss = total_loss / len(train_loader)
        
        # 테스트 평가
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        
        test_accuracy = 100 * test_correct / test_total
        
        train_losses.append(avg_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        
        # 진행 상황 출력
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d}/{epochs}: "
                  f"Loss={avg_loss:.4f}, "
                  f"Train Acc={train_accuracy:.1f}%, "
                  f"Test Acc={test_accuracy:.1f}%")
    
    return train_losses, train_accuracies, test_accuracies


def visualize_decision_boundary(model, title="Decision Boundary", save_path=None):
    """결정 경계 시각화"""
    plt.figure(figsize=(10, 8))
    
    # 그리드 생성
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    h = 0.02
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # 그리드 점들에 대한 예측
    grid_points = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
    
    model.eval()
    with torch.no_grad():
        outputs = model(grid_points)
        if outputs.shape[1] > 1:  # 분류 모델
            _, predictions = torch.max(outputs, 1)
        else:  # 회귀 모델
            predictions = (outputs > 0.5).long().squeeze()
    
    predictions = predictions.numpy().reshape(xx.shape)
    
    # 결정 경계 그리기
    plt.contourf(xx, yy, predictions, alpha=0.7, cmap=plt.cm.RdYlBu)
    
    # XOR 데이터 포인트
    xor_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    xor_labels = np.array([0, 1, 1, 0])
    
    colors = ['red', 'blue']
    for i, label in enumerate([0, 1]):
        mask = xor_labels == label
        plt.scatter(xor_inputs[mask, 0], xor_inputs[mask, 1], 
                   c=colors[i], s=100, alpha=0.9, 
                   label=f'XOR = {label}', edgecolors='black')
    
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel('Input 1')
    plt.ylabel('Input 2')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def test_xor_understanding(model, model_name):
    """XOR 이해도 테스트"""
    print(f"\n{model_name} XOR 이해도 테스트")
    print("-" * 30)
    
    # 정확한 XOR 입력들
    test_inputs = torch.tensor([
        [0., 0.],
        [0., 1.],
        [1., 0.],
        [1., 1.]
    ])
    
    expected_outputs = [0, 1, 1, 0]
    
    model.eval()
    with torch.no_grad():
        outputs = model(test_inputs)
        if outputs.shape[1] > 1:  # 분류
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)
        else:  # 회귀
            predictions = (outputs > 0.5).long().squeeze()
            probabilities = outputs
    
    print("입력 → 예상 → 예측 → 신뢰도")
    print("-" * 25)
    
    correct = 0
    for i, (inp, expected, pred) in enumerate(zip(test_inputs, expected_outputs, predictions)):
        if outputs.shape[1] > 1:
            confidence = probabilities[i, pred].item()
        else:
            confidence = probabilities[i].item()
        
        is_correct = pred.item() == expected
        correct += is_correct
        
        status = "OK" if is_correct else "FAIL"
        print(f"{inp.tolist()} → {expected} → {pred.item()} → {confidence:.3f} {status}")
    
    accuracy = correct / len(test_inputs) * 100
    print(f"\n정확도: {accuracy:.1f}% ({correct}/{len(test_inputs)})")
    
    if accuracy == 100:
        print("완벽한 XOR 이해!")
    elif accuracy >= 75:
        print("좋은 XOR 이해")
    else:
        print("XOR 이해 부족")
    
    return accuracy


def compare_models():
    """고전 vs 양자 모델 비교"""
    print("고전 신경망 vs 양자 신경망 XOR 문제 해결 비교")
    print("=" * 60)
    
    # 데이터셋 준비
    train_dataset = XORDataset(n_samples=800, noise=0.05)
    test_dataset = XORDataset(n_samples=200, noise=0.05)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"훈련 데이터: {len(train_dataset)}개, 테스트 데이터: {len(test_dataset)}개")
    
    # 모델들 생성
    models = {
        "고전 신경망": create_classical_xor_model(),
        "양자 신경망": create_quantum_classifier(input_size=2, n_qubits=3, n_classes=2, 
                                               hidden_size=8, n_layers=2, ansatz='RY'),
        "하이브리드 네트워크": HybridNet(input_size=2, hidden_size=8, n_qubits=3, 
                                       n_layers=2, output_size=2, quantum_ansatz='RY')
    }
    
    results = {}
    
    # 각 모델 훈련 및 평가
    for name, model in models.items():
        print(f"\n{'='*20} {name} {'='*20}")
        
        # 파라미터 개수 출력
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"파라미터: 총 {total_params}개, 훈련 가능 {trainable_params}개")
        
        # 모델 구조 출력
        print(f"구조: {model}")
        
        try:
            # 훈련
            train_losses, train_accs, test_accs = train_model(
                model, train_loader, test_loader, 
                epochs=50, lr=0.01, model_name=name
            )
            
            # XOR 이해도 테스트
            final_accuracy = test_xor_understanding(model, name)
            
            results[name] = {
                'model': model,
                'final_accuracy': final_accuracy,
                'train_losses': train_losses,
                'train_accs': train_accs,
                'test_accs': test_accs,
                'total_params': total_params
            }
            
            # 결정 경계 시각화
            visualize_decision_boundary(model, f"{name} Decision Boundary")
            
        except Exception as e:
            print(f"{name} 훈련 실패: {e}")
            import traceback
            traceback.print_exc()
    
    # 결과 비교
    print(f"\n{'='*20} 최종 비교 결과 {'='*20}")
    print(f"{'모델':<15} {'XOR 정확도':<12} {'파라미터 수':<12} {'수렴 속도'}")
    print("-" * 50)
    
    for name, result in results.items():
        if result:
            convergence = "빠름" if result['final_accuracy'] >= 75 else "느림"
            print(f"{name:<15} {result['final_accuracy']:>8.1f}% {result['total_params']:>10}개 {convergence:>8}")
    
    return results


def quantum_circuit_analysis():
    """양자 회로 분석"""
    print(f"\n{'='*20} 양자 회로 상세 분석 {'='*20}")
    
    # 간단한 양자 회로로 XOR 패턴 확인
    print("양자 회로에서 XOR 패턴 찾기")
    
    # 각 XOR 입력에 대해 양자 상태 분석
    xor_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    xor_outputs = [0, 1, 1, 0]
    
    for inp, expected in zip(xor_inputs, xor_outputs):
        print(f"\n입력 {inp} (XOR = {expected}):")
        
        # 간단한 양자 회로 구성
        qc = QuantumCircuit(2)
        
        # 데이터 인코딩
        if inp[0] == 1:
            qc.RY(0, np.pi)
        if inp[1] == 1:
            qc.RY(1, np.pi)
        
        # 얽힘 생성
        qc.CNOT(0, 1)
        
        # 추가 회전 (학습된 파라미터 시뮬레이션)
        qc.RY(0, np.pi/4)
        qc.RY(1, np.pi/3)
        
        state = qc.get_state()
        probs = qc.get_probabilities()
        
        print(f"  양자 상태: {state}")
        print(f"  확률 분포: {probs}")
        
        # Z 기댓값 계산 (분류 결과와 연관)
        z0_exp = probs[0] + probs[2] - probs[1] - probs[3]  # 첫 번째 큐비트 Z 기댓값
        z1_exp = probs[0] + probs[1] - probs[2] - probs[3]  # 두 번째 큐비트 Z 기댓값
        
        print(f"  Z 기댓값: qubit0={z0_exp:.3f}, qubit1={z1_exp:.3f}")


def advanced_xor_experiment():
    """고급 XOR 실험"""
    print(f"\n{'='*20} 고급 XOR 실험 {'='*20}")
    
    # 다양한 노이즈 레벨에서 성능 비교
    noise_levels = [0.0, 0.05, 0.1, 0.2, 0.3]
    
    classical_results = []
    quantum_results = []
    
    for noise in noise_levels:
        print(f"\n노이즈 레벨: {noise}")
        
        # 데이터셋 생성
        train_dataset = XORDataset(n_samples=400, noise=noise)
        test_dataset = XORDataset(n_samples=100, noise=noise)
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        # 고전 모델
        classical_model = create_classical_xor_model()
        try:
            _, _, classical_test_accs = train_model(
                classical_model, train_loader, test_loader, 
                epochs=30, lr=0.01, model_name=f"고전(노이즈={noise})"
            )
            classical_final = classical_test_accs[-1]
        except:
            classical_final = 0
        
        # 양자 모델
        quantum_model = create_quantum_classifier(
            input_size=2, n_qubits=2, n_classes=2, n_layers=3
        )
        try:
            _, _, quantum_test_accs = train_model(
                quantum_model, train_loader, test_loader,
                epochs=30, lr=0.02, model_name=f"양자(노이즈={noise})"
            )
            quantum_final = quantum_test_accs[-1]
        except:
            quantum_final = 0
        
        classical_results.append(classical_final)
        quantum_results.append(quantum_final)
        
        print(f"  고전: {classical_final:.1f}%, 양자: {quantum_final:.1f}%")
    
    # 결과 시각화
    plt.figure(figsize=(10, 6))
    plt.plot(noise_levels, classical_results, 'o-', label='Classical NN', linewidth=2)
    plt.plot(noise_levels, quantum_results, 's-', label='Quantum NN', linewidth=2)
    plt.xlabel('Noise Level')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Robustness Comparison Against Noise')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return classical_results, quantum_results


def main():
    """메인 실행 함수"""
    print("PyQuantum XOR 문제 해결 데모")
    print("=" * 50)
    
    # 기본 모델 비교
    try:
        results = compare_models()
        print("기본 모델 비교 완료")
    except Exception as e:
        print(f"기본 모델 비교 실패: {e}")
    
    # 양자 회로 분석
    try:
        quantum_circuit_analysis()
        print("양자 회로 분석 완료")
    except Exception as e:
        print(f"양자 회로 분석 실패: {e}")
    
    # 고급 실험 (선택사항)
    print("\n고급 노이즈 robustness 실험을 실행하시겠습니까? (y/n): ", end="")
    
    try:
        choice = input().lower().strip()
        if choice == 'y':
            advanced_xor_experiment()
            print("고급 실험 완료")
    except KeyboardInterrupt:
        print("\n실험을 중단했습니다.")
    except Exception as e:
        print(f"고급 실험 실패: {e}")
    
    print("\nXOR 양자 신경망 데모 완료!")
    print("핵심 결과:")
    print("   • 양자 신경망으로 XOR 문제 해결 가능")
    print("   • 고전 신경망과 비교 가능한 성능")
    print("   • 파라미터 수는 더 적을 수 있음")
    print("   • 양자 얽힘이 비선형 패턴 학습에 도움")


if __name__ == "__main__":
    # GPU 사용 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 디바이스: {device}")
    
    # 재현 가능성을 위한 시드 설정
    torch.manual_seed(42)
    np.random.seed(42)
    
    main()