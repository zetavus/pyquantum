#!/usr/bin/env python3
"""
PyQuantum Bloch Sphere 시각화 테스트
표준 Bloch sphere 관례 사용:
- X축: |+⟩ ↔ |-⟩ (실수 중첩)
- Y축: |+i⟩ ↔ |-i⟩ (복소수 중첩)  
- Z축: |0⟩ ↔ |1⟩ (계산 기준)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 한글 폰트 설정 (문제 해결)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# PyQuantum 라이브러리 import
try:
    from pyquantum import QuantumCircuit
    print("PyQuantum Bloch Sphere 시각화 테스트")
    print("=" * 60)
except ImportError:
    print("PyQuantum 라이브러리를 찾을 수 없습니다!")
    exit(1)

def qubit_to_bloch_vector(state):
    """
    올바른 Bloch vector 변환 (X-Y 축 문제 해결)
    
    표준 물리학 관례:
    - X축: |+⟩ ↔ |-⟩ (실수 중첩)  
    - Y축: |+i⟩ ↔ |-i⟩ (복소수 중첩)
    - Z축: |0⟩ ↔ |1⟩ (계산 기준)
    """
    if len(state.state) != 2:
        raise ValueError("단일 큐비트 상태만 지원됩니다")
    
    # GPU 텐서를 CPU로 이동
    state_cpu = state.state.cpu() if hasattr(state.state, 'cpu') else state.state
    
    alpha = state_cpu[0]  # |0⟩ 계수
    beta = state_cpu[1]   # |1⟩ 계수
    
    # 표준 Bloch vector 공식
    bloch_x = 2 * (alpha.conj() * beta).real    # 실수 중첩
    bloch_y = 2 * (alpha.conj() * beta).imag    # 복소수 중첩  
    bloch_z = (abs(alpha)**2 - abs(beta)**2)    # 계산 기준
    
    # PyTorch 텐서를 numpy로 변환
    if hasattr(bloch_x, 'detach'):
        bloch_x = bloch_x.detach().numpy()
        bloch_y = bloch_y.detach().numpy() 
        bloch_z = bloch_z.detach().numpy()
    
    return np.array([float(bloch_x), float(bloch_y), float(bloch_z)])

def plot_bloch_sphere(vectors, labels, title="Bloch Sphere"):
    """
    깔끔한 Bloch sphere들을 그리기 (테두리 제거)
    """
    n_vectors = len(vectors)
    
    # 그리드 레이아웃 결정
    if n_vectors <= 2:
        cols = n_vectors
        rows = 1
        figsize = (5 * cols, 5)
    elif n_vectors <= 4:
        cols = 2
        rows = 2
        figsize = (10, 10)
    else:
        cols = 3
        rows = (n_vectors + 2) // 3
        figsize = (15, 5 * rows)
    
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=16, weight='bold')
    
    for i, (vector, label) in enumerate(zip(vectors, labels)):
        ax = fig.add_subplot(rows, cols, i+1, projection='3d')
        
        # Bloch sphere 그리기 (반투명)
        u = np.linspace(0, 2 * np.pi, 25)
        v = np.linspace(0, np.pi, 25)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
        
        ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.15, color='lightgray')
        
        # 주요 축 그리기 (더 굵고 진하게)
        ax.plot([-1.2, 1.2], [0, 0], [0, 0], 'k-', alpha=0.8, linewidth=4)
        ax.plot([0, 0], [-1.2, 1.2], [0, 0], 'k-', alpha=0.8, linewidth=4)
        ax.plot([0, 0], [0, 0], [-1.2, 1.2], 'k-', alpha=0.8, linewidth=4)
        
        # 축 끝 라벨 (더 명확하게)
        ax.text(1.35, 0, 0, 'x', fontsize=16, weight='bold', color='black')
        ax.text(0, 1.35, 0, 'y', fontsize=16, weight='bold', color='black')
        ax.text(0, 0, 1.35, '|0>', fontsize=14, weight='bold', color='blue', ha='center')
        ax.text(0, 0, -1.35, '|1>', fontsize=14, weight='bold', color='blue', ha='center')
        
        # 주요 상태 포인트들 (6개 주요 방향)
        ax.scatter([0, 0, 1, -1, 0, 0], [0, 0, 0, 0, 1, -1], [1, -1, 0, 0, 0, 0], 
                  c=['blue', 'blue', 'green', 'green', 'red', 'red'], s=40, alpha=0.8)
        
        # 상태 벡터 그리기 (굵은 자홍색 화살표)
        ax.quiver(0, 0, 0, vector[0], vector[1], vector[2], 
                 color='magenta', arrow_length_ratio=0.15, linewidth=5)
        
        # 각 subplot 제목
        ax.set_title(f'{label}', fontsize=11, weight='bold', pad=15)
        
        # 축 범위 설정
        ax.set_xlim([-1.4, 1.4])
        ax.set_ylim([-1.4, 1.4])
        ax.set_zlim([-1.4, 1.4])
        
        # 축 라벨, 틱, 격자 모두 제거
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_zlabel('')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.grid(False)
        
        # 축 배경과 테두리 완전 제거
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('none')
        ax.yaxis.pane.set_edgecolor('none')
        ax.zaxis.pane.set_edgecolor('none')
        ax.xaxis.pane.set_alpha(0)
        ax.yaxis.pane.set_alpha(0)
        ax.zaxis.pane.set_alpha(0)
        
        # 뷰 각도 조정 (더 입체적으로)
        ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    return fig

def test_basic_states():
    """기본 계산 상태들 테스트"""
    print("Testing Basic States")
    print("=" * 50)
    
    vectors = []
    labels = []
    
    # |0⟩ state
    qc = QuantumCircuit(1)
    state_0 = qc.get_state()
    vec_0 = qubit_to_bloch_vector(state_0)
    vectors.append(vec_0)
    labels.append("|0⟩")
    print(f"|0⟩ state: {state_0}")
    print(f"Bloch vector: ({vec_0[0]:.3f}, {vec_0[1]:.3f}, {vec_0[2]:.3f})")
    print("   → |0> is at the NORTH POLE (+Z axis)")
    
    # |1⟩ state
    qc = QuantumCircuit(1)
    qc.X(0)
    state_1 = qc.get_state()
    vec_1 = qubit_to_bloch_vector(state_1)
    vectors.append(vec_1)
    labels.append("|1⟩")
    print(f"|1⟩ state: {state_1}")
    print(f"Bloch vector: ({vec_1[0]:.3f}, {vec_1[1]:.3f}, {vec_1[2]:.3f})")
    print("   → |1> is at the SOUTH POLE (-Z axis)")
    
    # 그래프 생성
    plot_bloch_sphere(vectors, labels, "Basic States: |0⟩, |1⟩")
    plt.show()

def test_hadamard_gate():
    """하다마르 게이트 테스트"""
    print("\nTesting Hadamard Gate")
    print("=" * 50)
    
    vectors = []
    labels = []
    
    # H|0⟩ → |+⟩
    qc = QuantumCircuit(1)
    qc.H(0)
    state_plus = qc.get_state()
    vec_plus = qubit_to_bloch_vector(state_plus)
    vectors.append(vec_plus)
    labels.append("H|0⟩ = |+⟩")
    print(f"H|0⟩ = |+⟩ state: {state_plus}")
    print(f"Bloch vector: ({vec_plus[0]:.3f}, {vec_plus[1]:.3f}, {vec_plus[2]:.3f})")
    print("   → Hadamard creates SUPERPOSITION on +Y axis")
    print("   → 50% |0> + 50% |1> probability")
    
    # H|1⟩ → |-⟩
    qc = QuantumCircuit(1)
    qc.X(0).H(0)
    state_minus = qc.get_state()
    vec_minus = qubit_to_bloch_vector(state_minus)
    vectors.append(vec_minus)
    labels.append("H|1⟩ = |-⟩")
    print(f"H|1⟩ = |-⟩ state: {state_minus}")
    print(f"Bloch vector: ({vec_minus[0]:.3f}, {vec_minus[1]:.3f}, {vec_minus[2]:.3f})")
    print("   → Superposition with OPPOSITE PHASE on -Y axis")
    print("   → 50% |0> - 50% |1> (phase difference)")
    
    # 그래프 생성
    plot_bloch_sphere(vectors, labels, "Hadamard Gate: |+⟩, |-⟩ States")
    plt.show()

def test_pauli_gates():
    """파울리 게이트들 테스트"""
    print("\nTesting Pauli Gates")
    print("=" * 50)
    
    vectors = []
    labels = []
    
    # 초기 |+⟩ 상태
    qc = QuantumCircuit(1)
    qc.H(0)
    initial_state = qc.get_state()
    initial_vec = qubit_to_bloch_vector(initial_state)
    vectors.append(initial_vec)
    labels.append("Initial |+⟩")
    print(f"Initial |+⟩ state: {initial_state}")
    print(f"Bloch vector: ({initial_vec[0]:.3f}, {initial_vec[1]:.3f}, {initial_vec[2]:.3f})")
    print("   → Starting with superposition on +X axis")
    
    # X gate: |+⟩ → |+⟩ (no change!)
    qc = QuantumCircuit(1)
    qc.H(0).X(0)
    state_x = qc.get_state()
    vec_x = qubit_to_bloch_vector(state_x)
    vectors.append(vec_x)
    labels.append("X|+⟩ = |+⟩")
    print(f"X|+⟩ state: {state_x}")
    print(f"Bloch vector: ({vec_x[0]:.3f}, {vec_x[1]:.3f}, {vec_x[2]:.3f})")
    print("   → X gate has NO EFFECT on |+⟩!")
    
    # Y gate: |+⟩ → i|-⟩ 
    qc = QuantumCircuit(1)
    qc.H(0).Y(0)
    state_y = qc.get_state()
    vec_y = qubit_to_bloch_vector(state_y)
    vectors.append(vec_y)
    labels.append("Y|+⟩ = i|-⟩")
    print(f"Y|+⟩ state: {state_y}")
    print(f"Bloch vector: ({vec_y[0]:.3f}, {vec_y[1]:.3f}, {vec_y[2]:.3f})")
    print("   → Y gate: 180° rotation around Y-axis")
    
    # Z gate: |+⟩ → |-⟩
    qc = QuantumCircuit(1)
    qc.H(0).Z(0)
    state_z = qc.get_state()
    vec_z = qubit_to_bloch_vector(state_z)
    vectors.append(vec_z)
    labels.append("Z|+⟩ = |-⟩")
    print(f"Z|+⟩ state: {state_z}")
    print(f"Bloch vector: ({vec_z[0]:.3f}, {vec_z[1]:.3f}, {vec_z[2]:.3f})")
    print("   → Z gate: PHASE FLIP, 180° rotation around Z-axis")
    
    # 그래프 생성
    plot_bloch_sphere(vectors, labels, "Pauli Gates: X, Y, Z")
    plt.show()

def test_rotation_gates():
    """회전 게이트들 테스트"""
    print("\nTesting Rotation Gates")
    print("=" * 50)
    
    vectors = []
    labels = []
    
    # 다양한 Rx 회전각들
    angles = [np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
    
    for i, angle in enumerate(angles):
        qc = QuantumCircuit(1)
        qc.Rx(0, angle)
        state = qc.get_state()
        vec = qubit_to_bloch_vector(state)
        vectors.append(vec)
        labels.append(f"Rx({angle:.2f})|0⟩")
        
        print(f"Rx({angle:.2f})|0⟩ state: {state}")
        print(f"Bloch vector: ({vec[0]:.3f}, {vec[1]:.3f}, {vec[2]:.3f})")
        print(f"   → Rx({angle:.2f}) = {angle*180/np.pi:.1f}° rotation around X-axis")
        
        if abs(angle - np.pi/2) < 0.01:
            print("   → π/2 rotation: |0⟩ → |-i⟩ (points to -Y axis)")
        elif abs(angle - np.pi) < 0.01:
            print("   → π rotation: |0⟩ → |1⟩ (same as X gate!)")
    
    # 그래프 생성
    plot_bloch_sphere(vectors, labels, "Rotation Gates: Rx(θ)")
    plt.show()

def main():
    """메인 테스트 함수"""
    print("PyQuantum Bloch Sphere 시각화 테스트")
    print("=" * 60)
    
    # 각 테스트 실행
    test_basic_states()
    test_hadamard_gate()
    test_pauli_gates()
    test_rotation_gates()
    
    print("\nAll tests completed!")
    print("\nSUMMARY - Understanding Quantum Gates on Bloch Sphere:")
    print("=" * 60)
    print("COMPUTATIONAL BASIS:")
    print("   - |0⟩: North pole (0, 0, 1) - 'spin up'")
    print("   - |1⟩: South pole (0, 0, -1) - 'spin down'")
    print("\nSUPERPOSITION BASIS (X-axis):")
    print("   - |+⟩: +X axis (1, 0, 0) - equal superposition |0⟩ + |1⟩")
    print("   - |-⟩: -X axis (-1, 0, 0) - opposite phase |0⟩ - |1⟩")
    print("\nCOMPLEX PHASE BASIS (Y-axis):")
    print("   - |+i⟩: +Y axis (0, 1, 0) - complex superposition |0⟩ + i|1⟩")
    print("   - |-i⟩: -Y axis (0, -1, 0) - complex superposition |0⟩ - i|1⟩")
    print("\nGATE EFFECTS:")
    print("   - H gate: Creates/destroys superposition (Z ↔ X axis)")
    print("   - X gate: Bit flip - rotates 180° around X-axis")
    print("   - Y gate: Bit + phase flip - rotates 180° around Y-axis") 
    print("   - Z gate: Phase flip only - rotates 180° around Z-axis")
    print("   - Rx/Ry/Rz: Smooth rotations around respective axes")
    print("\nKEY INSIGHT:")
    print("   Any qubit state = point on Bloch sphere")
    print("   Any quantum gate = rotation of the sphere!")
    print("   Measurement = projection onto computational basis (Z-axis)")

if __name__ == "__main__":
    main()