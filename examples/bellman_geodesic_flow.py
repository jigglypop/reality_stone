#!/usr/bin/env python3
"""
Bellman-Lagrangian Geodesic Flow Example

벨만 가치 함수와 라그랑지안 에너지를 사용한 표현 흐름 예제
"""

import numpy as np
import reality_stone as rs
from reality_stone.utils.plotting import plot_energy_history

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Note: matplotlib not available, skipping visualizations")

def simulate_environment(state, action):
    """간단한 환경 시뮬레이션"""
    # 목표: 원점으로 이동
    next_state = state + action * 0.1
    distance_to_goal = np.linalg.norm(next_state)
    reward = -distance_to_goal  # 원점에 가까울수록 높은 보상
    done = distance_to_goal < 0.1
    return next_state, reward, done

def main():
    print("=" * 70)
    print("Reality Stone - Bellman-Lagrangian Geodesic Flow Example")
    print("=" * 70)
    
    # 1. 벨만 활성화된 레이어 생성
    print("\n1. Creating layer with Bellman value function...")
    layer = rs.UnifiedRiemannianLayer(
        metric_type="diagonal",  # 학습 가능한 메트릭
        curvature=0.0,
        input_dim=64,
        enable_bellman=True,
        gamma=0.99
    )
    print("   -> Layer created with Bellman enabled")
    print(f"   Discount factor (gamma): {0.99}")
    
    # 2. 초기 상태 생성
    print("\n2. Initializing states...")
    initial_state = np.random.randn(1, 64).astype(np.float32) * 0.5
    print(f"   Initial state norm: {np.linalg.norm(initial_state):.4f}")
    
    # 3. 상태-행동-보상 시퀀스 시뮬레이션
    print("\n3. Simulating episodes...")
    num_episodes = 10
    max_steps = 50
    
    history = {
        'states': [],
        'rewards': [],
        'energies': {
            'kinetic': [],
            'potential': [],
            'lagrangian': [],
            'bellman_residual': []
        }
    }
    
    for episode in range(num_episodes):
        state = np.random.randn(1, 64).astype(np.float32) * 0.5
        episode_reward = 0
        
        print(f"\n   Episode {episode + 1}/{num_episodes}")
        
        for t in range(max_steps):
            # 표현 흐름으로 다음 상태 예측
            next_state = layer.flow_step(state, num_steps=5, learning_rate=0.01)
            
            # 행동 계산
            action = (next_state - state).flatten()
            
            # 환경과 상호작용
            next_state_env, reward, done = simulate_environment(state.flatten(), action)
            next_state_env = next_state_env.reshape(1, -1)
            
            # 에너지 계산
            velocity = (next_state_env - state) / 0.1
            energy = layer.compute_energy(
                state,
                velocity,
                next_state_env,
                np.array([reward])
            )
            
            # 가치 함수 업데이트
            layer.update_value_function(
                state,
                next_state_env,
                np.array([reward]),
                learning_rate=0.01
            )
            
            # 메트릭 업데이트
            layer.update_metric(state, velocity, learning_rate=0.001)
            
            # 기록
            history['states'].append(state.copy())
            history['rewards'].append(reward)
            history['energies']['kinetic'].append(energy['kinetic'][0])
            history['energies']['potential'].append(energy['potential'][0])
            history['energies']['lagrangian'].append(energy['lagrangian'][0])
            history['energies']['bellman_residual'].append(energy['bellman_residual'][0])
            
            episode_reward += reward
            state = next_state_env
            
            if done:
                print(f"      Goal reached at step {t}!")
                break
        
        print(f"      Total reward: {episode_reward:.4f}")
    
    # 4. 결과 분석
    print("\n4. Analyzing results...")
    print(f"   Total steps: {len(history['rewards'])}")
    print(f"   Average reward: {np.mean(history['rewards']):.4f}")
    print(f"   Average Lagrangian: {np.mean(history['energies']['lagrangian']):.4f}")
    print(f"   Final Bellman residual: {history['energies']['bellman_residual'][-1]:.4f}")
    
    # 5. 시각화
    print("\n5. Visualizing energy evolution...")
    
    plot_energy_history(history['energies'], 'bellman_energy_evolution.png')
    
    # 6. 학습된 가치 함수 테스트
    print("\n6. Testing learned value function...")
    test_states = np.random.randn(100, 64).astype(np.float32) * 0.5
    
    values = []
    for state in test_states:
        # 더미 next_state와 reward로 에너지 계산
        next_state = state.reshape(1, -1)
        velocity = np.zeros_like(next_state)
        energy = layer.compute_energy(
            state.reshape(1, -1),
            velocity,
            next_state,
            np.array([0.0], dtype=np.float32)
        )
        # 벨만 잔차가 작을수록 가치가 정확
        values.append(-energy['bellman_residual'][0])
    
    # 상태 norm과 가치의 관계
    norms = np.linalg.norm(test_states, axis=1)
    
    if HAS_MATPLOTLIB:
        plt.figure(figsize=(10, 6))
        plt.scatter(norms, values, alpha=0.5)
        plt.xlabel('State Norm')
        plt.ylabel('Estimated Value')
        plt.title('Value Function: States closer to origin should have higher value')
        plt.grid(True, alpha=0.3)
        plt.savefig('value_function_test.png', dpi=150, bbox_inches='tight')
        print("   -> Saved plot to 'value_function_test.png'")
    else:
        print("   -> Skipping value function plot (matplotlib not available)")
    
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("Generated plots:")
    print("  - bellman_energy_evolution.png")
    print("  - value_function_test.png")
    print("=" * 70)

if __name__ == "__main__":
    main()

