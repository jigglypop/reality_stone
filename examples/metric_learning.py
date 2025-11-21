#!/usr/bin/env python3
"""
Metric Learning Example

학습 가능한 대각 메트릭을 사용한 예제
"""

import numpy as np
import reality_stone as rs
import matplotlib.pyplot as plt

def generate_hierarchical_data(num_samples=1000, dim=128, num_classes=10):
    """계층적 구조를 가진 데이터 생성"""
    data = []
    labels = []
    
    for cls in range(num_classes):
        # 각 클래스는 원점 근처의 특정 방향에 위치
        angle = 2 * np.pi * cls / num_classes
        center = np.zeros(dim)
        center[0] = 0.5 * np.cos(angle)
        center[1] = 0.5 * np.sin(angle)
        
        # 가우시안 노이즈
        samples = center + np.random.randn(num_samples // num_classes, dim).astype(np.float32) * 0.1
        data.append(samples)
        labels.extend([cls] * (num_samples // num_classes))
    
    data = np.vstack(data).astype(np.float32)
    labels = np.array(labels)
    
    return data, labels

def main():
    print("=" * 70)
    print("Reality Stone - Metric Learning Example")
    print("=" * 70)
    
    # 1. 데이터 생성
    print("\n1. Generating hierarchical data...")
    X_train, y_train = generate_hierarchical_data(num_samples=1000, dim=128, num_classes=10)
    X_test, y_test = generate_hierarchical_data(num_samples=200, dim=128, num_classes=10)
    
    print(f"   Training set: {X_train.shape}")
    print(f"   Test set: {X_test.shape}")
    print(f"   Classes: {np.unique(y_train)}")
    
    # 2. 학습 가능한 대각 메트릭 레이어 생성
    print("\n2. Creating learnable diagonal metric layer...")
    layer = rs.UnifiedRiemannianLayer(
        metric_type="diagonal",
        curvature=0.0,
        input_dim=128,
        enable_bellman=True,
        gamma=0.95
    )
    print("   -> Layer created with learnable metric")
    
    # 3. 학습
    print("\n3. Training metric...")
    num_epochs = 50
    batch_size = 32
    learning_rate_metric = 0.001
    learning_rate_value = 0.01
    
    history = {
        'lagrangian': [],
        'kinetic': [],
        'potential': []
    }
    
    print("   Epoch | Lagrangian | Kinetic | Potential")
    print("   " + "-" * 50)
    
    for epoch in range(num_epochs):
        epoch_lagrangian = []
        epoch_kinetic = []
        epoch_potential = []
        
        # 배치 학습
        indices = np.random.permutation(len(X_train))
        for i in range(0, len(X_train), batch_size):
            batch_idx = indices[i:i+batch_size]
            x_batch = X_train[batch_idx]
            
            # 타겟: 같은 클래스의 다른 샘플
            y_batch_idx = []
            for idx in batch_idx:
                same_class = np.where(y_train == y_train[idx])[0]
                y_idx = np.random.choice(same_class)
                y_batch_idx.append(y_idx)
            y_batch = X_train[y_batch_idx]
            
            # 순전파
            output, energy = layer.forward(x_batch, target=y_batch)
            
            if energy is not None:
                # 에너지 기록
                epoch_lagrangian.append(energy['lagrangian'].mean())
                epoch_kinetic.append(energy['kinetic'].mean())
                epoch_potential.append(energy['potential'].mean())
                
                # 메트릭 학습
                velocity = (output - x_batch) / 0.1
                layer.update_metric(x_batch, velocity, learning_rate_metric)
                
                # 가치 함수 학습
                rewards = -np.linalg.norm(output - y_batch, axis=1)  # 목표에 가까울수록 높은 보상
                layer.update_value_function(
                    x_batch,
                    output,
                    rewards,
                    learning_rate_value
                )
        
        # 에폭 통계
        if epoch % 10 == 0:
            mean_lagrangian = np.mean(epoch_lagrangian)
            mean_kinetic = np.mean(epoch_kinetic)
            mean_potential = np.mean(epoch_potential)
            
            history['lagrangian'].append(mean_lagrangian)
            history['kinetic'].append(mean_kinetic)
            history['potential'].append(mean_potential)
            
            print(f"   {epoch:5d} | {mean_lagrangian:10.4f} | {mean_kinetic:7.4f} | {mean_potential:9.4f}")
    
    print("   " + "-" * 50)
    print("   -> Training completed")
    
    # 4. 평가
    print("\n4. Evaluating learned metric...")
    
    # 클래스 내 거리 vs 클래스 간 거리
    intra_class_dist = []
    inter_class_dist = []
    
    for cls in range(10):
        class_samples = X_test[y_test == cls][:10]
        
        # 클래스 내 거리
        for i in range(len(class_samples)):
            for j in range(i + 1, len(class_samples)):
                dist = rs.geodesic_distance(
                    class_samples[i:i+1],
                    class_samples[j:j+1],
                    "diagonal",
                    0.0
                )
                intra_class_dist.append(dist[0])
        
        # 클래스 간 거리
        other_class = (cls + 5) % 10
        other_samples = X_test[y_test == other_class][:10]
        for i in range(len(class_samples)):
            for j in range(len(other_samples)):
                dist = rs.geodesic_distance(
                    class_samples[i:i+1],
                    other_samples[j:j+1],
                    "diagonal",
                    0.0
                )
                inter_class_dist.append(dist[0])
    
    print(f"   Intra-class distance: {np.mean(intra_class_dist):.4f} ± {np.std(intra_class_dist):.4f}")
    print(f"   Inter-class distance: {np.mean(inter_class_dist):.4f} ± {np.std(inter_class_dist):.4f}")
    print(f"   Separation ratio: {np.mean(inter_class_dist) / np.mean(intra_class_dist):.2f}x")
    
    # 5. 시각화
    print("\n5. Visualizing results...")
    
    # 학습 곡선
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(history['lagrangian'], 'g-o')
    axes[0].set_title('Lagrangian Evolution')
    axes[0].set_xlabel('Epoch (x10)')
    axes[0].set_ylabel('L = T - V')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history['kinetic'], 'b-o')
    axes[1].set_title('Kinetic Energy Evolution')
    axes[1].set_xlabel('Epoch (x10)')
    axes[1].set_ylabel('T')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(history['potential'], 'r-o')
    axes[2].set_title('Potential Energy Evolution')
    axes[2].set_xlabel('Epoch (x10)')
    axes[2].set_ylabel('V')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('metric_learning_curves.png', dpi=150, bbox_inches='tight')
    print("   -> Saved learning curves to 'metric_learning_curves.png'")
    
    # 거리 분포
    plt.figure(figsize=(10, 6))
    plt.hist(intra_class_dist, bins=30, alpha=0.6, label='Intra-class', color='blue')
    plt.hist(inter_class_dist, bins=30, alpha=0.6, label='Inter-class', color='red')
    plt.xlabel('Geodesic Distance')
    plt.ylabel('Frequency')
    plt.title('Distance Distribution: Intra-class vs Inter-class')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('distance_distribution.png', dpi=150, bbox_inches='tight')
    print("   -> Saved distance distribution to 'distance_distribution.png'")
    
    # 2D 임베딩 시각화 (PCA)
    print("\n6. Visualizing embeddings (PCA)...")
    from sklearn.decomposition import PCA
    
    # 학습된 메트릭으로 임베딩
    embeddings, _ = layer.forward(X_test)
    
    # PCA로 2D 투영
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=y_test, cmap='tab10', alpha=0.6, s=50)
    plt.colorbar(scatter, label='Class')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Learned Embeddings (2D PCA projection)')
    plt.grid(True, alpha=0.3)
    plt.savefig('embeddings_2d.png', dpi=150, bbox_inches='tight')
    print("   -> Saved embeddings visualization to 'embeddings_2d.png'")
    
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("Generated plots:")
    print("  - metric_learning_curves.png")
    print("  - distance_distribution.png")
    print("  - embeddings_2d.png")
    print("=" * 70)

if __name__ == "__main__":
    try:
        from sklearn.decomposition import PCA
        main()
    except ImportError:
        print("Error: scikit-learn is required for this example")
        print("Install it with: pip install scikit-learn")

