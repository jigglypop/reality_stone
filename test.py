import torch
import riemannian_manifold
import matplotlib.pyplot as plt
import numpy as np

def test_poincare_exp_log_maps():
    # 원점에서의 지수 맵과 로그 맵 테스트
    x = torch.zeros(1, 2)  # 원점
    v = torch.tensor([[0.3, 0.4]])  # 접벡터
    c = 1.0  # 곡률
    
    # 지수 맵 테스트
    y = riemannian_manifold.exp_map(x, v, c)
    print("원점으로부터의 지수 맵 결과:", y)
    
    # 로그 맵 테스트 (역방향)
    v_recovered = riemannian_manifold.log_map(x, y, c)
    print("원래 접벡터:", v)
    print("복원된 접벡터:", v_recovered)
    print("일치 여부:", torch.allclose(v, v_recovered, atol=1e-5))
    
    # 거리 테스트
    dist = riemannian_manifold.distance(x, y, c)
    print("거리:", dist.item())
    
def test_butterfly_transform():
    # 버터플라이 변환 테스트
    x = torch.randn(8)  # 8차원 입력
    params = torch.tensor([0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1])  # 파라미터
    
    # 첫 번째 레이어 (L=0)
    y0 = riemannian_manifold.butterfly_transform(x, params, 0)
    print("버터플라이 레이어 0 결과:", y0)
    
    # 두 번째 레이어 (L=1)
    y1 = riemannian_manifold.butterfly_transform(y0, params, 1)
    print("버터플라이 레이어 1 결과:", y1)
    
    # 세 번째 레이어 (L=2)
    y2 = riemannian_manifold.butterfly_transform(y1, params, 2)
    print("버터플라이 레이어 2 결과:", y2)

def test_hyper_butterfly_layer():
    # Hyper-Butterfly 레이어 테스트
    dim = 8
    num_layers = 3
    layer = riemannian_manifold.HyperButterflyLayer(dim, num_layers, curvature=0.5)
    
    # 포인카레 볼 내의 입력점 생성
    x = torch.randn(8) * 0.3  # 반지름이 작은 점들
    
    # 순전파 실행
    y = layer(x)
    print("입력:", x)
    print("출력:", y)
    
    # 하이퍼볼릭 거리 계산
    dist = riemannian_manifold.distance(torch.zeros_like(x), y, c=0.5)
    print("원점으로부터의 거리:", dist.item())

def visualize_poincare_disc():
    # 2D 포인카레 디스크 시각화
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 단위 원 그리기
    circle = plt.Circle((0, 0), 1, fill=False, color='black')
    ax.add_patch(circle)
    
    # 원점에서 시작하는 여러 접벡터 생성
    origin = torch.zeros(1, 2)
    angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
    magnitudes = [0.5, 1.0, 1.5]
    
    colors = ['red', 'blue', 'green']
    
    # 각 접벡터에 대해 지수 맵 적용 및 시각화
    for mag_idx, magnitude in enumerate(magnitudes):
        for angle in angles:
            v = torch.tensor([[magnitude * np.cos(angle), magnitude * np.sin(angle)]])
            y = riemannian_manifold.exp_map(origin, v)
            
            # 원점과 매핑된 점 사이에 선 그리기
            ax.plot([0, y[0, 0].item()], [0, y[0, 1].item()], 
                   color=colors[mag_idx], alpha=0.7)
            ax.scatter(y[0, 0].item(), y[0, 1].item(), 
                      color=colors[mag_idx], s=30)
    
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.set_title('Poincaré Disc: Exponential Map from Origin')
    ax.grid(True)
    
    plt.savefig('poincare_disc.png')
    print("포인카레 디스크 시각화가 'poincare_disc.png'로 저장되었습니다.")

if __name__ == "__main__":
    print("1. 포인카레 디스크 지수 및 로그 맵 테스트")
    test_poincare_exp_log_maps()
    
    print("\n2. 버터플라이 변환 테스트")
    test_butterfly_transform()
    
    print("\n3. Hyper-Butterfly 레이어 테스트")
    test_hyper_butterfly_layer()
    
    print("\n4. 포인카레 디스크 시각화")
    visualize_poincare_disc()
