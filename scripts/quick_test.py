"""빠른 빌드 검증 테스트"""
import torch
import reality_stone as rs

def test_basic():
    """기본 동작 테스트"""
    print("Reality Stone 기본 테스트")
    print("=" * 50)
    
    batch, dim = 4, 8
    c, t = 0.1, 0.5
    
    # 입력 생성
    u = torch.randn(batch, dim) * 0.3
    v = torch.randn(batch, dim) * 0.3
    
    print(f"Input: batch={batch}, dim={dim}, c={c}, t={t}")
    
    # Klein
    try:
        z_klein = rs.klein_layer(u, v, c, t)
        print(f"✓ Klein:    {z_klein.shape} - OK")
    except Exception as e:
        print(f"✗ Klein:    FAILED - {e}")
    
    # Lorentz
    try:
        u_lor = torch.cat([torch.ones(batch, 1), u], dim=1)
        v_lor = torch.cat([torch.ones(batch, 1), v], dim=1)
        z_lor = rs.lorentz_layer(u_lor, v_lor, c, t)
        print(f"✓ Lorentz:  {z_lor.shape} - OK")
    except Exception as e:
        print(f"✗ Lorentz:  FAILED - {e}")
    
    # Poincaré
    try:
        z_poin = rs.poincare_ball_layer(u, v, c, t)
        print(f"✓ Poincaré: {z_poin.shape} - OK")
    except Exception as e:
        print(f"✗ Poincaré: FAILED - {e}")
    
    # Distance 테스트
    print("\nDistance 함수:")
    try:
        d_klein = rs.klein_distance(u, v, c)
        print(f"✓ Klein distance:    {d_klein.shape}")
    except Exception as e:
        print(f"✗ Klein distance:    FAILED - {e}")
    
    try:
        d_lor = rs.lorentz_distance(u_lor, v_lor, c)
        print(f"✓ Lorentz distance:  {d_lor.shape}")
    except Exception as e:
        print(f"✗ Lorentz distance:  FAILED - {e}")
    
    try:
        d_poin = rs.poincare_distance(u, v, c)
        print(f"✓ Poincaré distance: {d_poin.shape}")
    except Exception as e:
        print(f"✗ Poincaré distance: FAILED - {e}")
    
    print("\n" + "=" * 50)
    print("테스트 완료!")

if __name__ == "__main__":
    test_basic()

