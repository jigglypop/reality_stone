#!/usr/bin/env python3
"""
Reality Stone Python Bindings 확인 스크립트

이 스크립트는 Rust에서 Python으로 바인딩된 모든 함수와 클래스가
올바르게 노출되었는지 확인합니다.
"""

import sys
from typing import List, Tuple

def check_rust_extension():
    """Rust 확장 모듈 로드 확인"""
    try:
        import reality_stone as rs
        print("✓ reality_stone 모듈 로드 성공")
        print(f"  - Rust extension: {rs._has_rust_ext}")
        print(f"  - CUDA support: {rs._has_cuda}")
        return rs._has_rust_ext, rs._has_cuda
    except Exception as e:
        print(f"✗ reality_stone 모듈 로드 실패: {e}")
        return False, False

def check_module_functions(module, module_name: str, expected_functions: List[str]) -> Tuple[int, int]:
    """모듈의 함수 존재 여부 확인"""
    print(f"\n[{module_name}]")
    found = 0
    missing = []
    
    for func_name in expected_functions:
        if hasattr(module, func_name):
            print(f"  ✓ {func_name}")
            found += 1
        else:
            print(f"  ✗ {func_name} (누락)")
            missing.append(func_name)
    
    print(f"  → {found}/{len(expected_functions)} 함수 확인됨")
    if missing:
        print(f"  → 누락된 함수: {', '.join(missing)}")
    
    return found, len(expected_functions)

def main():
    print("=" * 60)
    print("Reality Stone Python Bindings 확인")
    print("=" * 60)
    
    has_rust, has_cuda = check_rust_extension()
    
    if not has_rust:
        print("\n✗ Rust 확장이 로드되지 않았습니다.")
        print("  다음 명령으로 빌드하세요:")
        print("  uv run maturin develop --features cuda")
        sys.exit(1)
    
    from reality_stone import _rust
    
    total_found = 0
    total_expected = 0
    
    # 1. Mobius Operations (루트 레벨)
    mobius_functions = [
        'mobius_add_cpu',
        'mobius_scalar_cpu',
        'mobius_add_dynamic_cpu',
        'mobius_add_dynamic_backward_cpu',
        'mobius_add_layerwise_cpu',
        'mobius_add_layerwise_backward_cpu',
    ]
    
    if has_cuda:
        mobius_functions.extend([
            'mobius_add_cuda',
            'mobius_scalar_cuda',
        ])
    
    found, expected = check_module_functions(_rust, "Mobius Operations", mobius_functions)
    total_found += found
    total_expected += expected
    
    # 2. Poincaré Operations (서브모듈)
    poincare_functions = [
        'poincare_distance_cpu',
        'poincare_to_lorentz_cpu',
        'poincare_to_klein_cpu',
        'poincare_ball_layer_cpu',
        'poincare_exp_at_cpu',
        'poincare_log_at_cpu',
        'poincare_ball_layer_backward_cpu',
        'mobius_add_vjp_cpu',
        'mobius_scalar_vjp_cpu',
        'project_to_ball_cpu',
        'poincare_ball_layer_dynamic_cpu',
        'poincare_ball_layer_dynamic_backward_cpu',
        'poincare_ball_layer_layerwise_cpu',
        'poincare_ball_layer_layerwise_backward_cpu',
    ]
    
    if has_cuda:
        poincare_functions.extend([
            'poincare_distance_cuda',
            'poincare_ball_layer_cuda',
            'poincare_ball_layer_backward_cuda',
        ])
    
    found, expected = check_module_functions(_rust.poincare, "Poincaré Operations", poincare_functions)
    total_found += found
    total_expected += expected
    
    # 3. CUDA 심볼 (루트 레벨 재노출)
    if has_cuda:
        cuda_root_symbols = [
            'poincare_distance_cuda',
            'poincare_ball_layer_cuda',
            'poincare_ball_layer_backward_cuda',
        ]
        found, expected = check_module_functions(_rust, "CUDA Symbols (Root)", cuda_root_symbols)
        total_found += found
        total_expected += expected
    
    # 4. Lorentz Operations
    lorentz_functions = [
        'lorentz_add',
        'lorentz_scalar',
        'lorentz_distance',
        'lorentz_inner',
        'lorentz_to_poincare',
        'lorentz_to_klein',
        'lorentz_layer_forward',
        'lorentz_ball_layer_backward_cpu',
        'lorentz_layer_dynamic_cpu',
        'lorentz_layer_dynamic_backward_cpu',
        'lorentz_layer_layerwise_cpu',
        'lorentz_layer_layerwise_backward_cpu',
        'from_poincare_dynamic_cpu',
        'from_poincare_dynamic_backward_cpu',
    ]
    
    if has_cuda:
        lorentz_functions.extend([
            'lorentz_distance_cuda',
            'lorentz_layer_forward_cuda',
            'lorentz_ball_layer_backward_cuda',
        ])
    
    found, expected = check_module_functions(_rust, "Lorentz Operations", lorentz_functions)
    total_found += found
    total_expected += expected
    
    # 5. Klein Operations
    klein_functions = [
        'klein_add',
        'klein_scalar',
        'klein_distance',
        'klein_to_poincare',
        'klein_to_lorentz',
        'klein_layer_forward',
        'klein_ball_layer_backward_cpu',
        'from_poincare_dynamic_cpu',
        'from_poincare_dynamic_backward_cpu',
    ]
    
    if has_cuda:
        klein_functions.extend([
            'klein_distance_cuda',
            'klein_layer_forward_cuda',
            'klein_ball_layer_backward_cuda',
        ])
    
    found, expected = check_module_functions(_rust, "Klein Operations", klein_functions)
    total_found += found
    total_expected += expected
    
    # 6. Riemann Operations
    riemann_functions = [
        'riemann_lowrank_forward_cpu',
    ]
    
    found, expected = check_module_functions(_rust, "Riemann Operations", riemann_functions)
    total_found += found
    total_expected += expected
    
    # 7. Spline Layer
    try:
        spline_module = _rust.spline
        spline_items = ['SplineLayer']
        
        if has_cuda:
            spline_items.extend([
                'spline_interpolate_cuda',
                'spline_forward_cuda',
                'spline_backward_cuda',
            ])
        
        found, expected = check_module_functions(spline_module, "Spline Layer", spline_items)
        total_found += found
        total_expected += expected
    except AttributeError:
        print("\n[Spline Layer]")
        print("  ✗ spline 서브모듈을 찾을 수 없습니다")
    
    # 8. Suppression Field
    suppression_functions = [
        'compute_suppression_field',
    ]
    
    found, expected = check_module_functions(_rust, "Suppression Field", suppression_functions)
    total_found += found
    total_expected += expected
    
    # 9. Geodesic Attention
    if has_cuda:
        try:
            geodesic_module = _rust.geodesic
            geodesic_functions = [
                'geodesic_topk_attention',
                'batched_cholesky_cuda',
            ]
            
            found, expected = check_module_functions(geodesic_module, "Geodesic Attention", geodesic_functions)
            total_found += found
            total_expected += expected
        except AttributeError:
            print("\n[Geodesic Attention]")
            print("  ✗ geodesic 서브모듈을 찾을 수 없습니다")
    
    # 10. MetriKey Module
    try:
        metrikey_module = _rust.metrikey
        metrikey_functions = [
            'spd_metric_from_key',
            'metric_factor_cholesky',
            'mahalanobis_distance_sq_g',
            'mahalanobis_distance_sq_l',
            'block_orthogonal_from_key',
            'spd_block_metric_from_key',
            'spd_metric_from_key_weighted',
            'compose_layers_order_preserving',
            'compose_layers_gravity',
            'compose_layers_gravity_f64',
            'apply_linear',
            'apply_linear_f64',
        ]
        
        found, expected = check_module_functions(metrikey_module, "MetriKey Functions", metrikey_functions)
        total_found += found
        total_expected += expected
        
        # MetriKey 클래스
        metrikey_classes = [
            'CollapsedTransformF64',
            'CollapsedTransformF32',
            'CollapsedRunnerF64',
            'CollapsedRunnerF32',
        ]
        
        found, expected = check_module_functions(metrikey_module, "MetriKey Classes", metrikey_classes)
        total_found += found
        total_expected += expected
    except AttributeError:
        print("\n[MetriKey Module]")
        print("  ✗ metrikey 서브모듈을 찾을 수 없습니다")
    
    # 11. Python 레이어 래퍼
    print("\n" + "=" * 60)
    print("Python 레이어 래퍼 확인")
    print("=" * 60)
    
    import reality_stone as rs
    
    python_exports = [
        'MobiusAdd',
        'MobiusScalarMul',
        'PoincareBallLayer',
        'poincare_add',
        'poincare_scalar_mul',
        'poincare_distance',
        'poincare_to_lorentz',
        'poincare_to_klein',
        'poincare_ball_layer',
        'LorentzLayer',
        'lorentz_add',
        'lorentz_scalar_mul',
        'lorentz_distance',
        'lorentz_inner',
        'lorentz_to_poincare',
        'lorentz_to_klein',
        'lorentz_layer',
        'KleinLayer',
        'klein_add',
        'klein_scalar_mul',
        'klein_distance',
        'klein_to_poincare',
        'klein_to_lorentz',
        'klein_layer',
        'SplineLinear',
        'convert_to_hyperbolic',
        'metrikey',
    ]
    
    found, expected = check_module_functions(rs, "Python API", python_exports)
    total_found += found
    total_expected += expected
    
    # 최종 결과
    print("\n" + "=" * 60)
    print("최종 결과")
    print("=" * 60)
    print(f"총 {total_found}/{total_expected} 항목 확인됨")
    
    if total_found == total_expected:
        print("\n✓ 모든 바인딩이 정상적으로 노출되었습니다!")
        return 0
    else:
        print(f"\n✗ {total_expected - total_found}개 항목이 누락되었습니다.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

